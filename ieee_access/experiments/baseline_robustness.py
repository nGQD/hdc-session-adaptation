#!/usr/bin/env python3
"""Reproducible, finite boundary-sensitivity confirmation for gradient baselines.

Original result files are read-only. Checkpoints are required and are never trained
by this runner. Selection uses only the original held-in inner stream and the
three development seeds. Each adaptation run has an explicit RNG reset.
"""
from __future__ import annotations

import argparse
import copy
import random
import shutil
import gc
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import platform
import sys
import time

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / 'mjcs' / 'code'))
import numpy as np
import torch
from scipy import stats
import data
import engine
import run as original

OUT = HERE / 'results' / 'baseline_robustness'
TASKS = ('h1', 'm1', 'm2')
METHODS = tuple(original.GRADIENT) + ('ewc',)
NI_REFS = ('finetune_full', 'replay', 'full_tag', 'readout_tag')
EWC_EPOCHS = (1, 2, 5, 10, 20, 40, 80)
VERSION = 1
_ACTIVE_CONTEXT = None
_FISHER_CACHE = {}
_FEATURE_CACHE = {}


def write(path, value):
    data.write_json(path, value)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical(config):
    return json.dumps(config, sort_keys=True, separators=(',', ':'))


def candidate_id(config):
    return hashlib.sha256(canonical(config).encode()).hexdigest()[:12]


def source_selection(task):
    return json.loads((original.OUT / 'selection' / f'{task}.json').read_text())


def adjacent_lrs(lr):
    if math.isclose(lr, 1e-5):
        return [1e-6, 3e-6]
    if math.isclose(lr, 1e-3):
        return [3e-3]
    return []


def candidates(method, incumbent):
    """Fixed local extension; all candidates frozen before their scores exist."""
    rows = [dict(incumbent)]
    lrs = [incumbent['lr']] + adjacent_lrs(incumbent['lr'])
    if method == 'ewc':
        weights = [incumbent['ewc']]
        if incumbent['ewc'] == 1e6:
            weights += [1e7]
        elif incumbent['ewc'] == 1e2:
            weights += [1e1]
        # Independent epochs and joint local lr/penalty neighborhoods, no inherited
        # epoch choice from FT-full in the confirmation.
        rows += [dict(lr=lr, epochs=e, ewc=w)
                 for lr in lrs for w in weights for e in EWC_EPOCHS]
    else:
        epochs = [incumbent['epochs']]
        if incumbent['epochs'] == 40:
            epochs += [80]
        rows += [dict(lr=lr, epochs=e) for lr in lrs for e in epochs]
    seen, unique = set(), []
    for row in rows:
        k = canonical(row)
        if k not in seen:
            unique.append(dict(id=candidate_id(row), config=row,
                               incumbent=(row == incumbent)))
            seen.add(k)
    return unique


def plan():
    payload = dict(
        version=VERSION,
        design='Fixed boundary-sensitivity confirmation, not an unbounded hyperparameter search.',
        protocol='Original inner held-in stream; no held-out calibration or outer test data are used for selection.',
        dev_seeds=original.DEV_SEEDS, outer_seeds=original.FINAL_SEEDS,
        selection_metric='Mean final stream ACC across the three development seeds.',
        stopping_rule='Run exactly the candidates listed here; stop after this finite confirmation even if a new edge wins. No candidate is chosen using outer outcomes.',
        tie_rule='Highest mean inner ACC; exact ties prefer the original incumbent, then candidate order.',
        rng_rule='Reset Python, NumPy, CPU torch and CUDA torch RNGs to the declared seed immediately before each candidate stream, after loading the checkpoint.',
        historical_note='Legacy run.py reset RNGs during backbone pretraining but not explicitly after checkpoint loading. Original results remain unchanged; fresh incumbent and extension runs use explicit common seeds.',
        selection_implementation='Final-row-only scoring is validated against the original full stream before selection; scoring uses evaluation mode and consumes no RNG.',
        original_files={p.name:digest(p) for p in [REPO/'mjcs/code/engine.py', REPO/'mjcs/code/run.py', REPO/'mjcs/code/model.py', REPO/'mjcs/code/data.py']},
        tasks={})
    for task in TASKS:
        sel = source_selection(task)
        payload['tasks'][task] = dict(
            selection_file_sha256=digest(original.OUT/'selection'/f'{task}.json'),
            methods={m:candidates(m, sel['gradient'][m]) for m in METHODS})
    path = OUT / 'plan.json'
    if path.exists():
        old = json.loads(path.read_text())
        if old != payload:
            raise RuntimeError('Frozen plan differs; do not overwrite an executed plan.')
    else:
        write(path, payload)
    return payload


def load_checkpoint(task, seed, inner):
    c = original.base_config(task)
    name = 'inner_pretrain' if inner else 'outer'
    path = original.OUT/'checkpoints'/task/name/f'full_s{seed}.pt'
    if not path.exists():
        raise FileNotFoundError(f'Required existing checkpoint absent: {path}')
    model, rec, hi, ho, ys = original.checkpoint(task, 'full', c, seed, inner=inner)
    global _ACTIVE_CONTEXT
    _ACTIVE_CONTEXT = (task, seed, inner, digest(path))
    _FISHER_CACHE.clear()
    _FEATURE_CACHE.clear()
    return c, model, rec, hi, ho, ys, path


class CachedInitialFisher(engine.GradientMethod):
    """Reuse only the identical pretrained Fisher, preserving its RNG advance."""
    def start(self, model, held_in, c, seed):
        key = (_ACTIVE_CONTEXT, seed)
        if key not in _FISHER_CACHE:
            super().start(model, held_in, c, seed)
            _FISHER_CACHE[key] = dict(
                fisher=[x.detach().cpu().clone() for x in self.fisher],
                anchor=[x.detach().cpu().clone() for x in self.anchor],
                cpu_rng=torch.get_rng_state().clone(),
                cuda_rng=[x.clone() for x in torch.cuda.get_rng_state_all()] if torch.cuda.is_available() else [],
                numpy_rng=np.random.get_state(), python_rng=random.getstate())
            return
        cached=_FISHER_CACHE[key]
        self.base,self.c,self.pre_macs,self.store=model,c,0,{}
        self.model=copy.deepcopy(model)
        self.params=list(self.model.parameters())
        self.buffer=list(held_in)
        self.fisher=[x.to(engine.DEVICE).clone() for x in cached['fisher']]
        self.anchor=[x.to(engine.DEVICE).clone() for x in cached['anchor']]
        torch.set_rng_state(cached['cpu_rng'])
        if cached['cuda_rng']: torch.cuda.set_rng_state_all(cached['cuda_rng'])
        np.random.set_state(cached['numpy_rng']);random.setstate(cached['python_rng'])


class CachedReadout(engine.ReadoutMethod):
    """Reuse frozen features on bins used by training/scoring, never predictions."""
    def feats(self,s):
        if s.name in self.cache: return self.cache[s.name]
        task,seed,inner,_=_ACTIVE_CONTEXT
        split='inner' if inner else 'outer'
        folder=HERE/'results'/'feature_controls'/'features'/task/split/f's{seed}'
        manifest=folder/'metadata.json'
        if not manifest.exists(): return super().feats(s)
        meta=json.loads(manifest.read_text())
        expected_sha=_ACTIVE_CONTEXT[3]
        cached_sha=meta.get('checkpoint_sha256')
        if cached_sha is not None and cached_sha != expected_sha:
            raise RuntimeError('Feature cache checkpoint does not match loaded checkpoint.')
        entry=next((x for x in meta['sessions'] if x['name']==s.name),None)
        if entry is None: return super().feats(s)
        # Stored caches have only the feature rows used by the loss or metric.
        # All other readout inputs are immaterial and can be zero-filled.
        k=(_ACTIVE_CONTEXT,s.name)
        if k not in _FEATURE_CACHE:
            with np.load(folder/entry['file']) as z:
                f=np.zeros((len(s.x),self.model.features),dtype=np.float32)
                fit=np.flatnonzero(s.fit[:s.fit_end])
                if len(z['fit_gf']):
                    assert len(fit)==len(z['fit_gf'])
                    f[fit]=z['fit_gf'][:,128:]
                score=np.flatnonzero(s.score)
                assert len(score)==len(z['score_gf'])
                f[score]=z['score_gf'][:,128:]
            _FEATURE_CACHE[k]=torch.as_tensor(f,device=engine.DEVICE)
        self.cache[s.name]=_FEATURE_CACHE[k]
        return self.cache[s.name]


def make_method(name, config, native=False):
    if name == 'ewc':
        cls=engine.GradientMethod if native else CachedInitialFisher
        return cls('full', **config)
    if not native and name in ('finetune_readout','readout_tag','last_tag'):
        return CachedReadout('last' if name=='last_tag' else 'readout',
                             tag=name!='finetune_readout',**config)
    return original.GRADIENT[name](config)


def final_score(method, model, hi, ho, config, seed, ystats):
    """Same updates as engine.stream, scoring only the row used by selection."""
    method.start(model, hi, config, seed)
    costs = []
    for s in ho:
        costs.append(int(method.adapt(s)))
    scores=[]
    for s in hi + ho:
        pred=method.predict(s) * ystats['std'] + ystats['mean']
        scores.append(engine.r2(s.y,pred,s.score))
    return dict(acc=float(np.mean(scores)), final_scores=scores,
                held_in=len(hi), names=[s.name for s in hi+ho], adapt_macs=costs)


def environment():
    return dict(python=platform.python_version(), torch=torch.__version__, numpy=np.__version__,
                device=engine.DEVICE, cuda=torch.version.cuda,
                gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                runner_sha256=digest(__file__),plan_sha256=digest(OUT/'plan.json'),
                tf32_matmul=torch.backends.cuda.matmul.allow_tf32,
                tf32_cudnn=torch.backends.cudnn.allow_tf32,
                cpu_threads=torch.get_num_threads())


def validation(task='h1'):
    out = OUT/'validation.json'
    if out.exists():
        prior=json.loads(out.read_text())
        if prior['passed'] and prior.get('environment',{}).get('runner_sha256')==digest(__file__):
            return prior
    c, model, rec, hi, ho, ys, ckpt = load_checkpoint(task, original.DEV_SEEDS[0], True)
    checks=[]
    # Cover parameter sharing, session tags, and the Fisher-state path. Short
    # trajectories suffice to validate the execution-order optimization.
    for name, cfg in [('finetune_full',dict(lr=1e-5,epochs=1)),
                      ('readout_tag',dict(lr=1e-3,epochs=1)),
                      ('ewc',dict(lr=1e-5,epochs=1,ewc=100.0))]:
        engine.seed_all(original.DEV_SEEDS[0])
        full=engine.stream(make_method(name,cfg,native=True),model,hi,ho,c,original.DEV_SEEDS[0],ys)
        engine.seed_all(original.DEV_SEEDS[0])
        final=final_score(make_method(name,cfg),model,hi,ho,c,original.DEV_SEEDS[0],ys)
        delta=float(np.max(np.abs(np.asarray(full['R'][-1])-final['final_scores'])))
        engine.seed_all(original.DEV_SEEDS[0])
        repeated=final_score(make_method(name,cfg),model,hi,ho,c,original.DEV_SEEDS[0],ys)
        repeated_delta=float(np.max(np.abs(np.asarray(full['R'][-1])-repeated['final_scores'])))
        checks.append(dict(method=name,config=cfg,max_absolute_final_score_difference=max(delta,repeated_delta),
                           full_acc=engine.summarize(full)['acc'], final_only_acc=final['acc'],cached_repeat_acc=repeated['acc']))
        gc.collect(); torch.cuda.empty_cache()
    value=dict(task=task,seed=original.DEV_SEEDS[0],checks=checks,
               passed=all(v['max_absolute_final_score_difference']<=1e-7 for v in checks),
               environment=environment())
    write(out,value)
    if not value['passed']:
        raise RuntimeError('Final-row-only scoring did not match original full stream.')
    return value


def select(tasks, methods):
    p=plan()
    validation(tasks[0])
    for task in tasks:
        for seed in original.DEV_SEEDS:
            c,model,rec,hi,ho,ys,ckpt=load_checkpoint(task,seed,True)
            for name in methods:
                for candidate in p['tasks'][task]['methods'][name]:
                    path=OUT/'inner'/task/name/f"{candidate['id']}_s{seed}.json"
                    if path.exists(): continue
                    engine.seed_all(seed)
                    start=time.perf_counter()
                    result=final_score(make_method(name,candidate['config']),model,hi,ho,c,seed,ys)
                    if engine.DEVICE=='cuda': torch.cuda.synchronize()
                    finite=all(np.isfinite(result['final_scores']))
                    if not finite:
                        result={**result,'acc':None,'final_scores':[float(x) if np.isfinite(x) else None for x in result['final_scores']]}
                    record=dict(task=task,method=name,seed=seed,candidate=candidate,
                                status='complete' if finite else 'nonfinite_predictions',
                                checkpoint_sha256=digest(ckpt),rng_seed=seed,
                                wall_seconds=time.perf_counter()-start,environment=environment(),**result)
                    write(path,record)
                    print(json.dumps(dict(stage='inner',task=task,method=name,seed=seed,
                                          config=candidate['config'],acc=result['acc'],seconds=record['wall_seconds'])),flush=True)
                    gc.collect(); torch.cuda.empty_cache()
        aggregate_selection(task,methods,p)


def aggregate_selection(task, methods, p=None):
    p=p or plan()
    results={}
    for name in methods:
        rows=[]
        for candidate in p['tasks'][task]['methods'][name]:
            paths=[OUT/'inner'/task/name/f"{candidate['id']}_s{sd}.json" for sd in original.DEV_SEEDS]
            if not all(x.exists() for x in paths):
                raise RuntimeError(f'Incomplete inner candidate {task}/{name}/{candidate}')
            records=[json.loads(x.read_text()) for x in paths]
            scores=[r['acc'] for r in records]
            valid=all(x is not None and np.isfinite(x) for x in scores)
            rows.append(dict(**candidate,valid=valid,inner_mean_acc=float(np.mean(scores)) if valid else None,
                             inner_seed_acc=scores,inner_record_paths=[str(x.relative_to(OUT)) for x in paths]))
        valid_rows=[(i,r) for i,r in enumerate(rows) if r['valid']]
        if not valid_rows: raise RuntimeError(f'No finite candidates for {task}/{name}')
        selected=max(valid_rows,key=lambda pair:(pair[1]['inner_mean_acc'],pair[1]['incumbent'],-pair[0]))[1]
        incumbent=next(r for r in rows if r['incumbent'])
        results[name]=dict(selected=selected,incumbent=incumbent,candidates=rows,
                           changed=(selected['id']!=incumbent['id']),
                           inner_acc_gain=(selected['inner_mean_acc']-incumbent['inner_mean_acc']) if incumbent['valid'] else None)
    path=OUT/'selection'/f'{task}.json'
    old=json.loads(path.read_text()) if path.exists() else dict(task=task,methods={})
    old['methods'].update(results)
    old['gradient']={m:r['selected']['config'] for m,r in old['methods'].items()}
    old['complete']=set(old['methods'])==set(METHODS)
    write(path,old)
    compatible=source_selection(task)
    compatible['gradient'].update(old['gradient'])
    compatible['confirmation']=dict(complete=old['complete'],plan_sha256=digest(OUT/'plan.json'),
                                   selection_file=str(path.relative_to(REPO)),methods=old['methods'])
    write(OUT/'selection_for_analysis'/f'{task}.json',compatible)
    return old


def outer(tasks, methods):
    p=plan()
    for task in tasks:
        selected=aggregate_selection(task,methods,p)
        for seed in original.FINAL_SEEDS:
            c,model,rec,hi,ho,ys,ckpt=load_checkpoint(task,seed,False)
            for name in methods:
                choices={selected['methods'][name][role]['id']:selected['methods'][name][role]
                         for role in ['incumbent','selected']}
                for cid,candidate in choices.items():
                    path=OUT/'outer'/task/name/f'{cid}_s{seed}.json'
                    if path.exists(): continue
                    engine.seed_all(seed)
                    start=time.perf_counter()
                    result=engine.stream(make_method(name,candidate['config']),model,hi,ho,c,seed,ys)
                    if engine.DEVICE=='cuda': torch.cuda.synchronize()
                    record=dict(task=task,method=name,seed=seed,rng_seed=seed,variant='full',
                                backbone=dict(macs=rec['macs'],info=rec['info']),
                                timing_scope='Computation-reuse experiment timing; not an end-to-end deployment benchmark.',
                                config=candidate['config'],candidate_id=cid,
                                checkpoint_sha256=digest(ckpt),environment=environment(),
                                wall_seconds=time.perf_counter()-start,
                                summary=engine.summarize(result),**result)
                    write(path,record)
                    print(json.dumps(dict(stage='outer',task=task,method=name,seed=seed,
                                          config=candidate['config'],acc=record['summary']['acc'],
                                          seconds=record['wall_seconds'])),flush=True)
                    gc.collect(); torch.cuda.empty_cache()
            # An ordinary stream layout allows the endpoint runner to overlay the
            # selected confirmation baselines without touching legacy records.
            for name in methods:
                cid=selected['methods'][name]['selected']['id']
                src=OUT/'outer'/task/name/f'{cid}_s{seed}.json'
                dst=OUT/'export'/task/'full'/f'{name}_s{seed}.json'
                dst.parent.mkdir(parents=True,exist_ok=True)
                shutil.copy2(src,dst)
                authoritative=OUT/'stream'/task/'full'/f'{name}_s{seed}.json'
                authoritative.parent.mkdir(parents=True,exist_ok=True)
                shutil.copy2(src,authoritative)


def summarize(tasks, methods):
    report=dict(description='Explicitly seeded boundary confirmation; original results are separate historical records.',
                tasks={}, development_seeds=original.DEV_SEEDS, outer_seeds=original.FINAL_SEEDS)
    for task in tasks:
        selection=json.loads((OUT/'selection'/f'{task}.json').read_text())
        task_report={}
        for name in methods:
            row={}
            for role in ['incumbent','selected']:
                candidate=selection['methods'][name][role]
                records=[json.loads((OUT/'outer'/task/name/f"{candidate['id']}_s{sd}.json").read_text()) for sd in original.FINAL_SEEDS]
                fields=records[0]['summary']
                aggregated={k:dict(mean=float(np.mean([r['summary'][k] for r in records])),
                                   sd=float(np.std([r['summary'][k] for r in records],ddof=1))) for k in fields}
                hdc=[json.loads((original.OUT/'stream'/task/'full'/f'hdc_s{sd}.json').read_text()) for sd in original.FINAL_SEEDS]
                ni={}
                for metric in ['acc','acc_held_out']:
                    d=np.asarray([h['summary'][metric]-r['summary'][metric] for h,r in zip(hdc,records)])
                    se=float(d.std(ddof=1)/np.sqrt(len(d)))
                    p=float(stats.t.sf((d.mean()+.03)/se,len(d)-1)) if se else (0.0 if d.mean()>-.03 else 1.0)
                    ni[metric]=dict(hdc_minus_reference=float(d.mean()),margin=.03,p_one_sided=p,
                                    p_bonferroni_12=min(1.,12*p),noninferior_unadjusted=p<.05,
                                    noninferior_bonferroni_12=12*p<.05,
                                    scope='Bonferroni family applies to four named NI references across three tasks.' if name in NI_REFS else 'Exploratory reference outside prespecified twelve comparisons.')
                row[role]=dict(config=candidate['config'],inner_mean_acc=candidate['inner_mean_acc'],
                               summary=aggregated,hdc_noninferiority=ni)
            legacy=[json.loads((original.OUT/'stream'/task/'full'/f'{name}_s{sd}.json').read_text()) for sd in original.FINAL_SEEDS]
            row['legacy_mean']={k:float(np.mean([x['summary'][k] for x in legacy])) for k in legacy[0]['summary']}
            row['changed']=selection['methods'][name]['changed']
            task_report[name]=row
        report['tasks'][task]=task_report
    write(OUT/'summary.json',report)
    print(json.dumps(report,indent=2))
    return report


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('stage',choices=['plan','validate','select','outer','summarize','all'])
    ap.add_argument('--tasks',nargs='+',choices=TASKS,default=list(TASKS))
    ap.add_argument('--methods',nargs='+',choices=METHODS,default=list(METHODS))
    ap.add_argument('--threads',type=int,default=4)
    args=ap.parse_args()
    torch.set_num_threads(args.threads)
    if args.stage=='plan':
        p=plan()
        print(json.dumps({t:{m:len(v) for m,v in p['tasks'][t]['methods'].items()} for t in TASKS},indent=2))
    elif args.stage=='validate': validation(args.tasks[0])
    else:
        if args.stage in ('select','all'): select(args.tasks,args.methods)
        if args.stage in ('outer','all'): outer(args.tasks,args.methods)
        if args.stage in ('summarize','all'): summarize(args.tasks,args.methods)

if __name__=='__main__': main()
