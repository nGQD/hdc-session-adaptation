#!/usr/bin/env python3
"""Prior-centered frozen-feature controls, with held-in-only model selection.

Run from repository root using .venv/bin/python. This runner never changes
original data, checkpoints, selections, or experimental records.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

os.environ.setdefault('OMP_NUM_THREADS', '8')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '8')
os.environ.setdefault('MKL_NUM_THREADS', '8')
import numpy as np
import scipy.linalg as la
import torch

HERE = Path(__file__).resolve().parent
# Support this workspace and a standalone code/results reproduction repository.
CODE = next(p for a in HERE.parents for p in [a / 'mjcs' / 'code', a / 'code']
            if (p / 'engine.py').is_file() and (p / 'run.py').is_file())
STUDY = CODE.parent
REPO = STUDY.parent if STUDY.name == 'mjcs' else STUDY
sys.path.insert(0, str(CODE))
import run as study
import engine
from model import Backbone, Config, FHRR

OUT = Path(__file__).resolve().parent / 'results' / 'feature_controls'
SEEDS = study.FINAL_SEEDS
DEV = study.DEV_SEEDS
# Broad, fixed before outer evaluation; ridge penalty is per 1000 fitting bins.
LAMBDAS = [0.1, 1., 10., 30., 100., 300., 1000., 3000., 10000., 30000., 100000.]
KINDS = ['linear_gf', 'relu_matched']


def write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def path_for(task, split, seed):
    return OUT / 'features' / task / split / f's{seed}'


def extract(task, split, seed):
    dest = path_for(task, split, seed)
    if (dest / 'metadata.json').exists():
        saved = json.loads((dest / 'metadata.json').read_text())
        required = [dest / 'prior.npy', *[dest / row['file'] for row in saved['sessions']]]
        if all(path.is_file() and path.stat().st_size > 0 for path in required):
            print('cache present', task, split, seed, flush=True)
            return
        print('cache metadata present; extracting missing arrays', task, split, seed, flush=True)
    t0=time.perf_counter()
    c = study.base_config(task)
    inner = split == 'inner'
    checkpoint = STUDY / 'results/checkpoints' / task / ('inner_pretrain' if inner else 'outer') / f'full_s{seed}.pt'
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    model, rec, hi, ho, ys = study.checkpoint(task, 'full', c, seed, inner=inner)
    model.eval()
    dest.mkdir(parents=True, exist_ok=True)
    rows=[]
    for i,s in enumerate(hi+ho):
        sf = dest / f'{i:02d}_{s.name}.npz'
        if not sf.exists():
            h,gs,fs=None,[],[]
            with torch.no_grad():
                for z in torch.as_tensor(s.x, device=engine.DEVICE).split(2048):
                    f,h=model.encode(z[None],h)
                    gs.append(model.readout[:-1](f)[0].cpu())
                    fs.append(f[0].cpu())
            gf=torch.cat([torch.cat(gs),torch.cat(fs)],1).numpy()
            fitting=s.fit if i>=len(hi) else np.zeros_like(s.fit)
            np.savez(sf, fit_gf=gf[fitting],fit_y=s.yz[fitting],score_gf=gf[s.score],score_y=s.y[s.score])
            del gf, gs, fs
        rows.append(dict(file=sf.name,name=s.name,fit_bins=int(s.fit.sum()) if i>=len(hi) else 0,
                         prefix_bins=s.fit_end,score_bins=int(s.score.sum())))
        print('features',task,split,seed,i+1,'/',len(hi+ho),round(time.perf_counter()-t0,1),flush=True)
    last=model.readout[-1]
    w0=torch.cat([last.weight.T,last.bias[None]],0).detach().cpu().double().numpy()
    np.save(dest/'prior.npy',w0)
    metadata=dict(task=task,split=split,seed=seed,held_in=len(hi),sessions=rows,
                  target_mean=ys['mean'].tolist(),target_std=ys['std'].tolist(),
                  model_parameters=model.parameter_count(),backbone_macs=rec['macs'],
                  checkpoint=str(checkpoint.relative_to(REPO)),checkpoint_sha256=sha(checkpoint),
                  source_sha256={x:sha(CODE/x) for x in ['engine.py','run.py','model.py','data.py']},
                  extraction_device=engine.DEVICE,extraction_seconds=time.perf_counter()-t0,
                  projection_width=model.features,prior_width=w0.shape[0],
                  dtype='float32 features; float64 prior and solves',
                  torch_version=torch.__version__)
    write(dest/'metadata.json',metadata)
    print('extracted',task,split,seed,round(time.perf_counter()-t0,1),flush=True)
    study._MEMO.clear()
    del model,hi,ho
    if torch.cuda.is_available():torch.cuda.empty_cache()


def config_for(task, kind, lam):
    selected=json.loads((STUDY/'results/selection'/f'{task}.json').read_text())
    D=selected['hdc']['dimension']
    return dict(kind=kind,lam=lam,dimension=D,source='state',
                relu_width=2*D,scale=1.0)


class Map:
    def __init__(self, metadata, kind, dimension, seed, scale=1.):
        self.k=metadata['prior_width']-1
        self.nf=metadata['projection_width']
        self.kind=kind
        self.dimension=dimension
        self.parameters=0
        if kind=='relu_matched':
            g=torch.Generator().manual_seed(seed)
            self.w=(torch.randn(self.nf,2*dimension,generator=g)/math.sqrt(self.nf)).numpy()
            self.parameters=self.w.size
        elif kind=='hdc':
            enc=FHRR(self.nf,dimension,scale,seed)
            self.w,self.b=enc.w.numpy(),enc.b.numpy()
            self.parameters=self.w.size+self.b.size
        self.width=self.k+1+(self.nf if kind=='linear_gf' else 2*dimension if kind in ['relu_matched','hdc'] else 0)

    def __call__(self,gf):
        base=np.column_stack([gf[:,:self.k],np.ones(len(gf),np.float32)])
        if self.kind=='closed_linear':return base.astype(np.float64)
        f=gf[:,self.k:]
        if self.kind=='linear_gf':add=f
        elif self.kind=='relu_matched':add=np.maximum(f@self.w,0)*np.float32(math.sqrt(2.))
        elif self.kind=='hdc':
            phase=f@self.w+self.b
            # Original FHRR divides then multiplies by sqrt(D) in float32.
            add=np.concatenate([np.cos(phase),np.sin(phase)],1)/np.float32(math.sqrt(self.dimension))*np.float32(math.sqrt(self.dimension))
        return np.column_stack([base,add]).astype(np.float64)

    def macs(self,N):
        if self.kind=='relu_matched':return N*self.nf*2*self.dimension
        if self.kind=='hdc':return N*self.nf*self.dimension
        return 0


def prior_residual(X, Y, w0):
    """Subtract the pretrained readout; all expansion rows of its prior are zero."""
    return Y - X[:, :len(w0)] @ w0


def residual_equivalence_check():
    """Cheap prediction check of the sparse prior in both solve/grid branches."""
    rng = np.random.default_rng(20261024)
    errors, prediction_errors = [], []
    # Task-sized expansion widths and a small primal case; no dataset fit is run.
    for n, base, extra, outputs in [(17, 129, 304, 7), (11, 129, 8192, 16),
                                    (19, 129, 4096, 2), (67, 9, 13, 3)]:
        d = base + extra
        X = rng.normal(size=(n, d)).astype(np.float32).astype(np.float64)
        Z = rng.normal(size=(13, d)).astype(np.float32).astype(np.float64)
        Y = rng.normal(size=(n, outputs)).astype(np.float32).astype(np.float64)
        w0 = rng.normal(size=(base, outputs)).astype(np.float32).astype(np.float64)
        prior = np.zeros((d, outputs)); prior[:base] = w0
        old, new = Y - X @ prior, prior_residual(X, Y, w0)
        errors.append(float(np.max(np.abs(old - new))))
        if not np.allclose(old, new, atol=1e-12, rtol=1e-13):
            raise AssertionError('Sparse-prior residual changed')
        dual = n < d
        gram = X @ X.T if dual else X.T @ X
        vals, vecs = la.eigh(gram, check_finite=False)
        for grid in [False, True]:
            for lam in [0.1, 1000., 100000.]:
                predictions = []
                for residual in [old, new]:
                    rhs = residual if dual else X.T @ residual
                    leff = lam * n / 1000
                    if grid:
                        solved = vecs @ ((vecs.T @ rhs) / (np.maximum(vals, 0)[:, None] + leff))
                    else:
                        solved = la.cho_solve(la.cho_factor(gram + leff * np.eye(len(gram))), rhs)
                    delta = X.T @ solved if dual else solved
                    predictions.append((Z @ (prior + delta)).astype(np.float32))
                error = float(np.max(np.abs(predictions[0] - predictions[1])))
                prediction_errors.append(error)
                if error != 0:
                    raise AssertionError(('Sparse-prior float32 prediction changed', error))
    return dict(cases=len(prediction_errors), max_residual_absolute_difference=max(errors),
                max_float32_prediction_absolute_difference=max(prediction_errors),
                branches=['primal', 'dual'], solves=['Cholesky', 'eigenvalue grid'])


def evaluate_grid(task, split, seed, kind, lambdas, detailed=False):
    root=path_for(task,split,seed)
    meta=json.loads((root/'metadata.json').read_text())
    w0=np.load(root/'prior.npy')
    cfg=config_for(task,kind,lambdas[0])
    mapper=Map(meta,kind,cfg['dimension'],seed)
    d=mapper.width;o=w0.shape[1]
    prior=np.zeros((d,o),np.float64);prior[:len(w0)]=w0
    k=meta['held_in']; T=len(meta['sessions'])-k
    start=np.empty(k+T)
    finals=np.empty((len(lambdas),k+T))
    costs=[];legacy_costs=[];symmetric_costs=[];times=[]
    scale=np.asarray(meta['target_std'],np.float32)
    shift=np.asarray(meta['target_mean'],np.float32)
    for j,row in enumerate(meta['sessions']):
        with np.load(root/row['file']) as f:
            score_gf=f['score_gf']; truth=f['score_y']
            base=np.column_stack([score_gf[:,:mapper.k],np.ones(len(score_gf),np.float32)]).astype(np.float64)
            basepred=(base@w0).astype(np.float32)*scale+shift
            mask=np.ones(len(truth),bool)
            start[j]=engine.r2(truth,basepred,mask)
            finals[:,j]=start[j]
            if j<k:continue
            t0=time.perf_counter()
            X=mapper(f['fit_gf']); Y=f['fit_y'].astype(np.float64)
            N=len(X)
            Z=mapper(score_gf)
            residual=prior_residual(X,Y,w0)
            # Exact ridge algebra: dual form when fitting rows are fewer than head columns.
            dual=N<d
            gram=X@X.T if dual else X.T@X
            rhs=residual if dual else X.T@residual
            if len(lambdas)>1:
                vals,vecs=la.eigh(gram,check_finite=False,overwrite_a=True,driver='evd')
                vals=np.maximum(vals,0)
                projected=vecs.T@rhs
            for li,lam in enumerate(lambdas):
                leff=lam*N/1000
                if len(lambdas)>1:
                    solved=vecs@(projected/(vals[:,None]+leff))
                else:
                    gram.flat[::len(gram)+1]+=leff
                    solved=la.cho_solve(la.cho_factor(gram,lower=True,check_finite=False),rhs,check_finite=False)
                delta=X.T@solved if dual else solved
                pred=(Z@(prior+delta)).astype(np.float32)*scale+shift
                finals[li,j]=engine.r2(truth,pred,mask)
            elapsed=time.perf_counter()-t0
            bb=sum(meta['backbone_macs'].values())
            legacy_mac=row['prefix_bins']*bb+mapper.macs(N)+N*d*(d+1)//2+N*d*o+d**3//6+d*d*o
            q=min(N,d)
            symmetric_mac=row['prefix_bins']*bb+mapper.macs(N)+N*d*(q+1)//2+N*d*o+q**3//6+q*q*o
            mac=row['prefix_bins']*bb+mapper.macs(N)+N*d*q+N*d*o+q**3//6+q*q*o
            costs.append(int(mac));legacy_costs.append(int(legacy_mac));symmetric_costs.append(int(symmetric_mac));times.append(elapsed)
            print('solve',task,split,seed,kind,j-k+1,'/',T,'N',N,'d',d,'dual',dual,'sec',round(elapsed,2),flush=True)
    records=[]
    for li,lam in enumerate(lambdas):
        R=np.tile(start,(T+1,1))
        for t in range(1,T+1): R[t,k:k+t]=finals[li,k:k+t]
        legacy_memory=[meta['model_parameters']+mapper.parameters+d*o*t for t in range(1,T+1)]
        memory=[v+w0.size for v in legacy_memory]
        rec=dict(R=R.tolist(),names=[r['name'] for r in meta['sessions']],held_in=k,
                 adapt_macs=costs,legacy_primal_adapt_macs=legacy_costs,
                 legacy_symmetric_adapt_macs=symmetric_costs,adapt_seconds=times,
                 memory=memory,legacy_memory=legacy_memory,start_macs=0,start_seconds=0,
                 task=task,seed=seed,split=split,method=kind,configuration=config_for(task,kind,lam),
                 algorithm='prior-centered ridge; exact primal/dual solve chosen by min(N,d)',
                 computation_note='Analytic MACs count the implemented dense Gram product N*d*q, q=min(N,d). Retained scalars include the prior copy. Seconds are cached-feature analysis and are not deployment timings.',
                 feature_cache=str(root.relative_to(REPO)),checkpoint_sha256=meta['checkpoint_sha256'],
                 extra_feature_parameters=mapper.parameters,head_width=d)
        rec['summary']=engine.summarize(rec)
        records.append(rec)
    return records


def select(task, kind):
    dest=OUT/'selection'/f'{task}_{kind}.json'
    if dest.exists():return json.loads(dest.read_text())
    scores=[]
    for seed in DEV:
        p=OUT/'inner'/task/f'{kind}_s{seed}.json'
        if p.exists():r=json.loads(p.read_text())
        else:
            r=evaluate_grid(task,'inner',seed,kind,LAMBDAS)
            write(p,r)
        scores.append([v['summary']['acc'] for v in r])
    means=np.mean(scores,axis=0);ix=int(np.argmax(means))
    out=dict(task=task,kind=kind,lambda_grid=LAMBDAS,dev_seeds=DEV,inner_acc_by_seed=scores,
             mean_inner_acc=means.tolist(),selected_lambda=LAMBDAS[ix],selected_at_boundary=ix in [0,len(LAMBDAS)-1],
             selection_endpoint='mean all-session stream ACC on inner held-in-only pseudo-new sessions',
             formula='[g, 1, f]' if kind=='linear_gf' else '[g, 1, sqrt(2)*ReLU(fW)]',
             random_projection='W_ij iid Normal(0,1/n_f); fixed by seed; no bias; 2D expansion columns' if kind=='relu_matched' else None)
    write(dest,out)
    print('SELECTED',task,kind,out['selected_lambda'],means[ix],flush=True)
    return out


def evaluate(task,kind,seed):
    dest=OUT/'outer'/task/f'{kind}_s{seed}.json'
    if dest.exists():return
    sel=select(task,kind)
    rec=evaluate_grid(task,'outer',seed,kind,[sel['selected_lambda']])[0]
    write(dest,rec)
    print('OUTER',task,kind,seed,rec['summary'],flush=True)


def verify(task,seed):
    sel=json.loads((STUDY/'results/selection'/f'{task}.json').read_text())
    out=[]
    for kind in ['closed_linear','hdc']:
        lam=sel['linear' if kind=='closed_linear' else 'hdc']['lam']
        rec=evaluate_grid(task,'outer',seed,kind,[lam])[0]
        ref=json.loads((STUDY/'results/stream'/task/'full'/f'{kind}_s{seed}.json').read_text())
        diff=np.max(np.abs(np.array(rec['R'])-np.array(ref['R'])))
        row=dict(task=task,seed=seed,method=kind,max_abs_R_difference=float(diff),
                 reproduced_summary=rec['summary'],reference_summary=ref['summary'],passed=bool(diff<2e-5))
        out.append(row)
        print('VERIFY',row,flush=True)
        if not row['passed']:raise AssertionError(row)
    write(OUT/'verification'/f'{task}_s{seed}.json',out)


def aggregate():
    import scipy.stats as st
    out={}
    for task in ['h1','m1','m2']:
        out[task]={}
        for kind in KINDS:
            paths=[OUT/'outer'/task/f'{kind}_s{s}.json' for s in SEEDS]
            if not all(p.exists() for p in paths):continue
            rows=[json.loads(p.read_text()) for p in paths]
            sums={key:[r['summary'][key] for r in rows] for key in ['acc','acc_held_out','acc_held_in','bwt','adapt_macs','memory']}
            report=dict(n_seeds=len(rows),selected=select(task,kind),metrics={key:dict(mean=float(np.mean(v)),sd=float(np.std(v,ddof=1))) for key,v in sums.items()})
            ref=[json.loads((STUDY/'results/stream'/task/'full'/f'hdc_s{s}.json').read_text()) for s in SEEDS]
            for metric in ['acc','acc_held_out']:
                d=np.array([r['summary'][metric] for r in ref])-np.array(sums[metric])
                se=np.std(d,ddof=1)/math.sqrt(len(d));tc=st.t.ppf(.975,len(d)-1)
                report['hdc_minus_control_'+metric]=dict(mean=float(d.mean()),ci95=[float(d.mean()-tc*se),float(d.mean()+tc*se)],paired_t_p=float(st.ttest_1samp(d,0).pvalue),wilcoxon_p=float(st.wilcoxon(d,method='exact').pvalue))
            # Shared paired draws over sessions and seeds, preserving held-in/out counts.
            matrix=np.array([r['R'][-1] for r in ref])-np.array([r['R'][-1] for r in rows])
            k=rows[0]['held_in']; nout=matrix.shape[1]-k
            rng=np.random.default_rng(20261023+['h1','m1','m2'].index(task))
            seed_ix=rng.integers(0,len(rows),(10000,len(rows)))
            hi_ix=rng.integers(0,k,(10000,k))
            ho_ix=k+rng.integers(0,nout,(10000,nout))
            for metric,ix in [('acc',np.concatenate([hi_ix,ho_ix],1)),('acc_held_out',ho_ix)]:
                boot=matrix[seed_ix[:,:,None],ix[:,None,:]].mean((1,2))
                report['hdc_minus_control_'+metric]['paired_two_way_bootstrap_ci95']=np.quantile(boot,[.025,.975]).tolist()
            report['inference_note']='Paired t intervals and Wilcoxon tests condition on the fixed sessions; two-way bootstrap also resamples sessions. Controls are additional robustness experiments.'
            out[task][kind]=report
    # Conservative Holm adjustment across all six task-by-control contrasts per metric.
    from statsmodels.stats.multitest import multipletests
    for metric in ['acc','acc_held_out']:
        cells=[r['hdc_minus_control_'+metric] for rr in out.values() for r in rr.values()]
        if cells:
            adjusted=multipletests([r['wilcoxon_p'] for r in cells],method='holm')[1]
            for cell,padj in zip(cells,adjusted):
                cell['wilcoxon_holm_p']=float(padj)
                cell['holm_family_size']=len(cells)
    write(OUT/'aggregate.json',out)
    print(json.dumps(out,indent=2),flush=True)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('stage',choices=['extract','select','evaluate','verify','aggregate','self-test'])
    ap.add_argument('--tasks',nargs='+',default=['h1','m1','m2'])
    ap.add_argument('--splits',nargs='+',default=['inner','outer'])
    ap.add_argument('--seeds',nargs='+',type=int)
    ap.add_argument('--kinds',nargs='+',choices=KINDS,default=KINDS)
    ap.add_argument('--threads',type=int,default=8)
    args=ap.parse_args();torch.set_num_threads(args.threads)
    if args.stage=='self-test':return print(json.dumps(residual_equivalence_check(),indent=2))
    if args.stage=='aggregate':return aggregate()
    for task in args.tasks:
        if args.stage=='extract':
            for split in args.splits:
                for seed in (args.seeds or (DEV if split=='inner' else SEEDS)):extract(task,split,seed)
        elif args.stage=='verify':
            for seed in args.seeds or [101]:verify(task,seed)
        else:
            for kind in args.kinds:
                if args.stage=='select':select(task,kind)
                else:
                    for seed in args.seeds or SEEDS:evaluate(task,kind,seed)

if __name__=='__main__':main()
