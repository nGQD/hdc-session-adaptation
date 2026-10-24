"""Recompute dose and pretraining-budget gradient controls after inner selection.

The confirmed configuration is reused at every prefix length and pretraining
budget. Neither sensitivity experiment selects hyperparameters on outer scores.
Each stream resets all random generators explicitly, independent of cache state.
"""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import platform
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'mjcs' / 'code'))
import numpy as np
import torch
import data
import engine
import run

OUT = Path(__file__).resolve().parent / 'results' / 'confirmed_sensitivity'
SELECT = OUT.parent / 'baseline_robustness' / 'selection_for_analysis'


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def selection(task):
    p = SELECT / f'{task}.json'
    d = json.loads(p.read_text())
    if not d.get('confirmation', {}).get('complete'):
        raise ValueError(f'Incomplete inner selection: {task}')
    return d, hashlib.sha256(p.read_bytes()).hexdigest()


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while block := stream.read(2 ** 20):
            h.update(block)
    return h.hexdigest()


def checkpoint_path(task, variant, seed):
    return ROOT / 'mjcs/results/checkpoints' / task / 'outer' / f'{variant}_s{seed}.pt'


def computation_identity(task, method, variant, seed, selected, selection_sha):
    return dict(task=task, method=method, variant=variant, seed=seed, rng_seed=seed,
                config=selected['gradient'][method], selection_sha256=selection_sha,
                checkpoint_sha256=file_sha(checkpoint_path(task, variant, seed)),
                runner_sha256=file_sha(__file__),
                implementation_sha256={name:file_sha(ROOT/'mjcs/code'/name)
                                       for name in ['engine.py','run.py','model.py','data.py']},
                software_versions=dict(python=platform.python_version(),numpy=np.__version__,
                                       torch=torch.__version__,cuda_runtime=torch.version.cuda,device=engine.DEVICE),
                data_manifest_sha256=file_sha(ROOT/'mjcs/results/provenance'/f'manifest_{task}.json'))


def array_sha(value):
    value = np.ascontiguousarray(value)
    h = hashlib.sha256(json.dumps([value.dtype.str, list(value.shape)]).encode())
    h.update(memoryview(value).cast('B'))
    return h.hexdigest()


def input_identity(hi, ho, ys):
    """Bind cached evidence to the actual normalized inputs, targets and masks."""
    return dict(held_in=len(hi), sessions=[dict(name=s.name, fit_end=s.fit_end,
                    arrays_sha256={key:array_sha(getattr(s,key)) for key in ['x','y','yz','fit','score']})
                    for s in hi+ho], target_statistics_sha256={key:array_sha(ys[key]) for key in ['mean','std']})


def verify_saved(path, expected, stream=False):
    record = json.loads(Path(path).read_text())
    mismatches = [key for key, value in expected.items() if record.get(key) != value]
    if mismatches:
        raise ValueError(f'Stale or mismatched sensitivity record {path}: {mismatches}')
    if stream:
        inputs = expected['input_identity']
        names = [s['name'] for s in inputs['sessions']]
        R = np.asarray(record.get('R', []))
        if (record.get('names') != names or record.get('held_in') != inputs['held_in']
                or R.shape != (len(names)-inputs['held_in']+1, len(names)) or not np.isfinite(R).all()):
            raise ValueError(f'Malformed sensitivity stream: {path}')
        for key in ['adapt_macs','adapt_seconds','memory']:
            values = np.asarray(record.get(key, []))
            if values.shape != (len(names)-inputs['held_in'],) or not np.isfinite(values).all() or (values < 0).any():
                raise ValueError(f'Malformed sensitivity update measurements: {path}')
        if record.get('summary') != engine.summarize(record):
            raise ValueError(f'Sensitivity summary differs from its stream: {path}')
    return record


def checked_checkpoint(task, variant, config, seed):
    p = checkpoint_path(task, variant, seed)
    assert p.exists(), f'Required existing checkpoint missing: {p}'
    return run.checkpoint(task, variant, config, seed)


def dose(task, seed):
    sel, sha = selection(task)
    c = run.base_config(task)
    model, rec, _, _, _ = checked_checkpoint(task, 'full', c, seed)
    for name in ['finetune_full', 'replay']:
        target = OUT / 'dose' / task / f'{name}_s{seed}.json'
        identity = computation_identity(task, name, 'full', seed, sel, sha)
        if target.exists():
            verify_saved(target, dict(**identity, dose=run.DOSES))
        rows, streams = [], {}
        for fraction in run.DOSES:
            hi, ho, ys = run.prepare(task, c, prefix=fraction)
            for s in ho:
                mask = np.zeros(len(s.score), bool)
                mask[int(round(len(s.score) * 0.5)) + data.GAP:] = True
                if fraction == 0.5:
                    assert np.array_equal(s.score, s.score & mask), 'Half-prefix suffix differs from main stream'
                s.score &= mask
            start = time.perf_counter()
            reuse = {}
            if fraction == 0.5:
                source = OUT.parent/'baseline_robustness'/'stream'/task/'full'/f'{name}_s{seed}.json'
                main = json.loads(source.read_text())
                assert main['config'] == sel['gradient'][name]
                assert main['candidate_id'] == sel['confirmation']['methods'][name]['selected']['id']
                assert main['seed'] == seed and main['held_in'] == len(hi)
                assert main['names'] == [s.name for s in hi+ho]
                assert main['checkpoint_sha256'] == identity['checkpoint_sha256']
                reuse = dict(reused_main_stream=str(source.relative_to(ROOT)),
                             reused_main_stream_sha256=file_sha(source),
                             reuse_reason='Identical checkpoint, seed, configuration, fitting prefix and score mask.')
            stream_path = OUT / 'dose_stream' / task / f'{name}_p{fraction}_s{seed}.json'
            expected = dict(**identity, prefix_fraction=fraction, input_identity=input_identity(hi, ho, ys), **reuse)
            if stream_path.exists():
                saved = verify_saved(stream_path, expected, stream=True)
                summary = saved['summary']
            else:
                if target.exists():
                    raise ValueError(f'Completed dose record is missing stream evidence: {stream_path}')
                engine.seed_all(seed)
                if fraction == 0.5:
                    result = {k:main[k] for k in ['R','names','held_in','adapt_macs','adapt_seconds',
                                                 'memory','start_macs','start_seconds']}
                else:
                    result = engine.stream(run.GRADIENT[name](sel['gradient'][name]), model, hi, ho, c, seed, ys)
                summary = engine.summarize(result)
                write(stream_path, dict(**result, summary=summary, **expected,
                                       timing_scope='Stream computation; use isolated resource benchmark for deployment timing.'))
            rows.append(summary)
            streams[str(stream_path.relative_to(OUT))] = file_sha(stream_path)
            print(json.dumps(dict(stage='dose', task=task, seed=seed, method=name,
                                  fraction=fraction, acc_held_out=summary['acc_held_out'],
                                  seconds=time.perf_counter()-start)), flush=True)
            gc.collect(); torch.cuda.empty_cache()
        result = dict(**identity, dose=run.DOSES, stream_records_sha256=streams,
                      **{key:[r[key] for r in rows] for key in ['acc_held_out', 'acc', 'adapt_macs']})
        if target.exists():
            verify_saved(target, result)
        else:
            write(target, result)


def budget(task, seed):
    sel, sha = selection(task)
    for variant in ['epochs25', 'epochs100']:
        target = OUT / 'stream' / task / variant / f'finetune_full_s{seed}.json'
        identity = computation_identity(task, 'finetune_full', variant, seed, sel, sha)
        if target.exists():
            verify_saved(target, identity)
        c = run.variants(run.base_config(task))[variant]
        model, rec, hi, ho, ys = checked_checkpoint(task, variant, c, seed)
        identity['input_identity'] = input_identity(hi, ho, ys)
        if target.exists():
            verify_saved(target, identity, stream=True)
            continue
        engine.seed_all(seed)
        start = time.perf_counter()
        result = engine.stream(run.GRADIENT['finetune_full'](sel['gradient']['finetune_full']),
                               model, hi, ho, c, seed, ys)
        summary = engine.summarize(result)
        write(target, dict(**result, summary=summary, **identity,
                           backbone=dict(macs=rec['macs'], info=rec['info']),
                           timing_scope='Stream computation; use isolated resource benchmark for deployment timing.'))
        print(json.dumps(dict(stage='budget', task=task, variant=variant, seed=seed,
                              acc=summary['acc'], seconds=time.perf_counter()-start)), flush=True)
        gc.collect(); torch.cuda.empty_cache()


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--tasks', nargs='+', default=['h1', 'm1', 'm2'])
    ap.add_argument('--seeds', nargs='+', type=int, default=run.ABLATION_SEEDS)
    ap.add_argument('--stage', choices=['dose','budget','all'], default='all')
    args = ap.parse_args()
    torch.set_num_threads(4)
    for task in args.tasks:
        for seed in args.seeds:
            if args.stage in ('dose','all'):
                dose(task, seed)
            if args.stage in ('budget','all'):
                budget(task, seed)
