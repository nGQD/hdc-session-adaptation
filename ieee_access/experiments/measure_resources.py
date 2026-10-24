"""Measure cold-cache adaptation, including prefix feature extraction and solves.

Preparation is outside the measured interval. Each worker has a fresh process,
one task/method and one repetition. Only released adaptation prefixes are loaded.
CUDA peak allocated bytes include model parameters and all live tensor buffers;
CPU resident bytes are sampled at 1 ms and include the interpreter and input data.
These are workstation measurements, not implanted-device power measurements.
"""
import argparse
import copy
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import threading
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "results" / "resources"
CONFIRMED_SELECTION = OUT.parent / 'baseline_robustness' / 'selection_for_analysis'
sys.path.insert(0, str(ROOT / "mjcs" / "code"))


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def fingerprint(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def host_environment():
    cpu_model = platform.processor()
    try:
        cpu_model = next(line.split(':',1)[1].strip() for line in Path('/proc/cpuinfo').read_text().splitlines()
                         if line.startswith('model name'))
    except (OSError, StopIteration):
        pass
    return dict(cpu_model=cpu_model or 'unknown', cpu_affinity=sorted(os.sched_getaffinity(0)))


def selected_configuration(task, method, selection_root):
    path = Path(selection_root) / f'{task}.json'
    sel = json.loads(path.read_text())
    if not sel.get('confirmation', {}).get('complete'):
        raise ValueError(f'Confirmed selection required: {path}')
    key = 'hdc' if method in ('hdc','hdc_primal') else 'linear' if method == 'closed_linear' else None
    return sel, sel[key] if key else sel['gradient'][method], path


def included_identity(task, method, selection_root, resource_root=None):
    """Identity of the small files included in the records-only archive."""
    root = OUT if resource_root is None else Path(resource_root)
    _, config, selection = selected_configuration(task, method, selection_root)
    return dict(configuration=config, benchmark_source_sha256=fingerprint(__file__),
                    solver_source_sha256=fingerprint(Path(__file__).with_name('efficient_closed_form.py')),
                    selection_file_sha256=fingerprint(selection),
                    prefix_manifest_sha256=fingerprint(root/f'{task}_prefix_manifest.json'),
                    implementation_sha256={name:fingerprint(ROOT/'mjcs/code'/name)
                                           for name in ['engine.py','run.py','model.py','data.py']})


def current_identity(task, method, selection_root, resource_root=None):
    """Live benchmark validation also requires the actual large input files."""
    import numpy as np
    import torch
    root = OUT if resource_root is None else Path(resource_root)
    return dict(**included_identity(task, method, selection_root, root),
                    checkpoint_sha256=fingerprint(ROOT/'mjcs/results/checkpoints'/task/'outer/full_s101.pt'),
                    prefix_cache_sha256=fingerprint(root/f'{task}_prefix_cache.npz'),
                    software_versions=dict(python=platform.python_version(),numpy=np.__version__,
                                           torch=torch.__version__,cuda_runtime=torch.version.cuda))


def verify_record(path, task, method, threads, device, repeat, identity, resource_root=None):
    root = OUT if resource_root is None else Path(resource_root)
    rec = json.loads(path.read_text())
    if device not in {'cpu','cuda'} or not isinstance(repeat, int) or repeat < 0 or threads < 1:
        raise ValueError(f'Invalid benchmark run settings: {path}')
    expected = dict(identity)
    expected.update(task=task, method=method, threads=threads, device=device, repetition=repeat, seed=101)
    mismatches = [key for key, value in expected.items() if rec.get(key) != value]
    if mismatches:
        raise ValueError(f'Stale or mismatched benchmark {path}: {mismatches}')
    if path.name != f'{device}_{task}_{method}_r{repeat}.json':
        raise ValueError(f'Mismatched benchmark filename: {path}')
    environment = rec.get('measurement_environment', {})
    affinity = environment.get('cpu_affinity', [])
    if (set(environment) != {'cpu_model','cpu_affinity'} or not isinstance(environment['cpu_model'],str)
            or not environment['cpu_model'] or not isinstance(affinity,list) or not affinity
            or any(type(cpu) is not int or cpu < 0 for cpu in affinity) or affinity != sorted(set(affinity))):
        raise ValueError(f'Invalid benchmark CPU model or affinity: {path}')
    for key in ['system_load_start','system_load_end']:
        load = rec.get(key, [])
        if (not isinstance(load,list) or len(load) != 3
                or any(not isinstance(x,(int,float)) or not math.isfinite(x) or x < 0 for x in load)):
            raise ValueError(f'Invalid benchmark system load: {path}')
    manifest = json.loads((root / f'{task}_prefix_manifest.json').read_text())
    rows = rec.get('rows', [])
    if not manifest or len({m['name'] for m in manifest}) != len(manifest):
        raise ValueError('Empty or duplicated prefix manifest')
    if len(rows) != len(manifest) or any(any(row.get(k) != meta[k] for k in ['name','prefix_bins','labeled_bins'])
                                       for row, meta in zip(rows, manifest)):
        raise ValueError(f'Mismatched benchmark sessions: {path}')
    fields = ['seconds','analytic_macs','persistent_scalars_after_update',
              'process_rss_base_bytes','process_rss_peak_bytes','process_rss_increment_bytes']
    if device == 'cuda':
        fields += ['cuda_baseline_allocated_bytes','cuda_peak_allocated_bytes','cuda_increment_allocated_bytes']
    for row in rows:
        if any(not isinstance(row.get(key), (int,float)) or not math.isfinite(row[key]) or row[key] < 0
               for key in fields):
            raise ValueError(f'Invalid benchmark measurement: {path}')
        if row['seconds'] <= 0 or row['process_rss_peak_bytes'] < row['process_rss_base_bytes']:
            raise ValueError(f'Invalid benchmark time or memory peak: {path}')
        if row['process_rss_increment_bytes'] != row['process_rss_peak_bytes']-row['process_rss_base_bytes']:
            raise ValueError(f'Inconsistent resident-memory increment: {path}')
        if device == 'cuda' and row['cuda_increment_allocated_bytes'] != row['cuda_peak_allocated_bytes']-row['cuda_baseline_allocated_bytes']:
            raise ValueError(f'Inconsistent CUDA-memory increment: {path}')
    return rec


def verify_existing(path, task, method, selection_root, threads, device, repeat, resource_root=None, match_host=True):
    record = verify_record(path, task, method, threads, device, repeat,
                           current_identity(task, method, selection_root, resource_root), resource_root)
    if match_host and record['measurement_environment'] != host_environment():
        raise ValueError(f'Benchmark resume CPU model or inherited affinity changed: {path}')
    return record


def prepare(task):
    import numpy as np
    import run
    c = run.base_config(task)
    hi, ho, ys = run.prepare(task, c)
    arrays = {}
    manifest = []
    for i, s in enumerate(ho):
        n = s.fit_end
        arrays[f"x{i}"] = s.x[:n]
        arrays[f"y{i}"] = s.yz[:n]
        arrays[f"mask{i}"] = s.fit[:n]
        manifest.append(dict(name=s.name, prefix_bins=n, labeled_bins=int(s.fit[:n].sum())))
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / f"{task}_prefix_cache.npz", **arrays)
    write(OUT / f"{task}_prefix_manifest.json", manifest)


class ResidentMonitor:
    def __init__(self):
        self.stop = threading.Event()
        self.base = self.rss()
        self.peak = self.base
        self.thread = threading.Thread(target=self.sample, daemon=True)

    def sample(self):
        while not self.stop.is_set():
            self.peak = max(self.peak, self.rss())
            self.stop.wait(0.001)

    @staticmethod
    def rss():
        # Linux resident pages; avoids an optional monitoring dependency.
        return int(Path('/proc/self/statm').read_text().split()[1]) * os.sysconf('SC_PAGE_SIZE')

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.peak = max(self.peak, self.rss())
        self.stop.set()
        self.thread.join()


def worker(task, method_name, device, repeat, threads, selection_root):
    import numpy as np
    import torch
    import engine
    from efficient_closed_form import EfficientPriorMethod
    from model import Backbone, Config
    measurement_environment = host_environment()
    system_load_start = list(os.getloadavg())
    engine.DEVICE = device
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)
    identity = current_identity(task, method_name, selection_root)
    checkpoint = ROOT / "mjcs" / "results" / "checkpoints" / task / "outer" / "full_s101.pt"
    rec = torch.load(checkpoint, map_location=device, weights_only=False)
    c = Config(**rec["config"])
    model = Backbone(c).to(device)
    model.load_state_dict(rec["state"])
    del rec
    model.eval()
    sel, selected, selection_path = selected_configuration(task, method_name, selection_root)
    def make():
        if method_name == "hdc":
            return EfficientPriorMethod("hdc", **sel["hdc"])
        if method_name == "hdc_primal":
            return engine.PriorMethod("hdc", **sel["hdc"])
        if method_name == "closed_linear":
            return EfficientPriorMethod("linear", **sel["linear"])
        if method_name == "readout_tag":
            return engine.ReadoutMethod("readout", tag=True, **sel["gradient"]["readout_tag"])
        if method_name == "finetune_full":
            return engine.GradientMethod("full", **sel["gradient"]["finetune_full"])
        raise ValueError(method_name)

    cache = np.load(OUT / f"{task}_prefix_cache.npz")
    manifest = json.loads((OUT / f"{task}_prefix_manifest.json").read_text())
    rows = []
    for i, meta in enumerate(manifest):
        s = SimpleNamespace(name=meta["name"], x=cache[f"x{i}"], yz=cache[f"y{i}"],
                            fit=cache[f"mask{i}"], fit_end=meta["prefix_bins"])
        # Warm initialization, matrix factorization and forward/backward kernels on
        # small arrays, without precomputing any benchmark-prefix features.
        engine.seed_all(101)
        model.train()
        warm = torch.tensor(s.x[:min(256, len(s.x))][None], device=device)
        model.zero_grad(set_to_none=True)
        pred, warm_hidden = model(warm)
        pred.square().mean().backward()
        model.zero_grad(set_to_none=True)
        model.eval()
        chol = torch.linalg.cholesky(torch.eye(64, dtype=torch.float64, device=device))
        del warm, pred, warm_hidden, chol
        gc.collect()
        if device == "cuda":
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            base_alloc = torch.cuda.memory_allocated()
            torch.cuda.reset_peak_memory_stats()
        else:
            base_alloc = None
            try:
                import ctypes
                ctypes.CDLL("libc.so.6").malloc_trim(0)
            except OSError:
                pass
        engine.seed_all(101)
        with ResidentMonitor() as monitor:
            t0 = time.perf_counter()
            method = make()
            method.start(model, [], c, 101)
            # No score(0), no suffix, and an empty feature cache: the full prefix
            # pass is inside the measurement for every adaptation family.
            macs = method.adapt(s)
            if device == "cuda":
                torch.cuda.synchronize()
            seconds = time.perf_counter() - t0
        native_persistent = int(method.memory())
        persistent = native_persistent
        if method_name == 'hdc_primal':
            persistent += method.W0.numel()
        elif method_name == 'readout_tag':
            persistent += sum(p.numel() for p in method.shared.parameters())
        elif method_name == 'finetune_full':
            persistent += method.base.parameter_count()
        returned_macs = int(macs)
        if method_name == 'hdc_primal':
            n, d = meta['labeled_bins'], 129 + 2 * sel['hdc']['dimension']
            macs += n*d*(d-1)//2
        row = dict(**meta, seconds=seconds, analytic_macs=int(macs),
                   method_returned_macs=returned_macs,
                   process_rss_base_bytes=monitor.base, process_rss_peak_bytes=monitor.peak,
                   process_rss_increment_bytes=monitor.peak-monitor.base,
                   cuda_baseline_allocated_bytes=base_alloc,
                   cuda_peak_allocated_bytes=int(torch.cuda.max_memory_allocated()) if device == "cuda" else None,
                   cuda_peak_reserved_bytes=int(torch.cuda.max_memory_reserved()) if device == "cuda" else None,
                   persistent_scalars_after_update=int(persistent),
                   method_memory_return_value=native_persistent)
        if row["cuda_peak_allocated_bytes"] is not None:
            row["cuda_increment_allocated_bytes"] = row["cuda_peak_allocated_bytes"]-base_alloc
        rows.append(row)
        del method, s
        gc.collect()
    if current_identity(task, method_name, selection_root) != identity:
        raise RuntimeError('Benchmark source or inputs changed while the worker was running')
    if host_environment() != measurement_environment:
        raise RuntimeError('CPU model or affinity changed while the benchmark worker was running')
    result = dict(task=task, method=method_name, device=device, repetition=repeat,
                  **identity,
                  seed=101, threads=threads, rows=rows, torch=torch.__version__,
                  numpy=np.__version__, python=platform.python_version(),
                  platform=platform.platform(),
                  cpu=measurement_environment['cpu_model'], measurement_environment=measurement_environment,
                  system_load_start=system_load_start, system_load_end=list(os.getloadavg()),
                  gpu=torch.cuda.get_device_name() if torch.cuda.is_available() else None,
                  protocol="fresh process; pretrained model; normalized prefixes prepared before timing; isolated session update; empty feature cache; prefix only; method initialization, forward pass and update timed; no evaluation suffix; data loading and normalization excluded equally for all methods")
    path = OUT / "raw" / f"{device}_{task}_{method_name}_r{repeat}.json"
    write(path, result)
    print(json.dumps(dict(file=str(path), mean_seconds=float(np.mean([r["seconds"] for r in rows])))), flush=True)


def collect_summary(expected_repeats=3, selection_root=None, threads=4, resource_root=None):
    """Revalidate every raw run against current inputs before deriving any summary."""
    root = OUT if resource_root is None else Path(resource_root)
    selection_root = CONFIRMED_SELECTION if selection_root is None else Path(selection_root)
    grouped = {}
    raw_hashes = {}
    for p in sorted((root / "raw").glob("*.json")):
        d = json.loads(p.read_text())
        d = verify_existing(p, d['task'], d['method'], selection_root, threads,
                            d['device'], d['repetition'], root, match_host=False)
        grouped.setdefault((d["device"], d["task"], d["method"]), []).append(d)
        raw_hashes[p.name] = fingerprint(p)
    if not grouped:
        raise ValueError('No resource measurements to aggregate')
    identities = {key:current_identity(key[1],key[2],selection_root,root) for key in grouped}
    return summarize_groups(grouped, raw_hashes, identities, expected_repeats, threads)


def summarize_groups(grouped, raw_hashes, identities, expected_repeats, threads):
    """Derive the same summaries from either live-verified or manifested records."""
    import numpy as np
    environments = {json.dumps(r['measurement_environment'],sort_keys=True) for rows in grouped.values() for r in rows}
    if len(environments) != 1:
        raise ValueError('CPU model or inherited affinity differs across benchmark runs')
    summary = []
    for (device, task, method), records in grouped.items():
        if sorted(r['repetition'] for r in records) != list(range(expected_repeats)):
            raise ValueError((device, task, method, 'incomplete repetitions'))
        sessions = {}
        for d in records:
            for r in d["rows"]:
                sessions.setdefault(r["name"], []).append(r)
        per = []
        for name, rows in sessions.items():
            per.append(dict(session=name, repeats=len(rows), median_seconds=float(np.median([r["seconds"] for r in rows])),
                            max_peak_bytes=max(r["cuda_peak_allocated_bytes"] if device=="cuda" else r["process_rss_peak_bytes"] for r in rows),
                            max_increment_bytes=max(r["cuda_increment_allocated_bytes"] if device=="cuda" else r["process_rss_increment_bytes"] for r in rows)))
        summary.append(dict(device=device, task=task, method=method, sessions=per,
                            threads=threads, **identities[(device,task,method)],
                            measurement_environment=records[0]['measurement_environment'],
                            system_load_by_repetition=[dict(repetition=r['repetition'],start=r['system_load_start'],end=r['system_load_end']) for r in records],
                            raw_records_sha256={f'{device}_{task}_{method}_r{r["repetition"]}.json':
                                raw_hashes[f'{device}_{task}_{method}_r{r["repetition"]}.json'] for r in records},
                            mean_session_median_seconds=float(np.mean([r["median_seconds"] for r in per])),
                            max_peak_bytes=max(r["max_peak_bytes"] for r in per),
                            max_increment_bytes=max(r["max_increment_bytes"] for r in per)))
    return summary


def aggregate(expected_repeats=3, selection_root=None, threads=4, resource_root=None):
    root = OUT if resource_root is None else Path(resource_root)
    summary = collect_summary(expected_repeats, selection_root, threads, root)
    write(root / "summary.json", summary)
    identity_keys = set(included_identity(summary[0]['task'],summary[0]['method'],
                        CONFIRMED_SELECTION if selection_root is None else selection_root,root))
    identity_keys |= {'checkpoint_sha256','prefix_cache_sha256','software_versions'}
    validation = dict(schema_version=1, all_live_input_checks_passed=True,
                      validation_scope='Actual checkpoints and prefix caches, included sources and selections, and every raw repetition verified before aggregation.',
                      expected_repeats=expected_repeats, threads=threads,
                      summary_sha256=fingerprint(root/'summary.json'),
                      raw_records_sha256={key:value for group in summary for key,value in group['raw_records_sha256'].items()},
                      groups=[dict(device=g['device'],task=g['task'],method=g['method'],
                                   measurement_environment=g['measurement_environment'],
                                   system_load_by_repetition=g['system_load_by_repetition'],
                                   identity={key:g[key] for key in sorted(identity_keys)}) for g in summary])
    write(root / 'validation_manifest.json', validation)
    print(json.dumps(summary, indent=2))
    return summary


def validate_recorded_resources(expected_repeats=3, selection_root=None, threads=4, resource_root=None):
    """Validate a relocated records archive without checkpoints, caches or CUDA.

    This requires the manifest emitted by successful live aggregation. Omitted
    large-input hashes remain bound to that evidence; included files are rehashed.
    """
    root = OUT if resource_root is None else Path(resource_root)
    selection_root = CONFIRMED_SELECTION if selection_root is None else Path(selection_root)
    validation = json.loads((root/'validation_manifest.json').read_text())
    if (validation.get('schema_version') != 1 or validation.get('all_live_input_checks_passed') is not True
            or validation.get('expected_repeats') != expected_repeats or validation.get('threads') != threads):
        raise ValueError('Missing or incompatible live resource-validation evidence')
    if validation.get('summary_sha256') != fingerprint(root/'summary.json'):
        raise ValueError('Resource summary differs from live-validated evidence')
    paths = sorted((root/'raw').glob('*.json'))
    raw_hashes = {p.name:fingerprint(p) for p in paths}
    if not paths or raw_hashes != validation.get('raw_records_sha256'):
        raise ValueError('Resource raw records differ from live-validated evidence')
    identities = {}
    for group in validation.get('groups', []):
        key = group['device'],group['task'],group['method']
        identity = group['identity']
        included = included_identity(key[1],key[2],selection_root,root)
        if key in identities or set(identity) != set(included)|{'checkpoint_sha256','prefix_cache_sha256','software_versions'}:
            raise ValueError('Duplicated or malformed resource validation group')
        if any(identity.get(k) != value for k,value in included.items()):
            raise ValueError(f'Included resource inputs differ from live-validated evidence: {key}')
        for field in ['checkpoint_sha256','prefix_cache_sha256']:
            digest = identity[field]
            if not isinstance(digest,str) or len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
                raise ValueError(f'Malformed recorded large-input hash: {field}')
        identities[key] = identity
    grouped = {}
    for path in paths:
        rec = json.loads(path.read_text())
        key = rec['device'],rec['task'],rec['method']
        if key not in identities:
            raise ValueError('Raw resource group absent from live validation manifest')
        verified = verify_record(path,key[1],key[2],threads,key[0],rec['repetition'],identities[key],root)
        grouped.setdefault(key,[]).append(verified)
    if set(grouped) != set(identities):
        raise ValueError('Resource validation manifest has missing raw groups')
    summary = summarize_groups(grouped,raw_hashes,identities,expected_repeats,threads)
    manifested = {(g['device'],g['task'],g['method']):g for g in validation['groups']}
    for group in summary:
        expected = manifested[(group['device'],group['task'],group['method'])]
        if any(group[key] != expected.get(key) for key in ['measurement_environment','system_load_by_repetition']):
            raise ValueError('Recorded benchmark environment differs from live-validation manifest')
    if summary != json.loads((root/'summary.json').read_text()):
        raise ValueError('Resource summary differs from manifested raw measurements')
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("stage", choices=["prepare", "worker", "run", "aggregate"])
    ap.add_argument("--tasks", nargs="+", default=["h1", "m1", "m2"])
    ap.add_argument("--methods", nargs="+", default=["hdc", "closed_linear", "readout_tag", "finetune_full", "hdc_primal"])
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--repeat", type=int, default=0)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--selection-root", default=str(CONFIRMED_SELECTION))
    args = ap.parse_args()
    if args.stage == "prepare":
        for task in args.tasks:
            prepare(task)
    elif args.stage == "worker":
        worker(args.tasks[0], args.methods[0], args.device, args.repeat, args.threads, args.selection_root)
    elif args.stage == "aggregate":
        aggregate(args.repeats, args.selection_root, args.threads)
    else:
        # Rotate method order across repetitions to reduce thermal/order bias.
        for rep in range(args.repeats):
            order = args.methods[rep % len(args.methods):]+args.methods[:rep % len(args.methods)]
            for task in args.tasks:
                for method in order:
                    target=OUT / "raw" / f"{args.device}_{task}_{method}_r{rep}.json"
                    if target.exists():
                        verify_existing(target, task, method, args.selection_root, args.threads, args.device, rep)
                        continue
                    subprocess.run([sys.executable, __file__, "worker", "--tasks", task, "--methods", method,
                                    "--device", args.device, "--repeat", str(rep), "--threads", str(args.threads),
                                    "--selection-root", args.selection_root], check=True)
        aggregate(args.repeats, args.selection_root, args.threads)
