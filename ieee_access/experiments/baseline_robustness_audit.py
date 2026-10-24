#!/usr/bin/env python3
"""Audit the fixed confirmation design and exported stream records without GPU use."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
DEFAULT_ROOT = HERE / 'results' / 'baseline_robustness'


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summary_from_matrix(record):
    r = np.asarray(record['R'], dtype=float)
    k, t = record['held_in'], r.shape[0] - 1
    last = r[-1]
    bwt_in = (last[:k] - r[0, :k]).mean()
    bwt_out = np.mean([last[k + i] - r[i + 1, k + i]
                       for i in range(t - 1)]) if t > 1 else 0.0
    return dict(acc=last.mean(), acc_held_in=last[:k].mean(),
                acc_held_out=last[k:].mean(),
                plasticity=np.mean([r[i + 1, k + i] for i in range(t)]),
                bwt=(bwt_in * k + bwt_out * (t - 1)) / (k + t - 1),
                bwt_held_in=bwt_in, bwt_held_out=bwt_out,
                zero_shot_held_out=r[0, k:].mean())


def audit(root, require_complete=True):
    plan = read(root / 'plan.json')
    errors, missing, counts = [], [], dict(inner=0, outer=0, selected_exports=0)
    expected = dict(inner=sum(len(candidates) for design in plan['tasks'].values()
                              for candidates in design['methods'].values()) * len(plan['dev_seeds']),
                    outer=0,
                    selected_exports=sum(len(design['methods']) for design in plan['tasks'].values())
                                     * len(plan['outer_seeds']))
    plan_sha = sha(root / 'plan.json')
    runner_hashes = set()
    artifact_hashes = {'plan.json': plan_sha}
    def check(condition, message):
        if not condition:
            errors.append(message)
    def require(path):
        if path.exists():
            artifact_hashes[str(path.relative_to(root))] = sha(path)
            return read(path)
        missing.append(str(path.relative_to(root)))
        return None
    for task, design in plan['tasks'].items():
        selection = require(root / 'selection' / f'{task}.json')
        compatible = require(root / 'selection_for_analysis' / f'{task}.json')
        if selection is not None and compatible is not None:
            check(compatible['gradient'] == selection['gradient'],
                  task + ': analysis configuration matches selected configuration')
            check(compatible['confirmation']['complete'] == selection['complete'],
                  task + ': analysis selection completeness')
            check(compatible['confirmation']['plan_sha256'] == plan_sha,
                  task + ': analysis selection plan hash')
        for method, candidates in design['methods'].items():
            complete_candidates = []
            for i, candidate in enumerate(candidates):
                vals = []
                for seed in plan['dev_seeds']:
                    rel = f"inner/{task}/{method}/{candidate['id']}_s{seed}.json"
                    record = require(root / rel)
                    if record is None:
                        continue
                    counts['inner'] += 1
                    runner_hashes.add(record['environment']['runner_sha256'])
                    check(record['seed'] == seed == record['rng_seed'], rel + ': RNG identity')
                    check(record['candidate'] == candidate, rel + ': candidate identity')
                    check(record['environment']['plan_sha256'] == plan_sha, rel + ': plan hash')
                    if record['status'] == 'complete':
                        check(np.isfinite(record['final_scores']).all(), rel + ': finite score')
                        check(np.isclose(np.mean(record['final_scores']), record['acc'],
                                         rtol=0, atol=1e-12), rel + ': ACC consistency')
                        vals.append(record['acc'])
                if len(vals) == len(plan['dev_seeds']):
                    complete_candidates.append((np.mean(vals), candidate['incumbent'], -i, candidate))
            if selection is None or method not in selection['methods']:
                continue
            row = selection['methods'][method]
            if complete_candidates:
                best = max(complete_candidates, key=lambda x: x[:3])
                check(best[-1]['id'] == row['selected']['id'], f'{task}/{method}: inner-only winner')
                check(np.isclose(best[0], row['selected']['inner_mean_acc'], rtol=0, atol=1e-12),
                      f'{task}/{method}: selected mean')
            choices = {row[role]['id']: row[role] for role in ('incumbent', 'selected')}
            expected['outer'] += len(choices) * len(plan['outer_seeds'])
            for cid, candidate in choices.items():
                for seed in plan['outer_seeds']:
                    rel = f'outer/{task}/{method}/{cid}_s{seed}.json'
                    record = require(root / rel)
                    if record is None:
                        continue
                    counts['outer'] += 1
                    r = np.asarray(record['R'], dtype=float)
                    check(np.isfinite(r).all(), rel + ': finite matrix')
                    check(r.shape == (len(record['names']) - record['held_in'] + 1,
                                      len(record['names'])), rel + ': stream shape')
                    check(record['seed'] == seed == record['rng_seed'], rel + ': RNG identity')
                    check(record['config'] == candidate['config'], rel + ': config identity')
                    check(record['environment']['plan_sha256'] == plan_sha, rel + ': plan hash')
                    runner_hashes.add(record['environment']['runner_sha256'])
                    for key, val in summary_from_matrix(record).items():
                        check(np.isclose(val, record['summary'][key], rtol=0, atol=1e-12),
                              rel + ': summary ' + key)
                    if cid == row['selected']['id']:
                        for destination in ('stream', 'export'):
                            target = root / destination / task / 'full' / f'{method}_s{seed}.json'
                            export = require(target)
                            if export is not None:
                                check(export == record, str(target.relative_to(root)) + ': byte-equivalent record')
                        if (root / 'stream' / task / 'full' / f'{method}_s{seed}.json').exists():
                            counts['selected_exports'] += 1
    validation = require(root / 'validation.json')
    aggregate = require(root / 'summary.json')
    if aggregate is not None:
        check(set(aggregate['tasks']) == set(plan['tasks']), 'Aggregate summary task coverage')
    if validation is not None:
        check(validation['passed'], 'Native-versus-cached implementation validation')
        check(validation['environment']['runner_sha256'] == sha(HERE / 'baseline_robustness.py'),
              'Validation runner hash matches the executed runner')
        check(validation['environment']['plan_sha256'] == plan_sha,
              'Validation plan hash matches the frozen plan')
    check(len(runner_hashes) <= 1, 'All records were generated by one immutable runner version')
    check(not runner_hashes or runner_hashes == {sha(HERE / 'baseline_robustness.py')},
          'Record runner hash matches the executed runner')
    if require_complete:
        check(not missing, 'All planned candidate, incumbent, selected, and export files exist')
        for field, value in expected.items():
            check(counts[field] == value, 'Complete coverage: ' + field)
        for folder, field in [('inner', 'inner'), ('outer', 'outer'),
                              ('stream', 'selected_exports'), ('export', 'selected_exports')]:
            check(len(list((root / folder).glob('*/*/*.json'))) == expected[field],
                  'No missing or extra records in ' + folder)
    source_status = {name: sha(REPO / 'mjcs' / 'code' / name) == digest
                     for name, digest in plan['original_files'].items()}
    snapshot_status = {name: (root / 'source_snapshot' / name).exists()
                      and sha(root / 'source_snapshot' / name) == digest
                      for name, digest in plan['original_files'].items()}
    check(all(source_status.values()), 'Original source files match the frozen plan')
    check(all(snapshot_status.values()), 'Preserved source snapshots match the frozen plan')
    report = dict(passed=not errors, complete=not missing, counts=counts,
                  expected_counts=expected, plan_sha256=plan_sha,
                  audit_script_sha256=sha(Path(__file__)),
                  runner_sha256=sha(HERE / 'baseline_robustness.py'),
                  artifact_sha256=artifact_hashes,
                  errors=errors, missing=missing, runner_hashes=sorted(runner_hashes),
                  source_files_still_match_frozen_plan=source_status,
                  source_snapshots_match_frozen_plan=snapshot_status,
                  source_note='The original source is intentionally frozen; revised experiments are implemented in separate runners.')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    parser.add_argument('--partial', action='store_true')
    args = parser.parse_args()
    result = audit(args.root, not args.partial)
    if result['complete']:
        (args.root / 'audit.json').write_text(json.dumps(result, indent=2) + '\n')
    displayed = {k: v for k, v in result.items() if k != 'artifact_sha256'}
    print(json.dumps({**displayed, 'artifact_hash_count': len(result['artifact_sha256']),
                      'missing': result['missing'][:10],
                      'missing_count': len(result['missing'])}, indent=2))
    raise SystemExit(0 if result['passed'] else 1)


if __name__ == '__main__':
    main()
