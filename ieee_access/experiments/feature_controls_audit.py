"""Validate frozen-feature controls and derive consistent deployment counts.

Raw fit records preserve the original primal accounting. This audit adds exact
primal/dual MACs and the retained prior copy without changing fitted results.
"""
import datetime
import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import scipy
import statsmodels
import torch
import feature_controls as fc


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        while block := stream.read(2 ** 20):
            h.update(block)
    return h.hexdigest()


def main():
    out = fc.OUT
    previous_path = out / 'provenance.json'
    previous = json.loads(previous_path.read_text()) if previous_path.exists() else {}
    fit_paths = sorted([*(out / 'inner').rglob('*.json'), *(out / 'outer').rglob('*.json')])
    fit_hashes = {str(p.relative_to(out)): sha(p) for p in fit_paths}
    previous_hashes = previous.get('result_sha256', {})
    unchanged = bool(previous_hashes) and all(previous_hashes.get(k) == v for k, v in fit_hashes.items())
    equivalence = fc.residual_equivalence_check()
    residual_checks, maximum_residual_error = 0, 0.0
    checks = dict(feature_caches=0, session_arrays=0, outer_records=0,
                  inner_configurations=0, selection_records=0, reference_comparisons=0)
    metadata, resources = [], {}
    for task in ['h1', 'm1', 'm2']:
        resources[task] = {}
        for split, seeds in [('inner', fc.DEV), ('outer', fc.SEEDS)]:
            for seed in seeds:
                folder = fc.path_for(task, split, seed)
                p = folder / 'metadata.json'
                meta = json.loads(p.read_text())
                prior = np.load(folder / 'prior.npy')
                assert np.isfinite(prior).all()
                assert sha(fc.REPO / meta['checkpoint']) == meta['checkpoint_sha256']
                for name, expected in meta['source_sha256'].items():
                    assert sha(fc.CODE / name) == expected
                files = {}
                for row in meta['sessions']:
                    f = folder / row['file']
                    files[f.name] = sha(f)
                    with np.load(f) as arrays:
                        assert len(arrays['fit_gf']) == len(arrays['fit_y']) == row['fit_bins']
                        assert len(arrays['score_gf']) == len(arrays['score_y']) == row['score_bins']
                        width = meta['prior_width'] - 1 + meta['projection_width']
                        assert arrays['fit_gf'].shape[1] == arrays['score_gf'].shape[1] == width
                        assert all(np.isfinite(arrays[key]).all() for key in arrays.files)
                        # Actual cached fitting rows, with zero prior expansion rows:
                        # the expansion values cannot affect their product with zero.
                        if len(arrays['fit_gf']):
                            gf, y = arrays['fit_gf'][:32], arrays['fit_y'][:32].astype(np.float64)
                            base = np.column_stack([gf[:, :len(prior)-1], np.ones(len(gf), np.float32)])
                            for kind in fc.KINDS:
                                cfg = fc.config_for(task, kind, 1.)
                                extra = meta['projection_width'] if kind == 'linear_gf' else 2 * cfg['dimension']
                                X = np.column_stack([base, np.zeros((len(base), extra))])
                                P = np.vstack([prior, np.zeros((extra, prior.shape[1]))])
                                old, new = y - X @ P, fc.prior_residual(X, y, prior)
                                error = float(np.max(np.abs(old-new)))
                                assert np.allclose(old, new, atol=1e-12, rtol=1e-13)
                                maximum_residual_error = max(maximum_residual_error, error)
                                residual_checks += 1
                    checks['session_arrays'] += 1
                files['prior.npy'] = sha(folder / 'prior.npy')
                metadata.append(dict(path=str(p.relative_to(fc.REPO)), sha256=sha(p),
                                     checkpoint=meta['checkpoint'],
                                     checkpoint_sha256=meta['checkpoint_sha256'],
                                     cache_files_sha256=files))
                checks['feature_caches'] += 1
        for kind in fc.KINDS:
            selection = json.loads((out / 'selection' / f'{task}_{kind}.json').read_text())
            assert not selection['selected_at_boundary']
            assert selection['dev_seeds'] == fc.DEV
            scores = []
            for seed in fc.DEV:
                records = json.loads((out / 'inner' / task / f'{kind}_s{seed}.json').read_text())
                assert [r['configuration']['lam'] for r in records] == selection['lambda_grid']
                scores.append([r['summary']['acc'] for r in records])
                checks['inner_configurations'] += len(records)
            assert np.array_equal(scores, selection['inner_acc_by_seed'])
            index = int(np.argmax(np.mean(scores, axis=0)))
            assert selection['selected_lambda'] == selection['lambda_grid'][index]
            checks['selection_records'] += 1
            counts = []
            for seed in fc.SEEDS:
                r = json.loads((out / 'outer' / task / f'{kind}_s{seed}.json').read_text())
                R = np.asarray(r['R'])
                k, T = r['held_in'], len(R) - 1
                assert R.shape == (T + 1, k + T) and np.isfinite(R).all()
                assert r['configuration']['lam'] == selection['selected_lambda']
                assert r['summary']['bwt'] == 0 and np.all(R[:, :k] == R[0, :k])
                assert abs(R[-1].mean() - r['summary']['acc']) < 1e-14
                for i in range(T):
                    assert np.all(R[:i + 1, k + i] == R[0, k + i])
                    assert np.all(R[i + 1:, k + i] == R[i + 1, k + i])
                folder = fc.path_for(task, 'outer', seed)
                meta = json.loads((folder / 'metadata.json').read_text())
                prior = np.load(folder / 'prior.npy')
                d, o = r['head_width'], prior.shape[1]
                costs = []
                for row in meta['sessions'][k:]:
                    n = row['fit_bins']
                    q = min(n, d)
                    mapping = (n * meta['projection_width'] * 2 * r['configuration']['dimension']
                               if kind == 'relu_matched' else 0)
                    mac = int(row['prefix_bins'] * sum(meta['backbone_macs'].values())
                              + mapping + n * d * q + n * d * o
                              + q ** 3 // 6 + q * q * o)
                    costs.append(dict(session=row['name'], N=n, d=d, q=q,
                                      solver='dual' if n < d else 'primal', analytic_macs=mac))
                retained = r.get('legacy_memory', r['memory'])[-1] + prior.size
                assert r['summary']['memory'] == retained
                assert r['adapt_macs'] == [row['analytic_macs'] for row in costs]
                counts.append(dict(seed=seed, solver_records=costs,
                                   mean_adapt_macs=float(np.mean([x['analytic_macs'] for x in costs])),
                                   retained_scalars_after_final_session=int(retained),
                                   prior_scalars=int(prior.size)))
                checks['outer_records'] += 1
            assert len({x['mean_adapt_macs'] for x in counts}) == 1
            assert len({x['retained_scalars_after_final_session'] for x in counts}) == 1
            resources[task][kind] = dict(records=counts, mean_adapt_macs=counts[0]['mean_adapt_macs'],
                                        retained_scalars_after_final_session=counts[0]['retained_scalars_after_final_session'])
        for row in json.loads((out / 'verification' / f'{task}_s101.json').read_text()):
            assert row['passed'] and row['max_abs_R_difference'] < 2e-5
            checks['reference_comparisons'] += 1
    assert checks == dict(feature_caches=39, session_arrays=322, outer_records=60,
                          inner_configurations=198, selection_records=6, reference_comparisons=6)
    fc.write(out / 'corrected_resource_accounting.json', resources)
    hashes = {str(p.relative_to(out)): sha(p) for p in sorted(out.rglob('*.json'))
              if p.name != 'provenance.json'}
    assert all(hashes[k] == v for k, v in fit_hashes.items()), 'Audit altered a fitted result'
    provenance = dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                      runner_sha256=sha(Path(fc.__file__)), audit_source_sha256=sha(Path(__file__)),
                      checks=checks, all_checks_passed=True, seeds=fc.SEEDS, inner_seeds=fc.DEV,
                      data_manifest_sha256={t: sha(fc.STUDY / 'results/provenance' / f'manifest_{t}.json')
                                            for t in ['h1', 'm1', 'm2']},
                      environment=dict(python=platform.python_version(), numpy=np.__version__,
                                       scipy=scipy.__version__, torch=torch.__version__,
                                       statsmodels=statsmodels.__version__),
                      cache_provenance=metadata, result_sha256=hashes,
                      residual_optimization_validation=dict(
                          synthetic_predictions=equivalence, sampled_cached_residual_cases=residual_checks,
                          max_cached_residual_absolute_difference=maximum_residual_error,
                          fit_records_unchanged_from_previous_audit=unchanged,
                          previous_runner_sha256=previous.get('runner_sha256'),
                          unchanged_fit_record_sha256=fit_hashes,
                          reason='Only prior residual multiplication skips known-zero expansion rows; fitted scores are unchanged and no fits were rerun.'),
                      reproduction_stages=['extract --threads 4', 'verify --seeds 101 --threads 4',
                                           'select --threads 4', 'evaluate --threads 4', 'aggregate --threads 1'],
                      resource_note='Current analytic MACs count dense Gram N*d*q with q=min(N,d); retained state includes the prior copy. Earlier conventions remain in legacy_primal_adapt_macs, legacy_symmetric_adapt_macs, and legacy_memory. Cached-feature seconds are not deployment timing.')
    fc.write(out / 'provenance.json', provenance)
    print(json.dumps(checks))
    print(json.dumps({task: {kind: {key: value for key, value in record.items() if key != 'records'}
                             for kind, record in kinds.items()} for task, kinds in resources.items()}, indent=2))


if __name__ == '__main__':
    main()
