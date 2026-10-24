"""Update analytic accounting only; retain every fitted score and legacy count."""
import hashlib
import json
import numpy as np
import feature_controls as fc


def matrix_hash(record):
    return hashlib.sha256(np.asarray(record['R'], dtype='<f8').tobytes()).hexdigest()


def main():
    changes = []
    for stage, split in [('inner', 'inner'), ('outer', 'outer')]:
        for path in sorted((fc.OUT / stage).glob('*/*.json')):
            value = json.loads(path.read_text())
            records = value if isinstance(value, list) else [value]
            before_file = fc.sha(path)
            before_matrices = [matrix_hash(r) for r in records]
            for rec in records:
                root = fc.path_for(rec['task'], split, rec['seed'])
                meta = json.loads((root / 'metadata.json').read_text())
                prior = np.load(root / 'prior.npy')
                d, o = rec['head_width'], prior.shape[1]
                costs, symmetric_costs = [], []
                for row in meta['sessions'][meta['held_in']:]:
                    n = row['fit_bins']; q = min(n, d)
                    encoding = n * meta['projection_width'] * 2 * rec['configuration']['dimension'] if rec['method'] == 'relu_matched' else 0
                    common = row['prefix_bins'] * sum(meta['backbone_macs'].values()) + encoding + n * d * o + q ** 3 // 6 + q * q * o
                    costs.append(int(common + n * d * q))
                    symmetric_costs.append(int(common + n * d * (q + 1) // 2))
                rec.setdefault('legacy_primal_adapt_macs', rec['adapt_macs'])
                rec.setdefault('legacy_memory', rec['memory'])
                rec.setdefault('legacy_symmetric_adapt_macs', symmetric_costs)
                rec['adapt_macs'] = costs
                rec['memory'] = [v + prior.size for v in rec['legacy_memory']]
                rec['summary']['adapt_macs'] = float(np.mean(costs))
                rec['summary']['memory'] = rec['memory'][-1]
                rec['computation_note'] = 'Analytic MACs count the implemented dense Gram product N*d*q, q=min(N,d). Retained scalars include the prior copy. Seconds are cached-feature analysis and are not deployment timings.'
            assert before_matrices == [matrix_hash(r) for r in records]
            fc.write(path, records if isinstance(value, list) else records[0])
            changes.append(dict(path=str(path.relative_to(fc.OUT)), before_sha256=before_file,
                                after_sha256=fc.sha(path), R_sha256=before_matrices,
                                fitted_matrices_unchanged=True))
    assert len(changes) == 78
    fc.write(fc.OUT / 'dense_accounting_migration.json', dict(records=changes, fitted_record_count=258,
                                                       all_fitted_matrices_unchanged=True,
                                                       explanation=__doc__))
    print('Updated analytic accounting in 258 fitted records; all R hashes unchanged.')


if __name__ == '__main__':
    main()
