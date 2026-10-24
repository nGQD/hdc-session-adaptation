#!/usr/bin/env python3
"""Verify confirmed sensitivity coverage and provenance without rerunning training."""
import json

import confirm_sensitivity as c


RUNNER_SHA = '6f04f8d2d9b4e6d4e88253181390d823daa3ff5a80046bc9b7cde4cfaf123a48'
TASKS = ('h1', 'm1', 'm2')
SEEDS = (101, 113, 127, 131, 137)
METHODS = ('finetune_full', 'replay')
DOSES = (0.125, 0.25, 0.5)
STREAM_FIELDS = ('R', 'names', 'held_in', 'adapt_macs', 'adapt_seconds',
                 'memory', 'start_macs', 'start_seconds')


def main():
    assert c.file_sha(c.__file__) == RUNNER_SHA, 'Frozen sensitivity source differs'
    expected_paths, hashes = set(), {}
    counts = dict(dose_streams=0, dose_aggregates=0, budget_streams=0, main_reuses=0)
    def load(path):
        expected_paths.add(path.resolve())
        hashes[str(path.relative_to(c.OUT))] = c.file_sha(path)
        return json.loads(path.read_text())
    for task in TASKS:
        selected, selection_sha = c.selection(task)
        for seed in SEEDS:
            for method in METHODS:
                identity = c.computation_identity(task, method, 'full', seed, selected, selection_sha)
                rows, stream_hashes = [], {}
                for fraction in DOSES:
                    path = c.OUT / 'dose_stream' / task / f'{method}_p{fraction}_s{seed}.json'
                    record = load(path)
                    expected = dict(**identity, prefix_fraction=fraction,
                                    input_identity=record['input_identity'])
                    if fraction == 0.5:
                        source = c.OUT.parent / 'baseline_robustness' / 'stream' / task / 'full' / f'{method}_s{seed}.json'
                        original = json.loads(source.read_text())
                        expected.update(reused_main_stream=str(source.relative_to(c.ROOT)),
                                        reused_main_stream_sha256=c.file_sha(source),
                                        reuse_reason='Identical checkpoint, seed, configuration, fitting prefix and score mask.')
                        assert all(record[key] == original[key] for key in STREAM_FIELDS), str(path)
                        counts['main_reuses'] += 1
                    c.verify_saved(path, expected, stream=True)
                    counts['dose_streams'] += 1
                    rows.append(record['summary'])
                    stream_hashes[str(path.relative_to(c.OUT))] = c.file_sha(path)
                target = c.OUT / 'dose' / task / f'{method}_s{seed}.json'
                load(target)
                c.verify_saved(target, dict(**identity, dose=list(DOSES), stream_records_sha256=stream_hashes,
                                           **{key: [row[key] for row in rows]
                                              for key in ('acc_held_out', 'acc', 'adapt_macs')}))
                counts['dose_aggregates'] += 1
            for variant in ('epochs25', 'epochs100'):
                path = c.OUT / 'stream' / task / variant / f'finetune_full_s{seed}.json'
                record = load(path)
                identity = c.computation_identity(task, 'finetune_full', variant, seed, selected, selection_sha)
                c.verify_saved(path, dict(**identity, input_identity=record['input_identity']), stream=True)
                counts['budget_streams'] += 1
    actual = {path.resolve() for folder in ('dose', 'dose_stream', 'stream')
              for path in (c.OUT / folder).rglob('*.json')}
    assert actual == expected_paths, 'Missing or extra sensitivity records'
    expected_counts = dict(dose_streams=90, dose_aggregates=30, budget_streams=30, main_reuses=30)
    assert counts == expected_counts
    report = dict(passed=True, complete=True, counts=counts, expected_counts=expected_counts,
                  runner_sha256=RUNNER_SHA, audit_script_sha256=c.file_sha(__file__),
                  artifact_sha256=hashes,
                  checks=['Selected configuration and RNG seed identities',
                          'Current checkpoint, selection, implementation and manifest hashes',
                          'Finite complete stream matrices and reconstructed summaries',
                          'Dose aggregates equal their hashed supporting stream records',
                          'Every half-prefix stream equals its authoritative main stream',
                          'Exact expected file coverage with no missing or extra records'])
    c.write(c.OUT / 'audit.json', report)
    print(json.dumps({key: value for key, value in report.items() if key != 'artifact_sha256'}, indent=2))


if __name__ == '__main__':
    main()
