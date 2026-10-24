#!/usr/bin/env python3
"""Run the revised analyses and bind their outputs to the complete study overlay.

This wrapper runs the unchanged statistical engine and the endpoint analysis.
Publication loaders verify its manifest rather than recomputing the analyses.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
TASKS = ['h1', 'm1', 'm2']
STAT_NAMES = ['aggregate', 'paired', 'mixed', 'tost', 'pareto', 'ablation']
ANALYSIS_SOURCES = [ROOT/'mjcs/code/stats.py', HERE/'endpoint_analysis.py', Path(__file__)]


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def checked_hashes(base, hashes):
    assert hashes, 'Empty provenance hash map'
    for relative, expected in hashes.items():
        path = base/relative
        assert path.is_file() and sha(path) == expected, f'Stale or missing provenance input: {path}'


def verify_overlay(study):
    """Reject partial, modified, or extra directory-discovered study inputs."""
    provenance = read(study/'provenance.json')
    assert provenance['schema_version'] == 2 and provenance['complete'], 'Study overlay is incomplete'
    records = provenance['records']
    outputs = {r['output']: r['output_sha256'] for r in records}
    assert records and len(outputs) == len(records), 'Missing or duplicate overlay output records'
    actual = {str(p.relative_to(study)) for folder in ['stream', 'selection', 'dose']
              for p in (study/folder).rglob('*.json')}
    assert set(outputs) == actual, 'Untracked or missing overlay output files'
    checked_hashes(study, outputs)
    checked_hashes(ROOT, provenance['inputs'])
    checked_hashes(ROOT, provenance['sources'])
    for record in records:
        assert provenance['inputs'][record['source']] == record['sha256']
    return provenance


def verify_analysis(study, endpoint_path=None):
    """Verify output hashes and the exact stream/source inputs used to derive them."""
    verify_overlay(study)
    path = study/'analysis_provenance.json'
    manifest = read(path)
    assert manifest['schema_version'] == 1 and manifest['complete'], 'Revised analysis is incomplete'
    assert manifest['overlay_sha256'] == sha(study/'provenance.json'), 'Analysis uses an earlier study overlay'
    protocol = manifest['protocol']
    assert protocol['bootstrap_resamples'] == 10000, 'Publication captions require 10,000 resamples'
    expected_inputs = {str(p.relative_to(study)) for p in (study/'stream').glob('*/*/*_s*.json')}
    assert set(manifest['inputs']) == expected_inputs, 'Analysis stream input coverage differs'
    checked_hashes(study, manifest['inputs'])
    expected_sources = {str(p.relative_to(ROOT)) for p in ANALYSIS_SOURCES}
    assert set(manifest['sources']) == expected_sources, 'Analysis source coverage differs'
    checked_hashes(ROOT, manifest['sources'])
    expected_stats = {f'stats/{task}_{name}.json' for task in TASKS for name in STAT_NAMES}
    expected_stats.add('stats/summary.json')
    assert set(manifest['statistics']) == expected_stats, 'Statistical output coverage differs'
    checked_hashes(study, manifest['statistics'])
    checked_hashes(study, manifest['endpoint_outputs'])
    endpoint = study/manifest['endpoint_json']
    assert str(manifest['endpoint_json']) in manifest['endpoint_outputs']
    if endpoint_path is not None:
        assert endpoint.resolve() == endpoint_path.resolve(), 'Endpoint is not the manifest-bound analysis'
    return manifest


def run(study, endpoint_dir, bootstrap=10000, statistics_seed=0, endpoint_seed=20261023):
    study, endpoint_dir = study.resolve(), endpoint_dir.resolve()
    verify_overlay(study)
    manifest_path = study/'analysis_provenance.json'
    # Invalidate any previous successful analysis before an interrupted rerun.
    if manifest_path.exists():
        manifest_path.unlink()
    before = dict(overlay_sha256=sha(study/'provenance.json'),
                  inputs={str(p.relative_to(study)): sha(p)
                          for p in sorted((study/'stream').glob('*/*/*_s*.json'))},
                  sources={str(p.relative_to(ROOT)): sha(p) for p in ANALYSIS_SOURCES})
    subprocess.run([sys.executable, str(ROOT/'mjcs/code/stats.py'), '--root', str(study),
                    '--out', str(study/'stats'), '--boot', str(bootstrap), '--seed', str(statistics_seed)], check=True)
    subprocess.run([sys.executable, str(HERE/'endpoint_analysis.py'), '--raw-root', str(study/'stream'),
                    '--output-dir', str(endpoint_dir), '--bootstrap', str(bootstrap),
                    '--bootstrap-seed', str(endpoint_seed)], check=True)
    verify_overlay(study)
    assert before['overlay_sha256'] == sha(study/'provenance.json'), 'Study changed during analysis'
    checked_hashes(study, before['inputs'])
    checked_hashes(ROOT, before['sources'])
    statistics = {f'stats/{task}_{name}.json': sha(study/'stats'/f'{task}_{name}.json')
                  for task in TASKS for name in STAT_NAMES}
    statistics['stats/summary.json'] = sha(study/'stats/summary.json')
    # The JSON is the authoritative input to tables and the supplement. The
    # convenience Markdown rendering is not included in the records-only bundle.
    endpoint_files = {os.path.relpath(endpoint_dir/'endpoint_analysis.json', study):
                      sha(endpoint_dir/'endpoint_analysis.json')}
    import numpy, scipy, statsmodels
    manifest = dict(schema_version=1, complete=True, **before, statistics=statistics,
                    endpoint_outputs=endpoint_files,
                    endpoint_json=os.path.relpath(endpoint_dir/'endpoint_analysis.json', study),
                    protocol=dict(bootstrap_resamples=bootstrap, statistics_seed=statistics_seed,
                                  endpoint_seed=endpoint_seed),
                    software=dict(python=sys.version, numpy=numpy.__version__, scipy=scipy.__version__,
                                  statsmodels=statsmodels.__version__))
    temporary = manifest_path.with_suffix('.tmp')
    temporary.write_text(json.dumps(manifest, indent=2)+'\n')
    temporary.replace(manifest_path)
    print(json.dumps(dict(analysis_manifest=str(manifest_path), streams=len(before['inputs']),
                          statistics=len(statistics), endpoint=str(endpoint_dir/'endpoint_analysis.json'))))
    return manifest


def self_test():
    """Exercise the integrity gates without accepting or analysing partial results."""
    with tempfile.TemporaryDirectory(prefix='revision_provenance_test_') as temporary:
        study = Path(temporary)
        (study/'stream/h1/full').mkdir(parents=True)
        record = study/'stream/h1/full/hdc_s101.json'
        record.write_text('{}\n')
        source = Path(__file__)
        rel = str(source.relative_to(ROOT))
        provenance = dict(schema_version=2, complete=True,
                          records=[dict(output=str(record.relative_to(study)), output_sha256=sha(record),
                                        source=rel, sha256=sha(source))],
                          inputs={rel:sha(source)}, sources={rel:sha(source)})
        path = study/'provenance.json'
        path.write_text(json.dumps(provenance))
        verify_overlay(study)
        (study/'stats').mkdir()
        statistics = {}
        for relative in [*[f'stats/{task}_{name}.json' for task in TASKS for name in STAT_NAMES],
                         'stats/summary.json']:
            output = study/relative
            output.write_text('{}\n')
            statistics[relative] = sha(output)
        endpoint = study/'endpoint_analysis.json'
        endpoint.write_text('{}\n')
        manifest = dict(schema_version=1, complete=True, overlay_sha256=sha(path),
                        inputs={str(record.relative_to(study)):sha(record)},
                        sources={str(p.relative_to(ROOT)):sha(p) for p in ANALYSIS_SOURCES},
                        statistics=statistics, endpoint_json=endpoint.name,
                        endpoint_outputs={endpoint.name:sha(endpoint)},
                        protocol=dict(bootstrap_resamples=10000))
        (study/'analysis_provenance.json').write_text(json.dumps(manifest))
        verify_analysis(study, endpoint)
        paired = study/'stats/h1_paired.json'
        paired.write_text('{"p_holm":0}\n')
        try:
            verify_analysis(study, endpoint)
        except AssertionError:
            pass
        else:
            raise AssertionError('Modified paired p value was accepted')
        paired.write_text('{}\n')
        endpoint.write_text('{"p_noninferiority":0}\n')
        try:
            verify_analysis(study, endpoint)
        except AssertionError:
            pass
        else:
            raise AssertionError('Modified endpoint p value was accepted')
        endpoint.write_text('{}\n')
        record.write_text('{"changed":true}\n')
        try:
            verify_overlay(study)
        except AssertionError:
            pass
        else:
            raise AssertionError('Modified overlay output was accepted')
        record.write_text('{}\n')
        extra = study/'stream/extra.json'
        extra.write_text('{}\n')
        try:
            verify_overlay(study)
        except AssertionError:
            pass
        else:
            raise AssertionError('Untracked overlay output was accepted')
        extra.unlink()
        provenance['complete'] = False
        path.write_text(json.dumps(provenance))
        try:
            verify_overlay(study)
        except AssertionError:
            pass
        else:
            raise AssertionError('Partial overlay was accepted')
    print(json.dumps(dict(passed=True, rejects_modified_untracked_and_partial_overlays=True,
                          rejects_modified_statistics_and_endpoint=True)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study-root', type=Path, default=HERE/'results/revised_study')
    parser.add_argument('--endpoint-dir', type=Path, default=HERE/'results/revised_endpoint')
    parser.add_argument('--bootstrap', type=int, default=10000)
    parser.add_argument('--statistics-seed', type=int, default=0)
    parser.add_argument('--endpoint-seed', type=int, default=20261023)
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        self_test()
    else:
        run(args.study_root, args.endpoint_dir, args.bootstrap, args.statistics_seed, args.endpoint_seed)
