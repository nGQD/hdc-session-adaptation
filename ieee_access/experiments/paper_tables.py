#!/usr/bin/env python3
"""Render manuscript tables from the completed revised study, with source checks.

Production input must have complete overlay provenance and endpoint hashes that
match its streams. --self-test uses a temporary, visibly marked schema fixture.
No original gradient result can silently substitute for a revised result.
"""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import re
import tempfile

import numpy as np
from analyze_revision import verify_analysis

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
TASKS = ['h1', 'm1', 'm2']
SEEDS = [101, 113, 127, 131, 137, 149, 157, 163, 173, 181]
MAIN = ['frozen', 'finetune_full', 'finetune_readout', 'ewc', 'replay',
        'full_tag', 'readout_tag', 'last_tag', 'closed_linear', 'closed_zero',
        'hdc', 'hdc_zero', 'hdc_shared', 'hdc_bound']
GRADIENT = MAIN[1:8]
FALCON = ['wf_zeroshot', 'wf_oracle', 'wf_continual',
          'rnn_zeroshot', 'rnn_oracle', 'rnn_finetune']
REFERENCES = ['finetune_full', 'replay', 'full_tag', 'readout_tag']
LABEL = dict(frozen='Frozen', finetune_full='FT-full', finetune_readout='FT-readout',
             ewc='EWC', replay='Replay', full_tag='FT-full-tag', readout_tag='FT-readout-tag',
             last_tag='FT-last-tag', closed_linear='Closed form', closed_zero='Closed form (zero prior)',
             hdc='HDC', hdc_zero='HDC (zero prior)', hdc_shared='HDC-shared', hdc_bound='HDC-bound',
             wf_zeroshot='WF zero-shot', wf_oracle='WF per-session', wf_continual='WF sequential',
             rnn_zeroshot='RNN zero-shot', rnn_oracle='RNN per-session', rnn_finetune='RNN sequential')


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fmt(value, digits=3):
    if abs(value) < 0.5 * 10 ** -digits:
        value = 0.0
    return f'{value:.{digits}f}'.replace('-', '−')


def pvalue(value):
    assert math.isfinite(value) and 0 <= value <= 1
    if value == 0:
        return '$<10^{-300}$'
    if value < 0.001:
        mantissa, exponent = f'{value:.1e}'.split('e')
        return '$' + mantissa + r'\times10^{' + str(int(exponent)) + '}$'
    return f'{value:.2g}'


def concise(value):
    return re.sub(r'e([+-])0+(\d+)', r'e\1\2', f'{value:g}').replace('e+', 'e')


def table(caption, label, headers, rows, fixture=False):
    assert rows and all(len(r) == len(headers) for r in rows)
    prefix = 'TEST FIXTURE ONLY. ' if fixture else ''
    result = [f'Table: {prefix}{caption} {{#tab:{label}}}', '',
              '| ' + ' | '.join(headers) + ' |',
              '|:--|' + ':-:|' * (len(headers) - 1)]
    result += ['| ' + ' | '.join(map(str, r)) + ' |' for r in rows]
    return '\n'.join(result) + '\n'


def validate_gradient_record(record, selected, method):
    """Reject substituted legacy streams even if their summaries are self-consistent."""
    expected = selected['gradient'][method]
    candidate = selected['confirmation']['methods'][method]['selected']
    assert candidate['config'] == expected, 'Inconsistent confirmed selection'
    assert record.get('config') == expected, f'{method}: stream configuration differs from selected configuration'
    assert record.get('candidate_id') == candidate['id'], f'{method}: stream candidate differs from selected candidate'


def load_study(study, fixture=False, endpoint_path=None):
    if not fixture:
        verify_analysis(study, endpoint_path)
    aggregates, pareto, paired, selections, local = {}, {}, {}, {}, {}
    source_paths = [] if fixture else [study/'provenance.json', study/'analysis_provenance.json']
    for task in TASKS:
        paths = [study / 'stats' / f'{task}_{suffix}.json'
                 for suffix in ['aggregate', 'pareto', 'paired']]
        aggregates[task], pareto[task], paired[task] = map(read, paths)
        select_path = study / 'selection' / f'{task}.json'
        selections[task] = read(select_path)
        if not fixture:
            assert selections[task]['confirmation']['complete'], f'{task}: unconfirmed selection'
        source_paths += [*paths, select_path]
        assert set(aggregates[task]) == set(MAIN), f'{task}: unexpected/missing main methods'
        for method in MAIN:
            runs = []
            for seed in SEEDS:
                path = study / 'stream' / task / 'full' / f'{method}_s{seed}.json'
                record = read(path)
                assert record['task'] == task and record['seed'] == seed and record['method'] == method
                if not fixture and method in GRADIENT:
                    validate_gradient_record(record, selections[task], method)
                runs.append(record)
                source_paths.append(path)
            for metric in ['acc', 'acc_held_out', 'acc_held_in', 'bwt', 'plasticity', 'adapt_macs', 'memory']:
                values = [r['summary'][metric] for r in runs]
                stat = aggregates[task][method][metric]
                assert stat['n_seeds'] == 10
                assert math.isclose(stat['mean'], float(np.mean(values)), rel_tol=1e-12, abs_tol=1e-12), (task, method, metric, 'stale aggregate')
                assert math.isclose(stat['sd'], float(np.std(values, ddof=1)), rel_tol=1e-12, abs_tol=1e-12), (task, method, metric, 'stale SD')
            for metric in ['acc', 'bwt', 'adapt_macs']:
                assert math.isclose(pareto[task]['point'][method][metric], aggregates[task][method][metric]['mean'], rel_tol=1e-12, abs_tol=1e-12)
        rows = paired[task]['rows']
        assert len(rows) == 26
        for row in rows:
            assert row['ref'] == 'hdc' and row['n'] == 10
            metric = row['metric']
            expected = aggregates[task]['hdc'][metric]['mean'] - aggregates[task][row['other']][metric]['mean']
            assert math.isclose(row['mean_diff'], expected, abs_tol=1e-12)
        local[task] = {}
        for method in FALCON:
            runs = []
            for seed in SEEDS:
                path = study / 'stream' / task / 'falcon' / f'{method}_s{seed}.json'
                record = read(path)
                assert record['method'] == method and record['seed'] == seed
                runs.append(record)
                source_paths.append(path)
            local[task][method] = {key: float(np.mean([r['summary'][key] for r in runs]))
                                   for key in ['acc', 'acc_held_out', 'bwt']}
    return aggregates, pareto, paired, selections, local, source_paths


def generate(study, endpoint_path, control_path, output, fixture=False):
    if fixture:
        assert output.resolve().is_relative_to(Path(tempfile.gettempdir()).resolve())
    aggregate, pareto, paired, selected, local, sources = load_study(study, fixture, endpoint_path)
    endpoint = read(endpoint_path)
    controls = read(control_path)
    sources += [endpoint_path, control_path]
    if not fixture:
        expected_paths = {f'{task}/full/{method}_s{seed}.json'
                          for task in TASKS for method in ['hdc', *REFERENCES] for seed in SEEDS}
        assert {x['path'] for x in endpoint['sources']} == expected_paths
        assert len(endpoint['sources']) == len(expected_paths)
        for source in endpoint['sources']:
            assert sha(study / 'stream' / source['path']) == source['sha256'], 'Endpoint input differs from revised stream'
        control_provenance = read(control_path.with_name('provenance.json'))
        assert control_provenance['all_checks_passed']
        assert control_provenance['checks']['outer_records'] == 60
        assert control_provenance['result_sha256']['aggregate.json'] == sha(control_path)
        sources.append(control_path.with_name('provenance.json'))
    parts, counts = {}, {}
    keys = ['acc', 'acc_held_out', 'bwt']
    best = {(task, key): max(aggregate[task][method][key]['mean'] for method in MAIN)
            for task in TASKS for key in keys}
    rows = []
    for method in MAIN:
        cells = []
        for key in keys:
            for task in TASKS:
                value = aggregate[task][method][key]['mean']
                cell = fmt(value)
                if math.isclose(value, best[task, key], rel_tol=0, abs_tol=1e-12):
                    cell = '**' + cell + '**'
                cells.append(cell)
        rows.append([LABEL[method], *cells])
    for method in FALCON:
        rows.append([LABEL[method], *[fmt(local[t][method][key]) for key in keys for t in TASKS]])
    parts['compact'] = table('Final stream results. Stochastic methods are averaged over ten seeds; WF uses one deterministic run per task. ACC averages all sessions; HO is final held-out $R^2$. BWT measures retention. Bold marks the largest unrounded mean among the 14 shared-backbone methods, including exact ties. The last six rows are local FALCON reference implementations with their own architectures.',
                             'main', ['Method', *[f'{t.upper()} {label}' for label in ['ACC', 'HO', 'BWT'] for t in TASKS]], rows, fixture)
    counts['compact'] = len(rows)
    rows = []
    for method in MAIN:
        rows.append([LABEL[method], *[f"{aggregate[t][method]['adapt_macs']['mean'] / 1e9:.2g}" for t in TASKS],
                     *[fmt(aggregate[t][method]['plasticity']['mean']) for t in TASKS],
                     *[fmt(pareto[t]['prob_non_dominated'][method], 2) for t in TASKS]])
    parts['macs'] = table('Analytic adaptation cost per held-out session, plasticity immediately after adaptation, and Pareto frequency over 10,000 paired two-way bootstrap draws. Closed-form costs count dense Gram products with the smaller primal or dual factorization; shared-head accumulation remains primal. P(nd) considers ACC, BWT, and MACs jointly.',
                          'macs', ['Method', *[f'{t.upper()} {label}' for label in ['GMACs', 'plast.', 'P(nd)'] for t in TASKS]], rows, fixture)
    counts['macs'] = len(rows)
    compact_rows = [[LABEL[method], *[f"{aggregate[t][method]['adapt_macs']['mean'] / 1e9:.2g}" for t in TASKS]]
                    for method in MAIN]
    parts['macs_main'] = table('Analytic adaptation cost in GMACs per held-out session at the inner-selected configurations. Closed-form counts include dense Gram products and the smaller primal or dual factorization; shared-head accumulation remains primal. Values have two significant figures; cost ratios use unrounded counts. Plasticity and Pareto frequencies are reported in the supplementary material.',
                               'macs', ['Method', *[f'{t.upper()} GMACs' for t in TASKS]], compact_rows, fixture)
    counts['macs_main'] = len(compact_rows)
    lookup = {t: {(r['other'], r['metric']): r for r in paired[t]['rows']} for t in TASKS}
    rows = []
    for method in MAIN:
        if method == 'hdc':
            continue
        cells = []
        for task in TASKS:
            row = lookup[task][method, 'acc']
            cells.append(f"{fmt(row['mean_diff'])} / {fmt(row['cliff_delta'], 2)} / {pvalue(row['p_holm'])}")
        bw = ' / '.join(pvalue(lookup[t][method, 'bwt']['p_holm']) for t in TASKS)
        rows.append([LABEL[method], *cells, bw])
    parts['tests'] = table('Paired Wilcoxon tests over ten seed-level session means. Task cells give the ACC difference (HDC minus reference), descriptive Cliff $\\delta_{\\mathrm{C}}$, and Holm-adjusted $p$. Adjustment covers 13 comparisons within each task and metric. All-zero BWT differences receive $p=1$.',
                           'tests', ['HDC vs', *[f'{t.upper()} (diff / $\\delta_{{\\mathrm{{C}}}}$ / $p$)' for t in TASKS], 'BWT $p$ (H1 / M1 / M2)'], rows, fixture)
    counts['tests'] = len(rows)
    endpoint_rows = {(r['task'], r['reference'], r['endpoint']): r for r in endpoint['comparisons'] if r['margin'] == 0.03}
    assert len(endpoint_rows) == 24
    rows = []
    for task in TASKS:
        for method in REFERENCES:
            cells = []
            decisions = []
            for name, metric in [('stream_acc', 'acc'), ('held_out_r2', 'acc_held_out')]:
                row = endpoint_rows[task, method, name]
                if not fixture:
                    expected = aggregate[task]['hdc'][metric]['mean'] - aggregate[task][method][metric]['mean']
                    assert math.isclose(row['mean_difference'], expected, abs_tol=1e-12)
                assert row['noninferior_bonferroni_12'] == (row['p_bonferroni_12'] < 0.05)
                cells += [fmt(row['mean_difference']), pvalue(row['p_bonferroni_12'])]
                decisions.append('yes' if row['noninferior_bonferroni_12'] else 'no')
            rows.append([task.upper(), LABEL[method], *cells, ' / '.join(decisions)])
    parts['tost'] = table('Non-inferiority at the analysis margin 0.03, separately for stream ACC and final held-out $R^2$. Differences are HDC minus reference. One-sided paired $t$ tests use ten seeds; each $p$ is Bonferroni-adjusted over 12 task-by-reference contrasts within its endpoint. The last column gives the decisions for ACC / HO at adjusted $p<0.05$.',
                          'tost', ['Task', 'Reference', 'ACC diff', 'ACC $p$', 'HO diff', 'HO $p$', 'Non-inferior (ACC / HO)'], rows, fixture)
    counts['tost'] = len(rows)
    rows = []
    for task in TASKS:
        for kind, label in [('linear_gf', 'Linear $[g,1,f]$'), ('relu_matched', 'Random ReLU')]:
            cell = controls[task][kind]
            assert cell['n_seeds'] == 10 and not cell['selected']['selected_at_boundary']
            lo, hi = cell['hdc_minus_control_acc']['paired_two_way_bootstrap_ci95']
            rows.append([task.upper(), label, concise(cell['selected']['selected_lambda']),
                         fmt(cell['metrics']['acc']['mean']), fmt(cell['metrics']['acc_held_out']['mean']),
                         fmt(cell['hdc_minus_control_acc']['mean'], 4), f'({fmt(lo, 5)}, {fmt(hi, 5)})'])
    parts['feature_controls'] = table('Additional tagged, prior-centered feature controls with the same frozen backbone. Ridge penalties were selected using three inner seeds; reported means use ten outer seeds. Random ReLU uses $2D$ expansion columns, matching the HDC head width and input features. The interval resamples paired seeds and sessions for the ACC difference.',
                                      'feature_controls', ['Task', 'Control', '$\\lambda$', 'ACC', 'HO $R^2$', 'HDC minus control', '95% paired two-way interval'], rows, fixture)
    counts['feature_controls'] = len(rows)
    rows = []
    for method in [*GRADIENT, 'closed_linear', 'closed_zero', 'hdc', 'hdc_zero', 'hdc_shared']:
        cells = []
        for task in TASKS:
            s = selected[task]
            if method in GRADIENT:
                g = s['gradient'][method]
                cell = f"{concise(g['lr'])} / {g['epochs']}"
                if method == 'ewc':
                    cell += ' / ' + concise(g['ewc'])
            elif method == 'hdc':
                h = s['hdc']
                cell = f"{h['dimension']} / {concise(h['scale'])} / {concise(h['lam'])}"
            else:
                cell = concise(s['linear' if method == 'closed_linear' else method]['lam'])
            cells.append(cell)
        rows.append([LABEL[method], *cells])
    assert all(selected[t]['hdc']['source'] == 'state' for t in TASKS)
    parts['selection_paper'] = table('Inner-selected hyperparameters. Gradient rows give learning rate / epochs (/ EWC weight); closed-form rows give the ridge penalty per 1000 fitting bins. HDC gives dimension $D$ / scale $\\sigma$ / ridge penalty and uses $f_t$ on all tasks. HDC-bound shares the HDC configuration. Gradient settings include the completed boundary checks.',
                                     'selection', ['Method', *[t.upper() for t in TASKS]], rows, fixture)
    counts['selection_paper'] = len(rows)
    assert counts == dict(compact=20, macs=14, macs_main=14, tests=13, tost=12, feature_controls=6, selection_paper=12)
    output.mkdir(parents=True, exist_ok=True)
    for name, content in parts.items():
        (output / f'{name}.md').write_text(content)
    manifest = dict(test_fixture_only=fixture, row_counts=counts, study_root=str(study.resolve()),
                    inputs={str(p.resolve()): sha(p) for p in sources},
                    outputs={name + '.md': sha(output / (name + '.md')) for name in parts},
                    source_sha256=sha(Path(__file__)))
    (output / 'table_provenance.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return manifest


def self_test():
    selected = dict(gradient={'finetune_full': {'lr': 1e-4, 'epochs': 20}},
                    confirmation={'methods': {'finetune_full': {'selected':
                        {'id': 'selected', 'config': {'lr': 1e-4, 'epochs': 20}}}}})
    valid = dict(config={'lr': 1e-4, 'epochs': 20}, candidate_id='selected')
    validate_gradient_record(valid, selected, 'finetune_full')
    for invalid in [{**valid, 'candidate_id': 'incumbent'},
                    {**valid, 'config': {'lr': 1e-3, 'epochs': 20}},
                    {'config': valid['config']}]:
        try:
            validate_gradient_record(invalid, selected, 'finetune_full')
        except AssertionError:
            pass
        else:
            raise AssertionError('Substituted gradient stream was accepted')
    original = ROOT / 'mjcs/results'
    control = HERE / 'results/feature_controls/aggregate.json'
    with tempfile.TemporaryDirectory(prefix='paper_tables_schema_test_') as temp:
        folder = Path(temp)
        endpoint = dict(analysis='TEST FIXTURE ONLY', sources=[], comparisons=[])
        for task in TASKS:
            for method in REFERENCES:
                for name in ['stream_acc', 'held_out_r2']:
                    endpoint['comparisons'].append(dict(task=task, reference=method, endpoint=name,
                                                        margin=0.03, mean_difference=-0.0123,
                                                        p_bonferroni_12=0.12 if name == 'held_out_r2' else 0.01,
                                                        noninferior_bonferroni_12=name == 'stream_acc'))
        endpoint_path = folder / 'synthetic_endpoint_TEST_ONLY.json'
        endpoint_path.write_text(json.dumps(endpoint))
        manifest = generate(original, endpoint_path, control, folder / 'tables', fixture=True)
        assert manifest['test_fixture_only']
        assert 'yes / no' in (folder / 'tables/tost.md').read_text()
        assert all('TEST FIXTURE ONLY' in p.read_text() for p in (folder / 'tables').glob('*.md'))
        try:
            generate(original, endpoint_path, control, folder / 'must_reject_original', fixture=False)
        except (AssertionError, FileNotFoundError, KeyError):
            pass
        else:
            raise AssertionError('Original results accepted for production')
        print(json.dumps(dict(passed=True, schema_fixture_only=True, row_counts=manifest['row_counts'],
                              rejects_original_production_input=True)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study-root', type=Path, default=HERE / 'results/revised_study')
    parser.add_argument('--endpoint', type=Path, default=HERE / 'results/endpoint_analysis/endpoint_analysis.json')
    parser.add_argument('--controls', type=Path, default=HERE / 'results/feature_controls/aggregate.json')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'ieee_access/src/tables')
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        self_test()
    else:
        result = generate(args.study_root, args.endpoint, args.controls, args.output_dir)
        print(json.dumps(dict(output=str(args.output_dir), rows=result['row_counts'])))
