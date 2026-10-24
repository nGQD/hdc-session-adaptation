#!/usr/bin/env python3
"""Build a portable, record-derived supplement after the revised study is complete.

The self-test uses temporary schema fixtures marked TEST ONLY. Production checks
the revised streams, confirmed selections, endpoint hashes, solver validation,
resource repetitions, controls, and selected hidden-test provenance before writing.
"""
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import shutil
import subprocess
import tempfile

import numpy as np
import paper_tables as pt
import measure_resources as mr

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RESULTS = HERE / 'results'
METHODS = ['hdc', 'closed_linear', 'readout_tag', 'finetune_full', 'hdc_primal']
SOLVERS = ['hdc', 'closed_linear', 'hdc_zero', 'closed_zero', 'hdc_bound']


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def esc(value):
    value = str(value)
    return ''.join({'&': r'\&', '%': r'\%', '$': r'\$', '#': r'\#', '_': r'\_',
                    '{': r'\{', '}': r'\}', '\\': r'\textbackslash{}',
                    '~': r'\textasciitilde{}', '^': r'\textasciicircum{}'}.get(c, c) for c in value)


def num(value, digits=4):
    if value is None:
        return 'n/a'
    assert math.isfinite(float(value))
    if abs(value) < 0.5 * 10 ** -digits:
        value = 0.0
    return f'{value:.{digits}f}'


def prob(value):
    if value == 0:
        return '0'
    return f'{value:.3g}'


def interval(values, digits=4):
    return '(' + ', '.join(num(x, digits) for x in values) + ')'


def config(value):
    return ', '.join(f'{k}={v:g}' if isinstance(v, (int, float)) else f'{k}={v}' for k, v in value.items())


def tex_table(caption, headers, rows, alignment=None):
    assert rows and all(len(row) == len(headers) for row in rows)
    alignment = alignment or ('l' + 'r' * (len(headers) - 1))
    line = ' & '.join(esc(x) for x in headers) + r' \\'
    return '\n'.join([r'\begingroup\small\setlength{\tabcolsep}{4pt}',
                      r'\begin{longtable}{' + alignment + '}',
                      r'\caption{' + esc(caption) + r'}\\', r'\toprule', line,
                      r'\midrule\endfirsthead', r'\toprule', line,
                      r'\midrule\endhead', r'\bottomrule\endfoot',
                      *[' & '.join(esc(x) for x in row) + r' \\' for row in rows],
                      r'\end{longtable}\endgroup', ''])


def load_all(study, endpoint_path, resources, fixture=False):
    aggregate, pareto, paired, selection, local, sources = pt.load_study(study, fixture, endpoint_path)
    endpoint = read(endpoint_path)
    controls_path = RESULTS / 'feature_controls/aggregate.json'
    controls = read(controls_path)
    control_provenance = read(controls_path.with_name('provenance.json'))
    solver_path = RESULTS / 'efficient_closed_form/summary.json'
    solver = read(solver_path)
    hidden_path = RESULTS / 'hidden_m1_selected/selected_hidden_test_verification.json'
    hidden = read(hidden_path)
    sources += [endpoint_path, controls_path, controls_path.with_name('provenance.json'), solver_path, hidden_path]
    if not fixture:
        expected = {f'{t}/full/{m}_s{s}.json' for t in pt.TASKS for m in ['hdc', *pt.REFERENCES] for s in pt.SEEDS}
        assert len(endpoint['sources']) == len(expected)
        assert {x['path'] for x in endpoint['sources']} == expected
        for record in endpoint['sources']:
            assert sha(study / 'stream' / record['path']) == record['sha256']
        assert len(endpoint['comparisons']) == 96
        for row in endpoint['comparisons']:
            metric = 'acc' if row['endpoint'] == 'stream_acc' else 'acc_held_out'
            expected = aggregate[row['task']]['hdc'][metric]['mean'] - aggregate[row['task']][row['reference']][metric]['mean']
            assert math.isclose(row['mean_difference'], expected, abs_tol=1e-12)
            assert math.isclose(row['p_bonferroni_12'], min(1., 12 * row['p_noninferiority']), abs_tol=1e-14)
            assert row['noninferior_bonferroni_12'] == (row['p_bonferroni_12'] < .05)
        assert solver['complete'] and solver['runs'] == 150
        assert control_provenance['all_checks_passed'] and control_provenance['checks']['outer_records'] == 60
        assert control_provenance['result_sha256']['aggregate.json'] == sha(controls_path)
        assert hidden['all_private'] and hidden['new_submissions_created'] == 0
        assert hidden['m1_backbone_tensor_identity_verified']
        assert hidden['local_m1_submission_artifacts_identical_to_submitted_container']
    baseline, baseline_selection, mixed = {}, {}, {}
    baseline_summary_path = RESULTS / 'baseline_robustness/summary.json'
    baseline_report = read(baseline_summary_path) if not fixture else None
    if not fixture:
        sources.append(baseline_summary_path)
    for task in pt.TASKS:
        mixed_path = study / 'stats' / f'{task}_mixed.json'
        mixed[task] = read(mixed_path)
        sources.append(mixed_path)
        assert mixed[task]['reference'] == 'hdc'
        for method, row in mixed[task]['coefficients'].items():
            if method != 'intercept':
                expected = aggregate[task][method]['acc']['mean'] - aggregate[task]['hdc']['acc']['mean']
                assert math.isclose(row['estimate'], expected, abs_tol=2e-6), 'Mixed model differs from revised scores'
        if fixture:
            baseline_selection[task] = {'complete': True, 'methods': {}}
            baseline[task] = {}
            for method in pt.GRADIENT:
                cfg = selection[task]['gradient'][method]
                cand = dict(id='fixture', config=cfg, valid=True, incumbent=True,
                            inner_mean_acc=0.5, inner_seed_acc=[0.49, 0.5, 0.51])
                baseline_selection[task]['methods'][method] = dict(selected=cand, incumbent=cand, candidates=[cand], changed=False)
                v = dict(config=cfg, inner_mean_acc=0.5, summary={k: aggregate[task][method][k]
                                                              for k in ['acc', 'acc_held_out', 'bwt', 'adapt_macs']})
                baseline[task][method] = dict(selected=v, incumbent=v, changed=False)
            continue
        path = RESULTS / 'baseline_robustness/selection' / f'{task}.json'
        baseline_selection[task] = read(path)
        sources.append(path)
        assert baseline_selection[task]['complete']
        assert set(baseline_selection[task]['methods']) == set(pt.GRADIENT)
        baseline[task] = baseline_report['tasks'][task]
        for method in pt.GRADIENT:
            detail = baseline_selection[task]['methods'][method]
            assert detail['selected']['config'] == selection[task]['gradient'][method]
            for candidate in detail['candidates']:
                for rel in candidate['inner_record_paths']:
                    path = RESULTS / 'baseline_robustness' / rel
                    assert path.exists()
                    sources.append(path)
            for role in ['incumbent', 'selected']:
                candidate = detail[role]
                rows = []
                for seed in pt.SEEDS:
                    path = RESULTS / 'baseline_robustness/outer' / task / method / f"{candidate['id']}_s{seed}.json"
                    row = read(path)
                    assert row['config'] == candidate['config'] and row['seed'] == seed
                    rows.append(row)
                    sources.append(path)
                for metric in ['acc', 'acc_held_out', 'bwt', 'adapt_macs']:
                    assert math.isclose(baseline[task][method][role]['summary'][metric]['mean'],
                                        float(np.mean([r['summary'][metric] for r in rows])), rel_tol=1e-12, abs_tol=1e-12)
    resource_summary = read(resources / 'summary.json') if not fixture else []
    resource_environment = {'cpu_model': 'TEST ONLY', 'cpu_affinity': [0,2,4,6],
                            'gpu': 'TEST ONLY', 'torch': 'TEST ONLY', 'numpy': 'TEST ONLY',
                            'python': 'TEST ONLY', 'threads': 4}
    if fixture:
        for task in pt.TASKS:
            for method in METHODS:
                resource_summary.append(dict(device='cuda', task=task, method=method,
                    mean_session_median_seconds=1.234, max_peak_bytes=123456789,
                    max_increment_bytes=1234567,
                    sessions=[dict(session='TEST_ONLY', repeats=3, median_seconds=1.234,
                                   max_peak_bytes=123456789, max_increment_bytes=1234567)]))
    else:
        sources += [resources / 'summary.json', resources / 'validation_manifest.json']
        current_resources = mr.validate_recorded_resources(expected_repeats=3, threads=4, resource_root=resources)
        if resource_summary != current_resources:
            raise ValueError('Resource summary differs from currently validated raw measurements')
        groups = {(r['device'], r['task'], r['method']): r for r in resource_summary}
        assert all(('cuda', task, method) in groups for task in pt.TASKS for method in METHODS)
        for key, group in groups.items():
            device, task, method = key
            raw = []
            for repeat in range(3):
                path = resources / 'raw' / f'{device}_{task}_{method}_r{repeat}.json'
                record = read(path)
                assert record['repetition'] == repeat and record['threads'] == 4
                raw.append(record)
                sources.append(path)
            for field in ['benchmark_source_sha256', 'solver_source_sha256', 'checkpoint_sha256',
                          'selection_file_sha256', 'prefix_cache_sha256', 'prefix_manifest_sha256']:
                assert len({r[field] for r in raw}) == 1
            resource_environment = {key: raw[0][key] for key in ['gpu', 'torch', 'numpy', 'python', 'threads']}
            resource_environment = {**raw[0]['measurement_environment'], **resource_environment}
            for session in group['sessions']:
                rows = [next(x for x in r['rows'] if x['name'] == session['session']) for r in raw]
                assert session['repeats'] == 3
                assert math.isclose(session['median_seconds'], float(np.median([x['seconds'] for x in rows])), abs_tol=1e-12)
            assert math.isclose(group['mean_session_median_seconds'], float(np.mean([r['median_seconds'] for r in group['sessions']])), abs_tol=1e-12)
    return dict(aggregate=aggregate, pareto=pareto, paired=paired, mixed=mixed, endpoint=endpoint,
                controls=controls, control_provenance=control_provenance,
                baseline=baseline, baseline_selection=baseline_selection, solver=solver,
                resources=resource_summary, resource_environment=resource_environment,
                hidden=hidden, sources=sources)


def render(data, fixture=False):
    flag = 'TEST FIXTURE ONLY: NUMBERS ARE NOT SUBMISSION RESULTS. ' if fixture else ''
    lines = [r'\documentclass[10pt]{article}', r'\usepackage[T1]{fontenc}',
             r'\usepackage[utf8]{inputenc}', r'\usepackage{lmodern,amsmath,amssymb,booktabs,longtable,graphicx,pdflscape,needspace}',
             r'\usepackage[margin=18mm]{geometry}', r'\usepackage[hidelinks]{hyperref}',
             r'\setlength{\parindent}{0pt}\setlength{\parskip}{5pt}',
             r'\renewcommand{\thetable}{S\arabic{table}}\renewcommand{\thefigure}{S\arabic{figure}}',
             r'\begin{document}', r'\begin{center}\Large\textbf{Supplementary Material}\\[5pt]',
             r'\large Gradient-Free Hyperdimensional Session Adaptation of a Causal Recurrent Intracortical Decoder\end{center}',
             esc(flag + 'This supplement reports the completed confirmation experiments and expanded analyses accompanying the article. All tables and figures are generated from saved experimental records. The source bundle includes the input records and their checksums.'),
             r'\tableofcontents\clearpage', r'\section{Data and inference}',
             'HDC denotes the proposed hyperdimensional decoder. Full-model and readout fine-tuning use the labels FT-full and FT-readout; EWC denotes elastic weight consolidation. Backward transfer (BWT) measures retention. Rectified linear unit (ReLU) names the control activation. Multiply-accumulate operations (MACs) quantify analytic adaptation cost; one GMAC is one billion MACs. CI, SD, and SE denote confidence interval, standard deviation, and standard error, respectively.',
             'Outer evaluations use ten training seeds (101, 113, 127, 131, 137, 149, 157, 163, 173, 181). Inner selection uses seeds 11, 23, and 37. H1, M1, and M2 contain 7, 3, and 4 held-out sessions. Stream ACC includes both retained held-in sessions and adapted held-out sessions. The held-out endpoint isolates final new-session performance.',
             'Seed-level paired tests condition on the recorded sessions. Two-way bootstrap intervals independently resample paired seeds and sessions, retaining the held-in and held-out strata for ACC. Seeds are repeated training runs; they are not additional participants. The held-out endpoint and margin sensitivity are additional analyses of the saved predictions, with no outer-score model selection.',
             r'\section{Endpoint and margin sensitivity}',
             'A non-inferiority (NI) decision rejects the null that the mean HDC minus reference difference is at most the negative analysis margin. Each family contains 12 task-by-reference comparisons within one endpoint and margin.']
    rows = [[r['endpoint'], num(r['margin'], 2), r['unadjusted'], r['holm_12'], r['bonferroni_12']]
            for r in data['endpoint']['counts']]
    lines.append(tex_table('Numbers of non-inferiority decisions among 12 task-by-reference contrasts.',
                           ['Endpoint', 'Margin', 'Unadjusted', 'Holm', 'Bonferroni'], rows))
    for endpoint in ['stream_acc', 'held_out_r2']:
        lines.append(r'\Needspace{10\baselineskip}\subsection{' + ('Stream ACC' if endpoint == 'stream_acc' else 'Final held-out R-squared') + '}')
        for margin in [0.01, 0.02, 0.03, 0.05]:
            rows = []
            selected = [r for r in data['endpoint']['comparisons'] if r['endpoint'] == endpoint and r['margin'] == margin]
            assert len(selected) == 12
            for r in selected:
                rows.append([r['task'].upper(), pt.LABEL[r['reference']], num(r['mean_difference']),
                             interval(r['ci90_seed']), interval(r['ci95_crossed_bootstrap']),
                             prob(r['p_bonferroni_12']), 'yes' if r['noninferior_bonferroni_12'] else 'no'])
            lines.append(tex_table(f'Margin {margin:.2f}. Differences are HDC minus reference. The seed interval is two-sided 90%; the paired two-way interval is 95%. P values are Bonferroni-adjusted within endpoint and margin.',
                                   ['Task', 'Reference', 'Difference', '90% seed CI', '95% two-way CI', 'P adjusted', 'NI'], rows))
    lines += [r'\clearpage\section{Feature-source and head-width controls}',
              r'The linear control uses $[g_t,1,f_t]$, with a pretrained prior on $[g_t,1]$ and a zero prior on $f_t$. The random rectified-linear control uses $[g_t,1,\sqrt{2}\max(0,f_tW)]$, where $W_{ij}\sim\mathcal{N}(0,1/n_f)$ is fixed by seed and has $2D$ columns. It matches the HDC input features and fitted head width. The random projection contains twice as many columns as the HDC frequency matrix; total projection storage is therefore reported separately. This comparison isolates feature-map choice at a common fitted head width.',
              r'Both controls store separate tagged corrections and solve the same prior-centered ridge objective. Each penalty is selected from the fixed grid using only the three inner seeds. The smaller primal or dual factorization gives the same ridge solution. Analytic MACs count the implemented dense Gram product, $Ndq$ for $q=\min(N,d)$; cached-feature analysis times are not deployment measurements.']
    rows = []
    for task in pt.TASKS:
        for kind, label in [('linear_gf', 'Linear [g,1,f]'), ('relu_matched', 'Random ReLU')]:
            r = data['controls'][task][kind]; m = r['metrics']; d = r['hdc_minus_control_acc']
            rows.append([task.upper(), label, pt.concise(r['selected']['selected_lambda']),
                         num(m['acc']['mean']) + ' (' + num(m['acc']['sd']) + ')',
                         num(m['acc_held_out']['mean']) + ' (' + num(m['acc_held_out']['sd']) + ')',
                         interval(d['paired_two_way_bootstrap_ci95'], 5)])
    lines.append(tex_table('Control accuracy: mean (seed standard deviation), ten outer seeds. The final interval is for HDC minus control on ACC.',
                           ['Task', 'Control', 'Penalty', 'ACC (SD)', 'Held-out (SD)', '95% two-way difference CI'], rows))
    rows = []
    for task in pt.TASKS:
        for kind, label in [('linear_gf', 'Linear [g,1,f]'), ('relu_matched', 'Random ReLU')]:
            r = data['controls'][task][kind]; m = r['metrics']
            rows.append([task.upper(), label, num(m['adapt_macs']['mean'] / 1e9, 3),
                         str(int(m['memory']['mean'])), num(m['bwt']['mean']),
                         prob(r['hdc_minus_control_acc']['wilcoxon_holm_p'])])
    lines.append(tex_table('Control resource accounting and seed-conditional paired tests. Retained scalar counts include the backbone, map, prior copy, and session corrections. The Wilcoxon family contains six task-by-control contrasts.',
                           ['Task', 'Control', 'GMACs', 'Retained scalars', 'BWT', 'ACC Holm p'], rows))
    grid = data['controls']['h1']['linear_gf']['selected']['lambda_grid']
    rows = [[pt.concise(lam), *[num(data['controls'][task][kind]['selected']['mean_inner_acc'][i], 5)
                              for task in pt.TASKS for kind in ['linear_gf', 'relu_matched']]] for i, lam in enumerate(grid)]
    lines.append(tex_table('Mean inner ACC for every control penalty; the maximum within each column selects the outer configuration.',
                           ['Penalty', 'H1 linear', 'H1 ReLU', 'M1 linear', 'M1 ReLU', 'M2 linear', 'M2 ReLU'], rows))
    lines += [r'\clearpage\section{Finite boundary checks for gradient methods}',
              'The confirmation evaluates a fixed candidate list using the inner seeds. The largest inner mean ACC selects a configuration, with the incumbent preferred on an exact tie. The list is finite and is not extended after inspecting outer results. Selected and incumbent configurations are then evaluated under the same explicit seed-reset procedure. This comparison separates the configuration effect from historical random-generator state.']
    rows = []
    for task in pt.TASKS:
        for method in pt.GRADIENT:
            r = data['baseline'][task][method]
            for role in ['incumbent', 'selected']:
                v = r[role]; s = v['summary']
                rows.append([task.upper(), pt.LABEL[method], role, num(v['inner_mean_acc']),
                             num(s['acc']['mean']), num(s['acc_held_out']['mean']), num(s['bwt']['mean'])])
    lines.append(tex_table('Incumbent and selected configurations: inner objective and outer means over ten seeds. Selection uses the inner column only.',
                           ['Task', 'Method', 'Role', 'Inner ACC', 'Outer ACC', 'Outer held-out', 'Outer BWT'], rows))
    rows = []
    for task in pt.TASKS:
        for method in pt.GRADIENT:
            detail = data['baseline_selection'][task]['methods'][method]
            for c in detail['candidates']:
                rows.append([task.upper(), pt.LABEL[method], config(c['config']),
                             num(c['inner_mean_acc'], 5) if c['valid'] else 'nonfinite',
                             'yes' if c['id'] == detail['selected']['id'] else '',
                             'yes' if c['incumbent'] else ''])
    lines.append(tex_table('All candidates in the finite confirmation. Nonfinite predictions are ineligible for selection.',
                           ['Task', 'Method', 'Configuration', 'Inner ACC', 'Selected', 'Incumbent'], rows,
                           r'llp{70mm}rcc'))
    lines += [r'\clearpage\section{Exact solver and deployment resources}',
              r'For $N$ fitting rows and $d$ head features, the implementation factors a matrix of dimension $q=\min(N,d)$. It uses the residual form $\Delta=X^\top(XX^\top+\lambda I)^{-1}(Y-XW_0)$ when $N<d$, and the algebraically equivalent primal form otherwise. Shared-head accumulation retains its primal statistics. The validation compares saved score matrices and, where checked, native predictions.']
    rows = []
    for task in pt.TASKS:
        for method in SOLVERS:
            r = data['solver']['tasks'][task][method]
            rows.append([task.upper(), pt.LABEL[method], len(r['seeds']),
                         prob(r['max_abs_R_difference']),
                         'not checked' if r['max_abs_native_prediction_difference'] is None else prob(r['max_abs_native_prediction_difference']),
                         ', '.join(r['solvers'])])
    lines.append(tex_table('Exact-solver validation against original records; score-matrix tolerance is ' + str(data['solver']['tolerance']) + '.',
                           ['Task', 'Method', 'Seeds', 'Max score error', 'Max prediction error', 'Solver'], rows))
    test = data['solver']['self_test']
    lines.append(esc(f"The independent augmented least-squares check used {test['checks']} cases and had maximum absolute weight error {test['max_absolute_weight_error']:.3g}."))
    lines += ['Resource measurements use separate processes and three repetitions, with four CPU threads. Each isolated update starts from the pretrained model with an empty feature cache. Method initialization, the complete adaptation-prefix feature pass, and the update are timed; loading and normalization are outside the interval for every method. GPU execution is synchronized before recording elapsed time. Session medians are averaged across sessions. These measurements use seed 101 and describe the stated workstation.',
              'GPU peak allocated bytes include model tensors and live workspaces. CPU resident memory includes the interpreter and loaded prefix data and is sampled at 1 ms. Peak values are maxima over repetitions and sessions. Incremental peaks subtract the pre-update baseline. Transient workspace is distinct from the retained-state scalar count.']
    rows = []
    for method in pt.MAIN:
        rows.append([pt.LABEL[method],
                     *[num(data['aggregate'][t][method]['adapt_macs']['mean'] / 1e9, 2) for t in pt.TASKS],
                     *[num(data['aggregate'][t][method]['plasticity']['mean'], 3) for t in pt.TASKS],
                     *[num(data['pareto'][t]['prob_non_dominated'][method], 3) for t in pt.TASKS]])
    lines.append(r'\begin{landscape}')
    lines.append(tex_table('Complete analytic costs, plasticity, and Pareto frequencies for all 14 shared-backbone methods. Plasticity is held-out performance immediately after its own adaptation. P(nd) is the frequency of non-domination over ACC, BWT, and MACs jointly, using 10,000 paired two-way bootstrap draws.',
                           ['Method', 'H1 GMAC', 'M1 GMAC', 'M2 GMAC', 'H1 plast.', 'M1 plast.', 'M2 plast.',
                            'H1 P(nd)', 'M1 P(nd)', 'M2 P(nd)'], rows))
    lines.append(r'\end{landscape}')
    lines.append(tex_table('Recorded resource-measurement environment.', ['Property', 'Value'],
                           [[key, value] for key, value in data['resource_environment'].items()], 'll'))
    rows = []
    for r in data['resources']:
        rows.append([r['device'], r['task'].upper(), 'HDC primal' if r['method'] == 'hdc_primal' else pt.LABEL[r['method']],
                     num(r['mean_session_median_seconds'], 3), num(r['max_peak_bytes'] / 2**20, 1),
                     num(r['max_increment_bytes'] / 2**20, 1)])
    lines.append(tex_table('Prefix-inclusive adaptation time and peak memory. Time is the mean of session medians; memory is in MiB.',
                           ['Device', 'Task', 'Method', 'Seconds', 'Peak MiB', 'Increment MiB'], rows))
    rows = []
    for r in data['resources']:
        for s in r['sessions']:
            rows.append([r['device'], r['task'].upper(), 'HDC primal' if r['method'] == 'hdc_primal' else pt.LABEL[r['method']],
                         s['session'], num(s['median_seconds'], 3), num(s['max_peak_bytes'] / 2**20, 1)])
    lines.append(tex_table('Per-session resource measurements, each based on three repetitions.',
                           ['Device', 'Task', 'Method', 'Session', 'Median seconds', 'Peak MiB'], rows))
    lines += [r'\clearpage\section{Paired tests}',
              'The paired tables use HDC minus reference. Wilcoxon p values are adjusted over 13 comparisons within each task and metric. Cliff delta is descriptive and compares the seed-level distributions. Inference uses the paired seed tests and crossed-bootstrap intervals reported here.']
    for task in pt.TASKS:
        pairs = {(r['other'], r['metric']): r for r in data['paired'][task]['rows']}
        rows = []
        for method in pt.MAIN:
            if method == 'hdc': continue
            a, b = pairs[method, 'acc'], pairs[method, 'bwt']
            rows.append([pt.LABEL[method], num(a['mean_diff']), num(a['cliff_delta'], 2),
                         prob(a['p']), prob(a['p_holm']), num(b['mean_diff']), prob(b['p_holm'])])
        lines.append(tex_table(task.upper() + ': ten-seed paired tests.',
                               ['Reference', 'ACC diff', 'Cliff delta', 'Raw p', 'Holm p', 'BWT diff', 'BWT Holm p'], rows))
    lines += [r'\clearpage\section{Session-level results}',
              r'\begin{figure}[!ht]\centering\includegraphics[width=\textwidth]{figures/f1_per_session.pdf}',
              r'\caption{Final session scores for nine methods, with task-specific vertical axes. Large markers are ten-seed means and small points are individual seeds. The dashed divider separates held-in and held-out sessions. All 14 methods appear in the accompanying statistical tables.}\end{figure}',
              r'\begin{figure}[!ht]\centering\includegraphics[width=\textwidth]{figures/f2_heatmaps.pdf}',
              r'\caption{Mean score matrices for HDC, full-model fine-tuning, and Replay over ten seeds. Columns are sessions; rows give the starting decoder and successive adaptations. Colors share limits within each task; the lower limit is the first percentile of plotted cells.}\end{figure}',
              r'\clearpage\section{Selected hidden-test configuration and provenance}',
              'The recorded scores below use the inner-selected seed-101 configurations. They are descriptive hidden-test results, with standard deviations across sessions. The M1 score comes from the original dimension-4096 submission, which the platform now reports as finished. Its container artifacts match the selected configuration and checkpoint; the earlier failed-status snapshot is retained in the provenance record. No score-based configuration choice or new submission was made during this verification.']
    rows = []
    for r in data['hidden']['rows']:
        v = r['result']['test_split_' + r['task']]
        rows.append([r['task'].upper(), r['submission']['id'],
                     num(v['Held Out R2 Mean'], 6), num(v['Held Out R2 Std.'], 6),
                     num(v['Held In R2 Mean'], 6), num(v['Normalized Latency'], 6)])
    lines.append(tex_table('Verified hidden-test scores for the selected configurations.',
                           ['Task', 'Submission', 'Held-out mean', 'Session SD', 'Held-in mean', 'Norm. latency'], rows))
    lines += [r'\section{Reproduction bundle}',
              esc('The portable source contains the LaTeX manuscript, two figure files, input records, and a checksum manifest. The manifest records SHA-256 hashes of the inputs and generated artifacts. Compile supplement.tex with pdflatex twice. The experiment scripts accompanying the main code release regenerate the analyses from the recorded streams.'),
              r'\end{document}']
    return '\n\n'.join(lines) + '\n'


def build(study, endpoint, resources, output, fixture=False):
    if fixture:
        assert output.resolve().is_relative_to(Path(tempfile.gettempdir()).resolve())
    data = load_all(study, endpoint, resources, fixture)
    output.mkdir(parents=True, exist_ok=True)
    (output / 'figures').mkdir(exist_ok=True)
    spec = importlib.util.spec_from_file_location('supplement_figures', ROOT / 'mjcs/code/figures.py')
    plots = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(plots)
    plots.style()
    # Use Matplotlib's bundled TrueType serif font, avoiding mixed CFF/TrueType
    # declarations produced by some system Nimbus OpenType font installations.
    plots.plt.rcParams['font.serif'] = ['STIXGeneral']
    streams = plots.load(study)
    plots.f1_sessions(streams, output / 'figures')
    plots.f2_heatmaps(streams, output / 'figures')
    source = render(data, fixture)
    (output / 'supplement.tex').write_text(source)
    for _ in range(2):
        result = subprocess.run(['pdflatex', '-interaction=nonstopmode', '-halt-on-error', 'supplement.tex'],
                                cwd=output, capture_output=True, text=True)
        if result.returncode:
            raise RuntimeError(result.stdout[-5000:] + result.stderr[-1000:])
    log = (output / 'supplement.log').read_text()
    assert 'Undefined control sequence' not in log and 'undefined references' not in log.lower()
    inputs = {}
    for path in sorted(set(data['sources'])):
        rel = path.resolve().relative_to(ROOT)
        dest = output / 'data' / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dest)
        inputs[str(rel)] = dict(sha256=sha(path), bundled=str(dest.relative_to(output)))
    shutil.copy2(Path(__file__), output / 'build_supplement.py')
    shutil.copy2(HERE / 'paper_tables.py', output / 'paper_tables.py')
    shutil.copy2(HERE / 'analyze_revision.py', output / 'analyze_revision.py')
    shutil.copy2(HERE / 'measure_resources.py', output / 'measure_resources.py')
    manifest = dict(test_fixture_only=fixture, all_input_checks_passed=True,
                    study_root=str(study.resolve()), inputs=inputs,
                    generator_sha256=sha(Path(__file__)), figures_source_sha256=sha(ROOT / 'mjcs/code/figures.py'),
                    latex_overfull_box_count=log.count('Overfull \\hbox'),
                    outputs={str(p.relative_to(output)): sha(p) for p in output.rglob('*')
                             if p.is_file() and p.suffix in ['.tex', '.pdf', '.png']})
    (output / 'supplement_provenance.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return manifest


def self_test():
    with tempfile.TemporaryDirectory(prefix='supplement_schema_TEST_ONLY_') as temp:
        output = Path(temp) / 'supplement'
        manifest = build(ROOT / 'mjcs/results', RESULTS / 'endpoint_analysis/endpoint_analysis.json',
                         RESULTS / 'resources', output, fixture=True)
        assert manifest['test_fixture_only'] and (output / 'supplement.pdf').stat().st_size > 10000
        assert 'TEST FIXTURE ONLY' in (output / 'supplement.tex').read_text()
        print(json.dumps(dict(passed=True, test_fixture_only=True, overfull_boxes=manifest['latex_overfull_box_count'],
                              pdf_bytes=(output / 'supplement.pdf').stat().st_size)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study-root', type=Path, default=RESULTS / 'revised_study')
    parser.add_argument('--endpoint', type=Path, default=RESULTS / 'endpoint_analysis/endpoint_analysis.json')
    parser.add_argument('--resources', type=Path, default=RESULTS / 'resources')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'ieee_access/supplementary_material')
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        self_test()
    else:
        report = build(args.study_root, args.endpoint, args.resources, args.output_dir)
        print(json.dumps(dict(output=str(args.output_dir), overfull_boxes=report['latex_overfull_box_count'])))
