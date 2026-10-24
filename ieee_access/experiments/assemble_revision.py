"""Assemble an auditable study overlay without altering the original results.

Confirmed gradient configurations replace the historical main results. Exact
closed-form scores retain their original values after solver equivalence checks;
their costs use the smaller factorization. Legacy cached timing is excluded.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
OLD = ROOT / 'mjcs' / 'results'
RESULTS = HERE / 'results'
OUT = RESULTS / 'revised_study'
TASKS = ['h1','m1','m2']
GRAD = ['finetune_full','finetune_readout','ewc','replay','full_tag','readout_tag','last_tag']
ANALYTIC = ['closed_linear','closed_zero','hdc','hdc_zero','hdc_bound','hdc_shared']
VALIDATED_ANALYTIC = [m for m in ANALYTIC if m != 'hdc_shared']
SEEDS = [101,113,127,131,137,149,157,163,173,181]
sys.path.insert(0, str(HERE))
from efficient_closed_form import corrected_memory


def read(path):
    return json.loads(path.read_text())


def write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def revise_analytic(rec, task, method):
    meta = read(RESULTS / 'feature_controls' / 'features' / task / 'outer' / f's{rec["seed"]}' / 'metadata.json')
    sel = read(OLD / 'selection' / f'{task}.json')
    dimension = sel['hdc']['dimension'] if method.startswith('hdc') else 0
    d = 129 + 2 * dimension
    # All variants preserve channel and output counts. Prefix masks are shared.
    channels = {'h1':176,'m1':64,'m2':96}[task]
    outputs = {'h1':7,'m1':16,'m2':2}[task]
    feature_source = 128 if sel['hdc']['source'] == 'penultimate' else 128 + channels
    bb = sum(rec['backbone']['macs'].values())
    costs, details = [], []
    for row in meta['sessions'][meta['held_in']:]:
        n = row['fit_bins']
        q = d if method == 'hdc_shared' else min(n, d)
        prefix = row['prefix_bins']
        cost = int(prefix * bb + n * feature_source * dimension + n*d*q
                   + n*d*outputs + q**3//6 + q*q*outputs)
        costs.append(cost)
        details.append(dict(session=row['name'], N=n, d=d, q=q,
                            solver='primal_shared' if method == 'hdc_shared' else 'dual' if n < d else 'primal', analytic_macs=cost))
    rec['legacy_symmetric_primal_adapt_macs'] = rec['adapt_macs']
    rec['adapt_macs'] = costs
    rec['summary']['adapt_macs'] = sum(costs)/len(costs)
    rec['solver_records'] = details
    return rec


def revise_memory(rec, task, method, variant):
    meta = read(RESULTS / 'feature_controls' / 'features' / task / 'outer' / f's{rec["seed"]}' / 'metadata.json')
    if variant != 'full':
        frozen = read(OLD/'stream'/task/variant/f'frozen_s{rec["seed"]}.json')
        meta['model_parameters'] = frozen['memory'][0]
    correction = corrected_memory(rec, task, method, metadata=meta)
    rec['native_memory'] = correction['native_memory']
    rec['memory'] = correction['memory']
    rec['memory_accounting'] = correction
    rec['summary']['memory'] = rec['memory'][-1]
    return rec


def clean_time(rec):
    for key in ['adapt_seconds','start_seconds']:
        rec.pop(key, None)
        rec.get('summary', {}).pop(key, None)
    rec['timing_scope'] = 'No deployment timing in this record; use results/resources for isolated prefix-inclusive timing.'


def assemble(partial=False):
    provenance = []
    inputs = {}
    def track(path):
        inputs[str(path.relative_to(ROOT))] = sha(path)
    def record_output(dst, src):
        track(src)
        provenance.append(dict(output=str(dst.relative_to(OUT)), source=str(src.relative_to(ROOT)),
                               sha256=sha(src), output_sha256=sha(dst)))
    if not partial:
        audit_path = RESULTS/'baseline_robustness'/'audit.json'
        audit = read(audit_path)
        assert audit['passed'] and audit['complete'], 'Baseline confirmation audit incomplete'
        assert audit['counts'] == audit['expected_counts'], 'Baseline confirmation coverage differs'
        assert audit['counts']['inner'] == 474 and audit['counts']['selected_exports'] == 210
        assert audit['plan_sha256'] == sha(RESULTS/'baseline_robustness'/'plan.json')
        assert audit['audit_script_sha256'] == sha(HERE/'baseline_robustness_audit.py')
        assert audit['runner_sha256'] == sha(HERE/'baseline_robustness.py')
        assert all(audit['source_files_still_match_frozen_plan'].values())
        assert all(audit['source_snapshots_match_frozen_plan'].values())
        for rel, expected in audit['artifact_sha256'].items():
            assert sha(RESULTS/'baseline_robustness'/rel) == expected, f'Stale baseline audit: {rel}'
        track(audit_path)
        track(RESULTS/'baseline_robustness'/'plan.json')
        sensitivity_audit_path = RESULTS/'confirmed_sensitivity'/'audit.json'
        sensitivity_audit = read(sensitivity_audit_path)
        assert sensitivity_audit['passed'] and sensitivity_audit['complete'], 'Sensitivity audit incomplete'
        assert sensitivity_audit['counts'] == sensitivity_audit['expected_counts'] == dict(
            dose_streams=90, dose_aggregates=30, budget_streams=30, main_reuses=30)
        assert sensitivity_audit['runner_sha256'] == sha(HERE/'confirm_sensitivity.py')
        sensitivity_auditor = HERE/'baseline_robustness_sensitivity_audit.py'
        assert sensitivity_audit['audit_script_sha256'] == sha(sensitivity_auditor)
        for rel, expected in sensitivity_audit['artifact_sha256'].items():
            assert sha(RESULTS/'confirmed_sensitivity'/rel) == expected, f'Stale sensitivity audit: {rel}'
        track(sensitivity_audit_path)
        track(sensitivity_auditor)
        validation_path = RESULTS/'efficient_closed_form'/'summary.json'
        validation = read(validation_path)
        track(validation_path)
        assert validation['complete'] and validation['runs'] == 150, 'Solver validation incomplete'
        for task in TASKS:
            for method in VALIDATED_ANALYTIC:
                checks = validation['tasks'][task][method]
                assert set(checks['seeds']) == set(SEEDS)
                assert checks['max_abs_R_difference'] <= validation['tolerance']
    for task in TASKS:
        select_path = RESULTS / 'baseline_robustness' / 'selection_for_analysis' / f'{task}.json'
        if not partial:
            assert select_path.exists() and read(select_path)['confirmation']['complete'], f'{task}: selection incomplete'
            for method in GRAD:
                assert all((RESULTS/'baseline_robustness'/'stream'/task/'full'/f'{method}_s{s}.json').exists() for s in SEEDS)
        for src in sorted((OLD / 'stream' / task).glob('*/*_s*.json')):
            variant = src.parent.name
            method = src.stem.rsplit('_s', 1)[0]
            chosen = src
            if variant == 'full' and method in GRAD:
                new = RESULTS/'baseline_robustness'/'stream'/task/variant/src.name
                if new.exists(): chosen = new
                elif not partial: raise RuntimeError(f'Missing confirmation: {new}')
            elif variant in ['epochs25','epochs100'] and method == 'finetune_full':
                new = RESULTS/'confirmed_sensitivity'/'stream'/task/variant/src.name
                if new.exists(): chosen = new
                elif not partial: raise RuntimeError(f'Missing sensitivity: {new}')
            elif variant not in ['full','falcon','epochs25','epochs100'] and method in GRAD:
                # Component ablations compare the closed-form method and Frozen;
                # historical gradient configurations are not mixed into them.
                continue
            rec = read(chosen)
            if not partial and method in GRAD and variant in ['full','epochs25','epochs100']:
                confirmed = read(select_path)['gradient'][method]
                assert rec['config'] == confirmed, f'Configuration mismatch: {chosen}'
                if variant == 'full':
                    detail = read(select_path)['confirmation']['methods'][method]['selected']
                    assert rec['candidate_id'] == detail['id'], f'Candidate mismatch: {chosen}'
                else:
                    assert rec['selection_sha256'] == sha(select_path), f'Stale sensitivity: {chosen}'
            if method in ANALYTIC:
                rec = revise_analytic(rec, task, method)
                track(OLD/'selection'/f'{task}.json')
            if variant != 'falcon':
                rec = revise_memory(rec, task, method, variant)
                track(RESULTS/'feature_controls'/'features'/task/'outer'/f's{rec["seed"]}'/'metadata.json')
                if variant != 'full':
                    track(OLD/'stream'/task/variant/f'frozen_s{rec["seed"]}.json')
            clean_time(rec)
            rec['source_result'] = str(chosen.relative_to(ROOT))
            rec['source_sha256'] = sha(chosen)
            dst = OUT / 'stream' / task / variant / src.name
            write(dst, rec)
            record_output(dst, chosen)
        selection_source = select_path if select_path.exists() else OLD/'selection'/f'{task}.json'
        selection_output = OUT/'selection'/f'{task}.json'
        write(selection_output, read(selection_source))
        record_output(selection_output, selection_source)
        for src in sorted((OLD/'dose'/task).glob('*.json')):
            method = src.stem.rsplit('_s',1)[0]
            chosen = src
            if method in ['finetune_full','replay']:
                new = RESULTS/'confirmed_sensitivity'/'dose'/task/src.name
                if new.exists(): chosen = new
                elif not partial: raise RuntimeError(f'Missing dose confirmation: {new}')
            rec = read(chosen)
            if not partial and method in ['finetune_full','replay']:
                assert rec['config'] == read(select_path)['gradient'][method]
                assert rec['selection_sha256'] == sha(select_path)
            # Dose figure reports accuracy only; legacy primal costs are not reused.
            rec.pop('adapt_macs',None)
            rec.update(source_result=str(chosen.relative_to(ROOT)),source_sha256=sha(chosen))
            dst = OUT/'dose'/task/src.name
            write(dst,rec)
            record_output(dst, chosen)
    expected_outputs = {r['output'] for r in provenance}
    # Earlier partial overlays may contain obsolete generated streams. They must
    # not remain visible to stats.py's directory-based input discovery.
    for folder in ['stream', 'selection', 'dose']:
        for path in (OUT/folder).rglob('*.json'):
            if str(path.relative_to(OUT)) not in expected_outputs:
                path.unlink()
    sources = {str(p.relative_to(ROOT)): sha(p) for p in
               [Path(__file__), HERE/'efficient_closed_form.py', HERE/'baseline_robustness_audit.py']}
    write(OUT/'provenance.json', dict(schema_version=2, complete=not partial, description=__doc__,
                                     records=provenance, inputs=inputs, sources=sources))
    print(json.dumps(dict(records=len(provenance), complete=not partial, output=str(OUT))))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--allow-partial', action='store_true')
    args = ap.parse_args()
    assemble(args.allow_partial)
