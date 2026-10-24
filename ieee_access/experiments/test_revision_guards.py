"""Lightweight CPU fixtures for experiment resume and publication guards."""
import copy
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import confirm_sensitivity as cs
import feature_controls as fc
import measure_resources as mr


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def rejects(call):
    try:
        call()
    except ValueError:
        return
    raise AssertionError('Malformed or stale fixture was accepted')


def resource_checks(root):
    resources, selections = root/'resources', root/'selection'
    selected = dict(confirmation=dict(complete=True), hdc=dict(dimension=8, lam=1),
                    gradient=dict(finetune_full=dict(epochs=1, lr=0.001)))
    write(selections/'h1.json', selected)
    manifest = [dict(name='session', prefix_bins=4, labeled_bins=3)]
    write(resources/'h1_prefix_manifest.json', manifest)
    (resources/'h1_prefix_cache.npz').write_bytes(b'TEST ONLY prefix bytes')
    checkpoint = root/'mjcs/results/checkpoints/h1/outer/full_s101.pt'
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b'TEST ONLY checkpoint bytes')
    for name in ['engine.py','run.py','model.py','data.py']:
        path = root/'mjcs/code'/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('TEST ONLY implementation')
    with patch.object(mr, 'ROOT', root):
        paths, good = [], []
        for repeat in range(3):
            path = resources/'raw'/f'cpu_h1_hdc_r{repeat}.json'
            record = dict(**mr.current_identity('h1','hdc',selections,resources),
                          task='h1',method='hdc',device='cpu',threads=4,seed=101,repetition=repeat,
                          measurement_environment=dict(cpu_model='TEST ONLY CPU',cpu_affinity=[0,2,4,6]),
                          system_load_start=[1.,2.,3.],system_load_end=[2.,3.,4.],
                          rows=[dict(**manifest[0],seconds=1+repeat,analytic_macs=100,
                              persistent_scalars_after_update=10,process_rss_base_bytes=100,
                              process_rss_peak_bytes=110,process_rss_increment_bytes=10,
                              cuda_peak_allocated_bytes=None)])
            write(path,record); paths.append(path); good.append(record)
        collect = lambda: mr.collect_summary(resource_root=resources,selection_root=selections)
        summary = collect()
        assert summary[0]['mean_session_median_seconds'] == 2
        assert summary[0]['raw_records_sha256'] == {p.name:mr.fingerprint(p) for p in paths}
        failures = 0
        # Every repetition carries the same stale values: consistency is insufficient.
        for field in ['configuration','benchmark_source_sha256','solver_source_sha256',
                      'checkpoint_sha256','selection_file_sha256','prefix_cache_sha256',
                      'prefix_manifest_sha256','implementation_sha256','software_versions']:
            for path, record in zip(paths,good):
                changed = copy.deepcopy(record); changed[field] = 'KNOWN_STALE'; write(path,changed)
            rejects(collect); failures += 1
            for path, record in zip(paths,good): write(path,record)
        for field, value in [('seconds',float('nan')),('prefix_bins',9),('process_rss_increment_bytes',9)]:
            changed = copy.deepcopy(good[0]); changed['rows'][0][field] = value
            write(paths[0],changed); rejects(collect); failures += 1; write(paths[0],good[0])
        for field,value in [('cpu_model','DIFFERENT CPU'),('cpu_affinity',[1,3,5,7])]:
            changed = copy.deepcopy(good[0]); changed['measurement_environment'][field] = value
            write(paths[0],changed); rejects(collect); failures += 1; write(paths[0],good[0])
        changed = copy.deepcopy(good[0]); changed['system_load_start'] = [float('nan'),1.,2.]
        write(paths[0],changed); rejects(collect); failures += 1; write(paths[0],good[0])
        resume = lambda:mr.verify_existing(paths[0],'h1','hdc',selections,4,'cpu',0,resources)
        with patch.object(mr,'host_environment',return_value=dict(cpu_model='DIFFERENT CPU',cpu_affinity=[0])):
            rejects(resume); failures += 1
        with patch.object(mr,'host_environment',return_value=good[0]['measurement_environment']):
            resume()
        for path in [checkpoint, root/'mjcs/code/engine.py', resources/'h1_prefix_cache.npz']:
            previous = path.read_bytes(); path.write_bytes(previous+b' changed')
            rejects(collect); failures += 1; path.write_bytes(previous)
        changed = copy.deepcopy(selected); changed['hdc']['lam'] = 2
        write(selections/'h1.json',changed); rejects(collect); failures += 1
        write(selections/'h1.json',selected)
        changed = copy.deepcopy(good[0]); changed['rows'] = []
        write(paths[0],changed); rejects(collect); failures += 1; write(paths[0],good[0])
        assert collect() == summary
        with contextlib.redirect_stdout(io.StringIO()):
            assert mr.aggregate(selection_root=selections,resource_root=resources) == summary
        assert mr.validate_recorded_resources(selection_root=selections,resource_root=resources) == summary
    return dict(valid_groups=1, stale_or_malformed_groups_rejected=failures,
                records_only_archive=archive_checks(root))


def archive_checks(root):
    """Reload the actual validator from a relocated archive lacking .pt/.npz."""
    with tempfile.TemporaryDirectory(prefix='resource_records_archive_TEST_ONLY_') as temp:
        archive = Path(temp)/'archive'
        shutil.copytree(root,archive)
        scripts = archive/'ieee_access/experiments'; scripts.mkdir(parents=True)
        for name in ['measure_resources.py','efficient_closed_form.py']:
            shutil.copy2(Path(mr.__file__).with_name(name),scripts/name)
        for path in [*archive.rglob('*.pt'),*archive.rglob('*.npz')]: path.unlink()
        assert not list(archive.rglob('*.pt')) and not list(archive.rglob('*.npz'))
        spec = importlib.util.spec_from_file_location('relocated_resource_validator',scripts/'measure_resources.py')
        relocated = importlib.util.module_from_spec(spec); spec.loader.exec_module(relocated)
        resources,selections = archive/'resources',archive/'selection'
        validate = lambda:relocated.validate_recorded_resources(resource_root=resources,selection_root=selections)
        # Offline validation must never inspect the auditing process's affinity.
        with patch.object(relocated,'host_environment',side_effect=AssertionError('Offline host queried')):
            summary = validate()
        failures = 0
        for path in [resources/'raw/cpu_h1_hdc_r0.json',resources/'summary.json',
                     archive/'mjcs/code/engine.py',scripts/'measure_resources.py',
                     scripts/'efficient_closed_form.py',selections/'h1.json',
                     resources/'h1_prefix_manifest.json']:
            previous = path.read_bytes(); path.write_bytes(previous+b'\n')
            rejects(validate); failures += 1; path.write_bytes(previous)
        # A rehashed but numerically altered summary must still fail recomputation.
        summary_path,manifest_path = resources/'summary.json',resources/'validation_manifest.json'
        old_summary,old_manifest = summary_path.read_bytes(),manifest_path.read_bytes()
        altered = copy.deepcopy(summary); altered[0]['mean_session_median_seconds'] += 1
        write(summary_path,altered)
        manifest = json.loads(old_manifest); manifest['summary_sha256'] = relocated.fingerprint(summary_path)
        write(manifest_path,manifest); rejects(validate); failures += 1
        summary_path.write_bytes(old_summary); manifest_path.write_bytes(old_manifest)
        manifest = json.loads(old_manifest); manifest['groups'][0]['measurement_environment']['cpu_affinity'] = [1,3,5,7]
        write(manifest_path,manifest); rejects(validate); failures += 1; manifest_path.write_bytes(old_manifest)
        assert validate() == summary
        try:
            relocated.collect_summary(resource_root=resources,selection_root=selections)
        except FileNotFoundError:
            pass
        else:
            raise AssertionError('Live aggregation accepted missing checkpoint/cache inputs')
    return dict(relocated_without_large_inputs=True,valid_summary_accepted=True,
                changed_records_inputs_or_summary_rejected=failures,
                auditing_process_affinity_not_queried=True,
                live_aggregation_still_requires_large_inputs=True)


def sensitivity_checks(root):
    write(root/'mjcs/results/provenance/manifest_h1.json', [{'file':'TEST ONLY'}])
    selected = dict(gradient=dict(finetune_full=dict(epochs=1,lr=.001)))
    sessions = [SimpleNamespace(name=name,fit_end=2,x=np.arange(8,dtype=np.float32).reshape(4,2),
                    y=np.arange(4,dtype=np.float32)[:,None],yz=np.arange(4,dtype=np.float32)[:,None],
                    fit=np.array([1,1,0,0],bool),score=np.array([0,0,0,1],bool)) for name in ['old','new']]
    ys = dict(mean=np.zeros(1,np.float32),std=np.ones(1,np.float32))
    with patch.object(cs,'ROOT',root):
        expected = cs.computation_identity('h1','finetune_full','full',101,selected,'selection digest')
        expected['input_identity'] = cs.input_identity(sessions[:1],sessions[1:],ys)
        record = dict(**expected,R=[[.1,.2],[.1,.4]],names=['old','new'],held_in=1,
                      adapt_macs=[100],adapt_seconds=[1.],memory=[10],start_macs=0,start_seconds=0.)
        record['summary'] = cs.engine.summarize(record)
        path = root/'sensitivity.json'; write(path,record)
        verify = lambda: cs.verify_saved(path,expected,stream=True)
        verify(); failures = 0
        for field in ['config','checkpoint_sha256','runner_sha256','implementation_sha256',
                      'selection_sha256','data_manifest_sha256','software_versions','seed','rng_seed','input_identity']:
            changed = copy.deepcopy(record); changed[field] = 'KNOWN_STALE'
            write(path,changed); rejects(verify); failures += 1
        for field,value in [('R',[[.1,.2],[.1,.5]]),('names',['wrong','new']),('adapt_seconds',[])]:
            changed = copy.deepcopy(record); changed[field] = value
            write(path,changed); rejects(verify); failures += 1
        write(path,record)
        changed_inputs = copy.deepcopy(sessions); changed_inputs[1].score[2] = True
        changed_expected = {**expected,'input_identity':cs.input_identity(changed_inputs[:1],changed_inputs[1:],ys)}
        rejects(lambda:cs.verify_saved(path,changed_expected,stream=True)); failures += 1
        assert verify() == record
    return dict(valid_streams=1,stale_or_malformed_streams_rejected=failures)


if __name__ == '__main__':
    with tempfile.TemporaryDirectory(prefix='revision_guards_TEST_ONLY_') as temp:
        root = Path(temp)
        report = dict(resources=resource_checks(root), sensitivity=sensitivity_checks(root),
                      control_residual=fc.residual_equivalence_check())
    print(json.dumps(report,indent=2))
