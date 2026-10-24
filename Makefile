.DEFAULT_GOAL := analysis
.NOTPARALLEL:

PY ?= .venv/bin/python
DANDI ?= .venv/bin/dandi
TASKS ?= h1 m1 m2
CPU_AFFINITY ?= 0,2,4,6
EXPERIMENTS := ieee_access/experiments
RESULTS := $(EXPERIMENTS)/results
STUDY := $(RESULTS)/revised_study
ENDPOINT := $(RESULTS)/revised_endpoint/endpoint_analysis.json

.PHONY: all analysis check-records check-manifest manifest stats tables figures supplement data cache experiments resources

all: analysis

analysis:
	$(MAKE) stats
	$(MAKE) tables
	$(MAKE) figures
	$(MAKE) supplement

check-records:
	$(PY) publication/check_records.py

stats:
	$(PY) $(EXPERIMENTS)/analyze_revision.py --study-root $(STUDY) --endpoint-dir $(RESULTS)/revised_endpoint

tables:
	$(PY) $(EXPERIMENTS)/paper_tables.py --study-root $(STUDY) --endpoint $(ENDPOINT) --output-dir ieee_access/src/tables

figures:
	$(PY) mjcs/code/figures.py --results $(STUDY) --out ieee_access/src/figures --only f1 f2 f3 f7

supplement:
	$(PY) $(EXPERIMENTS)/build_supplement.py --study-root $(STUDY) --endpoint $(ENDPOINT) --resources $(RESULTS)/resources --output-dir ieee_access/supplementary_material

data:
	mkdir -p falcon-challenge/data
	cd falcon-challenge/data && for id in 000954 000941 000953; do $(abspath $(DANDI)) download --existing skip DANDI:$$id; done

cache:
	$(PY) mjcs/code/data.py $(TASKS)

experiments: cache
	$(PY) $(EXPERIMENTS)/feature_controls.py extract
	$(PY) $(EXPERIMENTS)/feature_controls.py select
	$(PY) $(EXPERIMENTS)/feature_controls.py evaluate
	$(PY) $(EXPERIMENTS)/feature_controls.py verify --seeds 101
	$(PY) $(EXPERIMENTS)/feature_controls.py aggregate
	$(PY) $(EXPERIMENTS)/feature_controls_audit.py
	$(PY) $(EXPERIMENTS)/efficient_closed_form.py validate
	$(PY) $(EXPERIMENTS)/baseline_robustness.py all
	$(PY) $(EXPERIMENTS)/baseline_robustness_audit.py
	$(PY) $(EXPERIMENTS)/confirm_sensitivity.py
	$(PY) $(EXPERIMENTS)/baseline_robustness_sensitivity_audit.py
	$(PY) $(EXPERIMENTS)/assemble_revision.py

resources:
	$(PY) $(EXPERIMENTS)/measure_resources.py prepare
	taskset -c $(CPU_AFFINITY) $(PY) $(EXPERIMENTS)/measure_resources.py run --device cuda
	taskset -c $(CPU_AFFINITY) $(PY) $(EXPERIMENTS)/measure_resources.py run --device cpu --methods hdc closed_linear readout_tag

check-manifest:
	$(PY) publication/manifest.py

manifest:
	$(PY) publication/manifest.py --write
