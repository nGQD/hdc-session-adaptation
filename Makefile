PY ?= ../.venv/bin/python
DANDI ?= $(dir $(PY))dandi
DATA := ../falcon-challenge/data
TASKS ?= h1 m1 m2
AUTHORS ?= Qi Ding Ng

analysis: data cache overlap select main epochs ablation budget dose baselines stats tables figures theory macs

.PHONY: analysis data cache overlap select main epochs ablation budget dose baselines stats tables figures tikz strip theory macs docx audit manifest all

data:
	mkdir -p $(DATA)
	cd $(DATA) && for id in 000954 000941 000953; do $(abspath $(DANDI)) download --existing skip DANDI:$$id; done

cache: data
	$(PY) -c "import sys; sys.path.insert(0, 'code'); import data; [data.build_cache(t) for t in sys.argv[1:]]" $(TASKS)

overlap: data
	$(PY) code/minival_overlap.py

select: cache
	$(PY) code/run.py select --tasks $(TASKS)

main: select
	$(PY) code/run.py main --tasks $(TASKS)

epochs: main
	$(PY) code/run.py epochs --tasks $(TASKS)

ablation: main
	$(PY) code/run.py ablation --tasks $(TASKS)

budget: main
	$(PY) code/run.py budget --tasks $(TASKS)

dose: main
	$(PY) code/run.py dose --tasks $(TASKS)

baselines: cache
	$(PY) code/falcon_baselines.py --tasks $(TASKS)

stats: main ablation budget baselines
	$(PY) code/stats.py

tables: stats epochs
	$(PY) code/tables.py

figures: stats dose epochs theory
	$(PY) code/figures.py

tikz:
	bash paper/tikz/build.sh
	cp paper/tikz/d*.png paper/figures/

strip: figures tikz
	$(PY) code/strip_metadata.py paper/figures paper/tikz --authors "$(AUTHORS)"

theory:
	$(PY) code/theory_check.py

macs:
	$(PY) code/verify_macs.py

docx: strip tables
	python3 code/assemble.py
	python3 code/build_docx.py
	python3 code/strip_metadata.py paper/manuscript.docx --authors "$(AUTHORS)"

audit: docx
	python3 code/audit_docx.py

manifest:
	$(PY) code/manifest.py

all: data cache overlap select main epochs ablation budget dose baselines stats tables figures tikz strip theory macs docx audit manifest
