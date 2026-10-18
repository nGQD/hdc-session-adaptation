#!/bin/bash
# Runs every experiment stage for one task after its selection is final; stages resume from existing records.
cd "$(dirname "$0")/.."
task=$1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
until ../.venv/bin/python -c "import json,sys; sys.exit(0 if json.load(open('results/selection/$task.json')).get('grids') else 1)" 2>/dev/null; do sleep 120; done
for stage in main epochs ablation budget dose; do
  for try in 1 2 3 4 5; do
    ../.venv/bin/python code/run.py $stage --tasks $task >> results/${stage}_$task.log 2>&1 && break
    echo "retry $try" >> results/${stage}_$task.log; sleep 60
  done
done
echo done > results/pipeline_$task.done
