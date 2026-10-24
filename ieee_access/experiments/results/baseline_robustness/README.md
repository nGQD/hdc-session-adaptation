# Fixed boundary confirmation

This directory preserves a finite confirmation of the seven gradient adaptation baselines. The original study records remain in `mjcs/results`; they are never overwritten by this experiment.

The confirmation uses the original held-in development split and seeds 11, 23, and 37. It selects the largest mean final stream accuracy across those three seeds. The outer evaluation uses seeds 101, 113, 127, 131, 137, 149, 157, 163, 173, and 181, after selection. Outer results do not affect configuration selection.

`plan.json` fixes every candidate before the sweep. The original selected configuration is retained as an incumbent. A selected lower learning-rate boundary of 1e-5 is extended to 3e-6 and 1e-6; a selected upper boundary of 1e-3 is extended to 3e-3. A selected 40-epoch boundary is extended to 80 epochs. Elastic weight consolidation also searches epochs 1, 2, 5, 10, 20, 40, and 80 independently of full fine-tuning, and extends its selected penalty boundary from 100 to 10 or from 1e6 to 1e7. These rules yield 58, 41, and 59 candidate configurations on H1, M1, and M2, respectively: 474 development streams in total. The finite sweep stops after these candidates even if its winner lies at a new boundary.

Python, NumPy, CPU PyTorch, and CUDA PyTorch random states are reset immediately before every stream, after loading the existing checkpoint. The historical runner reset those states during pretraining but did not explicitly reset them after loading a checkpoint. Historical scores are therefore preserved as separate observations. Fresh incumbent scores, rather than historical scores, enter the confirmation selection and incumbent-versus-selected comparison.

The runner reuses frozen readout features and an identical initial Fisher estimate. The Fisher cache restores the random states reached by the native computation, so later adaptation draws remain the same. Development selection omits intermediate score rows that are unused by its selection metric. `validation.json` records the comparison against the native stream implementation. Outer runs retain full score matrices. Timings in these cached-computation records are experiment execution times; deployment timing is measured separately by the native resource benchmark.

Files:

- `source_snapshot/`: exact copies of the four original source files named by the frozen plan.
- `inner/`: candidate scores and random-state seed identities for every development stream.
- `selection/`: all candidate scores, fresh incumbents, and selected configurations.
- `selection_for_analysis/`: selected configurations in the format expected by the original analysis.
- `outer/`: complete outer score matrices for both the fresh incumbent and selected configuration; identical configurations share one result.
- `stream/` and `export/`: identical selected-result overlays in the original stream directory format.
- `summary.json`: historical, fresh-incumbent, and selected aggregates, plus noninferiority sensitivity summaries.
- `audit.json`: independent checks of completeness, inner-only selection, seeds, configurations, finite matrices, summary reconstruction, and exported-record identity.

Run from the repository root with the recorded Python environment and existing checkpoints:

```sh
.venv/bin/python ieee_access/experiments/baseline_robustness.py all
.venv/bin/python ieee_access/experiments/baseline_robustness_audit.py
```

The original `mjcs/code` source is intentionally frozen. The runner verifies its hashes against the plan. Its readout cache is optional; absent feature caches are computed by the native frozen backbone. Checkpoints are required and the runner never silently retrains a missing checkpoint. Completed records are reused when restarting an interrupted sweep.
