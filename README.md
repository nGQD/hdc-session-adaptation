# Gradient-free hyperdimensional session adaptation

Code and recorded results for **Gradient-Free Hyperdimensional Session Adaptation of a Causal Recurrent Intracortical Decoder**.

A frozen causal recurrent decoder adapts through a prior-centered correction on random phasor features. Corrections are fitted in closed form and retrieved by recording-session tag. The study compares fourteen methods starting from the same pretrained backbone on the FALCON human reaching (H1), monkey muscle activity (M1), and monkey finger movement (M2) tasks. Gradient baselines update their designated parameters; closed-form methods keep the backbone frozen.

At an analysis tolerance of 0.03 in the coefficient of determination, the proposed method meets non-inferiority after multiplicity correction in all six comparisons with tagged full-network and readout fine-tuning, for both stream and final held-out accuracy. At the selected configurations, it uses 10.9 to 49.4 times fewer adaptation multiply-accumulate operations than tagged full-network tuning. The records include the full accuracy, retention, cost, and measured-resource comparisons.

## Reproduce the analyses

Run commands from this repository's root. The supplied records are sufficient to regenerate the statistics, tables, figures, and supplement without neural recordings, checkpoints, or a GPU.

```sh
python3.10 -m venv .venv
.venv/bin/python -m pip install -r requirements-records.txt
make check-records
make analysis
```

The supplement also requires `pdflatex` with the standard LaTeX packages used in `build_supplement.py`. On Debian or Ubuntu, these are provided by `texlive-latex-extra` and `lmodern`. Matplotlib uses an installed serif font or its bundled fallback.

`make analysis` runs its stages sequentially. Individual targets are available:

| Target | Output |
| --- | --- |
| `check-records` | Validates the supplied study, source identities, and recorded resource measurements |
| `stats` | Revised statistics and endpoint analyses |
| `tables` | Seven Markdown tables in `ieee_access/src/tables/` |
| `figures` | Main and supplementary plots in `ieee_access/src/figures/` |
| `supplement` | `ieee_access/supplementary_material/supplement.pdf` and its source records |

Set `PY=/path/to/python` to use another environment. The analysis uses 10,000 bootstrap resamples and the recorded random seeds.

## Repository layout

| Directory | Contents |
| --- | --- |
| `mjcs/code/` | Original decoder, training, and statistical implementation |
| `mjcs/results/` | Original experiment records and data manifests |
| `mjcs/environment/` | Full experiment environment lock |
| `mjcs/submission/` | Calibration-artifact fitting and causal benchmark decoder |
| `ieee_access/experiments/` | Smaller-system solver, controls, confirmed baseline search, sensitivity experiments, resource measurements, and analysis tools |
| `ieee_access/experiments/results/revised_study/` | Authoritative stream, selection, and data-dose records for the manuscript |
| `ieee_access/experiments/results/revised_endpoint/` | Endpoint-specific non-inferiority and margin analyses |

The revised study incorporates the confirmed gradient configurations. Original records remain available for provenance; use `revised_study` for the current comparisons. The original epoch sweeps are retained in `mjcs/results/epochs/` as historical records. Source hashes bind the recorded fits to their implementations.

## Run the experiments

The recorded workstation used Python 3.10.20, PyTorch 2.5.1 with CUDA 12.4, and an NVIDIA GeForce RTX 4080 Laptop GPU. Install the full environment and the benchmark loader before fitting models:

```sh
.venv/bin/python -m pip install -r mjcs/environment/requirements.lock
.venv/bin/python -m pip install falcon-challenge==1.0.2
make data
```

For container builds, clone the benchmark source into `falcon-challenge/` before downloading data; see `mjcs/submission/README.txt`.

The data target downloads DANDI datasets 000954 (H1), 000941 (M1), and 000953 (M2) to `falcon-challenge/data/`. The expected file hashes are in `mjcs/results/provenance/manifest_*.json`.

Download [the released checkpoints](https://github.com/nGQD/hdc-session-adaptation/releases/download/v1.0/checkpoints.tar.gz) and extract them **inside `mjcs/`**, so the files occupy `mjcs/results/checkpoints/`. The experiment records identify the exact checkpoint hashes.

```sh
make experiments
make resources CPU_AFFINITY=0,2,4,6
make analysis
```

`make experiments` runs the feature controls, solver checks, confirmed gradient search, sensitivity experiments, and study assembly using the released checkpoints. It does not retrain the backbone. Runners validate and reuse completed records; to perform new fits, first move the corresponding completed result directories aside. Keep the original core implementation and selection files unchanged when reproducing the fixed confirmation plan.

Resource measurements must run after the fitting workers have stopped. For fresh timing measurements, move the completed `ieee_access/experiments/results/resources/` directory aside before running `make resources`. The affinity above selects four distinct performance cores on the reported workstation; choose suitable logical processor IDs on another machine. The benchmark records prefix-inclusive adaptation time, GPU tensor allocations, and processor resident memory separately.

## Hidden-test decoder

The selected hidden-test records are H1 submission 586729, M1 submission 586750, and M2 submission 586751. M1 uses the selected 4096-dimensional code. The completed-status and artifact checks are recorded in `ieee_access/experiments/results/hidden_m1_selected/selected_hidden_test_verification.json`.

`mjcs/submission/fit_corrections.py` fits calibration corrections, and `mjcs/submission/decoder.py` runs causal inference. See `mjcs/submission/README.txt` for commands and container inputs. The [released decoder artifacts](https://github.com/nGQD/hdc-session-adaptation/releases/download/v1.0/decoder_artifacts.tar.gz) contain the selected models. Extract the archive inside `mjcs/`; use `mjcs/submission/artifacts/h1/`, `m1/`, and `m2/`. The selected M1 directory is `m1/`, with the 4096-dimensional code.

## Checksums and license

`MANIFEST.sha256` records the files in this release. Check the downloaded tree before regenerating outputs with `make check-manifest`.

Study code is distributed under [AGPL-3.0](LICENSE). The neural recordings retain their original data-use terms and are downloaded from DANDI rather than redistributed here.
