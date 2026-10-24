FALCON submission package for H1, M1, M2

Method
  Pretrained backbone (code/model.py Backbone, checkpoint results/checkpoints/{task}/outer/full_s101.pt)
  with a per-session closed-form output correction fitted on the full released held-out calibration files
  (PriorMethod in code/engine.py). Session identity comes from the dataset tag passed to reset():
  H1 uses the day (S0, S1, ...), M1 the date, M2 the date. Held-in sessions use the backbone readout with
  session normalization priors from held-in calibration data. Unknown tags fall back to pooled held-in priors.
  Inference is causal, CPU only, one step per bin. Recurrent state and running statistics carry over
  between consecutive files of the same session. Use batch size 1.

Files
  fit_corrections.py   computes artifacts/{task}/corrections.pt and copies artifacts/{task}/backbone.pt
  decoder.py           BCIDecoder implementation and evaluator entry point
  {h1,m1,m2}.Dockerfile, {h1,m1,m2}.Dockerfile.dockerignore

Run the commands below from the repository root, which contains mjcs/, falcon-challenge/ and .venv/.

1. Fit corrections (requires mjcs/results/selection/{task}.json and mjcs/results/checkpoints/{task}/outer/full_s101.pt)

  for t in h1 m1 m2; do
    CUDA_VISIBLE_DEVICES= .venv/bin/python mjcs/submission/fit_corrections.py --task $t
  done

  Options: --method {hdc,hdc_bound,closed_linear} (default hdc), --checkpoint, --selection, --out.

2. Local format check on minival (CPU). Minival overlaps calibration data and is not the paper's held-out evaluation.

  export CUDA_VISIBLE_DEVICES= EVAL_DATA_PATH=$PWD/falcon-challenge/data
  export PREDICTION_PATH_LOCAL=/tmp/pred.pkl GT_PATH=/tmp/gt.pkl
  .venv/bin/python mjcs/submission/decoder.py --evaluation local --split h1 --phase minival --batch-size 1

  Replace h1 with m1 or m2. Explicit files: --checkpoint PATH --corrections PATH.

3. Build images (build context is the repository root). These Dockerfiles require the benchmark source in falcon-challenge/, not only its data subdirectory. Clone the source before downloading data:

  git clone https://github.com/snel-repo/falcon-challenge.git falcon-challenge

  The container uses PyTorch 2.1.2 and numpy<2; the separate analysis environment is recorded in mjcs/environment/requirements.lock.

  docker build -f mjcs/submission/h1.Dockerfile -t mjcs-h1 .
  docker build -f mjcs/submission/m1.Dockerfile -t mjcs-m1 .
  docker build -f mjcs/submission/m2.Dockerfile -t mjcs-m2 .

4. Local container evaluation on minival

  docker run --rm -e EVALUATION_LOC=local -e PHASE=minival -e EVAL_DATA_PATH=/data \
    -v $PWD/falcon-challenge/data:/data:ro mjcs-h1

  Use mjcs-m1 and mjcs-m2 for the other tracks.

5. Submission to EvalAI uses the images from step 3 with the standard evalai push workflow for each track.

The selected hidden-test submissions are H1 586729, M1 586750, and M2 586751.
The selected M1 artifacts use D=4096 and occupy submission/artifacts/m1 in the v1.0
release archive. Extract that archive inside mjcs. Use the h1, m1 and m2 directories
for the reported models; the additional m1_d2048 directory is a historical variant.
