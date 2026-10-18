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

Paths below are relative to the parent directory of this repository (the directory that contains this repository cloned as hdc-session-adaptation/, a clone of https://github.com/snel-repo/falcon-challenge as falcon-challenge/, and .venv/).

1. Fit corrections (requires results/selection/{task}.json and results/checkpoints/{task}/outer/full_s101.pt)

  for t in h1 m1 m2; do
    CUDA_VISIBLE_DEVICES= .venv/bin/python hdc-session-adaptation/submission/fit_corrections.py --task $t
  done

  Options: --method {hdc,hdc_bound,closed_linear} (default hdc), --checkpoint, --selection, --out.

2. Local format check on the minival split (CPU). Minival is not a held-out split and its score is not reported in the paper.

  export CUDA_VISIBLE_DEVICES= EVAL_DATA_PATH=$PWD/falcon-challenge/data
  export PREDICTION_PATH_LOCAL=/tmp/pred.pkl GT_PATH=/tmp/gt.pkl
  .venv/bin/python hdc-session-adaptation/submission/decoder.py --evaluation local --split h1 --phase minival --batch-size 1

  Replace h1 with m1 or m2. Explicit files: --checkpoint PATH --corrections PATH.

3. Build images. The images use pytorch/pytorch:2.1.2-cuda11.8-cudnn8-devel, numpy<2 and the falcon-challenge package from the clone; they are separate from the analysis environment (environment/requirements.lock).

  docker build -f hdc-session-adaptation/submission/h1.Dockerfile -t hdc-h1 .
  docker build -f hdc-session-adaptation/submission/m1.Dockerfile -t hdc-m1 .
  docker build -f hdc-session-adaptation/submission/m2.Dockerfile -t hdc-m2 .

4. Local container evaluation on minival

  docker run --rm -e EVALUATION_LOC=local -e PHASE=minival -e EVAL_DATA_PATH=/data \
    -v $PWD/falcon-challenge/data:/data:ro hdc-h1

  Use hdc-m1 and hdc-m2 for the other tracks.

5. Submission to EvalAI uses the images from step 3 with the standard evalai push workflow for each track.

M1 hidden-test submission 586752 used corrections fitted with D = 2048 (all other settings as in
results/selection/m1.json). Those files are in the release archive under submission/artifacts/m1_d2048;
copy them to submission/artifacts/m1 before building m1.Dockerfile to reproduce that image.
