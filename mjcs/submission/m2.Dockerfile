FROM pytorch/pytorch:2.1.2-cuda11.8-cudnn8-devel

WORKDIR /workspace

RUN apt-get update \
    && apt-get install -y --no-install-recommends git python3-venv \
    && python3 -m venv /venv \
    && /venv/bin/pip install --upgrade pip setuptools wheel \
    && rm -rf /var/lib/apt/lists/*

COPY falcon-challenge/falcon_challenge ./falcon-challenge/falcon_challenge
COPY falcon-challenge/decoder_demos ./falcon-challenge/decoder_demos
COPY falcon-challenge/preproc ./falcon-challenge/preproc
COPY falcon-challenge/setup.py ./falcon-challenge/
COPY falcon-challenge/pyproject.toml ./falcon-challenge/
COPY falcon-challenge/README.md ./falcon-challenge/

RUN /venv/bin/pip install --no-cache-dir -e falcon-challenge \
    && /venv/bin/pip install --no-cache-dir "numpy<2" scipy

COPY mjcs/code/model.py ./model.py
COPY mjcs/submission/decoder.py ./decode.py
COPY mjcs/submission/artifacts/m2/backbone.pt ./artifacts/m2/backbone.pt
COPY mjcs/submission/artifacts/m2/corrections.pt ./artifacts/m2/corrections.pt

ENV PREDICTION_PATH="/submission/submission.csv"
ENV PREDICTION_PATH_LOCAL="/tmp/submission.pkl"
ENV GT_PATH="/tmp/ground_truth.pkl"
ENV EVALUATION_LOC=remote
ENV EVAL_DATA_PATH="/dataset/evaluation_data"
ENV SPLIT=m2
ENV PHASE=test
ENV BATCH_SIZE=1
ENV CUDA_VISIBLE_DEVICES=""

CMD ["/bin/bash", "-c", "source /venv/bin/activate && python decode.py --evaluation $EVALUATION_LOC --split $SPLIT --phase $PHASE --batch-size $BATCH_SIZE"]
