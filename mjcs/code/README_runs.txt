Run from the mjcs directory.

Baselines under the shared stream protocol (results/stream/{task}/falcon/{method}_s{seed}.json)

Wiener filter, CPU (deterministic; computed once per task and copied across seeds). The ridge penalty per regime is chosen from {1e-6..10} on the inner held-in protocol (first 80 percent of held-in sessions, last two as pseudo-new sessions) and stored in results/selection/wf_{task}.json; delete that file to reselect, and pass --force to overwrite existing result files:
CUDA_VISIBLE_DEVICES= ../.venv/bin/python code/falcon_baselines.py --tasks h1 m1 m2

Recurrent decoder, GPU (zero-shot, per-session, sequential fine-tune), one process per task:
../.venv/bin/python code/falcon_baselines.py --tasks h1 --methods rnn_zeroshot rnn_oracle rnn_finetune
../.venv/bin/python code/falcon_baselines.py --tasks m1 --methods rnn_zeroshot rnn_oracle rnn_finetune
../.venv/bin/python code/falcon_baselines.py --tasks m2 --methods rnn_zeroshot rnn_oracle rnn_finetune

Base decoders are cached in results/checkpoints/falcon_rnn/{task}/s{seed}.pt and shared by the three regimes.
