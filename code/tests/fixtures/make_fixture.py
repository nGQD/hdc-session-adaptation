import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
METHODS = dict(frozen=(0.0, 0.0, 0.0, 0.0), finetune_full=(0.10, 0.70, 0.28, 3e9), finetune_readout=(0.06, 0.45, 0.10, 4e8),
               ewc=(0.09, 0.62, 0.20, 3.4e9), replay=(0.10, 0.68, 0.12, 6e9), full_tag=(0.09, 0.66, 0.00, 3e9),
               readout_tag=(0.08, 0.60, 0.00, 4e8), last_tag=(0.06, 0.50, 0.00, 1e8), closed_linear=(0.07, 0.62, 0.02, 2e7),
               closed_zero=(0.06, 0.55, 0.02, 2e7), hdc=(0.11, 0.78, 0.01, 9e7), hdc_zero=(0.10, 0.70, 0.01, 9e7),
               hdc_shared=(0.10, 0.72, 0.03, 9e7), hdc_bound=(0.11, 0.80, 0.01, 1.1e8))
SIZES = dict(h1=(4, 6), m1=(3, 8), m2=(3, 8))
FIXTURE_ABLATIONS = ["linear", "mlp", "standard_gru", "no_gru", "no_online_norm", "no_smoothing", "no_refinement"]


def record(task, variant, method, seed, penalty):
    rng = np.random.default_rng(
    sum(map(ord, task + variant + method)) * 131 + seed)
    k, t = SIZES[task]
    n = k + t
    base = np.clip(0.75 - 0.03 * np.arange(n) + rng.normal(0, 0.03, n), 0.2, 0.9) - penalty
    zero, gain, forget, macs = METHODS[method]
    R = np.zeros((t + 1, n))
    R[0, :k] = base[:k]
    R[0, k:] = base[k:] * (0.15 + 0.1 * rng.random(t)) - (0.3 if method == "frozen" else 0)
    for i in range(1, t + 1):
        R[i] = R[i - 1]
        R[i, k + i - 1] = base[k + i - 1] * (gain if method != "frozen" else 0) + zero + rng.normal(0, 0.02)
        R[i, :k] -= forget * rng.random(k) * 0.12
        R[i, k:k + i - 1] -= forget * 0.08 * rng.random(i - 1)
    R += rng.normal(0, 0.01, R.shape)
    out = dict(R=R.tolist(), names=[f"s{j + 1}" for j in range(n)], held_in=k, adapt_macs=[macs * (1 + 0.1 * rng.random())] * t,
               adapt_seconds=[max(macs, 1) / 4e9 * (1 + 0.1 * rng.random())] * t, memory=[1000] * t,
               start_macs=0, start_seconds=0.0, task=task, variant=variant, seed=seed, method=method,
               backbone=dict(macs=dict(encode=1, head=1), info={}))
    from_matrix(out)
    return out


def from_matrix(out):
    R = np.array(out["R"])
    k, T = out["held_in"], R.shape[0] - 1
    diag = np.array([R[i + 1, k + i] for i in range(T)])
    bo = np.mean([R[T, k + i] - R[i + 1, k + i] for i in range(T - 1)])
    bi = (R[T, :k] - R[0, :k]).mean()
    out["summary"] = dict(acc=float(R[T].mean()), acc_held_in=float(R[T, :k].mean()), acc_held_out=float(R[T, k:].mean()),
                          plasticity=float(diag.mean()), bwt=float((bi * k + bo * (T - 1)) / (k + T - 1)), bwt_held_in=float(bi),
                          bwt_held_out=float(bo), zero_shot_held_out=float(R[0, k:].mean()),
                          adapt_macs=float(np.mean(out["adapt_macs"])), adapt_seconds=float(np.mean(out["adapt_seconds"])), memory=1000.0)


def build(root):
    root = Path(root)
    for task in SIZES:
        for m in METHODS:
            for s in range(10):
                put(root, task, "full", m, s, 0.0)
        for v_i, v in enumerate(FIXTURE_ABLATIONS):
            for m in ["frozen", "hdc", "hdc_bound", "finetune_full"]:
                for s in range(5):
                    put(root, task, v, m, s, 0.015 * (v_i + 1) * (1 + 0.3 * (task == "m2")))
    (root / "theory").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    D = [64, 128, 256, 512, 1024, 2048, 4096]
    S = [2, 4, 8, 16, 32, 64, 128, 256]
    pred = [[0.9 * np.sqrt(s / d) for s in S] for d in D]
    emp = [[v * (1 + rng.normal(0, 0.06)) for v in row] for row in pred]
    json.dump(dict(D=D, S=S, empirical=emp, predicted=pred), open(root / "theory" / "bound.json", "w"))
    for task in SIZES:
        (root / "epochs" / task).mkdir(parents=True, exist_ok=True)
        for m, top in (("finetune_full", 0.7), ("replay", 0.72), ("readout_tag", 0.6)):
            for e in [1, 2, 5, 10, 20, 40]:
                for s in range(3):
                    acc = top * (1 - np.exp(-e / 8)) + rng.normal(0, 0.01)
                    json.dump(dict(task=task, method=m, epochs=e, seed=s, summary=dict(acc=acc, acc_held_in=acc, acc_held_out=acc, bwt=-0.01 * e, adapt_macs=1e8 * e)),
                              open(root / "epochs" / task / f"{m}_e{e}_s{s}.json", "w"))
    (root / "dose").mkdir(parents=True, exist_ok=True)
    for task in SIZES:
        (root / "dose" / task).mkdir(parents=True, exist_ok=True)
        for m in ["hdc", "hdc_bound", "finetune_full", "closed_linear"]:
            for s in range(5):
                x = [1, 2, 4, 8, 16]
                scale = 0.8 if m.startswith("hdc") else 0.6
                y = [scale * (1 - np.exp(-xi / (3 if m.startswith("hdc") else 8))) + rng.normal(0, 0.015) for xi in x]
                json.dump(dict(task=task, method=m, seed=s, dose=x, acc_held_out=y), open(root / "dose" / task / f"{m}_s{s}.json", "w"))


def put(root, task, variant, method, seed, penalty):
    d = root / "stream" / task / variant
    d.mkdir(parents=True, exist_ok=True)
    json.dump(record(task, variant, method, seed, penalty), open(d / f"{method}_s{seed}.json", "w"))


if __name__ == "__main__":
    build(HERE / "data")
