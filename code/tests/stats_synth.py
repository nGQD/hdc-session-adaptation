import json
import shutil
from pathlib import Path

import numpy as np

TASKS = {"synth_a": (3, 6), "synth_b": (2, 5)}
LEVEL = dict(frozen=0.50, finetune_full=0.60, finetune_readout=0.58, ewc=0.60, replay=0.64, full_tag=0.61, readout_tag=0.57,
             last_tag=0.55, closed_linear=0.62, closed_zero=0.58, hdc=0.66, hdc_zero=0.62, hdc_shared=0.64, hdc_bound=0.66)
FORGET = dict(frozen=0.0, finetune_full=0.03, finetune_readout=0.01, ewc=0.015, replay=0.004, full_tag=0.0, readout_tag=0.0,
              last_tag=0.0, closed_linear=0.0, closed_zero=0.0, hdc=0.0, hdc_zero=0.0, hdc_shared=0.002, hdc_bound=0.0)
MACS = dict(frozen=0, finetune_full=9e9, finetune_readout=1e9, ewc=1.2e10, replay=1.4e10, full_tag=9e9, readout_tag=1e9,
            last_tag=5e8, closed_linear=2e6, closed_zero=2e6, hdc=4e6, hdc_zero=4e6, hdc_shared=4e6, hdc_bound=5e6)
MEMORY = dict(frozen=1e5, finetune_full=1e5, finetune_readout=1e5, ewc=3e5, replay=4e6, full_tag=9e5, readout_tag=5e5,
              last_tag=3e5, closed_linear=1.1e5, closed_zero=1.1e5, hdc=1.5e5, hdc_zero=1.5e5, hdc_shared=1.2e5, hdc_bound=1.6e5)
VARIANTS = {"full": (0.0, 10), "linear": (-0.05, 5), "no_gru": (-0.03, 5)}


def summarize(R, k, macs, memory):
    T = R.shape[0] - 1
    final = R[T]
    diag = np.array([R[i + 1, k + i] for i in range(T)])
    bwt_out = np.mean([R[T, k + i] - R[i + 1, k + i] for i in range(T - 1)])
    bwt_in = (R[T, :k] - R[0, :k]).mean()
    return dict(acc=float(final.mean()), acc_held_in=float(final[:k].mean()), acc_held_out=float(final[k:].mean()),
                plasticity=float(diag.mean()), bwt=float((bwt_in * k + bwt_out * (T - 1)) / (k + T - 1)),
                bwt_held_in=float(bwt_in), bwt_held_out=float(bwt_out),
                zero_shot_held_out=float(np.mean([R[0, k + i] for i in range(T)])),
                adapt_macs=float(np.mean(macs)), adapt_seconds=1.0, memory=float(memory[-1]))


def make(dest, seed=0):
    dest = Path(dest)
    shutil.rmtree(dest / "stream", ignore_errors=True)
    for ti, (task, (k, T)) in enumerate(TASKS.items()):
        eff = np.random.default_rng([seed, ti, 99]).normal(0, 0.04, k + T)
        names = [f"in{j}" for j in range(k)] + [f"out{j}" for j in range(T)]
        for vi, (variant, (shift, n)) in enumerate(VARIANTS.items()):
            folder = dest / "stream" / task / variant
            folder.mkdir(parents=True, exist_ok=True)
            for mi, m in enumerate(LEVEL):
                for s in range(n):
                    rng = np.random.default_rng([seed, ti, vi, mi, s])
                    shared = np.random.default_rng([seed, ti, 98, s]).normal(0, 0.01)
                    R = np.zeros((T + 1, k + T))
                    for i in range(T + 1):
                        for j in range(k + T):
                            v = LEVEL[m] + shift + eff[j] + shared - FORGET[m] * i * (1 if j < k else max(i - (j - k), 0) / T * T)
                            if j >= k and i < j - k + 1:
                                v -= 0.08
                            R[i, j] = v + rng.normal(0, 0.008)
                    macs = [int(MACS[m])] * T
                    memory = [int(MEMORY[m]) + 1000 * i for i in range(1, T + 1)]
                    out = dict(synthetic=True, R=R.tolist(), names=names, held_in=k, adapt_macs=macs, adapt_seconds=[1.0] * T,
                               memory=memory, start_macs=0, start_seconds=0.0, summary=summarize(R, k, macs, memory),
                               task=task, variant=variant, seed=s, method=m, backbone=dict(macs=dict(encode=1, head=1), info={}))
                    (folder / f"{m}_s{s}.json").write_text(json.dumps(out))


if __name__ == "__main__":
    make(Path(__file__).resolve().parent / "fixtures" / "stats_data")
