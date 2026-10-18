import argparse
import os
import re
import sys
from pathlib import Path
from typing import List

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
for p in (HERE, HERE.parent / "code", HERE.parent.parent / "falcon-challenge"):
    if p.exists() and str(p) not in sys.path:
        sys.path.insert(0, str(p))

from falcon_challenge.config import FalconConfig, FalconTask
from falcon_challenge.interface import BCIDecoder
from model import Backbone, Config

torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))


def session_name(task, tag):
    tag = str(tag)
    if task == FalconTask.h1:
        return tag.split("_set_")[0]
    if task == FalconTask.m2:
        return tag.split("_")[-1]
    return tag


class MJCSDecoder(BCIDecoder):
    def __init__(self, task_config: FalconConfig, batch_size: int = 1, checkpoint=None, corrections=None):
        super().__init__(task_config=task_config, batch_size=batch_size)
        checkpoint = checkpoint or os.environ.get("MJCS_CHECKPOINT", str(HERE / "artifacts" / task_config.task.name / "backbone.pt"))
        corrections = corrections or os.environ.get("MJCS_CORRECTIONS", str(HERE / "artifacts" / task_config.task.name / "corrections.pt"))
        rec = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if "state" in rec and "config" in rec:
            self.c = Config(**rec["config"])
            state = rec["state"]
        else:
            self.c = Config(channels=task_config.n_channels, outputs=task_config.out_dim)
            state = rec
        self.model = Backbone(self.c)
        self.model.load_state_dict(state)
        self.model.eval()
        self.corr = torch.load(corrections, map_location="cpu", weights_only=False)
        self.kind = self.corr["kind"]
        self.ystd = self.corr["ystats"]["std"].astype(np.float32)
        self.ymean = self.corr["ystats"]["mean"].astype(np.float32)
        enc = self.corr["enc"]
        if enc is not None:
            self.enc_w = torch.as_tensor(enc["w"], dtype=torch.float32)
            self.enc_b = torch.as_tensor(enc["b"], dtype=torch.float32)
            self.dimension = enc["dimension"]
            self.source = enc["source"]
        self.last_session = [None] * self.batch_size
        self.reset_rows(range(self.batch_size))
        self.entries = [self.entry(None)] * self.batch_size

    def entry(self, name):
        s = self.corr["sessions"].get(name)
        if s is None:
            s = self.corr["default"]
        return dict(mean=s["mean"], std=s["std"], W=torch.as_tensor(s["W"], dtype=torch.float64))

    def reset_rows(self, rows):
        b, n = self.batch_size, self._task_config.n_channels
        if not hasattr(self, "hidden"):
            self.hidden = torch.zeros(1, b, self.c.hidden)
            self.count = np.zeros(b)
            self.sum = np.zeros((b, n))
            self.sumsq = np.zeros((b, n))
            self.smooth = np.zeros((b, n))
        for i in rows:
            self.hidden[:, i] = 0
            self.count[i] = 0
            self.sum[i] = 0
            self.sumsq[i] = 0
            self.smooth[i] = 0

    def reset(self, dataset_tags: List[Path] = [""]):
        tags = [self._task_config.hash_dataset(Path(t)) if str(t) else "" for t in dataset_tags]
        names = [session_name(self._task_config.task, t) if t else None for t in tags]
        names += [None] * (self.batch_size - len(names))
        changed = [i for i in range(self.batch_size) if names[i] != self.last_session[i]]
        self.reset_rows(changed)
        for i in changed:
            self.entries[i] = self.entry(names[i])
        self.last_session = names

    def on_done(self, dones: np.ndarray):
        pass

    def normalize(self, x):
        z = np.empty_like(x, dtype=np.float64)
        for i in range(self.batch_size):
            e = self.entries[i]
            self.count[i] += 1
            self.sum[i] += x[i]
            self.sumsq[i] += x[i] * x[i]
            n = self.count[i]
            mean, std = e["mean"], e["std"]
            if self.c.online_norm:
                om = self.sum[i] / n
                ov = max_zero(self.sumsq[i] / n - om * om)
                blend = min(0.7, n / 2000) if n >= 30 else 0.0
                mean = (1 - blend) * mean + blend * om
                std = (1 - blend) * std + blend * np.maximum(np.sqrt(ov), 0.01)
            v = (x[i] - mean) / std
            a = self.c.smoothing
            if a:
                self.smooth[i] = v if n == 1 else a * self.smooth[i] + (1 - a) * v
                v = self.smooth[i]
            z[i] = v
        return z.astype(np.float32)

    def phi(self, g, f):
        one = torch.ones(len(g), 1)
        if self.kind == "linear":
            return torch.cat([g, one], 1)
        src = g if self.source == "penultimate" else f
        theta = src @ self.enc_w + self.enc_b
        e = torch.cat([theta.cos(), theta.sin()], -1) / self.dimension ** 0.5
        return torch.cat([g, one, e * self.dimension ** 0.5], 1)

    @torch.no_grad()
    def predict(self, neural_observations: np.ndarray):
        x = np.asarray(neural_observations, dtype=np.float64)
        if x.shape[0] < self.batch_size:
            x = np.pad(x, ((0, self.batch_size - x.shape[0]), (0, 0)))
        z = torch.as_tensor(self.normalize(x))
        f, h = self.model.encode(z[:, None, :], self.hidden)
        self.hidden = h
        g = self.model.readout[:-1](f)[:, 0] if f.dim() == 3 else self.model.readout[:-1](f)
        f = f[:, 0]
        p = self.phi(g, f).double()
        out = torch.stack([(p[i] @ self.entries[i]["W"]).float() for i in range(self.batch_size)]).numpy()
        return out * self.ystd + self.ymean


def max_zero(a):
    return np.maximum(a, 0)


def main():
    from falcon_challenge.evaluator import FalconEvaluator

    ap = argparse.ArgumentParser()
    ap.add_argument("--evaluation", choices=["local", "remote"], default=os.environ.get("EVALUATION_LOC", "remote"))
    ap.add_argument("--split", choices=["h1", "m1", "m2"], default=os.environ.get("SPLIT", "h1"))
    ap.add_argument("--phase", choices=["minival", "test"], default=os.environ.get("PHASE", "test"))
    ap.add_argument("--batch-size", type=int, default=int(os.environ.get("BATCH_SIZE", "1")))
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--corrections", default=None)
    a = ap.parse_args()
    config = FalconConfig(task=getattr(FalconTask, a.split))
    decoder = MJCSDecoder(config, a.batch_size, a.checkpoint, a.corrections)
    evaluator = FalconEvaluator(eval_remote=a.evaluation == "remote", split=a.split)
    evaluator.evaluate(decoder, phase=a.phase)


if __name__ == "__main__":
    main()
