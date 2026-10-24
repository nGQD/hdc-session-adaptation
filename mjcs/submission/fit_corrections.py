import argparse
import json
import os
import shutil
import sys
from dataclasses import asdict
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "code"))

import numpy as np
import torch

import data
from engine import PriorMethod, Session, prior_of, target_stats
from model import Backbone, Config


def load_checkpoint(path, task):
    rec = torch.load(path, map_location="cpu", weights_only=False)
    if "state" in rec and "config" in rec:
        return Config(**rec["config"]), rec["state"], rec.get("seed", 101)
    rows = data.manifest(task)
    return Config(channels=rows[0]["channels"], outputs=rows[0]["outputs"]), rec, 101


def effective_heads(method, kind, names):
    k0 = method.W0.shape[0]
    out = {}
    for name in names:
        d = k0 + (2 * method.dimension if method.enc is not None else 0)
        W = method.prior(d)
        if name in method.heads:
            if kind == "bound":
                W = W.clone()
                W[:k0] += method.heads[name]
                W[k0:] += method.enc.bind(method.memory_W.T.float(), method.keys[name]).T.double()
            else:
                W = W + method.heads[name]
        out[name] = W
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=["h1", "m1", "m2"])
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--selection", default=None)
    ap.add_argument("--method", default="hdc", choices=["hdc", "hdc_bound", "closed_linear"])
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    task = a.task
    checkpoint = Path(a.checkpoint or ROOT / "results" / "checkpoints" / task / "outer" / "full_s101.pt")
    selection = Path(a.selection or ROOT / "results" / "selection" / f"{task}.json")
    out = Path(a.out or HERE / "artifacts" / task)
    if not selection.exists():
        sys.exit(f"missing selection file: {selection}")
    sel = json.loads(selection.read_text())
    kind = dict(hdc="hdc", hdc_bound="bound", closed_linear="linear")[a.method]
    params = sel["linear"] if kind == "linear" else sel["hdc"]

    c, state, seed = load_checkpoint(checkpoint, task)
    model = Backbone(c)
    model.load_state_dict(state)
    model.eval()

    held_in_parts = [(k, data.held_in_split(rs)) for k, rs in data.by_session(data.load(task, "held-in-calib"))]
    ystats = target_stats([p for _, p in held_in_parts])
    held_in = [Session(k, p, "train", c, ystats) for k, p in held_in_parts]
    held_out_parts = [(k, data.held_out_split(rs, prefix=1.0)) for k, rs in data.by_session(data.load(task, "held-out-calib"))]
    held_out = [Session(k, p, "adapt", c, ystats) for k, p in held_out_parts]

    method = PriorMethod(kind, **params)
    method.start(model, [], c, seed)
    for s in held_out:
        method.adapt(s)

    names = [s.name for s in held_in] + [s.name for s in held_out]
    heads = effective_heads(method, kind, names)
    sessions = {}
    for s in held_in + held_out:
        sessions[s.name] = dict(mean=s.prior["mean"].astype(np.float64), std=s.prior["std"].astype(np.float64),
                                W=heads[s.name].float().numpy(), held_out=s in held_out)
    pool = dict(mean=np.mean([s.prior["mean"] for s in held_in], 0), std=np.mean([s.prior["std"] for s in held_in], 0))
    default_W = method.prior(next(iter(heads.values())).shape[0]).float().numpy()

    payload = dict(task=task, kind=kind, params=params, seed=seed, config=asdict(c),
                   ystats=dict(mean=ystats["mean"], std=ystats["std"]), sessions=sessions,
                   default=dict(mean=pool["mean"], std=pool["std"], W=default_W),
                   enc=None if method.enc is None else dict(w=method.enc.w.cpu().numpy(), b=method.enc.b.cpu().numpy(),
                                                            dimension=method.dimension, source=method.source))
    out.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out / "corrections.pt")
    torch.save(dict(config=asdict(c), state=state, seed=seed), out / "backbone.pt")
    print(task, a.method, "sessions", len(held_in), "held-in", len(held_out), "held-out ->", out)


if __name__ == "__main__":
    main()
