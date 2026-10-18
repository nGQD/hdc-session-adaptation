import argparse
import copy
import json
import time

import numpy as np
import torch
from torch import nn

import data
from engine import DEVICE, batches, r2, seed_all, step_loss, stream, summarize, windows
from run import FINAL_SEEDS, OUT, base_config, prepare

LAGS = 5
LAMBDAS = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]
WF_MODES = ["zeroshot", "oracle", "continual"]
HIDDEN = 128


def lagged(x, lags=LAGS):
    out = np.zeros((len(x), x.shape[1] * lags + 1), np.float64)
    for i in range(lags):
        out[i:, i * x.shape[1]:(i + 1) * x.shape[1]] = x[:len(x) - i]
    out[:, -1] = 1.0
    return out


class Stats:
    def __init__(self, dim, outputs, lam):
        self.A = np.zeros((dim, dim))
        self.B = np.zeros((dim, outputs))
        self.lam, self.macs = lam, 0

    def add(self, p, t):
        w = 1.0 / len(p)
        self.A += w * p.T @ p
        self.B += w * p.T @ t
        self.macs += len(p) * p.shape[1] * (p.shape[1] + 1) // 2 + len(p) * p.shape[1] * t.shape[1]

    def solve(self, extra=None):
        A, B = (self.A, self.B) if extra is None else (self.A + extra.A, self.B + extra.B)
        d = len(A)
        L = np.linalg.cholesky(A + self.lam * np.eye(d))
        self.macs += d ** 3 // 6 + d * d * B.shape[1]
        return np.linalg.solve(L.T, np.linalg.solve(L, B))

    def numel(self):
        return self.A.size + self.B.size


class WienerMethod:
    def __init__(self, mode, lam):
        self.mode, self.lam = mode, lam

    def feats(self, s, rows=None):
        if s.name not in self.cache:
            self.cache[s.name] = lagged(s.x)
        p = self.cache[s.name]
        if rows is None:
            return p
        return p[rows], s.yz[rows].astype(np.float64)

    def rows(self, s):
        m = np.zeros(len(s.x), bool)
        m[:s.fit_end] = s.fit[:s.fit_end]
        return m

    def start(self, model, held_in, c, seed):
        self.cache, self.c, self.own = {}, c, {}
        pairs = [self.feats(s, self.rows(s)) for s in held_in]
        self.stats = Stats(pairs[0][0].shape[1], c.outputs, self.lam)
        for p, t in pairs:
            self.stats.add(p, t)
        self.W = self.stats.solve()
        self.pre_macs = self.stats.macs
        self.base = self.W

    def adapt(self, s):
        if self.mode == "zeroshot":
            return 0
        p, t = self.feats(s, self.rows(s))
        before = self.stats.macs
        if self.mode == "continual":
            self.stats.add(p, t)
            self.W = self.stats.solve()
        else:
            one = Stats(p.shape[1], t.shape[1], self.stats.lam)
            one.add(p, t)
            self.stats.macs += one.macs
            self.own[s.name] = self.stats.solve(one)
        return self.stats.macs - before

    def predict(self, s):
        W = self.own.get(s.name, self.W)
        return (self.feats(s) @ W).astype(np.float32)

    def memory(self):
        return self.stats.numel() + self.W.size * (1 + len(self.own))

    def inference_macs(self):
        return self.W.size


class Decoder(nn.Module):
    def __init__(self, n, outputs, hidden=HIDDEN):
        super().__init__()
        self.gru = nn.GRU(n, hidden, batch_first=True)
        self.head = nn.Linear(hidden, outputs)

    def forward(self, x, h=None):
        z, h = self.gru(x, h)
        return self.head(z), h

    def macs(self):
        g = self.gru
        return 3 * (g.input_size * g.hidden_size + g.hidden_size ** 2) + self.head.in_features * self.head.out_features


def fit(model, w, epochs, lr):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    processed = 0
    for epoch in range(epochs):
        model.train()
        for g in opt.param_groups:
            g["lr"] = lr * (0.01 + 0.99 * 0.5 * (1 + np.cos(np.pi * epoch / max(1, epochs))))
        for x, y, m in batches(w, 64):
            opt.zero_grad(set_to_none=True)
            step_loss(model, x, y, m).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            opt.step()
            processed += x.shape[0] * x.shape[1]
    return processed


@torch.no_grad()
def decode(model, x, chunk=4096):
    model.eval()
    h, out = None, []
    for z in torch.as_tensor(x, device=DEVICE).split(chunk):
        p, h = model(z[None], h)
        out.append(p[0])
    return torch.cat(out).cpu().numpy()


class RNNMethod:
    def __init__(self, mode, path=None, epochs=30, lr=1e-3, adapt_epochs=None, adapt_lr=1e-4):
        self.mode, self.path, self.epochs, self.lr = mode, path, epochs, lr
        self.adapt_epochs = adapt_epochs or (5 if mode == "oracle" else 20)
        self.adapt_lr = adapt_lr

    def start(self, model, held_in, c, seed):
        self.c, self.held_in, self.own, self.seed, self.calls = c, held_in, {}, seed, 0
        net = Decoder(c.channels, c.outputs).to(DEVICE)
        if self.path is not None and self.path.exists():
            rec = torch.load(self.path, map_location=DEVICE, weights_only=False)
            net.load_state_dict(rec["state"])
            self.pre_macs = rec["macs"]
        else:
            seed_all(seed)
            net = Decoder(c.channels, c.outputs).to(DEVICE)
            self.pre_macs = 3 * net.macs() * fit(net, windows(held_in, c), self.epochs, self.lr)
            if self.path is not None:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(dict(state=net.state_dict(), macs=self.pre_macs), self.path)
        self.base = net
        self.net = copy.deepcopy(net)

    def adapt(self, s):
        if self.mode == "zeroshot":
            return 0
        self.calls += 1
        seed_all(self.seed * 1000 + self.calls)
        if self.mode == "oracle":
            net = copy.deepcopy(self.base)
            w = windows(self.held_in + [s], self.c)
            self.own[s.name] = net
        else:
            net, w = self.net, windows([s], self.c, limit=s.fit_end)
        return 3 * net.macs() * fit(net, w, self.adapt_epochs, self.adapt_lr)

    def predict(self, s):
        return decode(self.own.get(s.name, self.net), s.x)

    def memory(self):
        n = sum(p.numel() for p in self.base.parameters())
        return n * (1 + len(self.own))


def select_lambda(task, c):
    path = OUT / "selection" / f"wf_{task}.json"
    if path.exists():
        return json.loads(path.read_text())["selected"]
    hi, ho, ys = prepare(task, c, inner=True)
    scores = {m: [summarize(stream(WienerMethod(m, lam), None, hi, ho, c, 0, ys))["acc"] for lam in LAMBDAS] for m in WF_MODES}
    selected = {m: LAMBDAS[int(np.argmax(v))] for m, v in scores.items()}
    data.write_json(path, dict(selected=selected, grid=LAMBDAS, scores=scores,
                               protocol="inner stream on the pretraining portion (first 80 percent) of held-in sessions; last two as pseudo-new sessions; score is mean inner ACC; no outer test bin is scored"))
    return selected


def methods(task, seed, lams=None):
    ckpt = OUT / "checkpoints" / "falcon_rnn" / task / f"s{seed}.pt"
    return dict(
        wf_zeroshot=lambda: WienerMethod("zeroshot", lams["zeroshot"]),
        wf_oracle=lambda: WienerMethod("oracle", lams["oracle"]),
        wf_continual=lambda: WienerMethod("continual", lams["continual"]),
        rnn_zeroshot=lambda: RNNMethod("zeroshot", ckpt),
        rnn_oracle=lambda: RNNMethod("oracle", ckpt),
        rnn_finetune=lambda: RNNMethod("finetune", ckpt),
    )


def info_of(method, c):
    if isinstance(method, WienerMethod):
        return dict(macs=dict(encode=0, head=method.inference_macs()), info=dict(lags=LAGS, lam=method.stats.lam))
    n = Decoder(c.channels, c.outputs)
    return dict(macs=dict(encode=n.macs() - HIDDEN * c.outputs, head=HIDDEN * c.outputs),
                info=dict(hidden=HIDDEN, parameters=sum(p.numel() for p in n.parameters())))


def run(task, names, seeds, force=False):
    c = base_config(task)
    hi, ho, ys = prepare(task, c)
    lams = select_lambda(task, c) if any(n.startswith("wf_") for n in names) else None
    done = {}
    for seed in seeds:
        for name in names:
            path = OUT / "stream" / task / "falcon" / f"{name}_s{seed}.json"
            if path.exists() and not force:
                continue
            if name.startswith("wf_") and name in done:
                data.write_json(path, dict(done[name], seed=seed))
                continue
            method = methods(task, seed, lams)[name]()
            t0 = time.perf_counter()
            out = stream(method, None, hi, ho, c, seed, ys)
            out.update(summary=summarize(out), task=task, variant="falcon", seed=seed, method=name,
                       backbone=info_of(method, c))
            data.write_json(path, out)
            if name.startswith("wf_"):
                done[name] = out
            s = out["summary"]
            print(f"{task} {name} s{seed} acc={s['acc']:.4f} in={s['acc_held_in']:.4f} out={s['acc_held_out']:.4f} "
                  f"zs={s['zero_shot_held_out']:.4f} {time.perf_counter() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="+", default=["h1", "m1", "m2"])
    ap.add_argument("--seeds", nargs="+", type=int, default=FINAL_SEEDS)
    ap.add_argument("--methods", nargs="+", default=["wf_zeroshot", "wf_oracle", "wf_continual"])
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    for task in a.tasks:
        run(task, a.methods, a.seeds, a.force)
