import copy
import random
import time

import numpy as np
from scipy.signal import lfilter
import torch
import torch.nn.functional as F

from model import Backbone, FHRR, session_key

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def prior_of(x):
    return dict(mean=x.mean(0), std=np.maximum(x.std(0), 0.01))


def normalize(x, prior, online=True, alpha=0.9):
    x = x.astype(np.float64)
    mean, std = prior["mean"], prior["std"]
    if online:
        n = np.arange(1, len(x) + 1)[:, None]
        om = np.cumsum(x, 0) / n
        ov = np.maximum(np.cumsum(x * x, 0) / n - om * om, 0)
        blend = np.where(n >= 30, np.minimum(0.7, n / 2000), 0)
        mean = (1 - blend) * mean + blend * om
        std = (1 - blend) * std + blend * np.maximum(np.sqrt(ov), 0.01)
    z = (x - mean) / std
    if alpha:
        z = lfilter([1 - alpha], [1, -alpha], z, axis=0, zi=(alpha * z[0])[None, :])[0]
    return z.astype(np.float32)


def r2(y, p, mask):
    y, p = y[mask], p[mask]
    resid = ((y - p) ** 2).sum(0)
    total = ((y - y.mean(0)) ** 2).sum(0)
    return float(1 - resid.sum() / total.sum())


class Session:
    def __init__(self, name, part, fit_key, c, ystats):
        self.name = name
        fit = part[fit_key]
        x = part["x"]
        self.prior = prior_of(x[:part["fit_end"]])
        self.x = normalize(x, self.prior, c.online_norm, c.smoothing)
        self.y = part["y"]
        self.yz = ((part["y"] - ystats["mean"]) / ystats["std"]).astype(np.float32)
        self.fit = fit
        self.score = part["score"]
        self.fit_end = part["fit_end"]


def target_stats(parts):
    y = np.concatenate([p["y"][p["train"]] for p in parts])
    return dict(mean=y.mean(0), std=np.maximum(y.std(0), 1e-6))


def windows(sessions, c, limit=None):
    xs, ys, ms, starts, off = [], [], [], [], 0
    for s in sessions:
        end = s.fit_end if limit is None else min(s.fit_end, limit)
        x, y, m = s.x[:end], s.yz[:end], s.fit[:end]
        ends = list(range(0, len(x) - c.window + 1, c.stride))
        if len(x) >= c.window and ends[-1] != len(x) - c.window:
            ends.append(len(x) - c.window)
        starts += [off + i for i in ends if m[i:i + c.window].sum() >= 10]
        xs.append(x); ys.append(y); ms.append(m); off += len(x)
    t = lambda a, d=torch.float32: torch.tensor(np.concatenate(a), device=DEVICE, dtype=d)
    return dict(x=t(xs), y=t(ys), m=t(ms, torch.bool), starts=torch.tensor(starts, device=DEVICE, dtype=torch.long),
                off=torch.arange(c.window, device=DEVICE))


def batches(w, batch):
    for part in torch.randperm(len(w["starts"]), device=DEVICE).split(batch):
        ix = w["starts"][part, None] + w["off"][None, :]
        yield w["x"][ix], w["y"][ix], w["m"][ix]


def step_loss(model, x, y, m, augment=True):
    if augment:
        x = x + torch.randn_like(x) * 0.15
        x = x * (torch.rand(x.shape[0], 1, x.shape[-1], device=DEVICE) > 0.1).float() / 0.9
    pred, _ = model(x)
    err = F.huber_loss(pred, y, reduction="none").mean(-1)
    return (err * m).sum() / m.sum().clamp_min(1)


def pretrain(sessions, c, seed):
    seed_all(seed)
    t0 = time.perf_counter()
    w = windows(sessions, c)
    model = Backbone(c).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=c.lr, weight_decay=c.weight_decay)
    losses, processed = [], 0
    for epoch in range(c.epochs):
        model.train()
        ratio = min((epoch + 1) / 3, 1) * (0.01 + 0.99 * 0.5 * (1 + np.cos(np.pi * max(0, epoch - 3) / max(1, c.epochs - 3))))
        for g in opt.param_groups:
            g["lr"] = c.lr * ratio
        run = []
        for x, y, m in batches(w, c.batch):
            opt.zero_grad(set_to_none=True)
            loss = step_loss(model, x, y, m)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            opt.step()
            run.append(loss.detach())
            processed += x.shape[0] * x.shape[1]
        losses.append(float(torch.stack(run).mean()))
    torch.cuda.synchronize()
    return model, dict(losses=losses, seconds=time.perf_counter() - t0, bins_processed=processed,
                       parameters=model.parameter_count())


@torch.no_grad()
def run_backbone(model, x, chunk=2048, head=False):
    model.eval()
    h, out = None, []
    for z in torch.as_tensor(x, device=DEVICE).split(chunk):
        f, h = model.encode(z[None], h)
        out.append(model.readout(f)[0] if head else f[0])
    return torch.cat(out)


def linear_macs(module):
    total = 0
    for m in module.modules():
        if isinstance(m, torch.nn.Linear):
            total += m.in_features * m.out_features
        elif isinstance(m, torch.nn.GRU):
            total += 3 * (m.input_size * m.hidden_size + m.hidden_size * m.hidden_size)
    return total


def backbone_macs(model):
    c = model.config
    extra = c.rank * c.hidden if c.projection == "multiplicative" else 0
    r = model.refine
    refine = 3 * (linear_macs(r.up) + linear_macs(r.gate) + linear_macs(r.down)) + linear_macs(r.mix) if hasattr(r, "up") else 0
    return dict(encode=linear_macs(model.projection) + linear_macs(model.recurrent) + refine + extra,
                head=linear_macs(model.readout))


class Ridge:
    def __init__(self, kind, n, outputs, dimension, scale, lam, seed, balance=True):
        self.kind, self.lam, self.balance, self.seed = kind, lam, balance, seed
        self.enc = FHRR(n, dimension, scale, seed).to(DEVICE) if kind != "raw" else None
        base = n + 1 if kind == "raw" else n + 2 * dimension + 1
        self.base = base
        self.dim = base + 2 * dimension if kind == "bound" else base
        self.A = torch.zeros(self.dim, self.dim, dtype=torch.float64, device=DEVICE)
        self.B = torch.zeros(self.dim, outputs, dtype=torch.float64, device=DEVICE)
        self.keys, self.W, self.macs, self.dimension = {}, None, 0, dimension

    def phi(self, f, key=None):
        one = torch.ones(len(f), 1, device=DEVICE)
        if self.kind == "raw":
            return torch.cat([f, one], 1)
        e = self.enc(f) * self.dimension ** 0.5
        h = torch.cat([f, e, one], 1)
        if self.kind == "hdc":
            return h
        if key is None:
            return torch.cat([h, torch.zeros_like(e)], 1)
        return torch.cat([h, self.enc.bind(e, self.keys[key])], 1)

    def key(self, name):
        if self.kind == "bound" and name not in self.keys:
            self.keys[name] = session_key(self.dimension, len(self.keys), self.seed).to(DEVICE)

    def add(self, name, f, y, mask):
        self.key(name)
        p = self.phi(f[mask], name if self.kind == "bound" else None).double()
        t = torch.as_tensor(y[mask], device=DEVICE, dtype=torch.float64)
        wgt = 1.0 / len(p) if self.balance else 1.0
        self.A += wgt * p.T @ p
        self.B += wgt * p.T @ t
        self.macs += len(p) * self.dim * (self.dim + 1) // 2 + len(p) * self.dim * t.shape[1]
        if self.enc is not None:
            self.macs += len(p) * self.enc.w.shape[0] * self.dimension
        self.solve()

    def solve(self):
        reg = self.lam * torch.eye(self.dim, dtype=torch.float64, device=DEVICE)
        L = torch.linalg.cholesky(self.A + reg)
        self.W = torch.cholesky_solve(self.B, L)
        self.macs += self.dim ** 3 // 6 + self.dim ** 2 * self.B.shape[1]

    def predict(self, name, f):
        key = name if (self.kind == "bound" and name in self.keys) else None
        return (self.phi(f, key).double() @ self.W).float()

    def memory(self):
        return self.A.numel() + self.B.numel() + (self.enc.w.numel() if self.enc is not None else 0)

    def inference_macs(self):
        enc = self.enc.w.shape[0] * self.dimension if self.enc is not None else 0
        return enc + self.dim * self.B.shape[1]


class FrozenMethod:
    def start(self, model, held_in, c, seed):
        self.model, self.pre_macs = model, 0

    def adapt(self, s):
        return 0

    def predict(self, s):
        return run_backbone(self.model, s.x, head=True).cpu().numpy()

    def memory(self):
        return self.model.parameter_count()


class GradientMethod:
    def __init__(self, mode, epochs=20, lr=1e-4, ewc=0.0, replay=0, tag=False):
        self.mode, self.epochs, self.lr, self.ewc, self.replay, self.tag = mode, epochs, lr, ewc, replay, tag

    def start(self, model, held_in, c, seed):
        self.base, self.c, self.pre_macs, self.store = model, c, 0, {}
        self.model = copy.deepcopy(model)
        self.params = list(self.model.readout.parameters()) if self.mode == "readout" else list(self.model.parameters())
        self.buffer = list(held_in)
        self.fisher, self.anchor = None, None
        if self.ewc:
            self.fisher, self.anchor = self.estimate_fisher(held_in), [p.detach().clone() for p in self.params]

    def estimate_fisher(self, sessions):
        w = windows(sessions, self.c)
        fisher = [torch.zeros_like(p) for p in self.params]
        self.model.train()
        for m in self.model.modules():
            if isinstance(m, torch.nn.Dropout):
                m.eval()
        count = 0
        for x, y, m in batches(w, self.c.batch):
            self.model.zero_grad(set_to_none=True)
            step_loss(self.model, x, y, m, augment=False).backward()
            for f, p in zip(fisher, self.params):
                f += p.grad.detach() ** 2
            count += 1
        return [f / max(count, 1) for f in fisher]

    def adapt(self, s):
        c = self.c
        if self.tag:
            self.model = copy.deepcopy(self.base)
            self.params = list(self.model.readout.parameters()) if self.mode == "readout" else list(self.model.parameters())
        w = windows([s], c, limit=s.fit_end)
        replay = windows(self.buffer, c) if self.replay else None
        for p in self.model.parameters():
            p.requires_grad_(False)
        for p in self.params:
            p.requires_grad_(True)
        opt = torch.optim.AdamW(self.params, lr=self.lr, weight_decay=1e-4)
        processed = 0
        for _ in range(self.epochs):
            self.model.train()
            for x, y, m in batches(w, c.batch):
                if replay is not None:
                    ix = torch.randint(len(replay["starts"]), (x.shape[0],), device=DEVICE)
                    jx = replay["starts"][ix, None] + replay["off"][None, :]
                    x = torch.cat([x, replay["x"][jx]]); y = torch.cat([y, replay["y"][jx]]); m = torch.cat([m, replay["m"][jx]])
                opt.zero_grad(set_to_none=True)
                loss = step_loss(self.model, x, y, m)
                if self.ewc:
                    loss = loss + self.ewc * sum((f * (p - a) ** 2).sum() for f, p, a in zip(self.fisher, self.params, self.anchor))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.params, 5)
                opt.step()
                processed += x.shape[0] * x.shape[1]
        if self.tag:
            self.store[s.name] = self.model
        macs = backbone_macs(self.model)
        forward = macs["encode"] + macs["head"]
        if self.mode == "readout":
            cost = processed * (forward + 2 * macs["head"])
        else:
            cost = processed * 3 * forward
        if self.ewc:
            new = self.estimate_fisher([s])
            self.fisher = [f + g for f, g in zip(self.fisher, new)]
            self.anchor = [p.detach().clone() for p in self.params]
            cost += 3 * forward * len(w["starts"]) * c.window + 3 * sum(p.numel() for p in self.params) * processed // c.window
        if self.replay:
            self.buffer.append(s)
        return cost

    def predict(self, s):
        model = self.store.get(s.name, self.base) if self.tag else self.model
        return run_backbone(model, s.x, head=True).cpu().numpy()

    def memory(self):
        if self.tag:
            return self.base.parameter_count() * (1 + len(self.store))
        extra = sum(p.numel() for p in self.params) * 2 if self.ewc else 0
        if self.replay:
            extra += sum(s.fit_end * (s.x.shape[1] + s.y.shape[1]) for s in self.buffer)
        return self.model.parameter_count() + extra


class ReadoutMethod:
    def __init__(self, scope="readout", epochs=20, lr=1e-4, tag=False):
        self.scope, self.epochs, self.lr, self.tag = scope, epochs, lr, tag

    def start(self, model, held_in, c, seed):
        self.model, self.c, self.pre_macs, self.cache, self.heads = model, c, 0, {}, {}
        self.base = model.readout
        self.shared = copy.deepcopy(model.readout)

    def feats(self, s):
        if s.name not in self.cache:
            self.cache[s.name] = run_backbone(self.model, s.x)
        return self.cache[s.name]

    def params(self, ro):
        return list(ro[-1].parameters()) if self.scope == "last" else list(ro.parameters())

    def adapt(self, s):
        n = s.fit_end
        f = self.feats(s)[:n]
        m = torch.as_tensor(s.fit[:n], device=DEVICE)
        X, Y = f[m], torch.as_tensor(s.yz[:n], device=DEVICE)[m]
        ro = copy.deepcopy(self.base) if self.tag else self.shared
        for p in ro.parameters():
            p.requires_grad_(False)
        params = self.params(ro)
        for p in params:
            p.requires_grad_(True)
        opt = torch.optim.AdamW(params, lr=self.lr, weight_decay=1e-4)
        batch, processed = self.c.batch * self.c.window, 0
        for _ in range(self.epochs):
            ro.train()
            for ix in torch.randperm(len(X), device=DEVICE).split(batch):
                opt.zero_grad(set_to_none=True)
                loss = F.huber_loss(ro(X[ix]), Y[ix])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 5)
                opt.step()
                processed += len(ix)
        ro.eval()
        if self.tag:
            self.heads[s.name] = ro
        macs = backbone_macs(self.model)
        grad = linear_macs(ro[-1]) if self.scope == "last" else macs["head"]
        return n * macs["encode"] + processed * (macs["head"] + 2 * grad)

    @torch.no_grad()
    def predict(self, s):
        ro = self.heads.get(s.name, self.base) if self.tag else self.shared
        ro.eval()
        return torch.cat([ro(z) for z in self.feats(s).split(16384)]).cpu().numpy()

    def memory(self):
        head = sum(p.numel() for p in self.params(self.base))
        return self.model.parameter_count() + head * (len(self.heads) if self.tag else 1)


def stream(method, model, held_in, held_out, c, seed, ystats):
    evals = held_in + held_out
    t0 = time.perf_counter()
    method.start(model, held_in, c, seed)
    start_seconds = time.perf_counter() - t0
    R = np.full((len(held_out) + 1, len(evals)), np.nan)
    costs, seconds, memory = [], [], []

    def score(row):
        for j, s in enumerate(evals):
            p = method.predict(s) * ystats["std"] + ystats["mean"]
            R[row, j] = r2(s.y, p, s.score)

    score(0)
    for i, s in enumerate(held_out, 1):
        if DEVICE == "cuda":
            torch.cuda.synchronize()
        t = time.perf_counter()
        costs.append(int(method.adapt(s)))
        if DEVICE == "cuda":
            torch.cuda.synchronize()
        seconds.append(time.perf_counter() - t)
        memory.append(int(method.memory()))
        score(i)
    return dict(R=R.tolist(), names=[s.name for s in evals], held_in=len(held_in),
                adapt_macs=costs, adapt_seconds=seconds, memory=memory,
                start_macs=int(method.pre_macs), start_seconds=start_seconds)


def summarize(out):
    R = np.array(out["R"])
    k, T = out["held_in"], R.shape[0] - 1
    final = R[T]
    held_in_final = final[:k].mean()
    held_out_final = final[k:].mean()
    acc = final.mean()
    diag = np.array([R[i + 1, k + i] for i in range(T)])
    bwt_out = np.mean([R[T, k + i] - R[i + 1, k + i] for i in range(T - 1)]) if T > 1 else 0.0
    bwt_in = (R[T, :k] - R[0, :k]).mean()
    bwt = (bwt_in * k + bwt_out * (T - 1)) / (k + T - 1)
    return dict(acc=float(acc), acc_held_in=float(held_in_final), acc_held_out=float(held_out_final),
                plasticity=float(diag.mean()), bwt=float(bwt), bwt_held_in=float(bwt_in), bwt_held_out=float(bwt_out),
                zero_shot_held_out=float(np.mean([R[0, k + i] for i in range(T)])),
                adapt_macs=float(np.mean(out["adapt_macs"])), adapt_seconds=float(np.mean(out["adapt_seconds"])),
                memory=float(out["memory"][-1]))


class PriorMethod:
    def __init__(self, kind, dimension=1024, scale=1.0, lam=1.0, source="penultimate", center=True, shared=False):
        self.kind, self.dimension, self.scale, self.lam, self.source = kind, dimension, scale, lam, source
        self.center, self.shared = center, shared

    def start(self, model, held_in, c, seed):
        self.model, self.cache, self.pre_macs, self.seed = model, {}, 0, seed
        last = model.readout[-1]
        self.W0 = torch.cat([last.weight.T, last.bias[None]], 0).detach().double()
        width = last.in_features if self.source == "penultimate" else model.features
        self.enc = FHRR(width, self.dimension, self.scale, seed).to(DEVICE) if self.kind != "linear" else None
        self.heads, self.keys, self.memory_W = {}, {}, None
        self.A, self.B, self.W_shared = None, None, None

    def run(self, s):
        if s.name not in self.cache:
            self.model.eval()
            h, gs, fs = None, [], []
            with torch.no_grad():
                for z in torch.as_tensor(s.x, device=DEVICE).split(2048):
                    f, h = self.model.encode(z[None], h)
                    gs.append(self.model.readout[:-1](f)[0])
                    fs.append(f[0])
            g, f = torch.cat(gs), torch.cat(fs)
            self.cache[s.name] = torch.cat([g, f], 1)
        return self.cache[s.name]

    def phi(self, gf):
        k = self.W0.shape[0] - 1
        g, f = gf[:, :k], gf[:, k:]
        one = torch.ones(len(g), 1, device=DEVICE)
        if self.kind == "linear":
            return torch.cat([g, one], 1)
        src = g if self.source == "penultimate" else f
        return torch.cat([g, one, self.enc(src) * self.dimension ** 0.5], 1)

    def prior(self, d):
        P = torch.zeros(d, self.W0.shape[1], dtype=torch.float64, device=DEVICE)
        if self.center:
            P[:self.W0.shape[0]] = self.W0
        return P

    def base(self, d):
        P = torch.zeros(d, self.W0.shape[1], dtype=torch.float64, device=DEVICE)
        P[:self.W0.shape[0]] = self.W0
        return P

    def adapt(self, s):
        g = self.run(s)
        n = s.fit_end
        m = torch.as_tensor(s.fit[:n], device=DEVICE)
        X = self.phi(g[:n][m]).double()
        Y = torch.as_tensor(s.yz[:n], device=DEVICE)[m].double()
        N, d, o = X.shape[0], X.shape[1], Y.shape[1]
        P = self.prior(d)
        lam = self.lam * max(N, 1) / 1000
        A = X.T @ X + lam * torch.eye(d, dtype=torch.float64, device=DEVICE)
        B = X.T @ Y + lam * P
        if self.shared:
            self.A = A if self.A is None else self.A + A
            self.B = B if self.B is None else self.B + B
            A, B = self.A, self.B
        W = torch.cholesky_solve(B, torch.linalg.cholesky(A))
        delta = W - self.base(d)
        if self.shared:
            self.W_shared = W
        elif self.kind == "bound":
            k = session_key(self.dimension, len(self.keys), self.seed).to(DEVICE)
            self.keys[s.name] = k
            self.heads[s.name] = delta[:self.W0.shape[0]]
            e = delta[self.W0.shape[0]:]
            bound = self.enc.bind(e.T.float(), -k).T.double()
            self.memory_W = bound if self.memory_W is None else self.memory_W + bound
        else:
            self.heads[s.name] = delta
        mac = backbone_macs(self.model)
        macs = n * (mac["encode"] + mac["head"]) + N * d * (d + 1) // 2 + N * d * o + d ** 3 // 6 + d * d * o
        if self.enc is not None:
            macs += N * self.enc.w.shape[0] * self.dimension
        return macs

    def predict(self, s):
        g = self.run(s)
        d = self.W0.shape[0] + (2 * self.dimension if self.enc is not None else 0)
        W = self.base(d)
        if self.shared and self.W_shared is not None:
            W = self.W_shared
        elif s.name in self.heads:
            if self.kind == "bound":
                k0 = self.W0.shape[0]
                W = W.clone()
                W[:k0] += self.heads[s.name]
                W[k0:] += self.enc.bind(self.memory_W.T.float(), self.keys[s.name]).T.double()
            else:
                W = W + self.heads[s.name]
        return torch.cat([(self.phi(z).double() @ W).float() for z in g.split(16384)]).cpu().numpy()

    def memory(self):
        base = self.model.parameter_count()
        if self.enc is not None:
            base += self.enc.w.numel() + self.enc.b.numel()
        extra = sum(h.numel() for h in self.heads.values())
        if self.A is not None:
            extra += self.A.numel() + self.B.numel()
        if self.memory_W is not None:
            extra += self.memory_W.numel()
        return base + extra
