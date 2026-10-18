import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import json
import math
from pathlib import Path

import numpy as np
import torch
from torch.utils.flop_counter import FlopCounterMode

from engine import backbone_macs
from model import Backbone, Config, FHRR, session_key

OUT = Path(__file__).resolve().parents[1] / "results" / "theory" / "bound.json"
torch.set_num_threads(2)


def rounded(v):
    if isinstance(v, (float, np.floating)):
        return float(f"{float(v):.8g}")
    if isinstance(v, dict):
        return {k: rounded(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [rounded(x) for x in v]
    return v


def slope(x, y):
    return float(np.polyfit(np.log(x), np.log(y), 1)[0])


def phase_rotate(h, k):
    d = h.shape[-1] // 2
    c, s = h[..., :d], h[..., d:]
    return np.concatenate([c * np.cos(k) - s * np.sin(k), c * np.sin(k) + s * np.cos(k)], -1)


def check_power_encoding(n=16, d=64, scale=2.0, points=5):
    enc = FHRR(n, d, scale, 3)
    w, b = enc.w.double().numpy(), enc.b.double().numpy()
    rng = np.random.default_rng(0)
    err = 0.0
    for _ in range(points):
        s = rng.standard_normal(n)
        z = np.prod(np.exp(1j * w * s[:, None]), axis=0) * np.exp(1j * b)
        direct = np.exp(1j * (s @ w + b))
        code = enc(torch.as_tensor(s[None], dtype=torch.float32)).double().numpy()[0] * math.sqrt(d)
        err = max(err, float(np.abs(z - direct).max()), float(np.abs(code[:d] - direct.real).max()), float(np.abs(code[d:] - direct.imag).max()))
    k = torch.rand(d).double()
    h = torch.randn(3, 2 * d).double()
    ref = (h[:, :d] + 1j * h[:, d:]) * torch.exp(1j * k)
    out = enc.bind(h, k)
    bind_err = float(max((out[:, :d] - ref.real).abs().max(), (out[:, d:] - ref.imag).abs().max()))
    return dict(max_abs_err_power_vs_phase=err, max_abs_err_bind_vs_complex_product=bind_err)


def check_kernel(n=304, points=120, seeds=12, spread=0.5):
    rng = np.random.default_rng(0)
    f = rng.standard_normal((points, n)) * spread
    sq = ((f[:, None] - f[None]) ** 2).sum(-1)
    off = ~np.eye(points, dtype=bool)
    out = []
    for scale in [1.0, 2.0, 4.0]:
        k = np.exp(-scale ** 2 * sq / (2 * n))
        var_pred = ((1 - k ** 2) ** 2 / 2)[off].mean()
        rows = []
        for d in [64, 128, 256, 512, 1024, 2048, 4096]:
            err = []
            for seed in range(seeds):
                enc = FHRR(n, d, scale, seed)
                phi = enc(torch.as_tensor(f, dtype=torch.float32)).double()
                err.append(((phi @ phi.T).numpy() - k)[off])
            err = np.stack(err)
            rows.append(dict(D=d, rms=float(np.sqrt((err ** 2).mean())), rms_pred=float(np.sqrt(var_pred / d)),
                             mean_bias=float(err.mean()), ratio=float(np.sqrt((err ** 2).mean() / (var_pred / d))),
                             max_abs=float(np.abs(err).max())))
        out.append(dict(sigma=scale, mean_k=float(k[off].mean()), rows=rows, slope_rms=slope([r["D"] for r in rows], [r["rms"] for r in rows])))
    return out


def check_dual(n=64, d_lin=129, nobs=150, o=3, lam=3.0, scale=1.0):
    rng = np.random.default_rng(5)
    g = rng.standard_normal((nobs, d_lin - 1))
    x = np.concatenate([g, np.ones((nobs, 1))], 1)
    src = rng.standard_normal((nobs, n)) * 0.5
    y = rng.standard_normal((nobs, o))
    prior = np.zeros((d_lin, o))
    prior[:] = rng.standard_normal((d_lin, o)) * 0.1
    q = rng.standard_normal((20, n)) * 0.5
    gq = np.concatenate([rng.standard_normal((20, d_lin - 1)), np.ones((20, 1))], 1)
    sq_tr = ((src[:, None] - src[None]) ** 2).sum(-1)
    sq_q = ((q[:, None] - src[None]) ** 2).sum(-1)
    rows = []
    for d in [256, 1024, 4096, 16384]:
        enc = FHRR(n, d, scale, 1)
        e = enc(torch.as_tensor(src, dtype=torch.float32)).double().numpy() * math.sqrt(d)
        eq = enc(torch.as_tensor(q, dtype=torch.float32)).double().numpy() * math.sqrt(d)
        phi, phiq = np.concatenate([x, e], 1), np.concatenate([gq, eq], 1)
        p = np.concatenate([prior, np.zeros((2 * d, o))], 0)
        kt = x @ x.T + e @ e.T
        kq = gq @ x.T + eq @ e.T
        dual = phiq @ p + kq @ np.linalg.solve(kt + lam * np.eye(nobs), y - phi @ p)
        gap = None
        if d <= 1024:
            w = np.linalg.solve(phi.T @ phi + lam * np.eye(phi.shape[1]), phi.T @ y + lam * p)
            gap = float(np.abs(phiq @ w - dual).max())
        k_tr = np.exp(-scale ** 2 * sq_tr / (2 * n))
        k_q = np.exp(-scale ** 2 * sq_q / (2 * n))
        lim = phiq @ p + (gq @ x.T + d * k_q) @ np.linalg.solve(x @ x.T + d * k_tr + lam * np.eye(nobs), y - phi @ p)
        rows.append(dict(D=d, primal_vs_dual=gap, finite_vs_gaussian_kernel=float(np.abs(dual - lim).max() / np.abs(lim).max())))
    return rows


def check_map_rls(nobs=300, d=80, o=3, lam=12.0):
    rng = np.random.default_rng(1)
    x = rng.standard_normal((nobs, d)) * np.exp(rng.standard_normal(d) * 0.5)
    prior = rng.standard_normal((d, o))
    y = x @ (prior + 0.2 * rng.standard_normal((d, o))) + 0.3 * rng.standard_normal((nobs, o))
    w_map = np.linalg.solve(x.T @ x + lam * np.eye(d), x.T @ y + lam * prior)
    w, p = prior.copy(), np.eye(d) / lam
    for i in range(nobs):
        u = x[i]
        pu = p @ u
        kgain = pu / (1.0 + u @ pu)
        w = w + np.outer(kgain, y[i] - u @ w)
        p = p - np.outer(kgain, pu)
    cho = np.linalg.cholesky(x.T @ x + lam * np.eye(d))
    z = np.linalg.solve(cho, x.T @ y + lam * prior)
    w_chol = np.linalg.solve(cho.T, z)
    return dict(N=nobs, d=d, max_abs_map_vs_rls=float(np.abs(w_map - w).max()),
                max_abs_map_vs_cholesky=float(np.abs(w_map - w_chol).max()), weight_scale=float(np.abs(w_map).max()))


def check_bias_variance(nobs=200, d=40, o=2, noise=0.5, draws=6000):
    rng = np.random.default_rng(2)
    spec = np.exp(-np.linspace(0, 4, d))
    basis = np.linalg.qr(rng.standard_normal((d, d)))[0]
    x = (rng.standard_normal((nobs, d)) * np.sqrt(spec * nobs / spec.sum() * 4)) @ basis.T
    a = x.T @ x
    lam = 0.05 * nobs
    inv = np.linalg.inv(a + lam * np.eye(d))
    w_true = rng.standard_normal((d, o))
    shift_dir = rng.standard_normal((d, o))
    rows = []
    for rho in [0.1, 0.3, 0.6, 1.0, 1.5]:
        shift = shift_dir * rho * np.linalg.norm(w_true) / np.linalg.norm(shift_dir)
        prior = w_true - shift
        out = {}
        for name, center in [("prior", prior), ("zero", np.zeros_like(prior))]:
            diff = w_true - center
            bias = lam ** 2 * np.linalg.norm(x @ inv @ diff) ** 2 / nobs
            var = noise ** 2 * o * np.trace(a @ inv @ a @ inv) / nobs
            eps = noise * rng.standard_normal((draws, nobs, o))
            tot = 0.0
            for i in range(0, draws, 1000):
                e = eps[i:i + 1000]
                rhs = np.einsum("dn,bno->bdo", x.T, x @ w_true + e) + lam * center
                west = np.einsum("de,beo->bdo", inv, rhs)
                tot += (np.linalg.norm(np.einsum("nd,bdo->bno", x, west - w_true), axis=(1, 2)) ** 2).sum() / nobs
            out[name] = dict(bias=float(bias), variance=float(var), formula=float(bias + var), monte_carlo=float(tot / draws))
        cnorm = lambda m: lam ** 2 * np.linalg.norm(x @ inv @ m) ** 2 / nobs
        rows.append(dict(rho=rho, ratio_c_norm=float(cnorm(shift) / cnorm(w_true)), prior=out["prior"], zero=out["zero"],
                         prior_better=bool(out["prior"]["monte_carlo"] < out["zero"]["monte_carlo"])))
    return rows


def r2w(y, p):
    return float(1 - ((y - p) ** 2).sum() / ((y - y.mean(0)) ** 2).sum())


def make_stream(rng, s_in, s_out, n_feat, o, nobs, d_hdc, scale, enc_seed=0):
    k0 = n_feat + 1
    w_base = rng.standard_normal((k0, o)) * 0.5
    enc = FHRR(n_feat, d_hdc, scale, enc_seed)
    d = k0 + 2 * d_hdc
    prior = np.zeros((d, o))
    prior[:k0] = w_base
    names = [f"in{i}" for i in range(s_in)] + [f"out{i}" for i in range(s_out)]
    data = {}
    for i, name in enumerate(names):
        shift = 0 if name.startswith("in") else 0.4
        g = rng.standard_normal((2 * nobs, n_feat)) * np.exp(0.2 * rng.standard_normal(n_feat))
        x1 = np.concatenate([g, np.ones((2 * nobs, 1))], 1)
        e = enc(torch.as_tensor(g, dtype=torch.float32)).double().numpy() * math.sqrt(d_hdc)
        w_s = w_base + shift * rng.standard_normal((k0, o)) / math.sqrt(k0)
        nl = np.tanh(g @ rng.standard_normal((n_feat, o)) / math.sqrt(n_feat)) * (0.5 if shift else 0.0)
        y = x1 @ w_s + nl + 0.3 * rng.standard_normal((2 * nobs, o))
        data[name] = (x1, e, y)
    return names, data, prior, enc, k0, d


def run_stream(variant, s_in=2, s_out=6, n_feat=32, o=3, nobs=300, d_hdc=512, scale=1.0, lam=1.0, seed=0):
    rng = np.random.default_rng(seed)
    names, data, prior, enc, k0, d = make_stream(rng, s_in, s_out, n_feat, o, nobs, d_hdc, scale)
    out_names = names[s_in:]
    corrections, keys, memory = {}, {}, np.zeros((d_hdc * 2, o))
    eps = {}

    def predict(name):
        x1, e, y = data[name]
        w = prior.copy()
        if name in corrections:
            c = corrections[name]
            w[:k0] += c[:k0]
            if variant == "bound":
                w[k0:] += phase_rotate(memory.T, keys[name]).T
            else:
                w[k0:] += c[k0:]
        phi = np.concatenate([x1[nobs:], e[nobs:]], 1)
        return phi @ w, y[nobs:]

    rows = []
    rows.append([r2w(y_, p_) for p_, y_ in (predict(n) for n in names)])
    for j, name in enumerate(out_names):
        x1, e, y = data[name]
        phi = np.concatenate([x1[:nobs], e[:nobs]], 1)
        lam_eff = lam * nobs / 1000
        w = np.linalg.solve(phi.T @ phi + lam_eff * np.eye(d), phi.T @ y[:nobs] + lam_eff * prior)
        delta = w - prior
        corrections[name] = delta
        if variant == "bound":
            keys[name] = np.random.default_rng(1000 * seed + j).uniform(0, 2 * math.pi, d_hdc)
            memory = memory + phase_rotate(delta[k0:].T, -keys[name]).T
            eps[name] = float((delta[k0:] ** 2).sum())
        pred = [predict(n) for n in names]
        rows.append([r2w(y_, p_) for p_, y_ in pred])
    r = np.array(rows)
    t = len(out_names)
    cols = {n: i for i, n in enumerate(names)}
    bwt_terms = [r[t, cols[n]] - r[j + 1, cols[n]] for j, n in enumerate(out_names[:-1])]
    bwt_terms += [r[t, i] - r[0, i] for i in range(s_in)]
    pred_drop = []
    if variant == "bound":
        for j, n in enumerate(out_names[:-1]):
            qn = (data[n][1][nobs:] ** 2).sum(1).mean() / d_hdc
            tail = sum(eps[m] for m in out_names[j + 1:])
            vy = ((data[n][2][nobs:] - data[n][2][nobs:].mean(0)) ** 2).sum(1).mean()
            pred_drop.append(qn * tail / (2 * vy))
    return r, bwt_terms, pred_drop, s_in, t


def check_bwt_zero(seeds=4):
    out = {}
    for variant in ["session", "bound"]:
        diffs_out, diffs_in, bwts = [], [], []
        drops, preds = [], []
        for seed in range(seeds):
            r, terms, pdrop, s_in, t = run_stream(variant, seed=seed)
            for j in range(t - 1):
                diffs_out.append(float(np.abs(r[t, s_in + j] - r[j + 1, s_in + j])))
            for i in range(s_in):
                diffs_in.append(float(np.abs(r[t, i] - r[0, i])))
            bwts.append(float(np.mean(terms)))
            if variant == "bound":
                drops.extend([float(r[j + 1, s_in + j] - r[t, s_in + j]) for j in range(t - 1)])
                preds.extend([float(x) for x in pdrop])
        res = dict(max_abs_R_change_old_heldout=max(diffs_out), max_abs_R_change_heldin=max(diffs_in), mean_BWT=float(np.mean(bwts)))
        if variant == "bound":
            res.update(measured_mean_r2_drop=float(np.mean(drops)), predicted_mean_r2_drop=float(np.mean(preds)),
                       measured_over_predicted=float(np.mean(drops) / np.mean(preds)), bwt_ge_negative_prediction=bool(np.mean(drops) <= 1.5 * np.mean(preds)))
        out[variant] = res
    return out


def check_bound(d_list=(256, 512, 1024, 2048), s_list=(2, 4, 8, 16, 32, 64), trials=1500, chunk=50):
    rng = np.random.default_rng(7)
    empirical, predicted = [], []
    for d in d_list:
        row, prow = [], []
        for s in s_list:
            sq = 0.0
            for _ in range(trials // chunk):
                q = np.exp(1j * rng.uniform(0, 2 * math.pi, (chunk, d))) / math.sqrt(d)
                delta = rng.standard_normal((chunk, s - 1, d)) + 1j * rng.standard_normal((chunk, s - 1, d))
                delta /= np.linalg.norm(delta, axis=2, keepdims=True)
                ks = rng.uniform(0, 2 * math.pi, (chunk, 1, d))
                kt = rng.uniform(0, 2 * math.pi, (chunk, s - 1, d))
                retrieved = (np.exp(1j * (ks - kt)) * delta).sum(1)
                xi = (q * np.conj(retrieved)).sum(1).real
                sq += (xi ** 2).sum()
            row.append(math.sqrt(sq / ((trials // chunk) * chunk)))
            prow.append(math.sqrt((s - 1) / (2 * d)))
        empirical.append(row)
        predicted.append(prow)
    ratio = [[e / p for e, p in zip(er, pr)] for er, pr in zip(empirical, predicted)]
    flat_e, flat_p = [x for r in empirical for x in r], [x for r in predicted for x in r]
    return dict(D=list(d_list), S=list(s_list), empirical=empirical, predicted=predicted, ratio=ratio,
                slope_vs_predicted=slope(flat_p, flat_e), mean_ratio=float(np.mean([x for r in ratio for x in r])))


def check_backward_ratio():
    c = Config()
    model = Backbone(c)
    x = torch.randn(64, model.features, requires_grad=True)
    with FlopCounterMode(display=False) as fwd:
        y = model.readout(x)
    f_fwd = fwd.get_total_flops()
    with FlopCounterMode(display=False) as full:
        y = model.readout(x)
        y.sum().backward()
    return dict(forward_flops=f_fwd, backward_flops=full.get_total_flops() - f_fwd, ratio=(full.get_total_flops() - f_fwd) / f_fwd)


def cost_numbers(d_list=(1024, 2048), epochs=20, bins=2392):
    c = Config()
    model = Backbone(c)
    m = backbone_macs(model)
    o = c.outputs
    f_enc, f_head = m["encode"], m["head"]
    ends = list(range(0, bins - c.window + 1, c.stride))
    if ends[-1] != bins - c.window:
        ends.append(bins - c.window)
    n_proc = len(ends) * c.window
    rows = []
    for src, ns in [("penultimate", 128), ("feature", model.features)]:
        for dh in d_list:
            d = 129 + 2 * dh
            c_cf = bins * (f_enc + f_head + ns * dh) + bins * d * (d + 1) // 2 + bins * d * o + d ** 3 // 6 + d * d * o
            c_ft = 3 * epochs * n_proc * (f_enc + f_head)
            rows.append(dict(source=src, n_s=ns, D=dh, d=d, C_cf=c_cf, C_ft=c_ft, ratio_ft_over_cf=c_ft / c_cf))
    return dict(F_enc=f_enc, F_head=f_head, N_s=bins, epochs=epochs, windows=len(ends), N_proc=n_proc, rows=rows)


def main():
    result = dict(power_encoding=check_power_encoding(), kernel=check_kernel(), dual=check_dual(), map_rls=check_map_rls(),
                  bias_variance=check_bias_variance(), bwt=check_bwt_zero(), backward=check_backward_ratio(), cost=cost_numbers())
    result.update(check_bound())
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rounded(result), indent=1, allow_nan=False) + "\n")
    print("power:", result["power_encoding"])
    for k in result["kernel"]:
        print("kernel sigma", k["sigma"], "slope", round(k["slope_rms"], 3), "ratios", [round(r["ratio"], 3) for r in k["rows"]], "bias", max(abs(r["mean_bias"]) for r in k["rows"]))
    print("dual:", result["dual"])
    print("map vs rls:", result["map_rls"])
    for r in result["bias_variance"]:
        print("rho", r["rho"], "cratio", round(r["ratio_c_norm"], 3), "prior", round(r["prior"]["formula"], 4), round(r["prior"]["monte_carlo"], 4),
              "zero", round(r["zero"]["formula"], 4), round(r["zero"]["monte_carlo"], 4))
    print("bwt:", result["bwt"])
    print("backward:", result["backward"])
    print("bound ratio:", [[round(x, 3) for x in r] for r in result["ratio"]], result["slope_vs_predicted"])
    print("cost:", result["cost"]["rows"])


if __name__ == "__main__":
    main()
