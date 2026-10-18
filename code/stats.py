import argparse
import json
import re
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats as st

ROOT = Path(__file__).resolve().parent.parent
METHODS = ["frozen", "finetune_full", "finetune_readout", "ewc", "replay", "full_tag", "readout_tag", "last_tag", "closed_linear",
           "closed_zero", "hdc", "hdc_zero", "hdc_shared", "hdc_bound"]
METRICS = ["acc", "acc_held_out", "acc_held_in", "bwt", "plasticity", "adapt_macs", "memory"]
PARETO = {"acc": 1, "bwt": 1, "adapt_macs": -1}
ABLATED = ["acc_held_in", "acc_held_out", "bwt"]
REFS = ["hdc"]
TOST_REFS = ["finetune_full", "replay", "full_tag", "readout_tag"]
OPTIMIZERS = ["lbfgs", "powell", "nm"]


def order(names):
    return sorted(names, key=lambda m: (METHODS.index(m) if m in METHODS else len(METHODS), m))


class Cell:
    def __init__(self, runs):
        runs = sorted(runs, key=lambda r: r["seed"])
        self.seeds = [r["seed"] for r in runs]
        R = np.array([r["R"] for r in runs], float)
        self.k = int(runs[0]["held_in"])
        self.T = R.shape[1] - 1
        self.names = list(runs[0]["names"])
        k, T = self.k, self.T
        idx = np.arange(T)
        self.final = R[:, T, :]
        self.fin = R[:, T, :k]
        self.fout = R[:, T, k:]
        self.first = R[:, 0, :k]
        self.diag = R[:, idx + 1, k + idx]
        self.bwc = self.fout - self.diag
        self.valid = idx < T - 1
        self.macs = np.array([r["adapt_macs"] for r in runs], float)
        self.mem = np.array([r["memory"][-1] for r in runs], float)


def values(c, seeds, ins, outs):
    s = seeds[:, :, None]
    o = outs[:, None, :]
    fin = c.fin[s, ins[:, None, :]]
    fout = c.fout[s, o]
    first = c.first[s, ins[:, None, :]]
    valid = c.valid[outs][:, None, :]
    nv = valid.sum(-1)
    k = ins.shape[1]
    bwt_in = (fin - first).mean(-1)
    bwt_out = (c.bwc[s, o] * valid).sum(-1)
    return dict(acc=(fin.sum(-1) + fout.sum(-1)) / (k + outs.shape[1]), acc_held_out=fout.mean(-1), acc_held_in=fin.mean(-1),
                bwt=(bwt_in * k + bwt_out) / (k + nv), plasticity=c.diag[s, o].mean(-1),
                adapt_macs=c.macs[s, o].mean(-1), memory=c.mem[seeds])


def per_seed(c, pos=None):
    pos = np.arange(len(c.seeds)) if pos is None else np.asarray(pos)
    v = values(c, pos[None], np.arange(c.k)[None], np.arange(c.T)[None])
    return {m: a[0] for m, a in v.items()}


class Draw:
    def __init__(self, k, T, smax, B, seed):
        rng = np.random.default_rng(seed)
        self.ins = rng.integers(0, k, (B, k))
        self.outs = rng.integers(0, T, (B, T))
        self.u = rng.random((B, smax))

    def seeds(self, n):
        return np.minimum((self.u[:, :n] * n).astype(int), n - 1)


def boot(c, d):
    v = values(c, d.seeds(len(c.seeds)), d.ins, d.outs)
    return {m: a.mean(1) for m, a in v.items()}


def interval(b):
    lo, hi = np.percentile(b, [2.5, 97.5])
    return float(lo), float(hi)


def sd(a):
    return float(np.std(a, ddof=1)) if len(a) > 1 else float("nan")


def holm(p):
    p = np.asarray(p, float)
    adj = np.empty(len(p))
    run = 0.0
    for r, i in enumerate(np.argsort(p)):
        run = max(run, (len(p) - r) * p[i])
        adj[i] = min(run, 1.0)
    return adj


def wilcoxon(d):
    d = np.asarray(d, float)
    d = d[np.isfinite(d)]
    if len(d) == 0 or np.all(d == 0):
        return 0.0, 1.0
    r = st.wilcoxon(d, zero_method="wilcox")
    return float(r.statistic), float(r.pvalue)


def dz(d):
    s = sd(d)
    return float(np.mean(d) / s) if s > 0 else (0.0 if np.mean(d) == 0 else float("nan"))


def cliff(x, y):
    ys = np.sort(y)
    lo = np.searchsorted(ys, x, "left")
    hi = np.searchsorted(ys, x, "right")
    return float((lo - (len(ys) - hi)).sum() / (len(x) * len(y)))


def common(a, b):
    s = sorted(set(a.seeds) & set(b.seeds))
    return s, [a.seeds.index(i) for i in s], [b.seeds.index(i) for i in s]


def aggregate(cells, d):
    out, boots = {}, {}
    for m in order(cells):
        c = cells[m]
        p, b = per_seed(c), boot(c, d)
        boots[m] = b
        row = {}
        for k in METRICS:
            lo, hi = interval(b[k])
            row[k] = dict(mean=float(p[k].mean()), sd=sd(p[k]), ci_low=lo, ci_high=hi, n_seeds=len(c.seeds))
        row["per_session_final_r2"] = dict(
            sessions=c.names, held_in=c.k, mean=c.final.mean(0), sd=[sd(c.final[:, j]) for j in range(c.final.shape[1])])
        out[m] = row
    return out, boots


def effect(x, y, extra=None):
    x, y = np.asarray(x, float), np.asarray(y, float)
    diff = x - y
    stat, p = wilcoxon(diff)
    row = dict(n=len(diff), mean_diff=float(np.nanmean(diff)), wilcoxon_stat=stat, p=p, cohen_dz=dz(diff[np.isfinite(diff)]),
               cliff_delta=cliff(x[np.isfinite(x)], y[np.isfinite(y)]))
    if extra is not None:
        row["p_session_mean"] = wilcoxon(extra)[1]
    return row


def paired(cells):
    rows = []
    for ref in REFS:
        if ref not in cells:
            continue
        for m in order(cells):
            if m == ref:
                continue
            s, i, j = common(cells[ref], cells[m])
            if not s:
                continue
            pr, pm = per_seed(cells[ref], i), per_seed(cells[m], j)
            cell = effect(cells[ref].final[i].ravel(), cells[m].final[j].ravel())
            rows.append(dict(ref=ref, other=m, metric="acc", **effect(pr["acc"], pm["acc"]),
                             p_cells=cell["p"], n_cells=cell["n"], cliff_delta_cells=cell["cliff_delta"]))
            rows.append(dict(ref=ref, other=m, metric="bwt", **effect(pr["bwt"], pm["bwt"])))
    for metric in ("acc", "bwt"):
        sel = [r for r in rows if r["metric"] == metric]
        for r, a in zip(sel, holm([r["p"] for r in sel]) if sel else []):
            r["p_holm"] = float(a)
    return dict(family="seed-level paired Wilcoxon (n seeds) on ACC and BWT, Holm within task per metric; p_cells is the session-by-seed test, reported for reference", rows=rows)


def mixed(cells):
    rows = []
    for m in order(cells):
        c = cells[m]
        for si, seed in enumerate(c.seeds):
            for j, name in enumerate(c.names):
                rows.append((float(c.final[si, j]), m, name, str(seed)))
    df = pd.DataFrame(rows, columns=["final_r2", "method", "session", "seed"])
    df["g"] = 1
    ref = "hdc" if "hdc" in cells else order(cells)[0]
    f = f"final_r2 ~ C(method, Treatment(reference='{ref}'))"
    df = df[np.isfinite(df["final_r2"])]
    fit, used, err = None, None, None
    for opt in OPTIMIZERS:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                cand = smf.mixedlm(f, df, groups="g", vc_formula={"session": "0 + C(session)", "seed": "0 + C(seed)"}).fit(reml=True, method=opt)
        except Exception as e:
            err = str(e)
            continue
        if fit is None:
            fit, used = cand, opt
        if cand.converged:
            fit, used = cand, opt
            break
    if fit is None:
        return dict(formula="final_r2 ~ method + (1 | session) + (1 | seed)", reference=ref, n_obs=len(df), converged=False, optimizer=None, error=err)
    ci = fit.conf_int()
    coef = {}
    for name in fit.fe_params.index:
        mt = re.search(r"\[T\.(.*)\]", name)
        key = mt.group(1) if mt else "intercept"
        coef[key] = dict(estimate=float(fit.fe_params[name]), se=float(fit.bse_fe[name]), z=float(fit.tvalues[name]),
                         p=float(fit.pvalues[name]), ci_low=float(ci.loc[name, 0]), ci_high=float(ci.loc[name, 1]))
    return dict(formula="final_r2 ~ method + (1 | session) + (1 | seed)", reference=ref, n_obs=len(df), converged=bool(fit.converged), optimizer=used,
                coefficients=coef, variance_components=dict(zip(fit.model.exog_vc.names, map(float, fit.vcomp))),
                residual_variance=float(fit.scale))


def tost(cells, margin=0.03, alpha=0.05):
    out = {}
    if "hdc" not in cells:
        return out
    for other in TOST_REFS:
        if other not in cells:
            continue
        s, i, j = common(cells["hdc"], cells[other])
        d = per_seed(cells["hdc"], i)["acc"] - per_seed(cells[other], j)["acc"]
        n, mu = len(d), float(d.mean())
        se = float(d.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0
        if se > 0:
            p_lo = float(1 - st.t.cdf((mu + margin) / se, n - 1))
            p_hi = float(st.t.cdf((mu - margin) / se, n - 1))
            tc = float(st.t.ppf(1 - alpha, n - 1))
        else:
            p_lo = 0.0 if mu > -margin else 1.0
            p_hi = 0.0 if mu < margin else 1.0
            tc = 0.0
        out[other] = dict(n=n, margin=margin, mean_diff=mu, se=se, ci90_low=mu - tc * se, ci90_high=mu + tc * se,
                          p_noninferior=p_lo, p_upper=p_hi, p_equivalence=max(p_lo, p_hi),
                          noninferior=bool(p_lo < alpha), equivalent=bool(max(p_lo, p_hi) < alpha),
                          smallest_margin_noninferior=max(0.0, -(mu - tc * se)))
    return out


def dominated(X):
    a = X[..., :, None, :]
    b = X[..., None, :, :]
    return ((a >= b).all(-1) & (a > b).any(-1)).any(-2)


def pareto(cells, boots, d):
    ms = order(cells)
    sign = np.array([PARETO[k] for k in PARETO])
    point = np.array([[per_seed(cells[m])[k].mean() for k in PARETO] for m in ms]) * sign
    B = np.stack([np.stack([boots[m][k] for k in PARETO], -1) for m in ms], 1) * sign
    prob = (~dominated(B)).mean(0)
    nd = ~dominated(point)
    return dict(objectives={k: "max" if v > 0 else "min" for k, v in PARETO.items()},
                non_dominated=[m for m, f in zip(ms, nd) if f], dominated=[m for m, f in zip(ms, nd) if not f],
                prob_non_dominated={m: float(p) for m, p in zip(ms, prob)},
                point={m: dict(zip(PARETO, (r * sign).tolist())) for m, r in zip(ms, point)})


def ablation(data, d):
    full = data.get("full", {})
    out = {}
    for var in sorted(v for v in data if v != "full"):
        for m in order(set(full) & set(data[var])):
            a, b = full[m], data[var][m]
            if a.names != b.names or a.k != b.k:
                continue
            s, i, j = common(a, b)
            if not s:
                continue
            pa, pb = per_seed(a, i), per_seed(b, j)
            sidx = d.seeds(len(s))
            va = values(a, np.array(i)[sidx], d.ins, d.outs)
            vb = values(b, np.array(j)[sidx], d.ins, d.outs)
            row = dict(n_seeds=len(s), min_possible_wilcoxon_p=2.0 ** (1 - len(s)))
            for k in ABLATED:
                delta = pb[k] - pa[k]
                lo, hi = interval((vb[k] - va[k]).mean(1))
                stat, p = wilcoxon(delta)
                row[k] = dict(delta=float(delta.mean()), sd=sd(delta), ci_low=lo, ci_high=hi, wilcoxon_stat=stat, p=p)
            out.setdefault(var, {})[m] = row
    for k in ABLATED:
        for m in {m for v in out.values() for m in v}:
            vs = [v for v in out if m in out[v]]
            for v, a in zip(vs, holm([out[v][m][k]["p"] for v in vs])):
                out[v][m][k]["p_holm"] = float(a)
    return out


def clean(o):
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, np.ndarray):
        return clean(o.tolist())
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, (float, np.floating)):
        return float(o) if np.isfinite(o) else None
    return o


def load(root):
    runs = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for f in sorted(Path(root, "stream").glob("*/*/*_s*.json")):
        mt = re.fullmatch(r"(.+)_s(\d+)", f.stem)
        if mt:
            runs[f.parent.parent.name][f.parent.name][mt.group(1)].append(json.loads(f.read_text()))
    return {t: {v: {m: Cell(r) for m, r in ms.items()} for v, ms in vs.items()} for t, vs in runs.items()}


def analyze(task_data, B, seed):
    full = task_data["full"]
    first = next(iter(full.values()))
    smax = max(len(c.seeds) for v in task_data.values() for c in v.values())
    d = Draw(first.k, first.T, smax, B, seed)
    agg, boots = aggregate(full, d)
    return dict(aggregate=agg, paired=paired(full), mixed=mixed(full), tost=tost(full),
                pareto=pareto(full, boots, d), ablation=ablation(task_data, d))


def run(root, out, B=10000, seed=0, tasks=None):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    data = load(root)
    summary = dict(meta=dict(bootstrap_resamples=B, seed=seed, root=str(Path(root).name)), tasks={})
    for task in sorted(data):
        if (tasks and task not in tasks) or "full" not in data[task]:
            continue
        res = analyze(data[task], B, seed)
        for name, v in res.items():
            (out / f"{task}_{name}.json").write_text(json.dumps(clean(v), indent=1))
        summary["tasks"][task] = res
    (out / "summary.json").write_text(json.dumps(clean(summary), indent=1))
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(ROOT / "results"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tasks", nargs="*")
    a = ap.parse_args()
    run(a.root, a.out or Path(a.root) / "stats", a.boot, a.seed, a.tasks)
