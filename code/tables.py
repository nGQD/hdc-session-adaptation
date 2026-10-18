import re
import glob
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
TASKS = ["h1", "m1", "m2"]
MAIN = ["frozen", "finetune_full", "finetune_readout", "ewc", "replay", "full_tag", "readout_tag", "last_tag", "closed_linear",
        "closed_zero", "hdc", "hdc_zero", "hdc_shared", "hdc_bound"]
GRADIENT_METHODS = ["finetune_full", "finetune_readout", "ewc", "replay", "full_tag", "readout_tag", "last_tag"]
EPOCH_METHODS = ["finetune_full", "replay", "readout_tag"]
EPOCH_GRID = [1, 2, 5, 10, 20, 40]
EPOCH_SEEDS = (101, 113, 127, 131, 137)
FALCON = ["wf_zeroshot", "wf_oracle", "wf_continual", "rnn_zeroshot", "rnn_oracle", "rnn_finetune"]
LABEL = dict(hdc="HDC", closed_linear="Closed form", frozen="Frozen", finetune_full="FT-full",
             finetune_readout="FT-readout", ewc="EWC", replay="Replay", hdc_bound="HDC-bound",
             full_tag="FT-full-tag", readout_tag="FT-readout-tag", last_tag="FT-last-tag", closed_zero="Closed form (zero prior)",
             hdc_zero="HDC (zero prior)", hdc_shared="HDC-shared",
             wf_zeroshot="WF zero-shot", wf_oracle="WF per-session", wf_continual="WF sequential",
             rnn_zeroshot="RNN zero-shot", rnn_oracle="RNN per-session", rnn_finetune="RNN sequential")
VARIANTS = ["linear", "mlp", "standard_gru", "no_gru", "no_online_norm", "no_smoothing", "no_refinement"]


def stat(task, name):
    p = RES / "stats" / f"{task}_{name}.json"
    return json.loads(p.read_text()) if p.exists() else None


def have(task, name):
    return stat(task, name) or {}


def fmt(x, nd=3):
    return f"{x:.{nd}f}".replace("-", "−")


def sci(x):
    if x == 0:
        return "0"
    e = int(np.floor(np.log10(x)))
    return f"{x / 10 ** e:.1f}e{e}"


def falcon_rows(task):
    d = defaultdict(list)
    for f in glob.glob(str(RES / "stream" / task / "falcon" / "*.json")):
        j = json.loads(Path(f).read_text())
        d[j["method"]].append(j["summary"])
    return {m: {k: float(np.mean([s[k] for s in v])) for k in ("acc", "acc_held_out", "acc_held_in", "bwt", "adapt_macs")} | {"sd": float(np.std([s["acc"] for s in v], ddof=1)), "n": len(v)}
            for m, v in d.items()}


def agg_of(t):
    return have(t, "aggregate")


def main_table():
    lines = []
    for t in TASKS:
        agg = agg_of(t)
        if not agg:
            continue
        lines += [f"\n### {t.upper()}\n", "| Method | ACC (95% CI) | Held-in | Held-out | BWT | Adapt MACs |", "|---|---|---|---|---|---|"]
        for m in MAIN:
            if m not in agg:
                continue
            a = agg[m]
            lines.append(f"| {LABEL[m]} | {fmt(a['acc']['mean'])} ({fmt(a['acc']['ci_low'])}, {fmt(a['acc']['ci_high'])}) | "
                         f"{fmt(a['acc_held_in']['mean'])} | {fmt(a['acc_held_out']['mean'])} | {fmt(a['bwt']['mean'])} | {sci(a['adapt_macs']['mean'])} |")
        for m, a in sorted(falcon_rows(t).items(), key=lambda kv: FALCON.index(kv[0])):
            lines.append(f"| {LABEL[m]} | {fmt(a['acc'])} | {fmt(a['acc_held_in'])} | {fmt(a['acc_held_out'])} | {fmt(a['bwt'])} | {sci(a['adapt_macs'])} |")
    return "\n".join(lines)


def compact_main():
    agg = {t: agg_of(t) for t in TASKS}
    fal = {t: falcon_rows(t) for t in TASKS}
    head = "| Method | " + " | ".join(f"{t.upper()} {k}" for k in ("ACC", "held-out", "BWT") for t in TASKS) + " |"
    lines = [head, "|:--|" + ":-:|" * 9]
    keys = ("acc", "acc_held_out", "bwt")
    for m in MAIN:
        if not any(m in agg[t] for t in TASKS):
            continue
        lines.append(f"| {LABEL[m]} | " + " | ".join(fmt(agg[t][m][k]["mean"]) if m in agg[t] else "n/a" for k in keys for t in TASKS) + " |")
    for m in FALCON:
        if not any(m in fal[t] for t in TASKS):
            continue
        lines.append(f"| {LABEL[m]} | " + " | ".join(fmt(fal[t][m][k]) if m in fal[t] else "n/a" for k in keys for t in TASKS) + " |")
    return "\n".join(lines)


def macs_table():
    agg = {t: agg_of(t) for t in TASKS}
    lines = ["| Method | " + " | ".join(f"{t.upper()} MACs" for t in TASKS) + " | " + " | ".join(f"{t.upper()} plasticity" for t in TASKS) + " |", "|:--|" + ":-:|" * 6]
    for m in MAIN:
        if m == "frozen" or not any(m in agg[t] for t in TASKS):
            continue
        lines.append(f"| {LABEL[m]} | " + " | ".join(sci(agg[t][m]["adapt_macs"]["mean"]) if m in agg[t] else "n/a" for t in TASKS) + " | " +
                     " | ".join(fmt(agg[t][m]["plasticity"]["mean"]) if m in agg[t] else "n/a" for t in TASKS) + " |")
    return "\n".join(lines)


def tests_table():
    lines = ["| Task | HDC vs | ACC diff | Cliff $\\delta$ | p | BWT diff | p (BWT) |", "|:--|:--|:-:|:-:|:-:|:-:|:-:|"]
    for t in TASKS:
        rows = have(t, "paired").get("rows", [])
        acc = {r["other"]: r for r in rows if r["metric"] == "acc"}
        bwt = {r["other"]: r for r in rows if r["metric"] == "bwt"}
        for m in MAIN:
            if m in acc and m in bwt:
                a, b = acc[m], bwt[m]
                lines.append(f"| {t.upper()} | {LABEL[m]} | {fmt(a['mean_diff'])} | {fmt(a['cliff_delta'], 2)} | {a['p_holm']:.2g} | {fmt(b['mean_diff'])} | {b['p_holm']:.2g} |")
    return "\n".join(lines)


def budget_table():
    lines = ["| Task | Epochs | Frozen held-in | Frozen held-out | HDC held-out | FT-full ACC |", "|:--|:-:|:-:|:-:|:-:|:-:|"]
    for t in TASKS:
        for v, e in (("epochs25", 25), ("full", 50), ("epochs100", 100)):
            fs = {m: [json.loads(Path(f).read_text())["summary"] for f in glob.glob(str(RES / "stream" / t / v / f"{m}_s[0-9]*.json"))
                      if int(Path(f).stem.split("_s")[-1]) in EPOCH_SEEDS] for m in ("frozen", "hdc", "finetune_full")}
            if not all(fs.values()):
                continue
            g = lambda m, k: float(np.mean([x[k] for x in fs[m]]))
            lines.append(f"| {t.upper()} | {e} | {fmt(g('frozen', 'acc_held_in'))} | {fmt(g('frozen', 'acc_held_out'))} | {fmt(g('hdc', 'acc_held_out'))} | {fmt(g('finetune_full', 'acc'))} |")
    return "\n".join(lines)


def tost_table():
    lines = ["| Task | Reference | Diff | 90% CI | Non-inferior at 0.03 | p |", "|---|---|---|---|---|---|"]
    for t in TASKS:
        for k, v in have(t, "tost").items():
            lines.append(f"| {t.upper()} | {LABEL.get(k, k)} | {fmt(v['mean_diff'])} | ({fmt(v['ci90_low'])}, {fmt(v['ci90_high'])}) | {'yes' if v['noninferior'] else 'no'} | {v['p_noninferior']:.2g} |")
    return "\n".join(lines)


def mixed_table():
    lines = ["| Task | Method | Estimate vs HDC | 95% CI | p |", "|---|---|---|---|---|"]
    for t in TASKS:
        mx = have(t, "mixed")
        for k, v in mx.get("coefficients", {}).items():
            if k == "intercept":
                continue
            lines.append(f"| {t.upper()} | {LABEL.get(k, k)} | {fmt(v['estimate'])} | ({fmt(v['ci_low'])}, {fmt(v['ci_high'])}) | {v['p']:.2g} |")
    return "\n".join(lines)


def pareto_table():
    lines = ["| Task | Non-dominated (point estimate) | P(non-dominated) |", "|---|---|---|"]
    for t in TASKS:
        p = stat(t, "pareto")
        if not p:
            continue
        prob = p["prob_non_dominated"]
        lines.append(f"| {t.upper()} | {', '.join(LABEL[m] for m in p['non_dominated'])} | " + ", ".join(f"{LABEL[m]} {prob[m]:.2f}" for m in MAIN if m in prob) + " |")
    return "\n".join(lines)


def ablation_table():
    lines = ["| Task | Variant | Method | Delta held-in | Delta held-out (95% CI) | Delta BWT |", "|---|---|---|---|---|---|"]
    for t in TASKS:
        a = have(t, "ablation")
        for v in VARIANTS:
            for m in ("hdc", "finetune_full", "frozen"):
                r = a.get(v, {}).get(m)
                if r:
                    o = r["acc_held_out"]
                    lines.append(f"| {t.upper()} | {v} | {LABEL[m]} | {fmt(r['acc_held_in']['delta'])} | {fmt(o['delta'])} ({fmt(o['ci_low'])}, {fmt(o['ci_high'])}) | {fmt(r['bwt']['delta'])} |")
    return "\n".join(lines)


def dose_table():
    lines = ["| Task | Method | " + " | ".join(f"Held-out R2 at {p}" for p in (0.125, 0.25, 0.5)) + " | Adapt MACs at 0.5 |", "|---|---|---|---|---|---|"]
    for t in TASKS:
        for m in ("hdc", "closed_linear", "finetune_full", "replay"):
            fs = glob.glob(str(RES / "dose" / t / f"{m}_s[0-9]*.json"))
            if not fs:
                continue
            js = [json.loads(Path(f).read_text()) for f in fs]
            ho = np.mean([j["acc_held_out"] for j in js], 0)
            mac = np.mean([j["adapt_macs"][-1] for j in js])
            lines.append(f"| {t.upper()} | {LABEL[m]} | " + " | ".join(fmt(x) for x in ho) + f" | {sci(mac)} |")
    return "\n".join(lines)


def epoch_summaries(task):
    out = defaultdict(lambda: defaultdict(list))
    for f in sorted(glob.glob(str(RES / "epochs" / task / "*_e*_s*.json"))):
        j = json.loads(Path(f).read_text())
        if j.get("method") in EPOCH_METHODS and j.get("epochs") in EPOCH_GRID:
            out[j["method"]][j["epochs"]].append(j["summary"])
    return out


def epochs_table():
    lines = ["| Task | Method | Epochs | n | ACC | BWT | Adapt MACs | MACs / HDC | Smallest E with ACC >= HDC |", "|---|---|:-:|:-:|:-:|:-:|:-:|:-:|:-:|"]
    for t in TASKS:
        ep = epoch_summaries(t)
        agg = agg_of(t)
        hdc_acc = agg["hdc"]["acc"]["mean"] if "hdc" in agg else None
        hdc_mac = agg["hdc"]["adapt_macs"]["mean"] if "hdc" in agg else None
        for m in EPOCH_METHODS:
            if m not in ep:
                continue
            es = sorted(ep[m])
            means = {e: {k: float(np.mean([x[k] for x in ep[m][e]])) for k in ("acc", "bwt", "adapt_macs")} for e in es}
            reach = next((e for e in es if means[e]["acc"] >= hdc_acc), "never") if hdc_acc is not None else "n/a"
            for i, e in enumerate(es):
                r = means[e]
                ratio = f"{r['adapt_macs'] / hdc_mac:.1f}x" if hdc_mac else "n/a"
                lines.append(f"| {t.upper()} | {LABEL[m]} | {e} | {len(ep[m][e])} | {fmt(r['acc'])} | {fmt(r['bwt'])} | {sci(r['adapt_macs'])} | {ratio} | {reach if i == 0 else ''} |")
    return "\n".join(lines)


def on_boundary(value, grid):
    g = sorted(grid)
    return "yes" if value in (g[0], g[-1]) else "no"


SELECTION_CAPTION = ("Table: Hyperparameters selected on the inner stream, per task and method. lam is the ridge penalty per 1000 bins, dimension the code dimension, "
                     "scale the FHRR scale, lr the learning rate, epochs the number of adaptation epochs and ewc the EWC penalty weight. The last column lists "
                     "the axes at which the selected value equals an end of the base grid. {#tab:selection}")


def selection_table():
    lines = [SELECTION_CAPTION, "", "| Task | Method | Selected | On grid boundary |", "|---|---|---|---|"]
    for t in TASKS:
        p = RES / "selection" / f"{t}.json"
        if not p.exists():
            continue
        sel = json.loads(p.read_text())
        grids = sel.get("grids", {})
        rows = []
        hg = grids.get("hdc", [])
        if "hdc" in sel:
            sh = sel["hdc"]
            flags = [k for k in ("source", "dimension", "scale", "lam") if hg and sh.get(k) in (sorted({g[k] for g in hg})[0], sorted({g[k] for g in hg})[-1]) and len({g[k] for g in hg}) > 1]
            rows.append(("hdc", ", ".join(f"{k}={v}" for k, v in sh.items()), ", ".join(flags) if flags else "no"))
        for key, name, grid in (("linear", "closed_linear", [g["lam"] for g in grids.get("linear", [])]),
                                ("closed_zero", "closed_zero", [g["lam"] for g in grids.get("closed_zero", [])]),
                                ("hdc_zero", "hdc_zero", [0.1, 1, 10, 100, 1000]),
                                ("hdc_shared", "hdc_shared", grids.get("code_lams", []))):
            if key in sel:
                v = sel[key]["lam"]
                rows.append((name, f"lam={v}", ("lam" if on_boundary(v, grid) == "yes" else "no") if grid else "n/a"))
        for name, h in sel.get("gradient", {}).items():
            flags = []
            for k, gk in (("lr", "lrs"), ("epochs", "epochs"), ("ewc", "ewc_weights")):
                if k in h and grids.get(gk) and h[k] in (min(grids[gk]), max(grids[gk])):
                    flags.append(k)
            rows.append((name, ", ".join(f"{k}={v}" for k, v in h.items()), ", ".join(flags) if flags else "no"))
        order = {m: i for i, m in enumerate(MAIN)}
        for name, desc, flag in sorted(rows, key=lambda r: order.get(r[0], 99)):
            lines.append(f"| {t.upper()} | {LABEL.get(name, name)} | {desc} | {flag} |")
    return "\n".join(lines)


SELECTION_PAPER_CAPTION = ("Table: Hyperparameters selected on the inner stream, per task and method. Gradient methods: learning rate / epochs "
                           "(/ EWC penalty weight); closed forms, HDC (zero prior) and HDC-shared: ridge penalty $\\lambda$ per 1000 bins; "
                           "HDC: code dimension $D$ / FHRR scale $\\sigma$ / $\\lambda$. HDC uses the feature $f_t$ as encoder input on every task, "
                           "and HDC-bound uses the HDC selection. {#tab:selection}")


def selection_paper_table():
    """Paper variant of the selection table: one row per method and one column per task, without the grid-boundary column."""
    rows = [[x.strip() for x in l.strip().strip("|").split("|")] for l in selection_table().split("\n") if l.startswith("| ")][1:]
    cell, order = {}, []
    for t, m, desc, _ in rows:
        kv = dict(x.split("=") for x in desc.split(", "))
        e = lambda v: re.sub(r"e([+-])0*(\d)", lambda q: "e" + ("-" if q.group(1) == "-" else "") + q.group(2), f"{float(v):.0e}")
        if "lr" in kv:
            cell[(m, t)] = " / ".join([e(kv["lr"]), kv["epochs"]] + ([e(kv["ewc"])] if "ewc" in kv else []))
        elif "dimension" in kv:
            cell[(m, t)] = f"{kv['dimension']} / {float(kv['scale']):g} / {float(kv['lam']):g}"
        else:
            cell[(m, t)] = f"{float(kv['lam']):g}"
        if m not in order:
            order.append(m)
    tasks = sorted({t for t, *_ in rows}, key=["H1", "M1", "M2"].index)
    lines = [SELECTION_PAPER_CAPTION, "", "| Method | " + " | ".join(tasks) + " |", "|:--|" + ":--|" * len(tasks)]
    for m in order:
        lines.append(f"| {m} | " + " | ".join(cell.get((m, t), "") for t in tasks) + " |")
    return "\n".join(lines)


def ratios():
    lines = []
    for t in TASKS:
        agg = agg_of(t)
        if "hdc" not in agg:
            continue
        h = agg["hdc"]["adapt_macs"]["mean"]
        parts = [f"{LABEL[m]}/HDC = {agg[m]['adapt_macs']['mean'] / h:.1f}x" for m in GRADIENT_METHODS if m in agg]
        line = f"- {t.upper()}: " + ", ".join(parts)
        if "closed_linear" in agg:
            line += f"; HDC/Closed form = {h / agg['closed_linear']['adapt_macs']['mean']:.1f}x"
        lines.append(line)
    return "\n".join(lines)


if __name__ == "__main__":
    out = RES / "tables"
    out.mkdir(exist_ok=True)
    parts = dict(budget=budget_table(), compact=compact_main(), macs=macs_table(), main=main_table(), tests=tests_table(), tost=tost_table(), mixed=mixed_table(),
                 pareto=pareto_table(), ablation=ablation_table(), dose=dose_table(), epochs=epochs_table(), selection=selection_table(), selection_paper=selection_paper_table(), ratios=ratios())
    for k, v in parts.items():
        (out / f"{k}.md").write_text(v + "\n")
    (out / "digest.md").write_text("\n\n".join(f"## {k}\n{v}" for k, v in parts.items()) + "\n")
    print(out / "digest.md")
