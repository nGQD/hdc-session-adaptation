import argparse
import glob
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.ticker import LogFormatterSciNotation, LogLocator, MaxNLocator
from scipy import stats as st

ROOT = Path(__file__).resolve().parent.parent
CM = 1 / 2.54
FULL, HALF = 15.5 * CM, 7.5 * CM
TASKS = ["h1", "m1", "m2"]
TASK_LABEL = dict(h1="H1", m1="M1", m2="M2")
TASK_COLOR = dict(h1="#1F4E79", m1="#E3A21A", m2="#4C9F70")
METHODS = ["frozen", "finetune_full", "finetune_readout", "ewc", "replay", "full_tag", "readout_tag", "last_tag", "closed_linear",
           "closed_zero", "hdc", "hdc_zero", "hdc_shared", "hdc_bound"]
SHOWN = ["frozen", "finetune_full", "ewc", "replay", "full_tag", "readout_tag", "closed_linear", "hdc", "hdc_shared"]
LABEL = dict(frozen="Frozen", finetune_full="FT-full", finetune_readout="FT-readout", ewc="EWC", replay="Replay",
             full_tag="FT-full-tag", readout_tag="FT-readout-tag", last_tag="FT-last-tag", closed_linear="Closed form",
             closed_zero="Closed form (zero prior)", hdc="HDC", hdc_zero="HDC (zero prior)", hdc_shared="HDC-shared",
             hdc_bound="HDC-bound")
COLOR = dict(frozen="#8A8B8C", finetune_full="#1F4E79", finetune_readout="#6FA3D0", ewc="#E69F00", replay="#4B4C4D",
             full_tag="#5E86AE", readout_tag="#8DB6DA", last_tag="#B3CDE5", closed_linear="#4C9F70", closed_zero="#8CC5A5",
             hdc="#A31F34", hdc_zero="#5E1424", hdc_shared="#6A4C93", hdc_bound="#D97A87")
MARKER = dict(frozen="x", finetune_full="s", finetune_readout="D", ewc="^", replay="v", full_tag="h", readout_tag="d",
              last_tag="p", closed_linear="P", closed_zero="X", hdc="o", hdc_zero="8", hdc_shared="<", hdc_bound="*")
LINESTYLE = dict(full_tag="--", readout_tag="--", last_tag="--", closed_zero="--", hdc_zero="--")
HOLLOW = {"frozen", "finetune_readout", "full_tag", "readout_tag", "last_tag", "closed_zero", "hdc_zero"}
EPOCH_METHODS = ["finetune_full", "replay", "readout_tag"]
VARIANTS = ["linear", "mlp", "standard_gru", "no_gru", "no_online_norm", "no_smoothing", "no_refinement"]
VARIANT_LABEL = dict(linear="Linear projection", mlp="MLP projection", standard_gru="Standard GRU", no_gru="No GRU",
                     no_online_norm="No online norm", no_smoothing="No smoothing", no_refinement="No refinement")
RED_MAP = LinearSegmentedColormap.from_list("mit", ["#F6F3F1", "#E2B7BD", "#A31F34", "#3F0610"])
FONTS = ["Times New Roman", "Nimbus Roman", "STIX Two Text", "STIXGeneral"]


def style():
    have = {f.name for f in font_manager.fontManager.ttflist}
    family = next((f for f in FONTS if f in have), "serif")
    plt.rcParams.update({
        "font.family": "serif", "font.serif": [family, "DejaVu Serif"], "mathtext.fontset": "stix",
        "font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 8.5, "xtick.labelsize": 8, "ytick.labelsize": 8,
        "legend.fontsize": 8, "legend.frameon": False, "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5,
        "ytick.major.size": 2.5, "axes.grid": True, "grid.color": "#E4E2E0", "grid.linewidth": 0.5,
        "axes.axisbelow": True, "lines.linewidth": 1.1, "lines.markersize": 4.5, "pdf.fonttype": 42, "ps.fonttype": 42,
        "svg.hashsalt": "mjcs", "figure.dpi": 150, "savefig.dpi": 600, "axes.unicode_minus": True,
        "figure.constrained_layout.use": True, "figure.constrained_layout.h_pad": 0.04,
        "figure.constrained_layout.w_pad": 0.04,
    })
    return family


def save(fig, out, name):
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{name}.png", dpi=600, metadata={"Software": None}, bbox_inches="tight", pad_inches=0.06)
    fig.savefig(out / f"{name}.pdf", metadata={"Creator": None, "Producer": None, "CreationDate": None}, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)


def panel(ax, i, dx=-0.02, dy=1.04):
    ax.text(dx, dy, f"({chr(97 + i)})", transform=ax.transAxes, fontsize=9, fontweight="bold", ha="right", va="bottom")


def load(results):
    data = defaultdict(list)
    for p in sorted(glob.glob(str(Path(results) / "stream" / "*" / "*" / "*_s*.json"))):
        r = json.load(open(p))
        if not all(k in r for k in ("task", "variant", "method", "seed", "R", "summary")):
            continue
        data[(r["task"], r["variant"], r["method"])].append(r)
    for k in data:
        data[k].sort(key=lambda r: r["seed"])
    return data


def present(data, task, variant="full", pool=None):
    return [m for m in (SHOWN if pool is None else pool) if data.get((task, variant, m))]


def mean_ci(x):
    x = np.asarray(x, float)
    n = len(x)
    if n < 2:
        return float(x.mean()), 0.0
    return float(x.mean()), float(st.t.ppf(0.975, n - 1) * x.std(ddof=1) / np.sqrt(n))


def summ(data, task, variant, method, key):
    return np.array([r["summary"][key] for r in data.get((task, variant, method), [])])


def legend(fig, methods, ncol=None, loc="outside upper center"):
    h = [Line2D([], [], color=COLOR[m], marker=MARKER[m], lw=1.1, ls=LINESTYLE.get(m, "-"), ms=5 if m != "hdc_bound" else 7, label=LABEL[m],
                markeredgewidth=1.0, mfc="none" if m in HOLLOW else COLOR[m]) for m in methods]
    fig.legend(handles=h, loc=loc, ncol=ncol or len(h), handlelength=1.6, columnspacing=1.0, handletextpad=0.4)


def short(names):
    return [n if len(n) <= 8 else str(i + 1) for i, n in enumerate(names)] if any(len(n) > 8 for n in names) else names


def f1_sessions(data, out):
    tasks = [t for t in TASKS if present(data, t)]
    if not tasks:
        return
    fig, axes = plt.subplots(1, len(tasks), figsize=(FULL, 5.4 * CM), squeeze=False, sharey=False)
    methods = [m for m in SHOWN if any(data.get((t, "full", m)) for t in tasks)]
    for i, (ax, t) in enumerate(zip(axes[0, :], tasks)):
        k = data[(t, "full", present(data, t)[0])][0]["held_in"]
        names = data[(t, "full", present(data, t)[0])][0]["names"]
        x = np.arange(len(names))
        lows = [np.array([r["R"][-1] for r in data[(t, "full", m)]]).min() for m in present(data, t)]
        floor = max(-0.3, np.floor((min(lows) - 0.05) * 10) / 10)
        for j, m in enumerate(present(data, t)):
            F = np.array([r["R"][-1] for r in data[(t, "full", m)]])
            off = (j - (len(SHOWN) - 1) / 2) * 0.055
            ax.plot(x, F.mean(0), color=COLOR[m], lw=0.9, ls=LINESTYLE.get(m, "-"), alpha=0.9, zorder=2 + (m.startswith("hdc")) * 2)
            ax.plot(x + off, F.T, ls="none", marker="o", ms=1.4, color=COLOR[m], alpha=0.35, zorder=1, clip_on=True)
            ax.plot(x, F.mean(0), ls="none", marker=MARKER[m], ms=4.2 if m != "hdc_bound" else 6, color=COLOR[m],
                    mec=COLOR[m], mew=0.9, mfc="none" if m in HOLLOW else COLOR[m],
                    zorder=3 + (m.startswith("hdc")) * 2)
        ax.axvline(k - 0.5, color="#8A8B8C", ls="--", lw=0.7)
        ax.set_ylim(bottom=floor)
        ax.set_xlim(-0.5, len(names) - 0.5)
        step = 1 if len(names) <= 12 else 3
        ax.set_xticks(x[::step])
        ax.set_xticklabels([f"S{j + 1}" for j in range(0, len(names), step)])
        ax.set_xlabel("Session")
        if i == 0:
            ax.set_ylabel(r"Final $R^2$")
        ax.text(0.005, 0.04, "held-in", transform=ax.transAxes, color="#5E5F60", fontsize=7.5)
        ax.text((k - 0.3 + 0.5) / len(names), 0.04, "held-out", transform=ax.transAxes, color="#5E5F60", fontsize=7.5)
        ax.set_title(TASK_LABEL[t], loc="left", fontsize=9, fontweight="bold", pad=2)
        panel(ax, i, dx=-0.05, dy=1.0)
    legend(fig, methods, ncol=5 if len(methods) > 6 else len(methods))
    save(fig, out, "f1_per_session")


def f2_heatmaps(data, out, picks=("hdc", "finetune_full", "replay")):
    tasks = [t for t in TASKS if present(data, t)]
    if not tasks:
        return
    picks = [m for m in picks if any(data.get((t, "full", m)) for t in tasks)]
    widths = [len(data[(t, "full", present(data, t)[0])][0]["names"]) + 3 for t in tasks]
    fig, axes = plt.subplots(len(picks), len(tasks), figsize=(FULL, 2.7 * CM * len(picks) + 1.6 * CM), squeeze=False,
                             gridspec_kw=dict(width_ratios=widths))
    n = 0
    for c, t in enumerate(tasks):
        mats = {m: np.nanmean([r["R"] for r in data[(t, "full", m)]], 0) for m in picks if data.get((t, "full", m))}
        allv = np.concatenate([v.ravel() for v in mats.values()])
        lo, hi = max(np.nanmin(allv), np.nanpercentile(allv, 1)), np.nanmax(allv)
        im = None
        for r, m in enumerate(picks):
            ax = axes[r, c]
            ax.grid(False)
            if m not in mats:
                ax.axis("off")
                continue
            M = mats[m]
            k = data[(t, "full", m)][0]["held_in"]
            im = ax.imshow(M, aspect="auto", cmap=RED_MAP, vmin=lo, vmax=hi, interpolation="nearest")
            ax.axvline(k - 0.5, color="white", lw=1.4)
            ax.axvline(k - 0.5, color="#4B4C4D", lw=0.5, ls="--")
            step = 2 if M.shape[0] > 6 else 1
            ticks = [0] + list(range(step, M.shape[0], step))
            ax.set_yticks(ticks)
            ax.set_yticklabels(["start" if i == 0 else f"+{i}" for i in ticks], fontsize=8)
            ax.set_xticks(range(M.shape[1]))
            dense = M.shape[1] > 10
            ax.set_xticklabels([f"S{i + 1}" for i in range(M.shape[1])] if r == len(picks) - 1 else [],
                               fontsize=6.5 if dense else 7.5, rotation=90 if dense else 0)
            for s in ax.spines.values():
                s.set_visible(False)
            ax.tick_params(length=0)
            if c == 0:
                ax.set_ylabel(LABEL[m], fontweight="bold", color="#222222")
            if r == 0:
                ax.set_title(TASK_LABEL[t], fontsize=9, fontweight="bold", pad=2)
            if r == len(picks) - 1:
                ax.set_xlabel("Session")
            if c == 0:
                panel(ax, n, dx=0.0, dy=1.0)
                n += 1
        if im is not None:
            cb = fig.colorbar(im, ax=axes[:, c].tolist(), location="bottom", shrink=0.8, aspect=30, pad=0.02)
            cb.set_label(r"$R^2$", fontsize=8)
            cb.ax.tick_params(labelsize=8, width=0.5)
            cb.outline.set_linewidth(0.4)
    save(fig, out, "f2_heatmaps")


def front(points, minimize_x=True, floor=-np.inf):
    idx = np.lexsort((-points[:, 1], points[:, 0] if minimize_x else -points[:, 0]))
    best, keep = floor, []
    for i in idx:
        if points[i, 1] > best:
            keep.append(i)
            best = points[i, 1]
    return points[keep]


def collect(data, task, xkey):
    rows = []
    for m in present(data, task):
        a, x = summ(data, task, "full", m, "acc"), summ(data, task, "full", m, xkey)
        (ma, ca), (mx, cx) = mean_ci(a), mean_ci(x)
        rows.append((m, mx, cx, ma, ca))
    return rows


def f3_pareto(data, out):
    tasks = [t for t in TASKS if present(data, t)]
    if not tasks:
        return
    fig, axes = plt.subplots(2, len(tasks), figsize=(FULL, 9.4 * CM), squeeze=False)
    for c, t in enumerate(tasks):
        for r, (xkey, logx) in enumerate([("adapt_macs", True), ("bwt", False)]):
            ax = axes[r, c]
            rows = collect(data, t, xkey)
            floor = -np.inf
            if logx:
                zero = [q for q in rows if q[1] <= 0]
                rows = [q for q in rows if q[1] > 0]
                floor = max([q[3] for q in zero], default=-np.inf)
                for q in zero:
                    ax.axhline(q[3], color="#8A8B8C", ls=":", lw=0.9)
                ax.set_xscale("log")
                ax.xaxis.set_major_locator(LogLocator(base=10, numticks=15))
                if rows:
                    lo, hi = min(q[1] for q in rows), max(q[1] for q in rows)
                    ax.set_xlim(lo / 10 ** 0.7, hi * 10 ** 0.7)
            for m, mx, cx, ma, ca in rows:
                ax.errorbar(mx, ma, xerr=cx if not logx else None, yerr=ca, color=COLOR[m], marker=MARKER[m],
                            ms=5 if m != "hdc_bound" else 8, mfc="none" if m in HOLLOW else COLOR[m],
                            capsize=1.5, elinewidth=0.7, capthick=0.7, zorder=3)
            if rows:
                pts = np.array([[q[1], q[3]] for q in rows])
                f = front(pts, minimize_x=logx, floor=floor)
                if len(f) > 1:
                    ax.plot(f[:, 0], f[:, 1], color="#222222", lw=0.8, ls="--", zorder=1)
            ax.set_xlabel("MACs per held-out session" if logx else "Backward transfer")
            if c == 0:
                ax.set_ylabel(r"ACC ($R^2$)")
            if r == 0:
                ax.set_title(TASK_LABEL[t], fontsize=9, fontweight="bold", pad=2)
        panel(axes[0, c], c, dx=0.0, dy=1.1)
        panel(axes[1, c], len(tasks) + c, dx=0.0, dy=1.1)
    legend(fig, [m for m in SHOWN if any(data.get((t, "full", m)) for t in tasks)], ncol=5)
    save(fig, out, "f3_pareto")


def f3_pareto_3d(data, out):
    tasks = [t for t in TASKS if present(data, t)]
    if not tasks:
        return
    fig = plt.figure(figsize=(FULL, 7.4 * CM), layout="none")
    for c, t in enumerate(tasks):
        ax = fig.add_subplot(1, len(tasks), c + 1, projection="3d")
        rows = [(m, summ(data, t, "full", m, "adapt_macs").mean(), summ(data, t, "full", m, "bwt").mean(),
                 summ(data, t, "full", m, "acc").mean()) for m in present(data, t)]
        rows = [q for q in rows if q[1] > 0]
        if not rows:
            continue
        zmin = min(q[3] for q in rows)
        for m, mac, bwt, acc in rows:
            x = np.log10(mac)
            ax.plot([x, x], [bwt, bwt], [zmin, acc], color=COLOR[m], lw=0.5, alpha=0.6)
            hollow = m in HOLLOW
            ax.scatter([x], [bwt], [acc], facecolors="none" if hollow else COLOR[m], marker=MARKER[m] if MARKER[m] != "x" else "o",
                       s=26 if m != "hdc_bound" else 55, depthshade=False, edgecolors=COLOR[m] if hollow else "white",
                       linewidths=0.9 if hollow else 0.3)
        ax.set_xlabel(r"$\log_{10}$(MACs)", labelpad=-4, fontsize=7.5)
        ax.set_ylabel("BWT", labelpad=4, fontsize=7.5)
        ax.set_zlabel(r"$R^2$", labelpad=-7, fontsize=7.5)
        ax.xaxis.set_major_locator(MaxNLocator(4))
        ax.yaxis.set_major_locator(MaxNLocator(3))
        ax.zaxis.set_major_locator(MaxNLocator(4))
        ax.zaxis.set_major_formatter("{x:.2f}")
        ax.tick_params(labelsize=6.5, pad=-3)
        for a in (ax.xaxis, ax.yaxis, ax.zaxis):
            a.pane.set_facecolor((1, 1, 1, 0))
            a.pane.set_edgecolor("#C2C0BF")
            a._axinfo["grid"]["linewidth"] = 0.4
            a._axinfo["grid"]["color"] = "#E4E2E0"
        ax.set_box_aspect(None, zoom=0.84)
        ax.view_init(elev=22, azim=-58)
        ax.set_title(f"({chr(97 + c)}) {TASK_LABEL[t]}", fontsize=9, fontweight="bold", pad=-2, y=0.93)
    fig.subplots_adjust(left=0.0, right=0.95, bottom=0.0, top=1.17, wspace=0.2)
    fig.text(1.0, 0.5, ".", color="none")
    legend(fig, [m for m in SHOWN if any(data.get((t, "full", m)) for t in tasks) and m != "frozen"], ncol=4,
           loc="upper center")
    save(fig, out, "f3_pareto_3d")


def paired_delta(data, task, variant, method, key):
    a = {r["seed"]: r["summary"][key] for r in data.get((task, variant, method), [])}
    b = {r["seed"]: r["summary"][key] for r in data.get((task, "full", method), [])}
    d = [a[s] - b[s] for s in a if s in b]
    return mean_ci(d) if d else None


def f4_ablation(data, out, method="hdc"):
    tasks = [t for t in TASKS if any(paired_delta(data, t, v, method, "acc") for v in VARIANTS)]
    vs = [v for v in VARIANTS if any(paired_delta(data, t, v, method, "acc") for t in tasks)]
    if not tasks:
        return
    fig, axes = plt.subplots(1, 2, figsize=(FULL, (1.5 + 1.15 * len(vs)) * CM), sharey=True)
    for ax, key, title in zip(axes, ["acc_held_in", "acc_held_out"], ["Held-in", "Held-out"]):
        y = np.arange(len(vs))[::-1]
        ax.axvline(0, color="#222222", lw=0.7)
        for j, t in enumerate(tasks):
            off = (j - (len(tasks) - 1) / 2) * 0.22
            for yy, v in zip(y, vs):
                d = paired_delta(data, t, v, method, key)
                if d is None:
                    continue
                ax.errorbar(d[0], yy + off, xerr=d[1], color=TASK_COLOR[t], marker="o", ms=3.8, capsize=1.5,
                            elinewidth=0.9, capthick=0.8, ls="none")
        ax.set_yticks(y)
        ax.set_yticklabels([VARIANT_LABEL[v] for v in vs])
        ax.set_xlabel(r"$\Delta R^2$ versus full model")
        ax.set_title(title, fontsize=9, fontweight="bold", pad=2)
        ax.grid(axis="y", visible=False)
    panel(axes[0], 0, dx=-0.0, dy=1.0)
    panel(axes[1], 1, dx=-0.0, dy=1.0)
    h = [Line2D([], [], color=TASK_COLOR[t], marker="o", lw=0, ms=4, label=TASK_LABEL[t]) for t in tasks]
    fig.legend(handles=h, loc="outside upper center", ncol=len(h))
    save(fig, out, "f4_ablation")


def f5_cost(data, out):
    tasks = [t for t in TASKS if present(data, t)]
    if not tasks:
        return
    methods = [m for m in SHOWN if m != "frozen" and any(data.get((t, "full", m)) for t in tasks)]
    fig, axes = plt.subplots(1, 2, figsize=(FULL, 5.6 * CM))
    w = 0.8 / len(tasks)
    for ax, key, lab in zip(axes, ["adapt_macs", "adapt_seconds"], ["Adaptation MACs per session", "Adaptation seconds per session"]):
        for j, t in enumerate(tasks):
            xs, ms, cs = [], [], []
            for i, m in enumerate(methods):
                v = summ(data, t, "full", m, key)
                if len(v) and v.mean() > 0:
                    mu, ci = mean_ci(v)
                    xs.append(i + (j - (len(tasks) - 1) / 2) * w)
                    ms.append(mu)
                    cs.append(ci)
            ax.bar(xs, ms, width=w * 0.92, color=TASK_COLOR[t], label=TASK_LABEL[t], zorder=2)
            ax.errorbar(xs, ms, yerr=cs, ls="none", color="#222222", lw=0.6, capsize=1, zorder=3)
        ax.set_yscale("log")
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels([LABEL[m] for m in methods], rotation=40, ha="right", rotation_mode="anchor")
        ax.set_ylabel(lab)
        ax.grid(axis="x", visible=False)
    panel(axes[0], 0, dy=1.0)
    panel(axes[1], 1, dy=1.0)
    h = [Line2D([], [], color=TASK_COLOR[t], marker="s", lw=0, ms=5, label=TASK_LABEL[t]) for t in tasks]
    fig.legend(handles=h, loc="outside upper center", ncol=len(h))
    save(fig, out, "f5_cost")


def f6_bound(results, out):
    p = Path(results) / "theory" / "bound.json"
    if not p.exists():
        return
    obj = json.load(open(p))
    D, S = np.asarray(obj["D"], float), obj["S"]
    emp, pred = np.asarray(obj["empirical"], float), np.asarray(obj["predicted"], float)
    fig, ax = plt.subplots(figsize=(HALF, 5.6 * CM))
    cmap = LinearSegmentedColormap.from_list("s", ["#E2B7BD", "#A31F34", "#3F0610"])
    for j, s in enumerate(S):
        col = cmap(j / max(1, len(S) - 1))
        ax.plot(D, pred[:, j], color=col, lw=0.9, ls="--", zorder=2)
        ax.plot(D, emp[:, j], color=col, marker="o", ms=3.5, ls="none", label=f"$S={s}$", zorder=3)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(D)
    ax.set_xticklabels([f"{int(d)}" for d in D])
    ax.xaxis.set_minor_formatter(plt.NullFormatter())
    ax.yaxis.set_minor_formatter(LogFormatterSciNotation(minor_thresholds=(2, 0.4)))
    ax.tick_params(which="minor", labelsize=6.5)
    ax.set_xlabel("Hypervector dimension $D$")
    ax.set_ylabel("Cross-talk (RMS)")
    h, _ = ax.get_legend_handles_labels()
    h.append(Line2D([], [], color="#222222", ls="--", lw=0.9, label="predicted"))
    fig.legend(handles=h, loc="outside upper center", ncol=4, columnspacing=0.8, handletextpad=0.3, fontsize=7)
    save(fig, out, "f6_bound")


def f7_dose(results, out):
    files = sorted(glob.glob(str(Path(results) / "dose" / "*" / "*.json")))
    if not files:
        return
    groups = defaultdict(list)
    for f in files:
        d = json.load(open(f))
        key = next((k for k in ("acc_held_out", "acc", "r2", "y") if k in d), None)
        if key is None or "dose" not in d:
            continue
        groups[(d.get("task", "h1"), d.get("method", "hdc"))].append((np.asarray(d["dose"], float), np.asarray(d[key], float)))
    tasks = [t for t in TASKS if any(k[0] == t for k in groups)]
    if not tasks:
        return
    fig, axes = plt.subplots(1, len(tasks), figsize=(FULL, 5.2 * CM), sharey=False, squeeze=False)
    used = []
    for c, (ax, t) in enumerate(zip(axes[0], tasks)):
        for m in METHODS:
            if (t, m) not in groups:
                continue
            used.append(m)
            x = groups[(t, m)][0][0]
            Y = np.array([g[1] for g in groups[(t, m)]])
            mu = Y.mean(0)
            ax.plot(x, mu, color=COLOR[m], marker=MARKER[m], ms=4 if m != "hdc_bound" else 6, zorder=3)
            if len(Y) > 1:
                ci = st.t.ppf(0.975, len(Y) - 1) * Y.std(0, ddof=1) / np.sqrt(len(Y))
                ax.fill_between(x, mu - ci, mu + ci, color=COLOR[m], alpha=0.18, lw=0, zorder=2)
        ax.set_xlabel("Prefix fraction of session")
        ax.set_xticks([0.125, 0.25, 0.5])
        ax.set_xticklabels(["0.125", "0.25", "0.5"])
        ax.set_title(TASK_LABEL[t], fontsize=9, fontweight="bold", pad=2)
        if c == 0:
            ax.set_ylabel(r"Held-out $R^2$")
        panel(ax, c, dx=0.0, dy=1.1)
    legend(fig, [m for m in METHODS if m in used])
    save(fig, out, "f7_dose")


def f8_epochs(results, data, out):
    curves = defaultdict(lambda: defaultdict(list))
    for f in sorted(glob.glob(str(Path(results) / "epochs" / "*" / "*_e*_s*.json"))):
        d = json.load(open(f))
        if not all(k in d for k in ("task", "method", "epochs", "summary")) or d["method"] not in EPOCH_METHODS:
            continue
        curves[(d["task"], d["method"])][int(d["epochs"])].append((d["summary"]["adapt_macs"], d["summary"]["acc"]))
    tasks = [t for t in TASKS if any(k[0] == t for k in curves)]
    if not tasks:
        return
    fig, axes = plt.subplots(1, len(tasks), figsize=(FULL, 5.2 * CM), sharey=False, squeeze=False)
    used = []
    for c, (ax, t) in enumerate(zip(axes[0], tasks)):
        for m in EPOCH_METHODS:
            ep = curves.get((t, m))
            if not ep:
                continue
            used.append(m)
            es = sorted(ep)
            x = np.array([np.mean([v[0] for v in ep[e]]) for e in es])
            y = np.array([np.mean([v[1] for v in ep[e]]) for e in es])
            keep = x > 0
            es, x, y = [e for e, k in zip(es, keep) if k], x[keep], y[keep]
            ax.plot(x, y, color=COLOR[m], ls=LINESTYLE.get(m, "-"), marker=MARKER[m], ms=3.8,
                    mfc="none" if m in HOLLOW else COLOR[m], zorder=3)
            off, ha = {"finetune_full": ((0, -9), "center"), "replay": ((0, 4), "center"),
                       "readout_tag": {"h1": ((-4, -9), "right"), "m1": ((-3, 5), "right"), "m2": ((5, -2), "left")}[t]}[m]
            for e, xi, yi in zip(es, x, y):
                if m == "finetune_full" and t == "m1" and e == 40:
                    ax.annotate("40", (xi, yi), xytext=(6, 4), textcoords="offset points", ha="left", fontsize=6, color=COLOR[m])
                elif e == 40 or (e == 1 and m != "readout_tag"):
                    ax.annotate(str(e), (xi, yi), xytext=off, textcoords="offset points", ha=ha, fontsize=6, color=COLOR[m])
        for m in ("closed_linear", "hdc"):
            a, b = summ(data, t, "full", m, "adapt_macs"), summ(data, t, "full", m, "acc")
            if len(a) and a.mean() > 0:
                used.append(m)
                ax.plot(a.mean(), b.mean(), ls="none", marker=MARKER[m], ms=6 if m == "hdc" else 5, color=COLOR[m], zorder=4)
        ax.set_xscale("log")
        ax.set_xlabel("MACs per held-out session")
        if c == len(tasks) - 1:
            ax.text(0.98, 0.04, "labels: epochs", transform=ax.transAxes, ha="right", fontsize=6.5, color="#5E5F60")
        ax.set_title(TASK_LABEL[t], fontsize=9, fontweight="bold", pad=2)
        if c == 0:
            ax.set_ylabel(r"ACC ($R^2$)")
        panel(ax, c, dx=0.0, dy=1.1)
    legend(fig, [m for m in METHODS if m in used])
    save(fig, out, "f8_epochs")


def main(results, out, only=None):
    family = style()
    data = load(results)
    jobs = dict(f1=lambda: f1_sessions(data, out), f2=lambda: f2_heatmaps(data, out), f3=lambda: (f3_pareto(data, out), f3_pareto_3d(data, out)),
                f4=lambda: f4_ablation(data, out), f5=lambda: f5_cost(data, out), f6=lambda: f6_bound(results, out),
                f7=lambda: f7_dose(results, out), f8=lambda: f8_epochs(results, data, out))
    for k, fn in jobs.items():
        if only is None or k in only:
            fn()
    return family


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(ROOT / "results"))
    ap.add_argument("--out", default=str(ROOT / "paper" / "figures"))
    ap.add_argument("--only", nargs="*")
    a = ap.parse_args()
    print(main(Path(a.results), Path(a.out), a.only))
