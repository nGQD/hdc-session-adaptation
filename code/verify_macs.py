import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import json
import time
import warnings

import numpy as np
import torch
from fvcore.nn import FlopCountAnalysis

import data
import run
from engine import GradientMethod, Ridge, Session, backbone_macs, target_stats, windows
from model import Backbone

torch.set_num_threads(8)
warnings.filterwarnings("ignore")
OUT = data.ROOT / "results" / "macs"
BINS = 256
TASKS = ["h1", "m1", "m2"]
DIMS = [512, 1024, 1536, 2304]


def gru_handle(inputs, outputs):
    x = inputs[0].type().sizes()
    hidden = outputs[0].type().sizes()[-1]
    batch, steps, size = x if len(x) == 3 else (1, *x)
    return int(batch * steps * 3 * (size * hidden + hidden * hidden))


def profile(c):
    torch.manual_seed(0)
    model = Backbone(c).eval()
    x = torch.randn(1, BINS, c.channels)
    with torch.no_grad():
        counter = FlopCountAnalysis(model, x).set_op_handle("aten::gru", gru_handle)
        counter.unsupported_ops_warnings(False).uncalled_modules_warnings(False)
        total = counter.total()
        head = FlopCountAnalysis(model.readout, torch.randn(1, BINS, model.features)).unsupported_ops_warnings(False).total()
    analytic = backbone_macs(model)
    return dict(profiled_total_per_bin=total / BINS, profiled_head_per_bin=head / BINS,
                profiled_encode_per_bin=(total - head) / BINS,
                analytic_encode=analytic["encode"], analytic_head=analytic["head"],
                ratio_encode=(total - head) / BINS / analytic["encode"],
                ratio_total=total / BINS / (analytic["encode"] + analytic["head"]),
                parameters=model.parameter_count(), features=model.features)


def ridge_formulas(n, d, o, N, D):
    return dict(accumulate=N * d * (d + 1) // 2 + N * d * o, cholesky=d ** 3 // 6, solve=d ** 2 * o,
                encode=N * n * D if D else 0)


def check_ridge(task):
    c = run.base_config(task)
    n, o, N = c.hidden + c.channels, c.outputs, 600
    f = torch.randn(N, n)
    y = torch.randn(N, o).numpy()
    mask = np.ones(N, bool)
    rows = []
    for kind, D in [("raw", 0), ("hdc", 256), ("bound", 256)]:
        r = Ridge(kind, n, o, D or 1, 1.0, 1e-3, 0)
        r.add("a", f, y, mask)
        form = ridge_formulas(n, r.dim, o, N, D)
        total = form["accumulate"] + form["cholesky"] + form["solve"] + form["encode"]
        rows.append(dict(kind=kind, d=r.dim, N=N, engine_macs=int(r.macs), formula_macs=int(total),
                         match=int(r.macs) == int(total), **form))
    return rows


def fine_tune_prefix(task):
    c = run.base_config(task)
    rs = data.by_session(data.load(task, "held-out-calib"))[0][1]
    held_in = [data.held_in_split(r) for _, r in data.by_session(data.load(task, "held-in-calib"))[:2]]
    ystats = target_stats(held_in)
    part = data.held_out_split(rs)
    s = Session("probe", part, "adapt", c, ystats)
    return c, s


def time_it(fn, repeats=5):
    fn()
    best = []
    for _ in range(repeats):
        t = time.perf_counter()
        fn()
        best.append(time.perf_counter() - t)
    return float(np.median(best))


def timing(task="h1"):
    torch.manual_seed(0)
    c, s = fine_tune_prefix(task)
    model = Backbone(c).eval()
    n, o, N = model.features, c.outputs, s.fit_end
    with torch.no_grad():
        f = torch.cat([model.encode(z[None])[0][0] for z in torch.as_tensor(s.x[:N]).split(2048)])
    mask = np.zeros(len(s.x), bool)
    mask[:N] = s.fit[:N]
    y = s.yz[:N]
    used = int(mask.sum())
    ridge = []
    for d in DIMS:
        D = max(1, round((d - n - 1) / 2))
        def step():
            r = Ridge("hdc", n, o, D, 1.0, 1e-3, 0)
            r.add("a", f, y, mask[:N])
        sec = time_it(step)
        r = Ridge("hdc", n, o, D, 1.0, 1e-3, 0)
        r.add("a", f, y, mask[:N])
        ridge.append(dict(d=r.dim, D=D, seconds=sec, macs=int(r.macs)))
    gm = GradientMethod("full", epochs=1, lr=1e-4)
    gm.start(model, [], c, 0)
    t = time.perf_counter()
    macs = gm.adapt(s)
    seconds = time.perf_counter() - t
    t = time.perf_counter()
    gm.adapt(s)
    seconds = min(seconds, time.perf_counter() - t)
    forward = sum(backbone_macs(model).values())
    processed = len(windows([s], c, limit=s.fit_end)["starts"]) * c.window
    return dict(task=task, prefix_bins=N, fit_bins=used, ridge=ridge,
                full_finetune_epoch=dict(seconds=seconds, engine_macs=int(macs), forward_macs_per_bin=forward,
                                         processed_bins=processed, formula_macs=int(3 * 1 * forward * processed)))


def table(report):
    lines = ["| task | variant | encode/bin analytic | encode/bin profiled | ratio | head/bin | params |", "|---|---|---|---|---|---|---|"]
    for t, vs in report["backbone"].items():
        for v, r in vs.items():
            lines.append(f"| {t} | {v} | {r['analytic_encode']:,} | {r['profiled_encode_per_bin']:,.0f} | {r['ratio_encode']:.4f} | {r['analytic_head']:,} | {r['parameters']:,} |")
    lines += ["", "| task | kind | d | engine MACs | closed-form MACs | match |", "|---|---|---|---|---|---|"]
    for t, rows in report["ridge_check"].items():
        for r in rows:
            lines.append(f"| {t} | {r['kind']} | {r['d']} | {r['engine_macs']:,} | {r['formula_macs']:,} | {r['match']} |")
    tm = report["timing"]
    lines += ["", f"CPU timing on {tm['task']} prefix of {tm['prefix_bins']} bins (8 threads)", "",
              "| method | d | seconds | MACs |", "|---|---|---|---|"]
    for r in tm["ridge"]:
        lines.append(f"| ridge step | {r['d']} | {r['seconds']:.3f} | {r['macs']:,} |")
    g = tm["full_finetune_epoch"]
    lines.append(f"| full fine-tune epoch | - | {g['seconds']:.3f} | {g['engine_macs']:,} (3*E*N_proc*F = {g['formula_macs']:,}) |")
    return "\n".join(lines) + "\n"


def main():
    report = dict(backbone={}, ridge_check={})
    for task in TASKS:
        report["backbone"][task] = {v: profile(c) for v, c in run.variants(run.base_config(task)).items()}
        report["ridge_check"][task] = check_ridge(task)
    report["timing"] = timing()
    OUT.mkdir(parents=True, exist_ok=True)
    data.write_json(OUT / "verify.json", report)
    (OUT / "verify.md").write_text(table(report))
    print(table(report))


if __name__ == "__main__":
    main()
