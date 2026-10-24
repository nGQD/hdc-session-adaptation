#!/usr/bin/env python3
"""Reanalyse stream and held-out endpoints without refitting or selecting models.

Run from any directory with Python, NumPy and SciPy. The default inputs are the
original ten-seed outer-stream records; all input hashes are written to the report.
Margin sensitivity and the held-out endpoint are explicitly post hoc analyses.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy import stats


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SEEDS = [101, 113, 127, 131, 137, 149, 157, 163, 173, 181]
TASKS = ["h1", "m1", "m2"]
REFERENCES = ["finetune_full", "replay", "full_tag", "readout_tag"]
LABELS = {"finetune_full": "FT-full", "replay": "Replay",
          "full_tag": "FT-full-tag", "readout_tag": "FT-readout-tag"}
MARGINS = [0.01, 0.02, 0.03, 0.05]
ENDPOINTS = ["stream_acc", "held_out_r2"]


def paired_noninferiority(differences, margin):
    """H0: mean(HDC-reference) <= -margin, conditional on fixed sessions."""
    x = np.asarray(differences, dtype=float)
    assert x.ndim == 1 and len(x) > 1 and np.isfinite(x).all()
    n, mean = len(x), float(x.mean())
    se = float(x.std(ddof=1) / np.sqrt(n))
    if se == 0:
        p = 0.0 if mean > -margin else 1.0
        halfwidth = 0.0
    else:
        p = float(stats.t.sf((mean + margin) / se, n - 1))
        halfwidth = float(stats.t.ppf(0.95, n - 1) * se)
    return {"mean_difference": mean, "standard_error": se,
            "ci90_seed": [mean - halfwidth, mean + halfwidth],
            "p_noninferiority": p, "noninferior_unadjusted": p < 0.05,
            "smallest_margin_for_unadjusted_noninferiority": max(0.0, -(mean - halfwidth))}


def holm(pvalues):
    pvalues = np.asarray(pvalues, dtype=float)
    order = np.argsort(pvalues, kind="stable")
    ordered = pvalues[order] * (len(order) - np.arange(len(order)))
    result = np.empty_like(ordered)
    result[order] = np.minimum(1.0, np.maximum.accumulate(ordered))
    return result


def load_runs(raw_root, baseline_root=None):
    loaded, sources = {}, []
    for task in TASKS:
        loaded[task] = {}
        for method in ["hdc", *REFERENCES]:
            input_root = baseline_root if baseline_root is not None and method != "hdc" else raw_root
            runs = []
            for seed in SEEDS:
                path = input_root / task / "full" / f"{method}_s{seed}.json"
                raw = path.read_bytes()
                run = json.loads(raw)
                assert run["seed"] == seed and run["task"] == task
                assert run["method"] == method and run["variant"] == "full"
                runs.append(run)
                sources.append({"path": str(path.relative_to(input_root)),
                                "source": "baseline_overlay" if input_root != raw_root else "original_stream",
                                "sha256": hashlib.sha256(raw).hexdigest()})
            k = int(runs[0]["held_in"])
            names = runs[0]["names"]
            assert all(r["held_in"] == k and r["names"] == names for r in runs)
            matrices = np.asarray([r["R"] for r in runs], dtype=float)
            t = len(names) - k
            assert matrices.shape == (len(SEEDS), t + 1, k + t)
            assert np.isfinite(matrices).all()
            final = matrices[:, -1, :]
            assert np.allclose(final.mean(axis=1), [r["summary"]["acc"] for r in runs], atol=1e-10)
            assert np.allclose(final[:, k:].mean(axis=1),
                               [r["summary"]["acc_held_out"] for r in runs], atol=1e-10)
            loaded[task][method] = {"final": final, "k": k, "names": names}
        base = loaded[task]["hdc"]
        assert all(x["k"] == base["k"] and x["names"] == base["names"]
                   for x in loaded[task].values())
    return loaded, sources


def draw_bootstrap(k, t, number, seed):
    rng = np.random.default_rng(seed)
    return (rng.integers(0, len(SEEDS), size=(number, len(SEEDS))),
            rng.integers(0, k, size=(number, k)),
            rng.integers(0, t, size=(number, t)))


def crossed_bootstrap_difference(difference, k, draws, endpoint):
    seeds, ins, outs = draws
    sampled_out = difference[:, k:][seeds[:, :, None], outs[:, None, :]]
    if endpoint == "held_out_r2":
        return sampled_out.mean(axis=(1, 2))
    sampled_in = difference[:, :k][seeds[:, :, None], ins[:, None, :]]
    return (sampled_in.sum(axis=(1, 2)) + sampled_out.sum(axis=(1, 2))) / (
        len(SEEDS) * difference.shape[1])


def analyse(loaded, number, seed):
    rows = []
    for task_index, task in enumerate(TASKS):
        hdc = loaded[task]["hdc"]
        k, h = hdc["k"], hdc["final"]
        draws = draw_bootstrap(k, h.shape[1] - k, number, seed + task_index)
        for method in REFERENCES:
            reference = loaded[task][method]["final"]
            delta = h - reference
            for endpoint in ENDPOINTS:
                sl = slice(None) if endpoint == "stream_acc" else slice(k, None)
                per_seed = delta[:, sl].mean(axis=1)
                boot = crossed_bootstrap_difference(delta, k, draws, endpoint)
                summary = {"task": task, "reference": method, "endpoint": endpoint,
                           "n_seeds": len(SEEDS), "held_in_sessions": k,
                           "held_out_sessions": h.shape[1] - k,
                           "hdc_mean": float(h[:, sl].mean()),
                           "reference_mean": float(reference[:, sl].mean()),
                           "seed_differences": per_seed.tolist(),
                           "session_names": hdc["names"][sl],
                           "session_mean_differences": delta[:, sl].mean(axis=0).tolist(),
                           "ci95_crossed_bootstrap": np.percentile(boot, [2.5, 97.5]).tolist(),
                           "lower95_crossed_bootstrap": float(np.percentile(boot, 5)),
                           "held_in_mean_difference": float(delta[:, :k].mean()),
                           "held_out_mean_difference": float(delta[:, k:].mean())}
                for margin in MARGINS:
                    rows.append({**summary, "margin": margin,
                                 **paired_noninferiority(per_seed, margin)})
    counts = []
    for endpoint in ENDPOINTS:
        for margin in MARGINS:
            family = [r for r in rows if r["endpoint"] == endpoint and r["margin"] == margin]
            assert len(family) == 12
            adjusted = holm([r["p_noninferiority"] for r in family])
            for r, ph in zip(family, adjusted):
                r["p_holm_12"] = float(ph)
                r["p_bonferroni_12"] = min(1.0, 12 * r["p_noninferiority"])
                r["noninferior_holm_12"] = bool(ph < 0.05)
                r["noninferior_bonferroni_12"] = r["p_bonferroni_12"] < 0.05
            counts.append({"endpoint": endpoint, "margin": margin, "comparisons": 12,
                           "unadjusted": sum(r["noninferior_unadjusted"] for r in family),
                           "holm_12": sum(r["noninferior_holm_12"] for r in family),
                           "bonferroni_12": sum(r["noninferior_bonferroni_12"] for r in family)})
    return rows, counts


def markdown(report):
    lines = ["Endpoint and margin sensitivity of HDC versus gradient adaptation", "",
             "The original stream endpoint includes both held-in and held-out sessions. "
             "The held-out endpoint and margin sensitivity below are post hoc analyses of the same "
             "saved outer-stream predictions; no model is selected or refitted. "
             "Paired tests use ten seeds and are conditional on the fixed benchmark sessions.", "",
             "| Endpoint | Margin | Unadjusted | Holm, 12 tests | Bonferroni, 12 tests |",
             "|:--|--:|--:|--:|--:|"]
    for r in report["counts"]:
        lines.append(f"| {r['endpoint']} | {r['margin']:.2f} | {r['unadjusted']}/12 | "
                     f"{r['holm_12']}/12 | {r['bonferroni_12']}/12 |")
    lines += ["", "Held-out endpoint at margin 0.03", "",
              "| Task | Reference | Difference | 90% seed interval | Bonferroni p | 95% crossed-bootstrap interval |",
              "|:--|:--|--:|:--|--:|:--|"]
    for r in report["comparisons"]:
        if r["endpoint"] != "held_out_r2" or r["margin"] != 0.03:
            continue
        lo, hi = r["ci90_seed"]
        bl, bh = r["ci95_crossed_bootstrap"]
        lines.append(f"| {r['task'].upper()} | {LABELS[r['reference']]} | "
                     f"{r['mean_difference']:+.4f} | ({lo:+.4f}, {hi:+.4f}) | "
                     f"{r['p_bonferroni_12']:.4g} | ({bl:+.4f}, {bh:+.4f}) |")
    lines += ["", "The crossed bootstrap resamples seeds and sessions independently, retaining "
              "pairing between methods and separate held-in/held-out strata for stream accuracy. "
              "Its intervals describe sensitivity to the observed sessions. Only 7, 3, and 4 "
              "held-out sessions are available for H1, M1, and M2; seeds are repeated training runs, "
              "not additional participants or recording sessions. The formal non-inferiority "
              "decisions above use the seed-level test, not these descriptive bootstrap intervals.", ""]
    return "\n".join(lines)


def self_test():
    # Identical predictions must cancel under arbitrary paired session/seed resampling.
    zero = np.zeros((len(SEEDS), 7))
    draws = draw_bootstrap(4, 3, 100, 7)
    for endpoint in ENDPOINTS:
        assert np.array_equal(crossed_bootstrap_difference(zero, 4, draws, endpoint), np.zeros(100))
    # A fixed effect remains fixed under both endpoints and every paired resample.
    fixed = zero + 0.02
    assert np.allclose(crossed_bootstrap_difference(fixed, 4, draws, "stream_acc"), 0.02)
    assert np.allclose(crossed_bootstrap_difference(fixed, 4, draws, "held_out_r2"), 0.02)
    assert paired_noninferiority(np.zeros(10), 0.03)["noninferior_unadjusted"]
    assert not paired_noninferiority(np.full(10, -0.04), 0.03)["noninferior_unadjusted"]
    assert np.allclose(holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=ROOT / "mjcs/results/stream")
    parser.add_argument("--baseline-root", type=Path, help="Optional replacement gradient records with task/full/method_sSEED.json layout; HDC remains from raw-root")
    parser.add_argument("--output-dir", type=Path, default=HERE / "results/endpoint_analysis")
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20261023)
    parser.add_argument("--check-original", action="store_true", help="Check original 11/12 stream and 8/12 held-out counts")
    args = parser.parse_args()
    self_test()
    loaded, sources = load_runs(args.raw_root.resolve(), args.baseline_root.resolve() if args.baseline_root else None)
    rows, counts = analyse(loaded, args.bootstrap, args.bootstrap_seed)
    report = {"analysis": "post_hoc_endpoint_and_margin_sensitivity",
              "protocol": {"seeds": SEEDS, "margins": MARGINS, "alpha": 0.05,
                           "test": "one-sided paired Student t test over seeds",
                           "null": "mean HDC-reference <= -margin",
                           "multiplicity_family": "12 comparisons within each endpoint and margin",
                           "bootstrap_resamples": args.bootstrap,
                           "bootstrap_seed": args.bootstrap_seed,
                           "bootstrap": "paired crossed seeds and sessions; stream stratified held-in/out",
                           "inference_scope": "fixed observed FALCON sessions; no independent-subject inference",
                           "analysis_does_not_change_model_selection": True},
              "software": {"numpy": np.__version__, "scipy": __import__("scipy").__version__},
              "sources": sources, "counts": counts, "comparisons": rows}
    # Regression check against the independently audited original stream endpoint.
    stream = next(r for r in counts if r["endpoint"] == "stream_acc" and r["margin"] == 0.03)
    heldout = next(r for r in counts if r["endpoint"] == "held_out_r2" and r["margin"] == 0.03)
    if args.check_original:
        assert stream["bonferroni_12"] == 11 and heldout["bonferroni_12"] == 8
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "endpoint_analysis.json").write_text(json.dumps(report, indent=2) + "\n")
    (args.output_dir / "endpoint_tables.md").write_text(markdown(report))
    print(json.dumps(counts, indent=2))


if __name__ == "__main__":
    main()
