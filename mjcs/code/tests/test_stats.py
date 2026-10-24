import json
from pathlib import Path

import numpy as np
import pytest

import stats
from stats_synth import make

B = 400


@pytest.fixture(scope="module")
def fixture_root():
    root = Path(__file__).resolve().parent / "fixtures" / "stats_data"
    make(root)
    return root


@pytest.fixture(scope="module")
def data(fixture_root):
    return stats.load(fixture_root)


@pytest.fixture(scope="module")
def full(data):
    return data["synth_a"]["full"]


@pytest.fixture(scope="module")
def draw(data):
    c = data["synth_a"]["full"]["frozen"]
    return stats.Draw(c.k, c.T, 10, B, 0)


def test_metrics_match_engine_summary(fixture_root, data):
    for f in sorted(fixture_root.glob("stream/synth_a/*/*_s*.json")):
        j = json.loads(f.read_text())
        c = data[j["task"]][j["variant"]][j["method"]]
        got = stats.per_seed(c)
        i = c.seeds.index(j["seed"])
        for k in stats.METRICS:
            assert got[k][i] == pytest.approx(j["summary"][k], rel=1e-9, abs=1e-9)


def test_holm():
    assert stats.holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])


def test_effect_sizes():
    x = np.array([3.0, 4.0, 5.0, 6.0])
    y = np.array([1.0, 2.0, 3.0, 9.0])
    assert stats.cliff(x, y) == pytest.approx((np.sign(x[:, None] - y[None]).sum()) / 16)
    assert stats.cliff(x, x) == 0
    assert stats.dz(x - y) == pytest.approx((x - y).mean() / (x - y).std(ddof=1))
    assert stats.wilcoxon(np.zeros(5))[1] == 1.0


def test_aggregate_deterministic_and_covers_mean(full, draw):
    a, _ = stats.aggregate(full, draw)
    b, _ = stats.aggregate(full, stats.Draw(draw.ins.shape[1], draw.outs.shape[1], 10, B, 0))
    assert json.dumps(stats.clean(a)) == json.dumps(stats.clean(b))
    for m in a:
        for k in stats.METRICS:
            r = a[m][k]
            assert r["ci_low"] <= r["mean"] + 1e-6 and r["mean"] <= r["ci_high"] + 1e-6
            assert r["sd"] >= 0 or np.isnan(r["sd"])
    assert a["hdc"]["acc"]["mean"] > a["frozen"]["acc"]["mean"]
    assert len(a["hdc"]["per_session_final_r2"]["mean"]) == 9


def test_paired(full):
    rows = stats.paired(full)["rows"]
    r = next(r for r in rows if r["ref"] == "hdc" and r["other"] == "frozen" and r["metric"] == "acc")
    assert r["p"] < 0.01 and r["p_holm"] >= r["p"] and r["cliff_delta"] > 0 and r["cohen_dz"] > 0
    assert r["n_cells"] == 90 and r["n"] == 90 // 9
    assert all(-1 <= r["cliff_delta"] <= 1 for r in rows)
    assert all(r["p_holm"] <= 1 for r in rows)


def test_tost(full):
    t = stats.tost(full)
    assert set(t) == set(stats.TOST_REFS)
    for r in t.values():
        assert r["noninferior"] == (r["smallest_margin_noninferior"] < r["margin"])
        assert r["ci90_low"] <= r["mean_diff"] <= r["ci90_high"]
    assert t["finetune_full"]["noninferior"]
    assert t["finetune_full"]["smallest_margin_noninferior"] == 0.0
    assert t["replay"]["mean_diff"] > 0


def test_mixed(full):
    r = stats.mixed(full)
    assert set(r["coefficients"]) == set(full) - {"hdc"} | {"intercept"}
    assert r["coefficients"]["frozen"]["estimate"] < 0
    assert r["coefficients"]["frozen"]["p"] < 0.01
    assert set(r["variance_components"]) == {"session", "seed"}
    assert r["optimizer"] in stats.OPTIMIZERS and isinstance(r["converged"], bool)


def test_missing_methods_skipped(full):
    sub = {m: full[m] for m in ("frozen", "hdc", "replay")}
    assert {r["other"] for r in stats.paired(sub)["rows"]} == {"frozen", "replay"}
    assert set(stats.tost(sub)) == {"replay"}
    assert "hdc" in stats.mixed(sub)["reference"]
    assert len(stats.paired({m: full[m] for m in ("frozen", "hdc")})["rows"]) == 2


def test_holm_family_is_all_comparisons(full):
    rows = stats.paired(full)["rows"]
    assert len([r for r in rows if r["metric"] == "acc"]) == len(full) - 1 == 13


def test_dominance():
    X = np.array([[1.0, 1.0], [0.5, 0.5], [2.0, 0.0], [1.0, 1.0]])
    assert stats.dominated(X).tolist() == [False, True, False, False]
    assert stats.dominated(np.stack([X, X[::-1]])).shape == (2, 4)


def test_pareto(full, draw):
    _, boots = stats.aggregate(full, draw)
    p = stats.pareto(full, boots, draw)
    assert "frozen" in p["non_dominated"]
    assert set(p["non_dominated"]) | set(p["dominated"]) == set(full)
    assert all(0 <= v <= 1 for v in p["prob_non_dominated"].values())
    assert "finetune_full" in p["dominated"]


def test_ablation(data, draw):
    a = stats.ablation(data["synth_a"], draw)
    assert set(a) == {"linear", "no_gru"}
    r = a["linear"]["hdc"]
    assert r["n_seeds"] == 5
    assert r["acc_held_out"]["delta"] < -0.03
    assert r["acc_held_out"]["ci_high"] < 0
    assert r["acc_held_out"]["p_holm"] >= r["acc_held_out"]["p"]


def test_end_to_end(fixture_root, tmp_path):
    s = stats.run(fixture_root, tmp_path, B=200, seed=0)
    assert set(s["tasks"]) == {"synth_a", "synth_b"}
    names = {p.name for p in tmp_path.iterdir()}
    for t in s["tasks"]:
        for part in ("aggregate", "paired", "mixed", "tost", "pareto", "ablation"):
            assert f"{t}_{part}.json" in names
    text = (tmp_path / "summary.json").read_text()
    assert "NaN" not in text and "Infinity" not in text
    assert json.loads(text)["meta"]["bootstrap_resamples"] == 200
