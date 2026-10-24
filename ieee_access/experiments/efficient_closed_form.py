#!/usr/bin/env python3
"""Exact primal/dual implementation of the original session-specific ridge head.

The smaller of the row and feature dimensions determines the factorization.
No hyperparameters, features, training targets, or stored-head semantics change.
Shared-head accumulation retains the native primal implementation: a dual shared
head would require retaining examples, changing its memory protocol.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "8")
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
CODE = next(p for a in HERE.parents for p in [a / "mjcs/code", a / "code"]
            if (p / "engine.py").is_file() and (p / "run.py").is_file())
STUDY = CODE.parent
sys.path.insert(0, str(CODE))
import engine
from model import Backbone, Config, session_key

OUT = HERE / "results/efficient_closed_form"
SEEDS = [101, 113, 127, 131, 137, 149, 157, 163, 173, 181]
METHODS = ["hdc", "closed_linear", "hdc_zero", "closed_zero", "hdc_bound"]


@torch.no_grad()
def ridge_correction(X, Y, prior, lam, solver="auto", weights=None, prior_rows=None):
    """Return W-prior for weighted ridge, using float64 primal or dual algebra.

Minimize sum_i weights_i ||X_i W-Y_i||^2 + lam ||W-prior||^2.
Weights default to one. prior_rows skips known-zero trailing prior rows.
The production session method uses the original unit weights and lambda scaling.
"""
    if X.dtype != torch.float64 or Y.dtype != torch.float64 or prior.dtype != torch.float64:
        raise TypeError("Ridge inputs, targets and prior must be float64")
    n, d = X.shape
    if Y.shape[0] != n or prior.shape != (d, Y.shape[1]) or lam <= 0:
        raise ValueError("Invalid ridge dimensions or nonpositive penalty")
    if solver not in {"auto", "primal", "dual"}:
        raise ValueError(solver)
    rows = d if prior_rows is None else prior_rows
    residual = Y - X[:, :rows] @ prior[:rows] if rows else Y
    if weights is not None:
        weights = torch.as_tensor(weights, dtype=X.dtype, device=X.device)
        if weights.shape != (n,) or not torch.isfinite(weights).all() or (weights < 0).any():
            raise ValueError("Weights must be a finite nonnegative vector")
        root = weights.sqrt()[:, None]
        X, residual = X * root, residual * root
    branch = ("dual" if n < d else "primal") if solver == "auto" else solver
    q = n if branch == "dual" else d
    if not n:
        return torch.zeros_like(prior), {"solver": branch, "N": n, "d": d, "q": q}
    gram = X @ X.T if branch == "dual" else X.T @ X
    gram.diagonal().add_(lam)
    rhs = residual if branch == "dual" else X.T @ residual
    solved = torch.cholesky_solve(rhs, torch.linalg.cholesky(gram))
    correction = X.T @ solved if branch == "dual" else solved
    return correction, {"solver": branch, "N": n, "d": d, "q": q}


def adaptation_macs(prefix_bins, backbone_per_bin, n, d, outputs, encode_per_bin=0,
                    *, q=None, symmetric=False):
    """Analytic MACs with the dense Gram multiplication used by the implementation.

q=min(N,d), or q=d for the shared feature-space accumulator. N*d*q
covers the complete dense Gram product. symmetric=True is retained only for
explicitly labeled provenance of the former symmetric-triangle convention. N*d*o
covers X.T@residual (primal) or X.T@dual_solution (dual). Computing
the pretrained prediction for residual centering is included in the complete
backbone/readout prefix pass; elementwise subtraction is not a MAC.
"""
    q = min(n, d) if q is None else q
    gram = n * d * (q + 1) // 2 if symmetric else n * d * q
    return int(prefix_bins * backbone_per_bin + n * encode_per_bin
               + gram + n * d * outputs
               + q ** 3 // 6 + q * q * outputs)


class EfficientPriorMethod(engine.PriorMethod):
    """Original PriorMethod with an equivalent smaller ridge factorization."""
    def start(self, model, held_in, c, seed):
        super().start(model, held_in, c, seed)
        self.solver_records = []

    def memory(self):
        """Retained model/adaptation scalars, excluding the offline feature cache.

The cache accelerates repeated evaluation and can be discarded without changing
the fitted model. Peak working-memory measurements must still include it.
Native accounting omits the detached double-precision prior copy, stored session
keys, and the shared head in addition to its Gram/cross-product accumulators.
"""
        return (super().memory() + self.W0.numel()
                + sum(key.numel() for key in self.keys.values())
                + (self.W_shared.numel() if self.W_shared is not None else 0))

    def release_evaluation_cache(self):
        """Release offline frozen-feature caches after adaptation or prediction.

No fitted tensor is removed. A subsequent predict() or adapt() recomputes the
needed causal features from its session input. Deployment code can call this
after each operation; repeated offline stream scoring may retain the cache.
        """
        count = len(self.cache)
        self.cache.clear()
        return count

    @torch.no_grad()
    def adapt(self, s):
        if self.shared:
            symmetric_cost = super().adapt(s)
            n = int(np.asarray(s.fit[:s.fit_end]).sum())
            d = int(self.W_shared.shape[0])
            cost = int(symmetric_cost + n * d * d - n * d * (d + 1) // 2)
            self.solver_records.append(dict(session=s.name, solver="native_primal_shared",
                                           N=n, d=d, q=d, analytic_macs=cost,
                                           symmetric_analytic_macs=int(symmetric_cost),
                                           legacy_primal_symmetric_macs=int(symmetric_cost),
                                           dense_primal_macs=cost,
                                           gram_counting="dense_matrix_product"))
            return cost
        gf = self.run(s)
        n = s.fit_end
        mask = torch.as_tensor(s.fit[:n], device=engine.DEVICE)
        X = self.phi(gf[:n][mask]).double()
        Y = torch.as_tensor(s.yz[:n], device=engine.DEVICE)[mask].double()
        N, d, o = len(X), X.shape[1], Y.shape[1]
        P = self.prior(d)
        penalty = self.lam * max(N, 1) / 1000
        correction, detail = ridge_correction(X, Y, P, penalty,
                                               prior_rows=len(self.W0) if self.center else 0)
        # Native zero-prior methods still predict as pretrained_base + stored_delta.
        delta = correction if self.center else correction - self.base(d)
        if self.kind == "bound":
            key = session_key(self.dimension, len(self.keys), self.seed).to(engine.DEVICE)
            self.keys[s.name] = key
            # A view would keep the full hypervector correction's backing storage.
            self.heads[s.name] = delta[:len(self.W0)].clone()
            extra = delta[len(self.W0):]
            bound = self.enc.bind(extra.T.float(), -key).T.double()
            self.memory_W = bound if self.memory_W is None else self.memory_W + bound
        else:
            self.heads[s.name] = delta
        bb = engine.backbone_macs(self.model)
        encoding = self.enc.w.shape[0] * self.dimension if self.enc is not None else 0
        cost = adaptation_macs(n, bb["encode"] + bb["head"], N, d, o, encoding)
        args = (n, bb["encode"] + bb["head"], N, d, o, encoding)
        self.solver_records.append(dict(session=s.name, **detail,
                                       lambda_effective=float(penalty), prefix_bins=int(n),
                                       analytic_macs=cost,
                                       symmetric_analytic_macs=adaptation_macs(*args, symmetric=True),
                                       legacy_primal_symmetric_macs=adaptation_macs(*args, q=d, symmetric=True),
                                       dense_primal_macs=adaptation_macs(*args, q=d),
                                       gram_counting="dense_matrix_product"))
        return cost


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def self_test():
    """Independent augmented least-squares oracle, with both factorization branches."""
    gen = torch.Generator().manual_seed(20261023)
    errors = []
    for n, d in [(7, 13), (19, 6)]:
        X = torch.randn(n, d, generator=gen, dtype=torch.float64)
        Y = torch.randn(n, 3, generator=gen, dtype=torch.float64)
        weights = torch.linspace(0.2, 2.0, n, dtype=torch.float64)
        for centered in [False, True]:
            P = torch.randn(d, 3, generator=gen, dtype=torch.float64) if centered else torch.zeros(d, 3, dtype=torch.float64)
            for weighted in [False, True]:
                w = weights if weighted else torch.ones(n, dtype=torch.float64)
                lam = 2.3
                aug_x = torch.cat([X * w.sqrt()[:, None], lam ** 0.5 * torch.eye(d, dtype=torch.float64)])
                aug_y = torch.cat([Y * w.sqrt()[:, None], lam ** 0.5 * P])
                oracle = torch.linalg.lstsq(aug_x, aug_y, driver="gelsd").solution
                for branch in ["primal", "dual", "auto"]:
                    delta, detail = ridge_correction(X, Y, P, lam, solver=branch,
                                                    weights=weights if weighted else None)
                    error = float((P + delta - oracle).abs().max())
                    assert error < 1e-11, (n, d, branch, error)
                    assert not delta.requires_grad
                    if branch == "auto": assert detail["q"] == min(n, d)
                    errors.append(error)
    assert adaptation_macs(20, 100, 7, 13, 3, 11) == 20 * 100 + 7 * 11 + 13 * 7 * 7 + 7 * 13 * 3 + 7 ** 3 // 6 + 7 * 7 * 3
    assert adaptation_macs(20, 100, 7, 13, 3, 11, symmetric=True) == 20 * 100 + 7 * 11 + 13 * 7 * 8 // 2 + 7 * 13 * 3 + 7 ** 3 // 6 + 7 * 7 * 3
    original_device = engine.DEVICE
    engine.DEVICE = "cpu"
    try:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(17)
            c = Config(channels=3, outputs=2, hidden=4, rank=2, refine_width=8, dropout=0.0)
            model = Backbone(c).eval()
            session = SimpleNamespace(name="cache_test", x=torch.randn(24, 3).numpy(),
                                      yz=torch.randn(24, 2).numpy(), fit=np.arange(24) < 12,
                                      fit_end=12)
            for kind, center in [("hdc", True), ("bound", True), ("linear", False)]:
                method = EfficientPriorMethod(kind, dimension=8, lam=1000.0, center=center)
                method.start(model, [], c, 101)
                method.adapt(session)
                before = method.predict(session)
                retained = method.memory()
                assert method.release_evaluation_cache() == 1 and not method.cache
                assert method.memory() == retained
                assert np.array_equal(before, method.predict(session))
                head = method.heads[session.name]
                assert head.untyped_storage().nbytes() == head.numel() * head.element_size()
    finally:
        engine.DEVICE = original_device
    return {"checks": len(errors), "max_absolute_weight_error": max(errors),
            "reference": "augmented weighted least squares via SVD",
            "cache_release_checks": 3, "stored_heads_own_exact_sized_storage": True}


class CachedEfficient(EfficientPriorMethod):
    def run(self, s):
        return s.gf


class CachedNative(engine.PriorMethod):
    def run(self, s):
        return s.gf


def method_config(selection, method):
    if method == "closed_linear": return dict(kind="linear", **selection["linear"])
    if method == "closed_zero": return dict(kind="linear", center=False, **selection["closed_zero"])
    if method == "hdc": return dict(kind="hdc", **selection["hdc"])
    if method == "hdc_zero": return dict(kind="hdc", center=False, **{**selection["hdc"], **selection["hdc_zero"]})
    if method == "hdc_bound": return dict(kind="bound", **selection["hdc"])
    raise ValueError(method)


def corrected_memory(native_record, task, kind, *, metadata=None):
    """Correct full-backbone retained-state scalar counts, without refitting.

Returns both conventions and their per-update differences. An existing
native_memory field makes this operation idempotent on validation records.
    Optional metadata has the feature-cache schema (including prefix lengths).
    Gradient workspaces and offline dataset/cache arrays are excluded; retained
    model copies, adaptation statistics, replay prefixes and fit masks count.
"""
    prior_kinds = {*METHODS, "hdc_shared"}
    gradient_kinds = {"finetune_full", "finetune_readout", "ewc", "replay",
                      "full_tag", "readout_tag", "last_tag"}
    if kind not in prior_kinds | gradient_kinds | {"frozen"}:
        raise ValueError("Expected one of the 14 original shared-backbone method names")
    if native_record.get("variant", "full") != "full" and metadata is None:
        raise ValueError("Non-full variants require explicit metadata with their model parameter count")
    if metadata is None:
        seed = native_record.get("seed", 101)
        path = HERE / "results/feature_controls/features" / task / "outer" / f"s{seed}" / "metadata.json"
        metadata = json.loads(path.read_text())
    selection = json.loads((STUDY / "results/selection" / f"{task}.json").read_text())
    outputs = len(metadata["target_mean"])
    prior_width = metadata["prior_width"]
    M = metadata["model_parameters"]
    L = prior_width * outputs
    H = (metadata["projection_width"] * 256 + 256 + 256 * 128 + 128 + L)
    dimension = selection["hdc"]["dimension"]
    native = list(native_record.get("native_memory", native_record["memory"]))
    prior_copy = L if kind in prior_kinds else 0
    shared_head = (prior_width + 2 * dimension) * outputs if kind == "hdc_shared" else 0
    redundant_base = M if kind in {"finetune_full", "ewc", "replay"} else 0
    unused_shared_head = H if kind in {"readout_tag", "last_tag"} else 0
    k = metadata["held_in"]
    if "names" in native_record:
        assert native_record["names"] == [s["name"] for s in metadata["sessions"]]
    components = []
    for t in range(1, len(native) + 1):
        replay_masks = sum(s["prefix_bins"] for s in metadata["sessions"][:k + t]) if kind == "replay" else 0
        components.append(dict(prior_copy=prior_copy, shared_head=shared_head,
                               bound_keys=t * dimension if kind == "hdc_bound" else 0,
                               redundant_pretrained_model=redundant_base,
                               unused_tagged_shared_head=unused_shared_head,
                               copied_frozen_readout=t * (H - L) if kind == "last_tag" else 0,
                               replay_fit_masks=replay_masks))
    added = [sum(row.values()) for row in components]
    return dict(native_memory=native, memory=[v + a for v, a in zip(native, added)],
                added_scalars=added, prior_copy_scalars=prior_copy,
                shared_head_scalars=shared_head,
                bound_key_scalars_per_session=dimension if kind == "hdc_bound" else 0,
                model_parameters=M, full_readout_parameters=H, last_layer_parameters=L,
                added_components=components,
                definition="Actually retained model copies/adaptation tensors and replay prefixes/fit masks; disposable offline dataset/feature cache and gradient/optimizer working space excluded")


def validate(task, seed, methods, native_methods, tolerance=1e-6):
    """Use identical cached float32 features; never read raw or hidden-test data."""
    engine.DEVICE = "cpu"
    cache = HERE / "results/feature_controls/features" / task / "outer" / f"s{seed}"
    meta = json.loads((cache / "metadata.json").read_text())
    cp = STUDY / "results/checkpoints" / task / "outer" / f"full_s{seed}.pt"
    checkpoint = torch.load(cp, map_location="cpu", weights_only=False)
    c = Config(**checkpoint["config"])
    model = Backbone(c)
    model.load_state_dict(checkpoint["state"])
    model.eval()
    del checkpoint
    selection = json.loads((STUDY / "results/selection" / f"{task}.json").read_text())
    shift, scale = np.asarray(meta["target_mean"], np.float32), np.asarray(meta["target_std"], np.float32)
    sessions, targets, fitting = [], [], []
    for row in meta["sessions"]:
        with np.load(cache / row["file"]) as f:
            sessions.append(SimpleNamespace(name=row["name"], gf=torch.from_numpy(f["score_gf"])))
            targets.append(f["score_y"])
            N = len(f["fit_gf"])
            fitting.append(SimpleNamespace(name=row["name"], gf=torch.from_numpy(f["fit_gf"]),
                                           yz=f["fit_y"], fit=np.ones(N, bool), fit_end=N))
    k, T = meta["held_in"], len(sessions) - meta["held_in"]
    rows = []
    for name in methods:
        started = time.perf_counter()
        method = CachedEfficient(**method_config(selection, name))
        method.start(model, [], c, seed)
        native = CachedNative(**method_config(selection, name)) if name in native_methods else None
        if native is not None: native.start(model, [], c, seed)
        R = np.empty((T + 1, len(sessions)))
        for j, s in enumerate(sessions):
            pred = method.predict(s) * scale + shift
            R[0, j] = engine.r2(targets[j], pred, np.ones(len(pred), bool))
        costs, memories, native_memories, max_prediction_difference = [], [], [], 0.0
        for t, j in enumerate(range(k, len(sessions)), 1):
            before = {key: value.clone() for key, value in method.heads.items()}
            cost = method.adapt(fitting[j])
            # Cached fitting rows omit the unmasked prefix bins, but the analytical
            # deployment cost includes the original complete recurrent prefix pass.
            prefix_adjustment = (meta["sessions"][j]["prefix_bins"] - fitting[j].fit_end) * sum(meta["backbone_macs"].values())
            cost += prefix_adjustment
            method.solver_records[-1]["prefix_bins"] = meta["sessions"][j]["prefix_bins"]
            for field in ["analytic_macs", "symmetric_analytic_macs", "legacy_primal_symmetric_macs", "dense_primal_macs"]:
                method.solver_records[-1][field] += int(prefix_adjustment)
            costs.append(int(cost)); memories.append(int(method.memory()))
            native_memories.append(int(engine.PriorMethod.memory(method)))
            if name != "hdc_bound":
                assert all(torch.equal(method.heads[key], value) for key, value in before.items())
            if native is not None: native.adapt(fitting[j])
            R[t] = R[t - 1]
            for q in (range(k, j + 1) if name == "hdc_bound" else [j]):
                pred_z = method.predict(sessions[q])
                if native is not None:
                    pred_native = native.predict(sessions[q])
                    difference = float(np.max(np.abs(pred_z - pred_native)))
                    max_prediction_difference = max(max_prediction_difference, difference)
                    assert difference <= tolerance, (task, seed, name, "prediction", difference)
                pred = pred_z * scale + shift
                R[t, q] = engine.r2(targets[q], pred, np.ones(len(pred), bool))
        rec = dict(task=task, seed=seed, method=name, variant="full", R=R.tolist(),
                   names=[s.name for s in sessions], held_in=k, adapt_macs=costs,
                   adapt_seconds=[0.0] * T, memory=memories, native_memory=native_memories,
                   start_macs=0, start_seconds=0)
        rec["summary"] = engine.summarize(rec)
        # These are correctness records, not adaptation timing measurements.
        rec.pop("adapt_seconds")
        rec.pop("start_seconds")
        rec["summary"].pop("adapt_seconds")
        reference_path = STUDY / "results/stream" / task / "full" / f"{name}_s{seed}.json"
        reference = json.loads(reference_path.read_text())
        matrix_error = float(np.max(np.abs(R - np.asarray(reference["R"]))))
        assert matrix_error <= tolerance, (task, seed, name, "R", matrix_error)
        if name != "hdc_bound": assert rec["summary"]["bwt"] == 0.0
        rec.update(solver_records=method.solver_records,
                   mac_counting="dense_gram_N*d*q",
                   max_abs_R_difference=matrix_error,
                   max_abs_native_prediction_difference=max_prediction_difference if native is not None else None,
                   native_prediction_checked=native is not None, tolerance=tolerance,
                   original_result_sha256=hashlib.sha256(reference_path.read_bytes()).hexdigest(),
                   checkpoint_sha256=meta["checkpoint_sha256"],
                   validation_seconds=time.perf_counter() - started,
                   memory_note="Retained model/adaptation scalars include the detached prior, bound keys, and shared head; exclude disposable offline feature cache. native_memory preserves the original counting convention.",
                   timing_note="validation_seconds includes cached-feature scoring and native reference solves; no deployment adaptation timing is reported here.")
        rows.append(rec)
        print(json.dumps(dict(task=task, seed=seed, method=name, max_abs_R_difference=matrix_error,
                              max_abs_native_prediction_difference=rec["max_abs_native_prediction_difference"],
                              seconds=rec["validation_seconds"])), flush=True)
        write(OUT / "verification" / f"{task}_s{seed}.json", rows)
    return rows


def aggregate():
    records = []
    for path in sorted((OUT / "verification").glob("*.json")):
        rows = json.loads(path.read_text())
        changed = False
        for row in rows:
            if "adapt_seconds" in row:
                row.pop("adapt_seconds")
                row.pop("start_seconds", None)
                row["summary"].pop("adapt_seconds", None)
                row["timing_note"] = "validation_seconds includes cached-feature scoring and native reference solves; no deployment adaptation timing is reported here."
                changed = True
            # Upgrade the counting convention of validations completed before the
            # retained-buffer audit. Predictions and fitted weights are untouched.
            if "native_memory" not in row:
                fixed = corrected_memory(row, row["task"], row["method"])
                row["native_memory"] = fixed["native_memory"]
                row["memory"] = fixed["memory"]
                row["summary"]["memory"] = row["memory"][-1]
                row["memory_note"] = "Retained model/adaptation scalars corrected to include detached prior and bound keys; excludes disposable offline feature cache. native_memory preserves original convention. No prediction recomputation needed."
                changed = True
            records.append(row)
        if changed: write(path, rows)
    out = {"analysis": "exact primal/dual implementation validation", "tolerance": 1e-6,
           "runs": len(records), "tasks": {}, "self_test": self_test(),
           "mac_counting": "dense_gram_N*d*q",
           "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    if not all(r.get("mac_counting") == "dense_gram_N*d*q" for r in records):
        raise RuntimeError("Run migrate-costs before aggregating historical symmetric-Gram validation records")
    for task in ["h1", "m1", "m2"]:
        out["tasks"][task] = {}
        for method in METHODS:
            rr = [r for r in records if r["task"] == task and r["method"] == method]
            if not rr: continue
            out["tasks"][task][method] = dict(seeds=[r["seed"] for r in rr],
                max_abs_R_difference=max(r["max_abs_R_difference"] for r in rr),
                max_abs_native_prediction_difference=max([r["max_abs_native_prediction_difference"] for r in rr if r["native_prediction_checked"]], default=None),
                acc=float(np.mean([r["summary"]["acc"] for r in rr])),
                bwt=float(np.mean([r["summary"]["bwt"] for r in rr])),
                mean_adapt_macs=float(np.mean([r["summary"]["adapt_macs"] for r in rr])),
                solvers=sorted({s["solver"] for r in rr for s in r["solver_records"]}))
    out["complete"] = all(set(out["tasks"][t].get(m, {}).get("seeds", [])) == set(SEEDS)
                          for t in ["h1", "m1", "m2"] for m in METHODS)
    write(OUT / "summary.json", out)
    print(json.dumps(out, indent=2))


def migrate_costs():
    """Change accounting only; preserve and hash every prediction/score field."""
    score_keys = ["acc", "acc_held_in", "acc_held_out", "plasticity", "bwt",
                  "bwt_held_in", "bwt_held_out", "zero_shot_held_out"]
    def score_hash(row):
        payload = dict(task=row["task"], seed=row["seed"], method=row["method"], R=row["R"],
                       scores={key: row["summary"][key] for key in score_keys},
                       max_abs_R_difference=row["max_abs_R_difference"],
                       max_abs_native_prediction_difference=row["max_abs_native_prediction_difference"])
        return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    audit, hdc_costs = [], {}
    for path in sorted((OUT / "verification").glob("*.json")):
        rows = json.loads(path.read_text())
        for row in rows:
            before = score_hash(row)
            task, seed, method = row["task"], row["seed"], row["method"]
            meta = json.loads((HERE / "results/feature_controls/features" / task / "outer" / f"s{seed}" / "metadata.json").read_text())
            selected = json.loads((STUDY / "results/selection" / f"{task}.json").read_text())
            cfg = method_config(selected, method)
            encoded = cfg["kind"] != "linear"
            dimension = cfg.get("dimension", 0) if encoded else 0
            d = meta["prior_width"] + 2 * dimension
            width = meta["prior_width"] - 1 if cfg.get("source", "penultimate") == "penultimate" else meta["projection_width"]
            encoding = width * dimension
            o, bb = len(meta["target_mean"]), sum(meta["backbone_macs"].values())
            costs, symmetric, legacy, primal = [], [], [], []
            session_costs = []
            for saved, session in zip(row["solver_records"], meta["sessions"][meta["held_in"]:]):
                n, N = session["prefix_bins"], session["fit_bins"]
                assert saved["N"] == N and saved["d"] == d and saved["q"] == min(N, d)
                args = (n, bb, N, d, o, encoding)
                dense = adaptation_macs(*args)
                sym = adaptation_macs(*args, symmetric=True)
                old = adaptation_macs(*args, q=d, symmetric=True)
                dense_primal = adaptation_macs(*args, q=d)
                saved.update(analytic_macs=dense, symmetric_analytic_macs=sym,
                             legacy_primal_symmetric_macs=old, dense_primal_macs=dense_primal,
                             gram_counting="dense_matrix_product")
                costs.append(dense); symmetric.append(sym); legacy.append(old); primal.append(dense_primal)
                session_costs.append(dict(name=session["name"], N=N, d=d, q=min(N, d),
                                          prefix_bins=n, efficient_macs=dense, primal_macs=dense_primal,
                                          efficient_symmetric_macs=sym, legacy_primal_symmetric_macs=old))
            row.update(adapt_macs=costs, adapt_macs_symmetric=symmetric,
                       adapt_macs_legacy_primal_symmetric=legacy,
                       adapt_macs_dense_primal=primal, mac_counting="dense_gram_N*d*q")
            row["summary"]["adapt_macs"] = float(np.mean(costs))
            after = score_hash(row)
            assert before == after
            audit.append(dict(task=task, seed=seed, method=method,
                              score_sha256_before=before, score_sha256_after=after,
                              score_invariance=True))
            if method == "hdc" and seed == 101:
                hdc_costs[task] = dict(task=task, mac_counting="dense_gram_N*d*q",
                    efficient_mean_macs=float(np.mean(costs)), primal_mean_macs=float(np.mean(primal)),
                    efficient_symmetric_mean_macs=float(np.mean(symmetric)),
                    legacy_primal_symmetric_mean_macs=float(np.mean(legacy)), sessions=session_costs)
        write(path, rows)
    write(OUT / "cost_migration.json", dict(change="Use dense Gram N*d*q; retain explicitly named symmetric and legacy provenance", records=audit,
                                           all_scores_unchanged=all(r["score_invariance"] for r in audit),
                                           source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
    write(OUT / "analytic_costs.json", [hdc_costs[t] for t in ["h1", "m1", "m2"] if t in hdc_costs])
    aggregate()


def memory_audit():
    """Audit every original main-table record; verify dimensions against the model."""
    out = {"definition": "Retained model/adaptation scalars and replay fit masks, excluding disposable gradient/optimizer workspace and offline evaluation data/cache", "tasks": {}}
    all_methods = ["frozen", "finetune_full", "finetune_readout", "ewc", "replay",
                   "full_tag", "readout_tag", "last_tag", *METHODS, "hdc_shared"]
    for task in ["h1", "m1", "m2"]:
        meta = json.loads((HERE / "results/feature_controls/features" / task / "outer/s101/metadata.json").read_text())
        model = Backbone(Config(channels=meta["projection_width"] - 128,
                                outputs=len(meta["target_mean"])))
        M = model.parameter_count()
        H = sum(p.numel() for p in model.readout.parameters())
        L = sum(p.numel() for p in model.readout[-1].parameters())
        assert M == meta["model_parameters"] and L == meta["prior_width"] * len(meta["target_mean"])
        task_out = dict(model_parameters=M, full_readout_parameters=H,
                        last_layer_parameters=L, methods={})
        for method in all_methods:
            rows = []
            for seed in SEEDS:
                path = STUDY / "results/stream" / task / "full" / f"{method}_s{seed}.json"
                original = json.loads(path.read_text())
                fixed = corrected_memory(original, task, method, metadata=meta)
                assert fixed["model_parameters"] == M and fixed["full_readout_parameters"] == H
                assert fixed["last_layer_parameters"] == L
                # Applying a correction twice must preserve the original ledger.
                again = corrected_memory({**original, "memory": fixed["memory"],
                                          "native_memory": fixed["native_memory"]},
                                         task, method, metadata=meta)
                assert fixed["memory"] == again["memory"]
                rows.append(dict(seed=seed, native_memory=fixed["native_memory"],
                                 memory=fixed["memory"], added_components=fixed["added_components"],
                                 source_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
            task_out["methods"][method] = dict(
                native_final_scalars=float(np.mean([r["native_memory"][-1] for r in rows])),
                corrected_final_scalars=float(np.mean([r["memory"][-1] for r in rows])),
                final_added_components=rows[0]["added_components"][-1], runs=rows)
        out["tasks"][task] = task_out
    write(OUT / "memory_accounting.json", out)
    print(json.dumps({t: {k: v for k, v in r.items() if k != "methods"}
                      for t, r in out["tasks"].items()}, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("stage", choices=["self-test", "validate", "aggregate", "memory-audit", "migrate-costs"])
    ap.add_argument("--tasks", nargs="+", default=["h1", "m1", "m2"])
    ap.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    ap.add_argument("--methods", nargs="+", choices=METHODS, default=METHODS)
    ap.add_argument("--native-methods", nargs="+", choices=METHODS, default=["hdc", "closed_linear"])
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    if args.stage == "self-test": print(json.dumps(self_test(), indent=2))
    elif args.stage == "memory-audit": memory_audit()
    elif args.stage == "migrate-costs": migrate_costs()
    elif args.stage == "aggregate": aggregate()
    else:
        self_test()
        for task in args.tasks:
            for seed in args.seeds: validate(task, seed, args.methods, args.native_methods)
        aggregate()
