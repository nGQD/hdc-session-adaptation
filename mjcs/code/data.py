from pathlib import Path
import hashlib
import json
import re
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
DATA = REPO / "falcon-challenge" / "data"
CACHE = ROOT / "cache"
sys.path.insert(0, str(REPO / "falcon-challenge"))

DANDI = dict(h1="000954", m1="000941", m2="000953")
SPLITS = ("held-in-calib", "held-out-calib")
GAP = 256
PREFIX = 0.5
HELD_IN_TEST = 0.2


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, allow_nan=False) + "\n")
    tmp.replace(path)


def nwb_files(task, split):
    return sorted(p for p in (DATA / DANDI[task]).rglob("*.nwb") if f"{split}_" in p.name)


def session_of(task, path):
    tag = path.stem.split("ses-")[-1].split("_behavior")[0]
    if task == "h1":
        from falcon_challenge.config import H1_NEW_TO_OLD
        return H1_NEW_TO_OLD[tag].split("_set_")[0]
    return re.match(r"(\d{4}-?\d{2}-?\d{2})", tag).group(1).replace("-", "")


def build_cache(task):
    from falcon_challenge.config import FalconTask
    from falcon_challenge.dataloaders import load_nwb

    CACHE.mkdir(exist_ok=True)
    rows = []
    for split in SPLITS:
        for p in nwb_files(task, split):
            out = CACHE / f"{task}_{p.stem}.npz"
            if not out.exists():
                x, y, trial, mask = load_nwb(p, getattr(FalconTask, task))
                np.savez(out, x=np.asarray(x, np.float32), y=np.asarray(y, np.float32),
                         trial=np.asarray(trial, bool), mask=np.asarray(mask, bool))
            with np.load(out) as z:
                rows.append(dict(task=task, split=split, file=p.name, session=session_of(task, p),
                                 sha256=sha(p), cache=out.name, bins=int(len(z["x"])),
                                 channels=int(z["x"].shape[1]), outputs=int(z["y"].shape[1]),
                                 scored=int(z["mask"].sum())))
    write_json(ROOT / "results" / "provenance" / f"manifest_{task}.json", rows)
    return rows


def manifest(task):
    path = ROOT / "results" / "provenance" / f"manifest_{task}.json"
    return json.loads(path.read_text()) if path.exists() else build_cache(task)


def load(task, split):
    out = []
    for row in manifest(task):
        if row["split"] != split:
            continue
        with np.load(CACHE / row["cache"]) as z:
            out.append(dict(row, **{k: z[k].copy() for k in z.files}))
    return out


def by_session(records):
    keys = sorted({r["session"] for r in records}, key=lambda s: int(re.sub(r"\D", "", s)))
    return [(k, [r for r in records if r["session"] == k]) for k in keys]


def _concat(rs):
    return dict(x=np.concatenate([r["x"] for r in rs]), y=np.concatenate([r["y"] for r in rs]),
                mask=np.concatenate([r["mask"] for r in rs]),
                starts=np.cumsum([0] + [len(r["x"]) for r in rs[:-1]]))


def pretrain_portion(rs, test=HELD_IN_TEST):
    s = _concat(rs)
    cut = int(round(len(s["x"]) * (1 - test)))
    return [dict(x=s["x"][:cut], y=s["y"][:cut], mask=s["mask"][:cut])]


def held_in_split(rs, test=HELD_IN_TEST, gap=GAP):
    s = _concat(rs)
    n = len(s["x"])
    cut = int(round(n * (1 - test)))
    train = np.zeros(n, bool)
    train[:cut] = True
    score = s["mask"].copy()
    score[:cut + gap] = False
    return dict(s, train=train & s["mask"], score=score, fit_end=cut)


def held_out_split(rs, prefix=PREFIX, gap=GAP):
    s = _concat(rs)
    n = len(s["x"])
    prefix = int(round(n * prefix))
    adapt = np.zeros(n, bool)
    adapt[:prefix] = True
    score = s["mask"].copy()
    score[:prefix + gap] = False
    return dict(s, adapt=adapt & s["mask"], score=score, fit_end=prefix)


if __name__ == "__main__":
    for t in sys.argv[1:] or DANDI:
        rows = build_cache(t)
        print(t, len(rows), sorted({(r["split"], r["session"]) for r in rows}))
