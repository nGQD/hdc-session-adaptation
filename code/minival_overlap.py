import glob
import json

import numpy as np
from pynwb import NWBHDF5IO

import data


def series(path):
    with NWBHDF5IO(path, "r") as io:
        a = io.read().acquisition["OpenLoopKinematics"]
        t = a.timestamps[:] if a.timestamps is not None else a.starting_time + np.arange(a.data.shape[0]) / a.rate
        return np.asarray(a.data[:]), np.asarray(t)


def main():
    rows = []
    for f in sorted(glob.glob(str(data.DATA / "000954" / "sub-HumanPitt-held-in-minival" / "*.nwb"))):
        x1, t1 = series(f)
        x2, t2 = series(f.replace("minival", "calib"))
        i = np.minimum(np.searchsorted(t2, t1), len(t2) - 1)
        same = (t2[i] == t1) & np.all(np.isclose(x1, x2[i]), 1)
        rows.append(dict(file=f.split("/")[-1], bins=int(len(t1)), duplicated=int(same.sum())))
    out = dict(files=rows, bins=sum(r["bins"] for r in rows), duplicated=sum(r["duplicated"] for r in rows))
    data.write_json(data.ROOT / "results" / "provenance" / "minival_overlap.json", out)
    print(out["bins"], out["duplicated"])


if __name__ == "__main__":
    main()
