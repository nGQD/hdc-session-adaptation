import hashlib
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE / "fixtures"))

import figures
import make_fixture

NAMES = ["f1_per_session", "f2_heatmaps", "f3_pareto", "f3_pareto_3d", "f4_ablation", "f5_cost", "f6_bound", "f7_dose", "f8_epochs"]


def digest(out):
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(out.iterdir())}


def test_render_all_and_deterministic():
    data = HERE / "fixtures" / "data"
    shutil.rmtree(data, ignore_errors=True)
    make_fixture.build(data)
    runs = []
    for i in range(2):
        out = HERE / "fixtures" / f"out{i}"
        shutil.rmtree(out, ignore_errors=True)
        figures.main(data, out)
        runs.append(digest(out))
    for n in NAMES:
        for ext in ("png", "pdf"):
            assert f"{n}.{ext}" in runs[0]
    assert runs[0] == runs[1]


if __name__ == "__main__":
    test_render_all_and_deterministic()
    print("ok")
