import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INCLUDE = ("code", "results", "paper")
SKIP_DIRS = {"__pycache__", "checkpoints", "cache", "template_unpacked"}
SKIP_SUFFIX = {".pt", ".pth", ".ckpt", ".pyc", ".npz", ".nwb", ".tmp", ".aux", ".log", ".out", ".toc"}
OUT = ROOT / "MANIFEST.sha256"


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def files():
    for top in INCLUDE:
        for p in sorted((ROOT / top).rglob("*")):
            rel = p.relative_to(ROOT)
            if p.is_file() and not SKIP_DIRS & set(rel.parts) and p.suffix not in SKIP_SUFFIX:
                yield rel


def main():
    lines = [f"{digest(ROOT / rel)}  {rel.as_posix()}\n" for rel in files()]
    OUT.write_text("".join(lines))
    print(f"{len(lines)} files")


def check():
    bad = 0
    listed = set()
    for line in OUT.read_text().splitlines():
        h, name = line.split("  ", 1)
        listed.add(name)
        if not (ROOT / name).is_file() or digest(ROOT / name) != h:
            print("mismatch", name)
            bad += 1
    for rel in files():
        if rel.as_posix() not in listed:
            print("unlisted", rel.as_posix())
            bad += 1
    return bad


if __name__ == "__main__":
    sys.exit(check() if sys.argv[1:] == ["--check"] else main())
