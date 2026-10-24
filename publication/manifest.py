"""Write or verify checksums for the publication files."""
import argparse
import hashlib
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "MANIFEST.sha256"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    if args.write:
        listed = subprocess.check_output(
            ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
            cwd=ROOT,
        ).decode().split("\0")
        names = sorted({name for name in listed if name and name != MANIFEST.name
                        and (ROOT / name).is_file()})
        MANIFEST.write_text("".join(f"{digest(ROOT / name)}  {name}\n" for name in names))
    count = 0
    for line in MANIFEST.read_text().splitlines():
        expected, name = line.split("  ", 1)
        path = ROOT / name
        if not path.is_file() or digest(path) != expected:
            raise RuntimeError(f"Checksum mismatch: {name}")
        count += 1
    print(f"Verified {count} publication files")


if __name__ == "__main__":
    main()
