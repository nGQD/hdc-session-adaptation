import argparse
import re
import shutil
import struct
import subprocess
import sys
import tempfile
import zipfile
import zlib
from pathlib import Path

PNG_SIG = b"\x89PNG\r\n\x1a\n"
PNG_DROP = {b"tEXt", b"zTXt", b"iTXt", b"tIME", b"eXIf"}
FIXED = "2000-01-01T00:00:00Z"
ZIP_TIME = (2000, 1, 1, 0, 0, 0)
CORE = (
    '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
    '<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" '
    'xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:dcterms="http://purl.org/dc/terms/" '
    'xmlns:dcmitype="http://purl.org/dc/dcmitype/" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">'
    "{title}<dc:creator>{authors}</dc:creator><cp:lastModifiedBy>{authors}</cp:lastModifiedBy>"
    '<dcterms:created xsi:type="dcterms:W3CDTF">{t}</dcterms:created>'
    '<dcterms:modified xsi:type="dcterms:W3CDTF">{t}</dcterms:modified>'
    "</cp:coreProperties>"
)
APP_DROP = re.compile(rb"<(Application|AppVersion|Company|TotalTime)>.*?</\1>", re.S)


def strip_png(path):
    data = Path(path).read_bytes()
    if data[:8] != PNG_SIG:
        raise ValueError(f"{path}: not a PNG")
    out, pos = [PNG_SIG], 8
    while pos < len(data):
        n, kind = struct.unpack(">I4s", data[pos:pos + 8])
        end = pos + 12 + n
        if kind not in PNG_DROP:
            out.append(data[pos:end])
        pos = end
    Path(path).write_bytes(b"".join(out))


def strip_pdf(path):
    path = Path(path)
    try:
        import pikepdf
    except ImportError:
        pikepdf = None
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d) / "out.pdf"
        if pikepdf is not None:
            with pikepdf.open(path) as pdf:
                with pdf.open_metadata(set_pikepdf_as_editor=False) as meta:
                    for key in list(meta.keys()):
                        del meta[key]
                for key in list(pdf.docinfo.keys()):
                    del pdf.docinfo[key]
                if "/Metadata" in pdf.Root:
                    del pdf.Root["/Metadata"]
                pdf.save(tmp, deterministic_id=True, static_id=True)
        elif shutil.which("qpdf"):
            subprocess.run(["qpdf", "--static-id", "--deterministic-id", "--empty", "--pages", str(path), "--", str(tmp)], check=True)
            subprocess.run(["qpdf", "--static-id", "--deterministic-id", "--replace-input", str(tmp)], check=True)
        else:
            raise RuntimeError("pikepdf or qpdf is required")
        path.write_bytes(tmp.read_bytes())


def strip_docx(path, authors):
    path = Path(path)
    with zipfile.ZipFile(path) as src:
        old = src.read("docProps/core.xml").decode() if "docProps/core.xml" in src.namelist() else ""
    title = re.search(r"<dc:title>.*?</dc:title>", old, re.S)
    core = CORE.format(title=title.group(0) if title else "", authors=authors, t=FIXED).encode()
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d) / "out.docx"
        with zipfile.ZipFile(path) as src, zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as dst:
            names = src.namelist()
            for name in names:
                body = src.read(name)
                if name == "docProps/core.xml":
                    body = core
                elif name == "docProps/app.xml":
                    body = APP_DROP.sub(b"", body)
                info = zipfile.ZipInfo(name, ZIP_TIME)
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = 0o644 << 16
                dst.writestr(info, body)
            if "docProps/core.xml" not in names:
                raise ValueError(f"{path}: missing docProps/core.xml")
        path.write_bytes(tmp.read_bytes())


HANDLERS = {".png": lambda p, a: strip_png(p), ".pdf": lambda p, a: strip_pdf(p), ".docx": strip_docx}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--authors", default="Anonymous")
    a = ap.parse_args()
    for arg in a.paths:
        root = Path(arg)
        for p in sorted(root.rglob("*") if root.is_dir() else [root]):
            if p.is_file() and p.suffix.lower() in HANDLERS:
                HANDLERS[p.suffix.lower()](p, a.authors)
                print("stripped", p)


if __name__ == "__main__":
    sys.exit(main())
