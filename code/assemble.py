import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"
S = PAPER / "sections"


def read(name):
    return (S / name).read_text().strip() + "\n"


def split_backmatter():
    back = read("backmatter.md")
    main, bio = back.split("# Biography {-}")
    return main.strip() + "\n", bio.strip()


def frontmatter(bio):
    head = (PAPER / "trial.md").read_text().split("---")[1]
    head = re.sub(r"abstract: \|\n(?:  .*\n)+", "", head)
    head = re.sub(r"keywords: .*\n", "", head)
    abstract = read("abstract.md")
    bio = "biography: |\n" + "\n".join("  " + l if l else "" for l in bio.splitlines()) + "\n"
    return "---" + head.rstrip("\n") + "\n" + abstract + bio + "---\n"


def main():
    methods = read("methods.md").replace("<<THEORY>>", read("theory.md").strip())
    sel = ROOT / "results" / "tables" / "selection.md"
    if sel.exists():
        methods = methods.replace("<<TABLE:selection>>", sel.read_text().strip())
    back, bio = split_backmatter()
    body = [read("introduction.md"), read("literature.md"), methods, read("results.md"), read("discussion.md"), back]
    text = frontmatter(bio) + "\n" + "\n".join(body)
    left = sorted(set(re.findall(r"<<[^>]+>>", text)))
    (PAPER / "manuscript.md").write_text(text)
    print(PAPER / "manuscript.md", len(text.split()), "words", left)


if __name__ == "__main__":
    main()
