from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = Path(__file__).resolve().parent

sys.path.insert(0, str(SCRIPTS_DIR))
from merge_markdown import CHAPTERS, merge_markdown  # noqa: E402

DIST_DIR = ROOT / "dist"
TEMPLATE_PATH = ROOT / "docs" / "latex" / "template.tex"
OUTPUT_MD = DIST_DIR / "notes.md"
OUTPUT_TEX = DIST_DIR / "notes.tex"


def ensure_notes_md() -> None:
    if OUTPUT_MD.exists():
        return
    DIST_DIR.mkdir(parents=True, exist_ok=True)
    content = merge_markdown(CHAPTERS)
    with open(OUTPUT_MD, "w", encoding="utf-8", newline="\n") as f:
        f.write(content)


def build_tex() -> None:
    template = TEMPLATE_PATH.read_text(encoding="utf-8")
    if "{{CONTENT}}" not in template:
        raise RuntimeError("template.tex missing {{CONTENT}} placeholder")
    tex = template.replace("{{CONTENT}}", OUTPUT_MD.name)
    with open(OUTPUT_TEX, "w", encoding="utf-8", newline="\n") as f:
        f.write(tex)


if __name__ == "__main__":
    ensure_notes_md()
    build_tex()
