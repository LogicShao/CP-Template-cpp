from __future__ import annotations

from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
DIST_DIR = ROOT / "dist"
TEX_FILE = DIST_DIR / "notes.tex"


def parse_args(argv: list[str]) -> tuple[str, bool]:
    engine = "xelatex"
    use_latexmk = False
    for arg in argv:
        if arg == "--latexmk":
            use_latexmk = True
        elif arg.startswith("--engine="):
            engine = arg.split("=", 1)[1].strip()
        else:
            raise ValueError(f"Unknown аргумент: {arg}")
    return engine, use_latexmk


def latexmk_args(engine: str) -> list[str]:
    if engine == "xelatex":
        return ["latexmk", "-xelatex", "-shell-escape", "notes.tex"]
    if engine == "lualatex":
        return ["latexmk", "-lualatex", "-shell-escape", "notes.tex"]
    if engine == "pdflatex":
        return ["latexmk", "-pdf", "-shell-escape", "notes.tex"]
    raise ValueError(f"Unsupported engine for latexmk: {engine}")


def main() -> int:
    if not TEX_FILE.exists():
        print("notes.tex not found. Run scripts/export_latex.py first.")
        return 2

    try:
        engine, use_latexmk = parse_args(sys.argv[1:])
    except ValueError as exc:
        print(str(exc))
        print("Usage: python scripts/compile_latex.py [--latexmk] [--engine=xelatex]")
        return 2

    if use_latexmk:
        cmd = latexmk_args(engine)
    else:
        cmd = [engine, "--shell-escape", "notes.tex"]

    print("Running:", " ".join(cmd))
    result = subprocess.run(cmd, cwd=DIST_DIR)
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
