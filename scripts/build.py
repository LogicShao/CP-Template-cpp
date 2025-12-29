from pathlib import Path
import runpy

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = Path(__file__).resolve().parent


def run_script(name: str) -> None:
    runpy.run_path(str(SCRIPTS_DIR / name), run_name="__main__")


if __name__ == "__main__":
    run_script("make_menu.py")
    run_script("merge_markdown.py")
    run_script("export_latex.py")
