from pathlib import Path

CHAPTERS = [
    "00_menu.md",
    "01_basic_algorithms.md",
    "02_data_structures.md",
    "03_search.md",
    "04_dynamic_programming.md",
    "05_graph_theory.md",
    "06_math.md",
]

CHAPTERS_DIR = Path(__file__).resolve().parents[1] / "docs" / "chapters"
OUTPUT_PATH = Path(__file__).resolve().parents[1] / "dist" / "notes.md"


def read_lines(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8").splitlines(keepends=True)


def merge_markdown(files: list[str]) -> str:
    merged = []
    for name in files:
        merged.extend(read_lines(CHAPTERS_DIR / name))
        if merged and not merged[-1].endswith("\n"):
            merged[-1] += "\n"
        merged.append("\n")
    return "".join(merged)


if __name__ == "__main__":
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    content = merge_markdown(CHAPTERS)
    with open(OUTPUT_PATH, "w", encoding="utf-8", newline="\n") as f:
        f.write(content)
