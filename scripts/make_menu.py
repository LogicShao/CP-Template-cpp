from pathlib import Path

CHAPTERS = [
    "01_basic_algorithms.md",
    "02_data_structures.md",
    "03_search.md",
    "04_dynamic_programming.md",
    "05_graph_theory.md",
    "06_math.md",
]
OUTPUT = "00_menu.md"
CHAPTERS_DIR = Path(__file__).resolve().parents[1] / "docs" / "chapters"


def file_lines(path: Path) -> list[tuple[int, str]]:
    res = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) > 1 and parts[0].count("#") == len(parts[0]):
            res.append((len(parts[0]), " ".join(parts[1:])))
    return res


def make_menu(path: Path) -> str:
    title = path.stem[3:]
    res = f"## [{title}](./{path.name})\n\n"
    headings = file_lines(path)
    if not headings:
        return res + "\n"
    base = min(dep for dep, _ in headings)
    for dep, title in headings:
        indent = "    " * (dep - base)
        res += f"{indent}* [{title}](./{path.name}#{title})\n"
    return res + "\n"


if __name__ == "__main__":
    output_path = CHAPTERS_DIR / OUTPUT
    with open(output_path, "w", encoding="utf-8", newline="\n") as f:
        for name in CHAPTERS:
            f.write(make_menu(CHAPTERS_DIR / name))
