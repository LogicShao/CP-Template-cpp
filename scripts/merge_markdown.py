from pathlib import Path
import re

CHAPTERS = [
    # "00_menu.md",  # 不包含自动生成的目录索引
    "01_basic_algorithms.md",
    "02_data_structures.md",
    "03_search.md",
    "04_dynamic_programming.md",
    "05_graph_theory.md",
    "06_math.md",
]

CHAPTERS_DIR = Path(__file__).resolve().parents[1] / "docs" / "chapters"
OUTPUT_MD = Path(__file__).resolve().parents[1] / "dist" / "notes.md"
OUTPUT_TEX = Path(__file__).resolve().parents[1] / "dist" / "content.tex"


def read_lines(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8").splitlines(keepends=True)


def escape_latex_special_chars(text: str) -> str:
    """转义LaTeX特殊字符"""
    # 需要转义的字符及其替换
    replacements = [
        ('\\', r'\textbackslash{}'),  # 反斜杠必须最先处理
        ('#', r'\#'),
        ('$', r'\$'),
        ('%', r'\%'),
        ('&', r'\&'),
        ('{', r'\{'),
        ('}', r'\}'),
        ('^', r'\^{}'),
        ('~', r'\~{}'),
    ]
    for char, replacement in replacements:
        text = text.replace(char, replacement)
    return text


def convert_markdown_to_latex(content: str) -> str:
    """
    将Markdown转换为LaTeX

    转换规则：
    - # 标题 → \section{标题}
    - ## 标题 → \subsection{标题}
    - ### 标题 → \subsubsection{标题}
    - #### 标题 → \paragraph{标题}
    - ##### 标题 → \subparagraph{标题}
    - ```cpp...``` → \begin{lstlisting}...\end{lstlisting}
    - 普通文本转义特殊字符
    """
    lines = content.split('\n')
    result = []
    in_code_block = False
    code_buffer = []

    for line in lines:
        # 检测代码块
        if line.strip().startswith('```'):
            if not in_code_block:
                # 开始代码块
                in_code_block = True
                code_buffer = []
                result.append('')
                result.append('\\begin{lstlisting}[language=C++]')
            else:
                # 结束代码块
                in_code_block = False
                result.extend(code_buffer)
                result.append('\\end{lstlisting}')
                result.append('')
        elif in_code_block:
            # 代码块内容 - 不转义
            code_buffer.append(line)
        else:
            # 标题转换 - 从最多的 # 开始检测，避免错误匹配
            if line.strip().startswith('#####'):
                title = line.strip()[5:].strip()
                result.append(f'\\subparagraph{{{escape_latex_special_chars(title)}}}')
            elif line.strip().startswith('####'):
                title = line.strip()[4:].strip()
                result.append(f'\\paragraph{{{escape_latex_special_chars(title)}}}')
            elif line.strip().startswith('###'):
                title = line.strip()[3:].strip()
                result.append(f'\\subsubsection{{{escape_latex_special_chars(title)}}}')
            elif line.strip().startswith('##'):
                title = line.strip()[2:].strip()
                result.append(f'\\subsection{{{escape_latex_special_chars(title)}}}')
            elif line.strip().startswith('#'):
                title = line.strip()[1:].strip()
                result.append(f'\\section{{{escape_latex_special_chars(title)}}}')
            else:
                # 普通文本行 - 转义特殊字符
                result.append(escape_latex_special_chars(line))

    return '\n'.join(result)


def merge_markdown(files: list[str]) -> str:
    merged = []
    for name in files:
        merged.extend(read_lines(CHAPTERS_DIR / name))
        if merged and not merged[-1].endswith("\n"):
            merged[-1] += "\n"
        merged.append("\n")
    return "".join(merged)


if __name__ == "__main__":
    OUTPUT_MD.parent.mkdir(parents=True, exist_ok=True)

    # 合并所有章节
    content = merge_markdown(CHAPTERS)

    # 生成Markdown版本（保留原有功能）
    with open(OUTPUT_MD, "w", encoding="utf-8", newline="\n") as f:
        f.write(content)
    print(f"已生成 {OUTPUT_MD}")

    # 转换为LaTeX并生成content.tex
    latex_content = convert_markdown_to_latex(content)
    with open(OUTPUT_TEX, "w", encoding="utf-8", newline="\n") as f:
        f.write(latex_content)
    print(f"已生成 {OUTPUT_TEX}（LaTeX格式）")
