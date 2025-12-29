# CP-Template-cpp

个人算法竞赛模板与笔记整理，包含 C++ 模板代码、分章节的 Markdown 笔记，以及 LaTeX 导出模板。

## 目录结构

- `cpp/`：C++ 算法模板代码（示例：`cpp/basic_algorithms.cpp`）。
- `docs/chapters/`：按主题拆分的 Markdown 笔记。
- `docs/latex/`：LaTeX 模板（用于导出整合文档）。
- `dist/`：合并后的输出文件（`notes.md`、`notes.tex`、`template.pdf`）。
- `scripts/`：构建与导出脚本。

## 常用命令

```bash
python scripts/make_menu.py
python scripts/merge_markdown.py
python scripts/export_latex.py
python scripts/build.py
```

- `make_menu.py`：生成 `docs/chapters/00_menu.md` 的目录索引。
- `merge_markdown.py`：合并章节为 `dist/notes.md`。
- `export_latex.py`：基于模板生成 `dist/notes.tex`。
- `build.py`：一键执行以上流程。

## LaTeX 导出

`dist/notes.tex` 使用 `markdown` 包解析 Markdown。建议使用 `lualatex` 编译：

```bash
cd dist
lualatex --shell-escape notes.tex
```

如需完整 TeX 发行版，请安装包含 `ctex` 与 `markdown` 的 TeX Live。
