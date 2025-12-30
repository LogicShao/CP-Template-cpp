# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

这是一个算法竞赛（Competitive Programming）模板与笔记仓库，包含：
- C++ 算法模板代码
- 按主题组织的 Markdown 笔记文档
- LaTeX 导出与编译系统（使用预处理方案）
- 自动化构建脚本

核心目标：维护个人算法竞赛知识库，并支持导出为 PDF 格式的参考手册。

## 目录结构

```
cpp/                  # C++ 算法模板实现
docs/
  chapters/           # 章节化 Markdown 笔记（主要工作目录）
    00_menu.md        # 自动生成的目录索引
    01_*.md           # 各主题章节文件
  latex/
    template.tex      # LaTeX 导出模板
dist/                 # 构建输出目录
  notes.md            # 合并后的完整 Markdown
  content.tex         # 预处理后的 LaTeX 内容
  notes.tex           # 最终生成的 LaTeX 文件
  notes.pdf           # 最终 PDF 文档
scripts/              # Python 构建脚本
```

## 章节管理与构建流程

### 章节顺序

章节顺序由 `scripts/make_menu.py` 和 `scripts/merge_markdown.py` 中的 `CHAPTERS` 列表控制：

```python
CHAPTERS = [
    # "00_menu.md",  # 不包含在导出中
    "01_basic_algorithms.md",
    "02_data_structures.md",
    "03_search.md",
    "04_dynamic_programming.md",
    "05_graph_theory.md",
    "06_math.md",
]
```

**重要：** 新增章节或调整顺序时，必须同步更新这两个文件中的 `CHAPTERS` 列表。

### 构建命令

在仓库根目录执行：

```bash
# 一键完整构建（推荐）
python scripts/build.py

# 或分步执行
python scripts/make_menu.py        # 生成 00_menu.md 目录索引
python scripts/merge_markdown.py   # 合并章节并转换为 LaTeX
python scripts/export_latex.py     # 生成最终的 notes.tex
```

### LaTeX 编译

**编译工作流程（方案2 - Markdown 预处理）：**

本项目使用预处理方案：
1. `merge_markdown.py` 将 Markdown 转换为 LaTeX (`content.tex`)
2. `export_latex.py` 将 `content.tex` 嵌入模板生成 `notes.tex`
3. 直接编译 LaTeX 文件，无需 markdown 包的运行时处理

```bash
# 进入输出目录
cd dist

# 使用 xelatex 编译
xelatex notes.tex

# 需要运行两次以生成完整目录和交叉引用
xelatex notes.tex
```

**编译脚本：**
```bash
# 使用默认引擎（xelatex）
python scripts/compile_latex.py
```

**依赖要求：**
- 需要安装 TeX Live 发行版，包含以下宏包：
  - `ctex`（中文支持）
  - `listings`（代码高亮）
  - `hyperref`（超链接和目录）
  - `xcolor`（颜色支持）
  - `geometry`（页面布局）
  - `tocloft`（目录格式）
  - `underscore`（下划线字符支持）
- **不再需要 `--shell-escape` 选项**（已移除 markdown 包依赖）

## LaTeX 模板机制

### 预处理转换规则

`merge_markdown.py` 中的 `convert_markdown_to_latex()` 函数执行以下转换：

**标题转换：**
- `#` → `\section{}`
- `##` → `\subsection{}`
- `###` → `\subsubsection{}`
- `####` → `\paragraph{}`
- `#####` → `\subparagraph{}`

**代码块转换：**
- ` ```cpp ... ``` ` → `\begin{lstlisting}[language=C++] ... \end{lstlisting}`

**特殊字符转义：**
自动转义 LaTeX 特殊字符（`#`, `$`, `%`, `&`, `^`, `~`, `{`, `}`, `\`）以避免编译错误。

### 模板占位符

`docs/latex/template.tex` 中使用 `{{CONTENT}}` 作为占位符：

```latex
\input{{CONTENT}}
```

`export_latex.py` 会将其替换为：

```latex
\input{content.tex}
```

**代码高亮：** 使用 `listings` 包配置 C++ 语法高亮和自定义样式（背景色、边框等）。

## 编码规范

- **文件命名：** 小写 + 下划线（`basic_algorithms.cpp`、`01_basic_algorithms.md`）
- **缩进：** 4 空格（C++ 和 Python）
- **编码：** UTF-8
- **行尾：** LF（Unix 风格）
- **Markdown 标题：** 使用 `#` 层级（最多支持5级标题）

## 修改章节内容的工作流程

1. 编辑 `docs/chapters/` 中的相应 `.md` 文件
2. 运行 `python scripts/build.py` 更新输出
3. （可选）运行 `cd dist && xelatex notes.tex` 生成 PDF（需运行两次）

## 添加新章节的工作流程

1. 在 `docs/chapters/` 创建新文件（遵循命名格式 `0X_topic_name.md`）
2. **同步更新章节列表：**
   - 编辑 `scripts/make_menu.py` 的 `CHAPTERS` 列表
   - 编辑 `scripts/merge_markdown.py` 的 `CHAPTERS` 列表
3. 运行 `python scripts/build.py` 重新生成输出

## 目录索引生成规则

`make_menu.py` 从每个章节文件中提取前 3 级标题（`MAX_LEVELS = 3`）生成 `00_menu.md`：

- 章节文件名去除 `0X_` 前缀并转换为标题格式
- 文件内标题按层级缩进（4 空格/级）
- 空行和非标题内容会被忽略

## 注意事项

- `00_menu.md` 是自动生成的，不要手动编辑
- `dist/` 目录下的所有文件都是构建产物，不要直接修改
- 修改 `docs/latex/template.tex` 后需重新运行 `export_latex.py`
- LaTeX 编译错误通常与缺失宏包有关，确保安装完整的 TeX Live
- Markdown 中的 LaTeX 特殊字符会被自动转义，无需手动处理
