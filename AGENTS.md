# Repository Guidelines

## 项目结构与模块组织
本仓库以笔记与模板代码为主。核心内容在 `docs/chapters/`（分章节 Markdown），C++ 模板集中在 `cpp/`。导出产物放在 `dist/`，其中 `dist/notes.md` 与 `dist/notes.tex` 为合并与 LaTeX 输出。构建脚本位于 `scripts/`，例如 `scripts/build.py` 用于一键执行完整流程。

## 构建、测试与开发命令
常用命令如下（均在仓库根目录运行）：`python scripts/make_menu.py` 生成目录索引，`python scripts/merge_markdown.py` 合并章节，`python scripts/export_latex.py` 生成 `dist/notes.tex`，`python scripts/build.py` 一键执行。LaTeX 编译示例：`cd dist` 后执行 `lualatex --shell-escape notes.tex`。

## 编码风格与命名约定
文件与目录统一英文命名，采用小写加下划线（例：`01_basic_algorithms.md`、`basic_algorithms.cpp`）。缩进建议：C++ 与 Python 均使用 4 空格，Markdown 标题使用 `#` 级别分层。文本编码统一 UTF-8，行尾使用 LF。

## 测试指南
当前仓库未集成测试框架与覆盖率要求。若新增脚本或算法实现，建议提供最小可运行示例或简单输入输出用例，并在 README 中注明运行方式。

## 提交与拉取请求规范
当前环境无法读取 Git 历史，未发现明确的提交格式约定。建议提交信息使用简短动词 + 范围，例如 `docs: update graph theory notes` 或 `scripts: add latex export`。PR 描述应包含变更动机、影响范围与运行结果（如脚本输出或截图）。

## 额外说明
LaTeX 模板位于 `docs/latex/template.tex`，脚本会替换占位符并输出 `dist/notes.tex`。若调整章节顺序，请同步更新 `scripts/make_menu.py` 与 `scripts/merge_markdown.py` 的章节列表。
