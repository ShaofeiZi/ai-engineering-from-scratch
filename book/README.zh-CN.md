# 图书构建流水线

本课程可编译为一套六卷本图书。图书是配套读物,而非替代品:交互式图表、分级测验和可运行代码仍保留在网站与本仓库中,每章结尾都会给出引导读者前往这些内容的链接。

## 分卷

分卷定义在 `volumes.json` 中。每一卷对应一个阶段区间:

| 卷 | 书名 | 阶段 |
|-----|-------|--------|
| 1 | 基础 | 00-02 |
| 2 | 深度学习 | 03、04、06 |
| 3 | 语言 | 05、07 |
| 4 | 大语言模型 | 08-11 |
| 5 | 智能体 | 12-16 |
| 6 | 生产落地 | 17-19 |

## 构建

```bash
python3 scripts/build_book.py                  # 全部卷,EPUB
python3 scripts/build_book.py --volume language
python3 scripts/build_book.py --pdf            # 额外生成 PDF(需要 xelatex + DejaVu 字体)
```

依赖 pandoc。可选:`@mermaid-js/mermaid-cli`(mmdc)用于将 mermaid 图渲染为图片;若未安装,这些图会变为网页版指向链接。输出产物位于 `dist/book/`。

CI(`.github/workflows/build-book.yml`)会在每次触及 `phases/` 的推送上构建 EPUB,并在发布时构建 EPUB + PDF,将两者附加到发布资产中。

## 汇编器对每节课的处理

- 课程的 `# 标题` 会成为一章;各阶段会成为不带编号的分部页。
- `figure` 块(交互式 JS 控件)会变为指向该课网页版的框注链接。
- Mermaid 块在 mmdc 可用时渲染为 SVG,否则变为网页指向链接。
- `## Ship It` 小节会被替换为指向仓库产物的链接。
- `## Exercises` 小节会增加一个指向该课 `code/` 目录的起步代码链接。
- 每章结尾都有一个"继续在线学习"框:网页版、代码、测验。
- 资源图片路径会被改写,以便 pandoc 内嵌各课程的 SVG。

智能体用于导航本课程的机器可读索引(`site/llms.txt`,在部署时由 `site/build.js` 生成)链接了每节课的原始 Markdown,而图书的"与 AI 共学"首页会告诉读者如何让自己的助手指向该索引。
