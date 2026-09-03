# 贡献指南

欢迎提交课程、翻译、修复和产出物。每个 Pull Request 只包含一项贡献，这样评审更快，贡献者人数和署名统计也能正确运作。

## 重要：README 和 ROADMAP 会喂给网站

`site/build.js` 会解析 `README.md`、`ROADMAP.md` 和 `glossary/terms.md` 来生成 `site/data.js`。任何涉及这些文件的 Pull Request 都必须保持以下两种格式不变：

- 阶段标题，要么写成 `### Phase N: Name \`X lessons\`` 的形式，要么写成 `<details><summary><b>Phase N — Name</b> ... <code>X lessons</code> ... <em>Description</em></summary>` 的形式。
- 课程表格的列结构为 `| # | Lesson | Type | Lang |`（capstone 表格则用 `| # | Project | Combines | Lang |`）。`Lang` 列接受纯文本（`Python, TypeScript`）或旧版的 emoji 旗帜（`🐍 🟦 🦀 🟣 ⚛️`），二者对解析器是等价的。
- ROADMAP 的状态符号（`✅`、`🚧`、`⬚`）用在阶段标题和课程行上。不要用文字替换它们——解析器是靠这些精确字符来识别的。

编辑这些文件后运行 `node site/build.js`；如果你的改动在结构上是安全的，`git diff site/data.js` 应当只显示时间戳的变化。

## 贡献方式

### 1. 添加新课程

每门课程位于 `phases/XX-phase-name/NN-lesson-name/`，结构如下：

```text
NN-lesson-name/
├── code/           至少一个可运行的实现
├── notebook/       用于实验的 Jupyter notebook（可选）
├── docs/
│   └── en.md       课程文档（必需）
└── outputs/        本课程产出的 prompt、skill 或 agent（如适用）
```

**课程文档格式**（`en.md`）：

```markdown
# Lesson Title

> One-line motto — the core idea in one sentence.

## The Problem

这为什么重要？如果不了解它，你将无法完成什么？

## The Concept

先通过图表、视觉材料和直觉解释，稍后再引入代码。

## Build It

从零开始，逐步完成实现。

## Use It

现在使用真实框架或库完成同样的工作。

## Ship It

本课程产出的提示词、技能、智能体或工具。

## Exercises

1. Exercise one
2. Exercise two
3. Challenge exercise
```

### 2. 添加翻译

英文课程文件保留在 `main` 分支上。课程翻译请发布到仓库配置的 `translations` 分支，使用如下布局：

```text
i18n/<lang>/phases/<phase>/<lesson>/docs/<lang>.md
```

保持与英文源文件相同的结构，翻译正文，不要翻译代码。翻译必须包含 [`docs/i18n.md`](docs/i18n.md) 中描述的缓存来源信息。简体中文（`zh`）由人工评审和维护，请勿对其运行机器翻译 CLI。落地页翻译（如 `i18n/zh/README.md`）是例外，仍保留在 `main` 分支上。

### 3. 添加产出物

如果一门课程应当产出一个可复用的 prompt、skill、agent 或 MCP server：

1. 在该课程的 `outputs/` 文件夹中创建它
2. 在顶层的 `outputs/` 索引中添加引用

**Prompt 格式：**

```markdown
---
name: prompt-name
description: 此提示词的用途
phase: 14
lesson: 01
---

[System prompt or template here]
```

**Skill 格式：**

```markdown
---
name: skill-name
description: 此技能教授的内容
version: 1.0.0
phase: 14
lesson: 01
tags: [agents, loops]
---

[Skill content here]
```

### 4. 修复 Bug 或改进现有课程

- 修复无法运行的代码
- 改进讲解
- 添加更好的图表
- 更新过时的信息

### 5. 添加练习或项目

始终欢迎更多练习和项目，尤其是能串联多个阶段的那些。

## 准则

- **代码必须能运行。** 每个代码文件都应在所列依赖下无报错地执行。
- **注释要有用。** 每个 `code/main.*` 文件都需要那 4–6 行的课程/来源头部；除此之外，优先让代码自解释，把详细的教学内容放在文档里。
- **为任务选最合适的语言。** 当 TypeScript 或 Rust 更合适时，不要硬用 Python。
- **先从零实现。** 在展示框架版本之前，始终先从第一性原理实现该概念。
- **保持实用。** 理论服务于实践，而不是反过来。
- **不要 AI 腔。** 像人一样写。直截了当。砍掉废话。

## Pull Request 流程

1. Fork 本仓库
2. 创建功能分支（`git checkout -b add-lesson-phase3-gradient-descent`）
3. 做出修改
4. 确保所有代码都能运行
5. 提交 Pull Request，并附上清晰的描述

## 行为准则

见 [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md)。友善、乐于助人、富有建设性。

## 风格

- 行文直接。砍掉废话。匹配手册的语气，而非营销文案。
- 标题中不要装饰性 emoji。Lang 列的 emoji 旗帜是唯一例外，且仅因解析器要靠它们映射。
- 代码在课程所列依赖下能直接运行。
- 先从零实现，再讲框架。
