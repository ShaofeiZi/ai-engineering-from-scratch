---
name: start-learning
version: 1.0.0
description: >
  面向"从零开始学 AI 工程"课程（523 节课，20 个阶段）的一次性入门引导。
  对学习者进行访谈，运行分级测验，并生成 LEARNING.md——一个由 learn 技能
  驱动的持久化学习计划。触发短语："start learning"、"set up the course"、
  "begin the curriculum"、"onboard me"、"create my learning plan"
tags: [onboarding, curriculum, ai-engineering, learning-plan]
---

# 开始学习

你正在引导一位学习者进入 **从零开始学 AI 工程** 课程：共 523 节课、20 个阶段，内容从线性代数一直到自主智能体。你的任务是生成 `LEARNING.md`，一个位于当前目录下的文件，记录他们为什么学习、应该从哪里开始，以及他们的学习路径长什么样。此后每次 `learn` 会话都会读取并更新这个文件，因此请把它当作学习者的唯一事实来源。

适用于任何代理。如果你的环境有结构化的问答工具，请对每个问题使用该工具；否则以纯文本形式展示带字母编号的选项并等待回复。

## 宿主调用约定

技能名称是可移植的，但调用语法属于宿主。在展示下一条命令之前，请使用正确的形式：

- Codex：使用 `start-learning`、`learn`、`course-guide` 等 `skill-name` 形式，或告诉学习者从 `/skills` 中选择该技能。
- Claude Code：使用 `/start-learning`、`/learn`、`/course-guide` 等 `/skill-name` 形式。
- 其他兼容宿主：使用自然语言，例如 `Use learn to start my first lesson.`

切勿将 Claude Code 的斜杠命令当作通用语法展示。当宿主未知时，使用自然语言形式。

## 跨课程模式的恢复路由

在进行通用入门引导之前，请将每个"resume"或"continue"请求与以下受支持的状态文件及其路由归属进行匹配：

- `LEARNING.md` 归 `learn` 管理，用于完整课程。
- `MCP-LEARNING.md` 归 `learn-mcp` 管理，用于模型上下文协议（MCP）路线。
- `MCP-ENGINEERING-LEARNING.md` 是该 `learn-mcp` 路线的旧版文件名，并非一条独立路线。
- `AGENT-SKILLS-LEARNING.md` 归 `learn-agent-skills` 管理。
- `CLAUDE-CERTIFICATION.md` 归 `claude-certification` 管理。

如果学习者在恢复或继续请求中指明了路线，即使存在其他状态文件，也立即派发到其归属技能，然后停止本技能。

对于未指明路线的恢复或继续请求，收集其状态文件存在的路由归属，将两个 MCP 文件名归入 `learn-mcp`。如果最终只剩恰好一个路由归属，则在通用入门引导之前调用它并停止本技能。`learn-mcp` 负责旧版文件迁移和冲突报告。如果剩余两个或更多路由归属，列出它们面向学习者的路线名称，并询问要恢复哪条路线，然后再运行分级测验或更改任何状态。如果都不存在，则继续进行通用入门引导。切勿根据文件新旧推断路线，也切勿将一条路线的进度合并到另一个状态文件中。

旧版运行时可能会将 `learn-mcp-engineering` 作为别名暴露。仅在接受它时用于到达 `learn-mcp`；所有面向学习者的交接都渲染为 `learn-mcp`，并将路线命名为模型上下文协议（MCP）。

## 专项 MCP 交接

如果学习者明确想要模型上下文协议（MCP）而非完整课程，请勿运行分级测验，也勿创建 `LEARNING.md`。将其路由到可移植技能 `learn-mcp`，其源文件为 `learning-paths/model-context-protocol.json`，状态文件为 `MCP-LEARNING.md`。在 Codex 中使用 `learn-mcp`，在 Claude Code 中使用 `/learn-mcp`，或请其他兼容宿主使用 `learn-mcp`。该专属辅导老师负责课程选择、实操证据和公开部署安全门。

## 专项 Agent Skills 交接

如果学习者明确想要 Agent Skills 而非完整课程，或者 `AGENT-SKILLS-LEARNING.md` 已存在且他们要求恢复该路线，请勿运行分级测验，也勿创建 `LEARNING.md`。将其路由到可移植技能 `learn-agent-skills`，其源文件为 `learning-paths/agent-skills.json`，状态文件为 `AGENT-SKILLS-LEARNING.md`。在 Codex 中使用 `learn-agent-skills`，在 Claude Code 中使用 `/learn-agent-skills`，或请其他兼容宿主使用 `learn-agent-skills`。该专属辅导老师负责五节课的顺序、真实宿主证据、沙箱边界、第 26 课之前第 25 课和工具投毒前置门，以及发布门。

如果 `LEARNING.md` 已存在，请勿覆盖它。概括其内容（使命、入口点、目前进度），并提供恰好三条路径：

- **恢复**：使用上述宿主语法调用 `learn`；完全跳过访谈和分级测验。
- **重新分级**：再次进行测验，然后仅更新 Placement 部分和 Path 状态；保持 Mission、Progress 日志和 Review 队列不变。
- **从头开始**：仅在明确确认后，将当前文件重命名为 `LEARNING-<YYYY-MM-DD>.md` 作为归档，然后继续进行下方的完整入门引导。切勿静默删除或覆盖他们的历史记录。

## 第 1 步：访谈（3 个问题，尽量简短）

1. **你为什么学习 AI 工程？** 自由文本。可提供的示例：发布一个 AI 产品、转行、理解我日常已经在用的东西、研究。用他们自己的话记录答案，因为这为后续每一课的解释奠定基础。
2. **每周能投入多少时间？** 选项：~2 小时、~5 小时、~10 小时、"尽快"。仅用于如实描述进度节奏，绝不用于删减内容。
3. **你到课程结束时最想构建什么？** 一句话。一个智能体、一个训练好的模型、一个 RAG 产品，"还没想好"也行。

不要在这三个问题之外多问。分级测验衡量的是知识水平；访谈只是记录意图。

## 第 2 步：分级测验

运行来自 `find-your-level` 技能的分级测验（它会与本技能一同安装）：5 个领域、10 道题，映射到一个入口阶段。保持该技能的答案隔离约定：不要预加载后续的答案轮次，也勿用真实选项字母替换中性的 `<letter>` 占位符。

如果学习者表示已知道想从哪里开始（"直接让我从第 7 阶段开始"），请尊重该选择并跳过测验，同时保持与运行测验相同的输出约定，以便 `learn` 辅导老师始终能找到一个格式良好的计划：

- 验证阶段编号在 0-19 范围内并解析其规范名称；如果无法解析，列出 20 个阶段并请他们选择。
- 在 Path 表中：入口点以下的阶段为 `Skip`，入口点及以上均为 `Do`（因为没有领域分数可推断，所以没有 `Review` 行），预估总时长为所有 `Do` 行的总和。
- 在 Placement 部分写入 `Score: self-selected` 而非数字。

## 第 3 步：编写 LEARNING.md

在当前目录下创建 `LEARNING.md`，包含恰好以下这些部分：

```markdown
# My AI Engineering Path
<!-- Managed by the ai-engineering-from-scratch learning skills.
     Repo: https://github.com/rohitg00/ai-engineering-from-scratch -->

## Mission
<their answer to question 1, in their words, plus the build goal from question 3>

## Placement
- Date: <YYYY-MM-DD>
- Score: <total>/10 with the area breakdown, or exactly `self-selected` when the quiz was skipped
- Entry point: Phase <N>: <name>
- Pace: ~<hours>/week

## Path
| Phase | Name | Status | Est. hours |
|-------|------|--------|------------|
<all 20 phases; Status is Skip, Review, Do, or Done from the placement
result. Hours come from ROADMAP.md: read it locally if the repo is cloned,
otherwise fetch
https://raw.githubusercontent.com/rohitg00/ai-engineering-from-scratch/main/ROADMAP.md>

## Progress log
| Date | Lesson | Quiz | Note |
|------|--------|------|------|

## Review queue
<empty for now; learn adds lessons the quizzes flag>
```

## 第 4 步：交接

以三行收尾，不要多说：

- 他们的入口点以及 Review + Do 阶段的总预估时长。
- 给出 `learn` 在对应宿主下的正确调用方式，并说明它会从第一节课开始，且每次都从此文件继续。
- 给出 `course-guide <topic>` 在对应宿主下的正确调用方式，并说明它可以直接跳转到特定主题。
