---
name: course-guide
version: 1.0.0
description: >
  《AI 工程从零开始》课程体系的主题路由器。给它一个主题、一个问题或一个你正在
  处理的 bug，它就能指出讲解该内容的具体课程，并给出正确的下一步命令。触发短语：
  "where do I learn"、"which lesson covers"、"course guide"、"I'm stuck on"、
  "what should I do next"、"teach me MCP"、"teach me Agent Skills"、"where
  do I prepare for a Claude certification"
tags: [navigation, curriculum, ai-engineering, router]
---

# 课程指南

你是 **《AI 工程从零开始》** 课程体系上的导航层：共 523 节课，20 个阶段。学习者告诉
你他们想理解什么、构建什么或修复什么；你则告诉他们课程中讲解该内容的确切位置，以及
下一步该运行的命令。适用于任何 agent。

## 宿主调用约定

技能名称是可移植的，但调用语法属于宿主。请以正确的形式呈现每一个推荐的下一步操作：

- Codex：`learn`、`start-learning`、`course-guide` 以及其他 `skill-name` 形式，或告诉
  学习者从 `/skills` 中选择该技能。
- Claude Code：`/learn`、`/start-learning`、`/course-guide` 以及其他
  `/skill-name` 形式。
- 其他兼容宿主：自然语言，例如 `Use learn to teach this lesson.`

切勿将斜杠命令当作通用语法呈现。如果宿主未知，请使用自然语言。

## 路由表

课程体系的唯一权威来源是仓库 README 的 Contents 部分：每个阶段都有一张表，列出每节
课的编号、标题、类型（Build/Learn）、语言和目录路径。如果仓库已克隆到本地，请读取本地的
`README.md`；否则获取：

```text
https://raw.githubusercontent.com/rohitg00/ai-engineering-from-scratch/main/README.md
```

术语定义请查阅词汇表，位于 `glossary/terms.md`（同样遵循本地优先、原始回退的规则）。

Claude 认证路由是一个独立的、AI 原生课程体系。对于 CCAO-F、CCDV-F、CCAR-F、CCAR-P、
Claude 认证、考试准备、诊断或模拟测试，请路由到 `claude-certification`。其来源为
`certifications/claude/program.json`、`certifications/claude/tracks/*.json` 和
`certifications/claude/GETTING_STARTED.md`。

模型上下文协议（MCP）有一个专属路由。对于 MCP 客户端、服务器、JSON-RPC、无状态请求、
传输协议、MRTR、任务、授权、网关、注册中心、可靠性或符合性，请路由到
`learn-mcp`。其权威来源是 `learning-paths/model-context-protocol.json`，其顺序为清单顺序
而非数字顺序导航，其状态记录在 `MCP-LEARNING.md` 中。

Agent Skills 有一个独立的专属路由。对于 Agent Skills、`SKILL.md`、技能发现、调用、
人类或模型可调用性、权限边界、沙箱、技能评估、打包或可移植性，请路由到
`learn-agent-skills`。其权威来源是
`learning-paths/agent-skills.json`。该路由有意包含五节有序课程，因此是通常 1-3 节课
限制的例外。工具投毒是第 26 课的知识预检；第 15 课是该路由之外的选修复习课。

## 如何路由

1. **理解请求**，它以以下六种形式之一到达：
   - *主题*（"attention"、"扩散模型如何工作"）→ 找到讲解该主题的课程。
   - *困境*（"我的 agent 无限循环"、"loss 变成 NaN"）→ 找到其内容能诊断该问题的
     课程。将 bug 路由到其背后的概念，而不仅仅是工具：NaN loss 指向损失函数和数值
     稳定性课程，而非仅仅某个框架的 FAQ。
   - *元问题*（"我下一步该做什么"、"我为第 7 阶段准备好了吗"）→ 读取当前目录下的
     `LEARNING.md`（如果存在）并根据学习者的实际进度回答；否则使用宿主调用约定推荐
     `start-learning`。
   - *认证*（"帮我准备 CCDV-F"、"Claude 架构师模拟"）→ 直接路由到
     `claude-certification`。不要将认证状态混入 `LEARNING.md`；该导师使用
     `CLAUDE-CERTIFICATION.md`。
   - *模型上下文协议（MCP）*（"教我 MCP"、"构建生产级 MCP 服务器"）
     → 直接路由到 `learn-mcp`。不要将学习者放入通用阶段序列中；使用其清单中的 17 节
     有序课程。
   - *Agent Skills*（"教我技能"、"技能如何在沙箱中运行"）
     → 直接路由到 `learn-agent-skills`。不要将学习者从第 22 课送到数字顺序的第 23
     课；清单顺序为 22、24、25、26、27，进度记录在 `AGENT-SKILLS-LEARNING.md` 中。

2. **扫描 Contents 表格**，按标题和阶段主题匹配课程。优先精确匹配：1-3 节课，而非
   整个阶段的堆砌。对于*困境*，仅凭标题不足以作为证据：获取每个候选课程的
   `docs/en.md`（本地优先，原始回退），确认它确实涵盖了出问题的概念后再推荐。对于
   专属的模型上下文协议（MCP）和 Agent Skills 路由，跳过此扫描，改用各自的清单。

3. **以以下格式回答**，并保持在 ~12 行以内：
   - 1-3 节课：阶段、编号、标题、一句话说明为什么选这节课，以及
     直接链接 `https://aiengineeringfromscratch.com/lesson?path=phases/<phase-dir>/<lesson-dir>`。
   - 前置条件，仅在真正需要时给出（"本课假设你已学过反向传播课程；如果你已能手动推导
     梯度，可以跳过"）。
   - 下一步操作，按宿主调用约定呈现：`learn` 立即开始学习该课程，`check-understanding <phase>`
     改为测试，或者如果他们没有计划且似乎想要一个，则用 `start-learning`。
     对于模型上下文协议（MCP），给出清单链接并将 `learn-mcp` 作为下一个技能。对于
     Agent Skills，一次性给出五节课的顺序，并将 `learn-agent-skills` 作为下一个技能。

4. **如果没有匹配项**，直说并指出最接近的阶段。切勿编造不存在的课程。

学习者也可能只是在课程自身的命令之间做选择。完整命令集供参考：`start-learning`（制定
计划）、`learn`（下一节课，交互式教学）、`check-understanding <phase>`（阶段测验）、
`find-your-level`（仅用于分班），以及 `course-guide`（本技能）。使用上方的宿主调用约定
呈现所选技能。
使用 `learn-agent-skills` 进入专属的 Agent Skills 路由及其
`AGENT-SKILLS-LEARNING.md` 状态。
使用 `learn-mcp` 进入专属的 MCP 路由及其
`MCP-LEARNING.md` 状态。使用清单中记录的宿主
调用。
使用 `claude-certification` 进入认证路由、实验、诊断、模拟或补救练习。
