---
name: terminal-coding-agent
description: 构建并评估终端原生编程智能体，在 SWE-bench Pro 上实现有界成本、沙箱工具和完整的 2026 钩子面。
version: 1.0.0
phase: 19
lesson: 01
tags: [capstone, coding-agent, claude-code, swe-bench, mcp, hooks, sandbox]
---

给定一个目标仓库和一个自然语言任务，构建一个能够规划、在沙箱中执行并提交 Pull Request 的测试框架。在 30 任务的 SWE-bench Pro 子集上匹配或超越 mini-swe-agent 基线，同时保持在每任务 5 美元预算以内。

构建计划：

1. 搭建一个 Bun + Ink TUI 框架，包含计划面板、工具调用流和实时 token/美元预算显示。
2. 基于 Model Context Protocol StreamableHTTP 定义六个工具（read_file、edit_file、ripgrep、tree_sitter_symbols、run_shell、git）。每次调用最多返回 4k token。
3. 在 E2B 或 Daytona 沙箱中运行每次工具调用，使用全新的 `git worktree add` 分支。绝不触碰宿主文件系统。
4. 串联全部八个 2026 钩子事件：SessionStart、SessionEnd、PreToolUse、PostToolUse、UserPromptSubmit、Notification、Stop、PreCompact。交付至少四个用户自定义钩子（破坏性命令守卫、token 记账、OTel span 发射器、追踪包写入器）。
5. 强制执行三个预算：50 轮、200k token、5 美元。PreCompact 在 150k 时触发并摘要较早的轮次。
6. 按照 GenAI 语义约定发射 OpenTelemetry span 到自托管的 Langfuse。
7. 成功后推送分支并提交 PR，在 PR 正文中附上计划和追踪包。
8. 在 30 个 issue 的 SWE-bench Pro Python 子集上对照 mini-swe-agent 进行评估，记录每任务的 pass@1、轮次、token 和美元开销。

评估量表：

| 权重 | 标准 | 度量方式 |
|:-:|---|---|
| 25 | SWE-bench Pro pass@1 | 在匹配的 30 任务子集上对比 mini-swe-agent 基线 |
| 20 | 架构清晰度 | 计划/执行/观察分离、钩子面、工具模式可读性 |
| 20 | 安全性 | 沙箱逃逸红队测试 + 破坏性命令守卫审计 |
| 20 | 可观测性 | 100% 工具调用被 span 覆盖、每轮 token 记账 |
| 15 | 开发者体验 | 冷启动低于 2s、崩溃恢复、Ctrl-C 取消语义 |

硬性拒绝条件：

- 在宿主文件系统而非沙箱内直接 shell 调用 git 的框架。
- 任何可以在工作树之外写入文件或在没有显式允许列表钩子的情况下 curl 外部 URL 的智能体。
- 未在同一 30 个 issue 上运行匹配基线就报告评估数值。
- 依赖于重试之间 `git reset --hard` 的"通过率"声明；SWE-bench Pro 是 pass@1。

拒绝规则：

- 拒绝在任何配置下直接推送到 main。仅允许 PR 分支。
- 拒绝禁用破坏性命令守卫。它是评估量表的硬性要求。
- 拒绝在没有预算上限的情况下运行。开放式运行会污染评估对比。

输出：一个包含测试框架的仓库，包含固定的 30 任务 SWE-bench Pro 评估框架及匹配的 mini-swe-agent 基线运行结果，至少 5 次完整运行的 OpenTelemetry 追踪归档，以及一份分析报告，指出该框架解决了哪些基线未解决的任务以及反之亦然。最后以一节结尾，描述你观察到的排名前三的失败模式及修复每种的钩子变更。
