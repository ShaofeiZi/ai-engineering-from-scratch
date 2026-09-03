---
name: claude-agent-scaffold
description: 搭建一个 Claude Agent SDK 应用，包含子智能体、生命周期钩子、会话存储、MCP 服务器挂载以及 W3C trace 上下文传播。
version: 1.0.0
phase: 14
lesson: 17
tags: [claude-agent-sdk, subagents, hooks, session-store, mcp]
---

给定一个产品领域和一组 MCP 服务器，搭建一个 Claude Agent SDK 应用。

产出：

1. 一个主智能体定义，包含指令、内置工具访问（read_file、write_file、shell、grep、glob、web fetch）以及自定义函数工具。
2. 用于并行化和上下文隔离的子智能体生成器。当编排器否则会耗尽其上下文预算时使用。
3. 已注册的生命周期钩子：用于审计的 PreToolUse + PostToolUse、用于初始化的 SessionStart、用于清理的 SessionEnd、用于规则执行的 UserPromptSubmit（参见 pro-workflow 模式）。
4. 会话存储（默认为 SQLite），并将 `list_subkeys` 接好以渲染子智能体树。
5. 用于外部工具/资源面的 MCP 服务器挂载。
6. W3C trace 上下文传播，使来自调用方的 OTel span 能够贯穿 CLI 继续传递。

硬性拒绝：

- 为单工具任务生成子智能体。子智能体用于并行化或上下文隔离；不用于“一次 read_file 调用”。
- 钩子中包含同步的耗时操作。钩子应为微秒到毫秒级。长时间运行的工作应放在子智能体中。
- 没有级联删除策略的会话存储。孤立的子智能体会话会撑爆存储。

拒绝规则：

- 如果产品需要长时间运行的异步工作（数小时至数天），拒绝自托管 SDK，并路由到 Claude Managed Agents。
- 如果用户要求将 `--session-mirror` 指向共享位置，拒绝。会话记录包含 PII；应镜像到按用户划分的加密存储。
- 如果智能体依赖原始 LLM 流式传输来提供 UX 而不使用工具调用，拒绝 Agent SDK，并直接推荐 Client SDK。

输出：`agent.py`、`tools.py`、`hooks.py`、`session.py`、`README.md`，说明子智能体策略、钩子注册表、会话后端、MCP 挂载和 OTel 接线。最后以“下一步阅读”结尾，指向 Lesson 22（语音交接）、Lesson 23（OTel span 归因），或 Lesson 18（如果产品需要生产运行时形态）。
