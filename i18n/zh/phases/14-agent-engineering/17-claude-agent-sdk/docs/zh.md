# 将 Harness 作为库使用：子代理与会话存储

> 一个可以直接导入的 harness：内置工具、用于上下文隔离的子代理、hooks、W3C trace 传播、会话持久化。Claude Agent SDK 是这一模式的代表性实现，也是 Claude Code harness 的库形态；Claude Managed Agents 则是面向长时间异步任务的托管替代方案。

**Type:** 学习 + 构建
**Languages:** Python（标准库）
**Prerequisites:** 第 14 阶段 · 01（Agent Loop），第 14 阶段 · 10（技能库）
**Time:** 约 75 分钟

## 学习目标

- 解释 Anthropic Client SDK（原始 API）与 Claude Agent SDK（harness 形态）之间的区别。
- 描述子代理的两大用途：并行化与上下文隔离，以及何时应使用它们。
- 说出 Python SDK 的会话存储接口（`append`、`load`、`list_sessions`、`delete`、`list_subkeys`）以及 `--session-mirror` 的作用。
- 用 stdlib 实现一个包含内置工具、具备隔离上下文的子代理生成、生命周期 hooks 和会话存储的 harness。

## 问题

原始 LLM API 只能完成一次往返。一个生产级 agent 还需要工具执行、MCP 服务器、生命周期 hooks、子代理生成、会话持久化以及 trace 传播。Claude Agent SDK 把这种完整形态作为库提供出来，也就是把 Claude Code 所使用的同类 harness 公开给你，用于构建自定义 agent。

## 概念

### Client SDK 与 Agent SDK

- **Client SDK (`anthropic`)。** 原始 Messages API。循环、工具和状态都由你自己维护。
- **Agent SDK (`claude-agent-sdk`)。** 内置工具执行、MCP 连接、hooks、子代理生成与会话存储。相当于把 Claude Code 的循环做成了一个库。

### 内置工具

SDK 开箱即带有 10 多种工具：文件读写、shell、grep、glob、web fetch 等。自定义工具则通过标准的 tool-schema 接口注册。

### 子代理

Anthropic 文档里明确给出两个主要用途：

1. **并行化。** 把彼此独立的工作同时跑起来。比如“为这 20 个模块分别找到测试文件”，就可以拆成 20 个并行的子代理任务。
2. **上下文隔离。** 子代理使用自己的上下文窗口；只有结果返回给编排器。这样可以保住编排器的上下文预算。

Python SDK 最近还新增了 `list_subagents()` 与 `get_subagent_messages()`，用于读取子代理的转录内容。

### 会话存储

它与 TypeScript 版本在协议层面对齐：

- `append(session_id, message)`：追加一轮消息。
- `load(session_id)`：恢复一段对话。
- `list_sessions()`：枚举会话。
- `delete(session_id)`：删除会话。
- `list_subkeys(session_id)`：列出子代理键。

`--session-mirror`（CLI 参数）会在流式输出时把转录同步写入外部文件，便于调试。

### Hooks

你可以注册这些生命周期 hooks：

- `PreToolUse`、`PostToolUse`：拦截或审计工具调用。
- `SessionStart`、`SessionEnd`：做启动与清理。
- `UserPromptSubmit`：在模型看到用户输入之前先进行处理。
- `PreCompact`：在上下文压缩前执行。
- `Stop`：agent 退出时清理资源。
- `Notification`：发出旁路通知。

pro-workflow（Phase 14 课程中的参考系统）以及类似系统，就是通过 hooks 叠加这种横切行为的。

### W3C trace context

调用方当前活跃的 OTel span，会通过 W3C trace context headers 传播到 CLI 子进程。最终你在后端看到的是一条完整的跨多进程 trace。

### Claude Managed Agents

这是托管替代方案（beta header `managed-agents-2026-04-01`）。它适合长时间运行的异步任务，并且内置 prompt caching 与 compaction。你用更少控制权换取托管基础设施。

### 这种模式会出错的地方

- **子代理滥发。** 为 100 个微小任务生成 100 个子代理。最终是开销压倒收益，应该先做批处理。
- **Hook 蔓延。** 每个团队都加几个 hook，启动时间会迅速膨胀。应该按季度审查一次 hook。
- **会话膨胀。** 会话不断累积，体积越来越大。要配合 `list_sessions` 制定过期策略。

```figure
ae-subagent-isolation
```

## 动手构建

`code/main.py` 用 stdlib 实现了这种 SDK 形态：

- `Tool`、`ToolRegistry`，带有内置的 `read_file`、`write_file`、`list_dir`。
- `Subagent`：拥有私有上下文、隔离运行，并把结果返回回来。
- `SessionStore`：支持 append、load、list、delete、list_subkeys。
- `Hooks`：包含 `pre_tool_use`、`post_tool_use`、`session_start`、`session_end`。
- 一个演示：主 agent 并行生成 3 个子代理（彼此隔离），汇总结果并持久化会话。

运行它:

```
python3 code/main.py
```

输出 trace 会展示子代理的上下文隔离效果（编排器上下文大小保持有界）、hook 的执行，以及会话持久化。

## 如何使用

- **Claude Agent SDK**：适合希望直接采用 Claude Code harness 形态的 Claude-first 产品。
- **Claude Managed Agents**：适合托管的长时间异步任务。
- **OpenAI Agents SDK**（Lesson 16）：对应的 OpenAI-first 方案。
- **LangGraph + 自定义工具**：如果你更想要图状态机式的结构。

## 交付成果

`outputs/skill-claude-agent-scaffold.md` 提供了 Claude Agent SDK 应用脚手架，包含子代理、hooks、会话存储、MCP server 挂载，以及 W3C trace 传播。

## 练习

1. 添加一个子代理生成器，把 20 个任务按每批 5 个并行子代理来处理。比较这种方案与“一任务一子代理”时编排器上下文大小的差异。
2. 实现一个 `PreToolUse` hook，对 `write_file` 调用做限流（每个 session 每分钟 5 次）。把整个行为 trace 出来。
3. 把 `list_subkeys` 接到可视化渲染里，输出一棵子代理树。深度嵌套会是什么样？
4. 把这个 toy 移植到真正的 `claude-agent-sdk` Python 包上。工具注册方式会发生什么变化？
5. 去读 Claude Managed Agents 文档。你会在什么情况下从自托管切换到托管？

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------|----------|
| Agent SDK | “Claude Code as a library” | 一种 harness 形态：工具、MCP、hooks、子代理、会话存储 |
| Subagent | “Child agent” | 独立上下文、独立预算；结果向上返回 |
| Session store | “Conversation DB” | 持久化、加载、列举、删除消息轮次，并级联到子代理 |
| Hook | “Lifecycle callback” | 在工具调用前后、会话、提示提交、压缩、停止等时机执行 |
| W3C trace context | “Cross-process trace” | 父 span 被传播进 CLI 子进程 |
| Managed Agents | “Hosted harness” | Anthropic 托管的长时间异步工作 |
| `--session-mirror` | “Transcript mirror” | 将流式会话轮次写入外部文件 |
| MCP server | “Tool surface” | 挂接到 agent 上的外部工具/资源来源 |

## 延伸阅读

- [Claude Agent SDK overview](https://platform.claude.com/docs/en/agent-sdk/overview) — Claude Code 的库形态
- [Anthropic, Building agents with the Claude Agent SDK](https://www.anthropic.com/engineering/building-agents-with-the-claude-agent-sdk) — 生产实践模式
- [Claude Managed Agents overview](https://platform.claude.com/docs/en/managed-agents/overview) — 托管替代方案
- [OpenAI Agents SDK](https://openai.github.io/openai-agents-python/) — 对应方案
