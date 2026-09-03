# OpenAI Agents SDK：移交、护栏与追踪

> OpenAI Agents SDK 是一个建立在 Responses API 之上的轻量级多 agent 框架。它有五个核心原语：Agent、Handoff、Guardrail、Session、Tracing。Handoff 在模型看来就是名为 `transfer_to_<agent>` 的工具；Guardrail 会在输入或输出上触发；Tracing 默认开启。

**Type:** 学习 + 构建
**Languages:** Python（标准库）
**Prerequisites:** 第 14 阶段 · 01（Agent Loop），第 14 阶段 · 06（工具使用）
**Time:** 约 75 分钟

## 学习目标

- 说出 OpenAI Agents SDK 的五个核心原语。
- 解释 handoff：为什么它被建模成工具、模型实际看到的命名形式是什么、上下文又是如何转移的。
- 区分 input guardrails、output guardrails 和 tool guardrails，并解释 `run_in_parallel` 与阻塞模式的区别。
- 用 stdlib 实现一个带 handoffs、guardrails 和 span-style tracing 的 runtime。

## 问题

不会干净委派的 agent，最后往往会把所有事都塞进一个 prompt 里。没有 guardrails 的 agent，则可能直接把 PII 发出去、输出违反策略的内容，甚至陷入无穷循环。OpenAI 的 SDK 把多 agent 体系里真正难管的几个关键原语明确地产品化了，所以这类系统终于变得可实现、可约束。

## 概念

### 五个原语

1. **Agent。** LLM + instructions + tools + handoffs。
2. **Handoff。** 委派给另一个 agent。在模型看来，它就是一个名为 `transfer_to_<agent_name>` 的工具。
3. **Guardrail。** 对输入（只在第一个 agent 上）、输出（只在最后一个 agent 上）或工具调用（按 function tool）做校验。
4. **Session。** 自动维护跨轮次的对话历史。
5. **Tracing。** 为 LLM generations、tool calls、handoffs、guardrails 自动生成 spans。

### 把 handoff 当作工具

模型看到的是像 `transfer_to_billing_agent` 这样的工具名。模型一旦调用它，runtime 就会执行三件事：

1. 复制当前对话上下文，或者在 beta 能力 `nest_handoff_history` 下先做折叠。
2. 用目标 agent 自己的 instructions 初始化该 agent。
3. 由目标 agent 继续这次运行。

这其实就是把 supervisor pattern（Lesson 13 / Lesson 28）产品化了。

### 护栏

它有三种形态：

- **Input guardrails。** 在第一个 agent 的输入上运行，在任何 LLM 调用发生前拦住不安全或越界请求。
- **Output guardrails。** 在最后一个 agent 的输出上运行，拦截 PII 泄露、策略违规、格式错误等问题。
- **Tool guardrails。** 在每个 function tool 调用时运行，负责校验参数、检查权限、审计执行。

运行模式：

- **Parallel**（默认）。Guardrail LLM 与主 LLM 并行执行。尾延迟更低，但如果 guardrail 触发，主 LLM 的 token 就白花了。
- **Blocking**（`run_in_parallel=False`）。先跑 guardrail LLM。若 guardrail 触发，主调用根本不会发生，因此不会浪费 token。

触发时会抛出 `InputGuardrailTripwireTriggered` 或 `OutputGuardrailTripwireTriggered`。

### 追踪

默认开启。每次 LLM generation、tool call、handoff、guardrail 都会产生一个 span。设定 `OPENAI_AGENTS_DISABLE_TRACING=1` 可以关闭它。通过 `add_trace_processor(processor)`，还可以把这些 span 同时扇出到你自己的后端，而不只是交给 OpenAI。

### 会话

`Session` 会把对话历史存进某个后端，比如 SQLite、Redis 或自定义存储。调用 `Runner.run(agent, input, session=session)` 时，历史会被自动加载并附加新一轮内容。

### 这种模式会在哪些地方出错

- **Handoff drift。** Agent A 交给 Agent B，Agent B 又交回 Agent A。解决方法通常是加入 hop counter。
- **Guardrail bypass。** Tool guardrails 只覆盖 function tools；内建工具，比如 file reader、web fetch，仍然需要单独的策略层。
- **Over-tracing。** 敏感内容会被写进 span。应结合 OTel GenAI 的内容采集规则（Lesson 23），把正文外置存储，只在 trace 里保留引用 ID。

```figure
ae-agent-handoff
```

## 动手构建

`code/main.py` 在 stdlib 中实现了这个 SDK 的基本形状：

- `Agent`、`FunctionTool`、`Handoff`（后者本质上是一个带 transfer 语义的 function tool）。
- `Runner`，负责 input/output/tool guardrails、handoff dispatch 和 hop counter。
- 一个简单的 span emitter，用来展示 trace 的结构。
- 一个 triage agent：根据用户查询把请求交给 billing 或 support；其中有一个输入会触发 guardrail。

运行方式：

```
python3 code/main.py
```

trace 会展示两次成功的 handoff、一次 input guardrail 触发，以及一棵与真实 SDK 很接近的 span tree。

## 如何使用

- **OpenAI Agents SDK**：适合 OpenAI-first 的产品。
- **Claude Agent SDK**（Lesson 17）：适合 Claude-first 的产品。
- **LangGraph**（Lesson 13）：适合你明确需要显式状态和 durable resume 的场景。
- **Custom**：适合你需要精确控制的场景，比如语音、多 provider 或联邦式部署。

## 交付成果

`outputs/skill-agents-sdk-scaffold.md` 会生成一个 Agents SDK 脚手架，内含 triage agent、handoffs、input/output/tool guardrails、session store 和 trace processor。

## 练习

1. 加入 handoff hop counter：超过 N 次 transfer 就拒绝继续。然后观察 trace 的行为。
2. 实现 `nest_handoff_history` 选项：handoff 前先把前文压缩成一条总结，再转交给下一个 agent。
3. 写一个 blocking output guardrail。对比它在“会触发”与“不会触发”的 prompt 上各自带来的延迟差异。
4. 把 `add_trace_processor` 接到一个 JSON logger 上。每个 span 最终会发出什么形状？
5. 阅读 SDK 文档，把这个 stdlib toy 移植到真实的 `openai-agents-python`。你在哪些地方建模错了？

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------|----------|
| Agent | "LLM + instructions" | SDK 里的 agent 类型；拥有 tools 和 handoffs |
| Handoff | "Transfer" | 由模型调用的工具，用于把任务委派给另一个 agent |
| Guardrail | "Policy check" | 在输入 / 输出 / 工具调用时执行的校验 |
| Tripwire | "Guardrail trip" | guardrail 拒绝时抛出的异常 |
| Session | "History store" | 在多次运行间持久化对话记忆 |
| Tracing | "Spans" | 针对 LLM + tool + handoff + guardrail 的内建可观测性 |
| Blocking guardrail | "Sequential check" | 先跑 guardrail；触发时不会浪费主调用 token |
| Parallel guardrail | "Concurrent check" | 与主调用并行；延迟更低，但触发时会浪费 token |

## 延伸阅读

- [OpenAI Agents SDK docs](https://openai.github.io/openai-agents-python/) — 原语、handoffs、guardrails、tracing 的官方文档
- [Claude Agent SDK overview](https://platform.claude.com/docs/en/agent-sdk/overview) — Claude 侧的对应物
- [Anthropic, Building Effective Agents](https://www.anthropic.com/research/building-effective-agents) — 什么时候 handoff 值得引入
- [OpenTelemetry GenAI semantic conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/) — Agents SDK spans 对齐的标准语义约定
