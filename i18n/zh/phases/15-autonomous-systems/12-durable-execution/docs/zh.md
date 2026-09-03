# 长时间运行的后台代理：持久执行

> 生产级长周期代理不会靠 `while True` 裸循环运行。每一次 LLM 调用都应当成为带检查点、重试和重放能力的活动单元（activity）。Temporal 的 OpenAI Agents SDK 集成已于 2026 年 3 月 GA。Anthropic 的 Claude Code Routines 可以按计划触发 Claude Code 调用，而不需要一个长期驻留的本地进程。会话会在等待人工输入时暂停，能跨部署存活，并按 `thread_id` 关联的最新检查点恢复。这层新的易用体验背后，其实是一个老模式：工作流编排（workflow orchestration）。新的输入只有一个：LLM 调用是非确定性活动单元，恢复时必须以确定性的方式重放。

**Type:** 学习
**Languages:** Python（stdlib，最小持久执行状态机）
**Prerequisites:** 阶段 15 · 10（权限模式），阶段 15 · 01（长时程代理）
**Time:** 约 60 分钟

## 问题

设想一个代理要运行四个小时。它调用三个工具，向用户请求两次确认，并发起四十次 LLM 调用。运行到一半时，承载它的主机重启了。接下来会发生什么？

- 在天真的 `while True` 循环里：所有状态都会丢失。整次运行从头开始。三个已经产生真实副作用的工具调用会再次执行。用户会再次被要求批准他们已经批准过的事项。四十次 LLM 调用会重新计费。
- 在持久执行里：运行会从最近的检查点恢复。已经完成的活动单元不会再次执行；它们的结果会从持久日志中重放。用户不需要重复批准已经批准过的事情。已经发出的 LLM 调用也不会再次计费。

这是工作流引擎十多年来一直在交付的同一个模式，例如 Temporal、Cadence 和 Uber 的 Cherami。新的地方在于：LLM 调用现在也成了一类活动单元。它们非确定、昂贵、可能带副作用，而且非常适合放进这个模式中。

本课的主线是：长周期可靠性会衰减。METR 观察到一种“35 分钟退化”现象，即成功率会随任务时长大致呈二次下降。持久执行让运行时间可以超过原本可靠性曲线所能支撑的范围；如果设计正确，这是更安全地失败的新方式，如果设计错误，则会把失败拖得更长、更隐蔽。

## 概念

### Activity、workflow 和 replay

- **Workflow**：确定性的编排代码。它定义活动单元的顺序、分支和等待点。它必须保持确定性，这样才能从事件日志中重放，而不会出现意外分叉。
- **Activity**：一个非确定、可能失败的工作单元。LLM 调用、工具调用、文件写入、HTTP 请求都属于这一类。每个活动单元都会记录输入，并在完成后记录输出。
- **Event log**：持久化的后端存储。每一次活动单元的开始、完成、失败、重试，以及每一个 workflow 决策都会被记录下来。
- **Replay**：恢复时，workflow 代码从头重新运行；已经完成的活动单元直接返回日志中的结果，不会再次执行。只有尚未完成的活动单元才会真正运行。

它的形状类似 React 基于虚拟 DOM 重新渲染，或者 Git 从提交记录重建工作树。编排器的确定性，是让持久化变得便宜的关键。

### 为什么 LLM 调用适合这个模式

LLM 调用具备这些特征：

- 非确定性：temperature > 0 时明显如此；即使 temperature 为 0，也可能随模型版本漂移。
- 成本高：既花钱，也引入延迟。
- 可能失败：会遇到 rate limit、timeout 等问题。
- 可能有副作用：如果调用结果进一步触发工具，副作用就会外溢。

这正是活动单元的典型轮廓。把每次 LLM 调用包装为活动单元，可以获得指数退避重试、跨重启检查点，以及可重放的调试轨迹。

### 以 `thread_id` 为键的检查点

LangGraph、Microsoft Agent Framework、Cloudflare Durable Objects 和 Claude Code Routines 都收敛到了相同的 API 形状：用 `thread_id` 或等价标识符标记会话；每个状态转换都会持久化到后端，例如默认使用 PostgreSQL、开发环境使用 SQLite、缓存使用 Redis；恢复时读取最新检查点。

后端选择很重要：

- **PostgreSQL**：持久、可查询、能跨部署存活。LangGraph 默认采用这种形态。
- **SQLite**：只适合本地开发；跨主机时会丢状态。
- **Redis**：速度快，但如果没有配置 AOF 或 snapshot，状态是临时的。
- **Cloudflare Durable Objects**：透明分布式；按唯一 key 划定作用域；可以存活数小时到数周。

### 人工输入是一等状态

Propose-then-commit（第 15 课）需要一个持久的“等待人工输入”状态。workflow 暂停，外部队列保存待审批请求，审批到达后再从原点恢复。没有持久化时，这只能算尽力而为；有了持久化，隔夜审批第二天早上到达时，workflow 仍能从正确位置继续。

### 35 分钟退化

METR 观察到，所有被测代理类别在连续运行超过约 35 分钟后都会出现可靠性衰减。任务时长翻倍，失败率大致会增加到四倍。持久执行本身不修复这个问题；它只是让你能够运行得比可靠性曲线原本支持的时间更长。安全的做法，是把持久化与重新进入时的新一轮 HITL 检查点结合起来，并用预算熔断开关（第 13 课）限制总计算量，而不是只看实际经过时间。

### 什么时候不该用持久执行

- 运行时间短于几分钟，且没有人工输入。
- 严格只读的信息检索任务。
- 正确性要求整个过程在一个 context window 内端到端完成的任务，例如某些推理任务或一次性生成任务。

```figure
memory-consolidation
```

## 用它

`code/main.py` 用 stdlib Python 实现了一个最小持久执行引擎。它支持：

- `@activity` 装饰器，把输入和输出记录到 JSON 事件日志。
- 一个 workflow 函数，用来按顺序组织活动单元。
- `run_or_replay(workflow, event_log)` 函数，用来重放已经完成的活动单元，而不再次执行它们。

驱动程序会模拟一个包含三个活动单元的 workflow，在中途崩溃，并展示两种结果：（a）天真的重试会重新执行所有内容；（b）重放只运行缺失的活动单元。

## 交付成果

`outputs/skill-durable-execution-review.md` 用来审查一个拟议的长时间运行代理部署是否具备正确的持久执行形态：活动单元、确定性、检查点后端、人工输入状态，以及恢复时的 HITL 策略。

## 练习

1. 运行 `code/main.py`。观察天真重试和重放在活动单元执行次数上的差异。改变崩溃点，并展示重放次数如何随之变化。

2. 把这个玩具引擎改成显式使用 `thread_id`。模拟两个并发会话共享同一个引擎，并确认它们的事件日志不会互相冲突。

3. 选取玩具引擎中的一个活动单元。引入一种非确定性，例如在 workflow 决策里使用实际时间戳。演示重放时如何分叉。解释真实引擎如何处理这个问题，例如副作用注册、`Workflow.now()` API。

4. 阅读 LangChain 的 “Runtime behind production deep agents” 文章。列出运行时持久化的所有状态，并说明每种状态覆盖哪类失败模式。

5. 为一个 6 小时的自主编码任务设计检查点策略。你会在哪里设置检查点？崩溃后恢复是什么样子？哪些地方需要新一轮 HITL？

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|---|---|---|
| Workflow | “代理脚本” | 确定性的编排代码；可从事件日志重放 |
| Activity | “一个步骤” | 非确定性单元，例如 LLM 调用、工具调用；执行前后都要记录 |
| Event log | “后端存储” | 每个状态转换的持久记录 |
| Replay | “恢复” | 重新运行 workflow；已完成的活动单元返回日志结果，不重新执行 |
| Checkpoint | “保存点” | 以 thread_id 为键持久化的状态；恢复时以最新状态为准 |
| thread_id | “会话键” | 限定持久状态作用域的标识符 |
| 35-minute degradation | “可靠性衰减” | METR 观察到成功率会随任务时长大致呈二次下降 |
| Non-determinism | “重放漂移” | 实际时钟、随机数、LLM 输出等；必须注册为副作用 |

## 延伸阅读

- [Anthropic — Claude Code Agent SDK: agent loop](https://code.claude.com/docs/en/agent-sdk/agent-loop) — 预算、轮次和恢复语义。
- [Microsoft — Agent Framework: human-in-the-loop and checkpointing](https://learn.microsoft.com/en-us/agent-framework/workflows/human-in-the-loop) — RequestInfoEvent 形态。
- [LangChain — The Runtime Behind Production Deep Agents](https://www.langchain.com/conceptual-guides/runtime-behind-production-deep-agents) — 具体运行时要求。
- [OpenAI Agents SDK + Temporal integration (Trigger.dev announcement)](https://trigger.dev) — LLM 调用的活动单元形态。
- [Anthropic — Measuring agent autonomy in practice](https://www.anthropic.com/research/measuring-agent-autonomy) — 35-minute degradation 参考。
