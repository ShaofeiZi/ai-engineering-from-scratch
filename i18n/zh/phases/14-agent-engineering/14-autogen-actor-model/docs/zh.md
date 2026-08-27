# 面向 Agent 的 Actor Model：异步消息与类型化运行时

> 把 agent 看成 actor：它们通过异步消息交换协作，依赖事件驱动处理器、天然的故障隔离，以及自然形成的并发。AutoGen v0.4（Microsoft Research，2025 年 1 月）围绕这一模型重构了 agent orchestration；如今 AutoGen 已进入维护模式，而 Microsoft Agent Framework（2025 年 10 月公开预览）则成为它面向生产的继任者。

**Type:** 学习 + 构建
**Languages:** Python（标准库）
**Prerequisites:** 第 14 阶段 · 01（Agent Loop），第 14 阶段 · 12（工作流模式）
**Time:** 约 75 分钟

## 学习目标

- 描述 actor model：agent 作为 actor、message 作为唯一 IPC、故障按 actor 隔离。
- 说出 AutoGen v0.4 的三层 API：Core、AgentChat、Extensions，并解释各自用途。
- 解释为什么将消息投递和消息处理解耦，会自然带来故障隔离与并发能力。
- 用 Python stdlib 实现一个 actor runtime，并把一个双 agent 代码审查流程移植到上面。

## 问题

大多数 agent 框架天然是同步的：一个 agent 产出，另一个 agent 消费，整个流程依附在一条调用栈上。出错时，整条栈一起崩。并发能力往往是后补的，分布式能力通常意味着重写。

AutoGen v0.4 给出的答案是 actor model。每个 agent 都是一个拥有私有 inbox 的 actor，message 是唯一交互方式。runtime 把投递和处理拆开，失败只隔离在单个 actor 内，并发成为默认能力，而分布式只不过是换了一种 transport。

## 概念

### Actor

一个 actor 具备：

- 私有状态，外界不能直接读写。
- inbox，也就是消息队列。
- 处理器：`receive(message) -> effects`，其中 effects 可以是“reply”“send to other actor”“spawn new actor”“update state”“stop self”。

两个 actor 之间不能共享内存，它们只能通过 message 通信。

### 三层 API

AutoGen v0.4 把整个表面拆成了三层：

1. **Core。** 低层 actor framework，包括 `AgentRuntime`、`Agent`、`Message`、`Topic` 等原语，负责异步消息交换和事件驱动执行。
2. **AgentChat。** 面向任务的高层 API，用来替代 v0.2 的 ConversableAgent，包括 `AssistantAgent`、`UserProxyAgent`、`RoundRobinGroupChat`、`SelectorGroupChat`。
3. **Extensions。** 与 OpenAI、Anthropic、Azure、tools、memory 等外部能力的集成层。

### 为什么“解耦投递与处理”很重要

在 v0.2 模型里，调用 `agent_a.chat(agent_b)` 会同步阻塞 agent_a，直到 agent_b 返回。在 v0.4 里，`send(agent_b, msg)` 只是把消息放进 agent_b 的 inbox，然后立即返回，稍后再由 runtime 负责投递和处理。这样会直接带来三个结果：

- **Fault isolation。** 如果 Agent B 崩了，不会连带把 Agent A 也打崩。runtime 会在 B 的 handler 边界捕获失败，再决定记录日志、重试，还是送入 dead-letter。
- **Natural concurrency。** 多条消息可以同时在飞，多个 actor 也可以并发处理自己的 inbox。
- **Distribution-ready。** inbox + transport 是统一抽象，不论 actor 是进程内运行，还是部署在另一台主机上，模型都不需要改。

### 拓扑

- **RoundRobinGroupChat。** 多个 agent 按固定顺序轮流发言。
- **SelectorGroupChat。** 由一个 selector agent 按当前对话上下文决定下一位谁来处理。
- **Magentic-One。** 一个用于 web browsing、code execution、file handling 的参考多 agent 团队，建立在 AgentChat 之上。

### 可观测性

内建支持 OpenTelemetry。每一条 message 都会对应一个 span；tool calls 默认会带上 `gen_ai.*` 属性，与 2026 年的 OTel GenAI semantic conventions（Lesson 23）对齐。

### 状态：维护模式

到 2026 年初，AutoGen v0.7.x 依然适合研究与原型验证，整体也比较稳定。但 Microsoft 已把持续的主力投入切换到 Microsoft Agent Framework，这才是它面向生产的后继者：2025 年 10 月 1 日公开预览，1.0 GA 原计划在 2026 年 Q1 末完成。值得保留的不是某个旧 API，而是 actor model 这个长期有效的思想。

```figure
actor-mailbox
```

## 动手构建

`code/main.py` 实现了一个 stdlib actor runtime：

- `Message`：带类型的 payload，包含 `sender`、`recipient`、`topic`、`body`。
- `Actor`：抽象基类，定义 `receive(message, runtime)`。
- `Runtime`：带共享队列的事件循环，负责投递消息并隔离失败。
- 一个双 actor demo：`ReviewerAgent` 审查代码，`ChecklistAgent` 执行清单；它们通过消息交换直到达成一致。

运行方式：

```
python3 code/main.py
```

trace 会展示消息投递过程、其中一个 actor 的模拟故障如何不会把另一个 actor 一并打崩，以及双方如何最终收敛到共同结论。

## 如何使用

- **AutoGen v0.4/v0.7**（维护模式）：适合研究、原型验证，以及探索多 agent 模式。
- **Microsoft Agent Framework**：面向生产的继任者（2025 年 10 月公开预览）；actor model 思路保持不变，只是 API 刷新了。
- **LangGraph swarm topology**（Lesson 13）：通过共享工具 handoff 实现的相近思路。
- **Custom actor runtime**：当你需要特定 transport，比如 NATS、RabbitMQ、gRPC。

## 交付成果

`outputs/skill-actor-runtime.md` 会生成一个最小可用的 actor runtime，以及给定多 agent 任务所需的团队模板（RoundRobin 或 Selector）。

## 练习

1. 加一个 dead-letter queue：当 handler 抛异常时，把失败消息停放下来，等待人工检查。你的 toy 里 DLQ 命中频率有多高？
2. 实现 `SelectorGroupChat`：由 selector actor 根据当前对话状态决定下一条消息交给谁处理。
3. 加入分布式 transport：把进程内队列替换成一个 JSON-over-HTTP server，让 actors 能运行在不同进程中。
4. 为每条消息接一个 OTel span，或者至少做一个 no-op stand-in，并发出 `gen_ai.agent.name`、`gen_ai.operation.name` 这些属性，对齐 Lesson 23。
5. 阅读 AutoGen v0.4 的架构文章，把这个 toy 移植到真实的 `autogen_core` API。有哪些你在生产里不能忽略、但 toy 里跳过了的部分？

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------|----------|
| Actor | "Agent" | 私有状态 + inbox + handler；不共享内存 |
| Message | "Event" | 带类型 payload；actor 之间唯一的交互方式 |
| Inbox | "Mailbox" | 每个 actor 自己待处理的消息队列 |
| Runtime | "Agent host" | 路由消息并隔离故障的事件循环 |
| Topic | "Channel" | actor 之间具名的发布-订阅路由 |
| Fault isolation | "Let it crash" | 一个 actor 失败不会把其他 actor 一起带崩 |
| RoundRobinGroupChat | "Fixed-rotation team" | agent 按顺序轮流处理 |
| SelectorGroupChat | "Context-routed team" | 由 selector 决定下一位谁来处理 |
| Magentic-One | "Reference team" | 处理 web + code + files 的参考多 agent 团队 |

## 延伸阅读

- [AutoGen v0.4, Microsoft Research](https://www.microsoft.com/en-us/research/articles/autogen-v0-4-reimagining-the-foundation-of-agentic-ai-for-scale-extensibility-and-robustness/) — 架构重构文章
- [LangGraph overview](https://docs.langchain.com/oss/python/langgraph/overview) — 图结构替代方案
- [OpenTelemetry GenAI semantic conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/) — AutoGen 默认发出的 spans 对齐的语义规范
