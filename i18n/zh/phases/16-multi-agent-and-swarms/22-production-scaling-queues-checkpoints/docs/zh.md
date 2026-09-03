# 生产级扩展：队列、检查点与持久性

> 要把多代理系统扩展到数千个并发运行，核心前提是**持久化执行（durable execution）**：你需要工作队列与检查点，这样只要租约处理、幂等副作用和确定性重放都设计到位，任何 worker 在任何崩溃后都能接手并恢复任意一次运行。LangGraph 的 runtime 是一个典型参考：它在每个 super-step 之后按 `thread_id` 写入检查点（默认落在 Postgres）；worker 崩溃后会释放租约，再由其他 worker 继续执行。代理也可以无限期休眠，等待人工输入。**MegaAgent**（arXiv:2408.09955）则为每个 agent 运行一个 producer-consumer 队列，包含三种状态（Idle / Processing / Response），并采用双层协调（组内聊天 + 组间管理聊天）。对 LLM 流式调用而言，**fiber/async** 比“一任务一线程”更合适：线程 99% 的时间都在等待 token，而 fiber 会在 I/O 上协作式让出执行权。另一面，Ashpreet Bedi 在 “Scaling Agentic Software” 中主张，在负载真正证明复杂度必要之前，坚持 **FastAPI + Postgres + nothing else**。简单架构往往能撑得比你想象更久。本课会实现一个持久检查点日志、一个带状态迁移的 per-agent 工作队列、一个 async-vs-thread 演示，并最终落到那条务实原则：先从简单方案开始。

**Type:** 学习 + 构建
**Languages:** Python（标准库，`asyncio`、`sqlite3`）
**Prerequisites:** 阶段 16 · 09（并行群体网络），阶段 16 · 13（共享记忆）
**Time:** 约 75 分钟

## 问题

一个多代理原型系统在单台笔记本上运行正常，三名代理共享一个内存事件循环。现在你要把它推到生产环境：

- 代理有时要运行数小时（长周期研究、等待人工反馈）。
- worker 进程会崩溃，重启后内存状态全部丢失。
- 峰值负载是平均值的 10 倍，你需要横向扩展。
- 用户按每次 agent-run 付费，因此计费必须具备 exactly-once 语义。

内存事件循环对这些需求一个都解决不了。你需要在底层引入一层可持久化的执行机制。到 2026 年，典型方案有：

1. 带检查点的工作流引擎（Temporal、LangGraph runtime）。
2. 消息队列 + 状态存储（Postgres + SQS/RabbitMQ）。
3. Actor-model 框架（如 MegaAgent 的 per-agent producer-consumer 设计）。
4. 自己手写的 FastAPI + Postgres（Bedi 的主张）。

本课会为这四类思路各做一个微型实现。

## 概念

### 持久化执行：基本模式

持久化执行引擎会在每个“步骤”之后持久化完整程序状态（在 LangGraph 的术语里，这个步骤叫 super-step）。发生崩溃时：

```
worker crashes mid-step
  -> lease timeout
  -> another worker picks up the thread_id
  -> resumes from last checkpoint
  -> no duplicate side effects
```

要让这套机制成立，需要满足：

- **状态可序列化。** 所有代理状态都必须能被持久化。带着活跃数据库连接的函数闭包是存不下来的。
- **恢复是确定性的。** 给定相同状态和相同输入，代理应当产生相同动作；或者把 LLM 调用委托给一个外部的确定性 oracle。
- **副作用具备幂等性。** 外部调用（工具调用、支付）必须是幂等的，或者必须带去重键。

LangGraph 在每个 super-step 后写检查点；Temporal 在每个 activity 后写检查点；Restate 使用 event-sourced journal。三者实现的是同一套模式。

### 每步一个检查点的 runtime

LangGraph runtime 是最直观的例子：每个代理都有一个 `thread_id`；状态是 typed dict；每个 super-step 都会向 checkpoints 表写入一行。恢复时，runtime 会从最近一次检查点开始重放，而不是从头重跑。代理可以通过 `interrupt()` 挂起并等待人工输入；runtime 会把状态持久化并释放 worker。等到输入到达时，任何一台 worker 都能继续执行。

这是 2026 年 4 月最具代表性的生产设计之一。

### MegaAgent 的逐智能体队列

arXiv:2408.09955 描述了一次大规模实验：在单个集群中同时运行数千个代理。其架构是：

```
agent i:
  state ∈ {Idle, Processing, Response}
  in_queue   <- messages addressed to agent i
  out_queue  -> replies + side effects

coordinators:
  intra-group chat  (agents in the same group)
  inter-group admin chat  (high-level routing)
```

双层协调的意义在于：让组内对话可以高密度发生，而组间通信保持稀疏。这正是把数千个代理的成本维持在线性范围内的常见模式。

### 异步与一任务一线程

LLM 调用本质上是 I/O-bound。一个线程在等待下一个 token 时，99% 的时间都处于空闲状态。每个线程大约要占用 ~1MB RAM；如果你有 10,000 个并发调用，仅线程栈就需要 10GB 内存。

Fiber（Python `asyncio`、Go goroutines、Rust `tokio`）会在 I/O 上协作式让出执行权。同样的 10,000 次调用通常可以轻松放进一个进程里。在 LLM agent 这个规模上，async 不是性能优化，而是架构选择。

例外是：CPU-bound 的后处理（embedding、tokenizer 技巧）仍然更适合线程或进程。把你的 I/O 层和 CPU 层拆开。

### Bedi 的反向观点

“Scaling Agentic Software”（Ashpreet Bedi，2026）认为，大多数团队在真正测量负载之前就已经过度设计。更务实的默认方案是：

- FastAPI + Postgres。
- 每次 agent run 是一行记录；状态用乐观并发控制做原地更新。
- 后台任务通过 `pg_notify` 或一个简单的 Celery worker 驱动。
- 重试策略放在应用代码里。

如果负载低于大约 100 个并发 agent-run，且任务规模可控，这套方案往往已经足够。只有在你测到它失败的时候，再升级架构。

规则是：当简单架构已经无法解决某个具体问题时，再引入 durable-execution framework。过早采用复杂框架，只会把时间浪费在没有回报的流程和仪式上。

### 恰好一次语义

对于收费型 agent-run，你真正需要的是“exactly-once effective”：底层是 at-least-once delivery，但 consumer 通过幂等设计把效果收敛为一次。常见工程手段包括：

- **每次 run 一个 dedup key。** 把它带进每个副作用调用中。
- **Outbox pattern。** 副作用先写入一张表，再由单独的进程执行。两个阶段都必须是幂等的。
- **补偿事务。** 当副作用成功，但记录该副作用的写操作失败时，安排后续补偿。

这些都是数据库工程里的经典模式，不是 LLM 特有问题。LLM 唯一增加的“税”只是调用更慢；剩下的仍然是标准分布式系统问题。

### 彩虹部署

Anthropic 的多代理研究系统使用了 “rainbow deployments”：让 agent runtime 的多个版本并行运行，这样长生命周期代理就不必在每次部署时都被强行中断。你可以把新版本先作为 canary 放给一小部分流量；旧版本则等它上面的代理自然跑完后再退役。

这原本就是长生命周期有状态系统的标准做法。到 2026 年，它的特殊之处在于：代理可能连续活上几个小时，因此部署节奏必须容纳这一现实。

### 生产环境的标准检查清单

- 持久状态（检查点、快照，或 outbox + 可重放日志）。
- 幂等副作用。
- 面向 LLM 调用的 async I/O 层。
- at-least-once delivery + dedup。
- 面向有状态工作负载的 rainbow/canary deployment。
- 可观测性：per-agent trace、super-step 审计、重试计数器。

```figure
sw-checkpoint-replay
```

## 动手构建

`code/main.py` 实现了：

- `CheckpointStore`：一个基于 SQLite 的检查点日志，以 thread-id 为键。每个 super-step 都会追加一行。
- `run_with_checkpoint(agent, thread_id)`：模拟运行中途崩溃，再由第二个 worker 从最后一个检查点恢复。
- `AgentQueue`：一个 per-agent 的 Idle / Processing / Response 状态机，带一个小型工作队列。
- `demo_async_vs_threads()`：分别用 asyncio 和 threads 跑 500 个并发模拟 “LLM calls”，并报告 wall-clock 时间与 peak memory（近似值）。

运行：

```
python3 code/main.py
```

预期输出：在模拟崩溃后，检查点恢复成功；async 版本能在 < 1s 内处理 500 个并发调用；thread 版本要花几秒，而且每个并发单元消耗的内存高出若干数量级。

## 实际使用

`outputs/skill-scaling-advisor.md` 会根据负载、状态保留需求和部署频率，帮助你判断该选 FastAPI + Postgres、LangGraph runtime、Temporal，还是自定义实现。

## 交付上线

生产环境的加固原则：

- **从简单方案开始（Bedi 规则）。** 先用 FastAPI + Postgres，直到你测量到它不够用。
- **先把可观测性打全，再谈优化。** 至少要有 per-run 延迟直方图、per-step 耗时、重试次数和失败分类。
- **对副作用使用 outbox pattern。** 支付和外部 API 调用尤其如此。
- **使用 rainbow deploy。** 部署时不要杀死仍在运行中的 agent-run。
- **在真正遇到特定问题时，再引入 durable-execution engine（Temporal / LangGraph / Restate）。** 例如长达数小时的人在环等待、跨区域协调、复杂的重试与补偿策略。
- **async 用于 I/O 层。** 线程只留给 CPU-bound 后处理。

## 练习

1. 运行 `code/main.py`。确认检查点恢复正常，并测量 async 与 thread 的并发差异。
2. 实现一个 **outbox** 表：每次工具调用都先写入 outbox，再由单独的 goroutine/task 执行。通过重复执行同一工具调用来验证幂等性。
3. 模拟一次 **rainbow deploy**：同时存在两个 runtime 版本；把一半新的 thread_ids 路由到每个版本；确认旧版本上的 in-flight threads 不会被打断。
4. 阅读下面链接的 LangGraph runtime 文档。找出其中哪些能力如果用 FastAPI + Postgres 手写，复制成本最高。那是否足以成为立即采用它的理由，还是仍然可以延后？
5. 阅读 MegaAgent（arXiv:2408.09955）第 3 节。文中明确提出了双层协调（组内 + 组间管理聊天）。请草拟你会如何把它映射成两类消息队列。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------------|------------------------|
| Durable execution | “把程序状态持久化” | 引擎会在每个 super-step 后写入状态；崩溃恢复是确定性的。 |
| Super-step | “事务边界” | 两次检查点之间的一段工作。LangGraph 的术语。 |
| thread_id | “代理运行标识” | 用来绑定检查点与恢复逻辑的键。 |
| Idempotency | “可安全重试” | 重复执行一次副作用，其结果与执行一次相同。 |
| Outbox pattern | “把副作用解耦出来” | 先把意图写入表，再由独立执行器真正执行并标记完成。 |
| At-least-once delivery | “可能出现重复消息” | 消息队列语义；通过 dedup key 让 consumer 达到 effective-once。 |
| Rainbow deploy | “版本重叠部署” | 在长生命周期工作负载中，让多个 runtime 版本并行存在。 |
| Async fiber | “协作式让出执行权” | 用户态并发；处理 I/O-bound 负载时比线程便宜得多。 |
| Checkpoint | “状态快照” | super-step 边界上的序列化状态，是恢复的关键。 |

## 延伸阅读

- [LangChain — The runtime behind production deep agents](https://www.langchain.com/conceptual-guides/runtime-behind-production-deep-agents) - LangGraph runtime 设计
- [MegaAgent](https://arxiv.org/abs/2408.09955) - per-agent producer-consumer queue；面向数千并发代理的双层协调
- [Matrix](https://arxiv.org/abs/2511.21686) - 以消息队列为协调基底的去中心化框架
- [Temporal docs](https://docs.temporal.io/) - durable execution 的参考工作流引擎
- [Anthropic — Multi-agent research system](https://www.anthropic.com/engineering/multi-agent-research-system) - 生产经验，包括 rainbow deployment
