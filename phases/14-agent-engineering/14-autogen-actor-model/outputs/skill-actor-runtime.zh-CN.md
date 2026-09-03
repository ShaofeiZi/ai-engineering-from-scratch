---
name: actor-runtime
description: 构建一个符合 AutoGen v0.4 形态的 actor 运行时，具备私有状态、每个 actor 独立收件箱、纯消息式 IPC、故障隔离以及死信队列。
version: 1.0.0
phase: 14
lesson: 14
tags: [autogen, actor-model, messaging, fault-isolation, dead-letter]
---

给定一个多智能体任务，生成一个 actor 运行时以及所需的智能体 actor。

需要产出：

1. 一个 `Message` 类型，包含 `sender`、`recipient`、`topic`、`body`、`mid`。
2. 一个 `Actor` 基类，包含 `receive(message, runtime)`。Actor 状态是私有的。
3. 一个 `Runtime`，包含共享队列、`send()`、`run_until_idle()` 以及一个死信队列。处理器中的异常进入死信队列（DLQ）；不得传播。
4. 一个拓扑辅助工具：RoundRobin（固定轮转）、Selector（由 LLM 选择下一个）或自定义广播。
5. 每条消息的可观测性钩子：按照第 23 课的要求，发射带有 `gen_ai.agent.name` 和 `gen_ai.operation.name` 的 OTel span。

硬性拒绝：

- 阻塞发送方直到接收方返回的同步消息传递。这是 v0.2 模型；它会破坏故障隔离。
- 跨 actor 的共享可变状态。Actor 只能通过消息读取状态，否则根本不读取。
- 传播处理器异常的运行时。故障应进入死信队列（DLQ）；让其他 actor 继续运行。

拒绝规则：

- 如果任务只有两个 actor 且是固定来回交互，则拒绝 actor 框架，并建议使用提示链（第 12 课）。只有在 >=3 个 actor 或异步并发时，actor 才值得付出成本。
- 如果用户想要“同步模式”以便“更容易调试”，则拒绝。建议改用日志 + 追踪（第 23 课）。
- 如果领域严格是请求/响应式且有单一专家，则建议使用路由（第 12 课）而非 actor 团队。

输出：`message.py`、`actor.py`、`runtime.py`、`teams.py`、`README.md`，其中 README 需解释 DLQ 策略、拓扑选择以及 OTel span 的接入方式。最后以“延伸阅读”结尾：如果 actor 之间需要协商，指向第 25 课（多智能体辩论）；如果需要追踪，指向第 23 课（OTel）；如果你想使用前瞻性的运行时，指向 Microsoft Agent Framework。
