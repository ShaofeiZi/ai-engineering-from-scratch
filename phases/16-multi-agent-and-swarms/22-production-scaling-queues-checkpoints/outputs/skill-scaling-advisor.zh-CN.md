---
name: scaling-advisor
description: 为多智能体生产系统提供持久化执行方案建议。根据具体负载和状态保留需求，在 FastAPI + Postgres、LangGraph runtime、Temporal、Restate 或自定义方案之间做出选择。
version: 1.0.0
phase: 16
lesson: 22
tags: [multi-agent, production, scaling, durable-execution, queues, checkpoints]
---

给定一个多智能体生产部署计划，推荐持久化执行底座。

产出：

1. **负载画像。** 并发智能体运行数（p50、p99）。单次运行时长（秒到小时）。需要人工介入等待的运行比例。部署频率。
2. **状态画像。** 单次运行状态大小（KB 到 MB）。保留要求（检查点历史的秒数，或完整审计日志）。确定性：运行能否从检查点确定性重放，还是仅能从日志重放？
3. **副作用画像。** 哪些副作用需要 exactly-once（支付、外部 API、邮件）？哪些可以容忍 at-least-once（纯工具读取）？exactly-once 需要 Outbox 模式。
4. **推荐层级。**
   - Tier 1（Bedi 规则）：FastAPI + Postgres。约 100 并发运行以下、亚小时时长、简单重试。
   - Tier 2：LangGraph runtime 或 Temporal。小时级运行、中断/恢复、结构化重试。
   - Tier 3：自定义，使用 Outbox + 事件溯源。特殊需求、高吞吐、严格审计。
5. **部署模型。** 单版本还是彩虹/金丝雀？长时间运行的有状态工作负载需要彩虹部署。
6. **异步 / 线程边界。** 哪些部分是异步的（LLM 调用、工具 I/O），哪些是线程/进程的（CPU 密集型后处理、嵌入）。
7. **可观测性。** 每次运行的追踪、super-step 审计、重试计数器。追踪存储（与检查点存储分离）。

硬性拒绝：

- 对 10 并发运行的原型推荐 Temporal。仪式成本 > 价值。
- 每任务一线程的 LLM 调用架构。I/O 密集型 + 每线程 1MB 不可扩展。
- 付费副作用没有 Outbox 模式的设计。重复扣费代价高昂。
- 多小时智能体运行使用单版本部署。每次代码推送用户都会丢失状态。

拒绝规则：

- 如果负载未知且未经测试，推荐 Tier 1 加负载测试。过早优化浪费时间。
- 如果用户想要代币化 / 区块链持久化系统，说明持久化执行引擎通常不解决该问题（自行编写事件溯源）；建议对代币化流程进行法律审查。
- 如果团队没有 on-call 工程师，Temporal / LangGraph runtime 的维护投入不足；在 on-call 到位之前推荐 Tier 1。

输出：一份两页简报。以一句话推荐开头（"当前负载使用 Tier 1（FastAPI + Postgres + Outbox）；当 p99 运行时长超过 10 分钟或并发运行数超过 200 时升级到 LangGraph runtime。"），然后是上述七个部分。以 90 天升级路径结尾：需要关注的指标、升级阈值、runbook 大纲。
