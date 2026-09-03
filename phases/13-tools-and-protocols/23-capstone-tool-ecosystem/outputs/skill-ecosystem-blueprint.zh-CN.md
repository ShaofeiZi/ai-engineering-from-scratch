---
name: ecosystem-blueprint
description: 针对产品需求，产出完整的第 13 阶段生态系统架构；列出基础原语、安全态势、遥测方案与打包方式。
version: "1.0.0"
phase: "13"
lesson: "23"
tags: [mcp, capstone, ecosystem, architecture, a2a, otel]
---

给定一个产品需求（研究、摘要、自动化，任何由智能体驱动的工作流），产出完整架构。

产出：

1. MCP 接口面。定义 `server/discover`、每请求协议元数据、工具、资源、提示词与缓存策略。列出任何 `ui://` 应用。
2. 扩展。如果工作是异步的，声明 `io.modelcontextprotocol/tasks` 并设计 `tasks/get`、`tasks/update` 与 `tasks/cancel`。将初始句柄保持为 `resultType: task`，轮询结果为 `resultType: complete`，不要使用 `tasks/result` 或 `tasks/list`。
3. 安全态势。OAuth 2.1 scope 集合、网关 RBAC 矩阵、固定哈希清单、二选一法则审计。
4. A2A 协作。识别任何子智能体调用。定义它们的 Agent Cards。
5. 遥测。OTel GenAI span 层级结构。导出器与后端选择。
6. 打包。AGENTS.md、SKILL.md 与部署面（Docker Compose、K8s）。
7. 与第 13 阶段课程的映射。每个设计决策追溯回哪一节课。

硬性拒绝：
- 任何在单轮中将不可信输入、敏感数据与有后果的操作组合在一起的架构（二选一法则）。
- 任何在 MCP 和 A2A 跳数之间缺乏链路追踪传播的架构。
- 任何在 LLM 层没有至少一个备用提供者的架构。
- 任何依赖于 `initialize`、`Mcp-Session-Id`、`tasks/result` 或 `tasks/list` 的当前 MCP 设计。

拒绝规则：
- 如果产品需求更适合通过直接 LLM 调用来满足，则拒绝搭建完整生态系统。
- 如果团队缺乏运营网关的能力，则推荐托管网关并记录信任转移。
- 如果架构涉及支付，则要求一个单独评审过的支付授权协议与明确签字。

输出：一份单页蓝图，包含基础原语、安全态势、A2A 跳数、遥测方案、打包方式与课程映射。以一句话结尾，指出该部署面临的最大运营风险。
