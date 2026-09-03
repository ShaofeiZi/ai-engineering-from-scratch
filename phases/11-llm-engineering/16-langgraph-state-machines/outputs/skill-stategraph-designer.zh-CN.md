---
name: stategraph-designer
description: 将智能体任务转换为 LangGraph StateGraph，包含命名节点、类型化状态、reducers、checkpointer 和人工中断。
version: 1.0.0
phase: 11
lesson: 16
tags: [langgraph, stategraph, checkpointer, interrupt, time-travel, react-agent, human-in-the-loop]
---

给定智能体任务（面向用户的目标、可用工具、预期轮数、带安全爆炸半径的副作用、持久化要求、目标延迟预算），输出：

1. 节点列表。为每个离散步骤命名：LLM 思考节点、每个工具运行节点、每一步人工审核、任何摘要器或评审器、任何检索器。若任何节点触及多于一个关注点则拒绝该设计；将其拆分。
2. 状态 schema。TypedDict（或 Pydantic）字段，每个列表都配一个 reducer。消息日志始终使用 Annotated[list, add_messages]。将任何任务专属的列表从 messages 中提升出来（一个计划、一个预算计数器、一个已检索文档列表），以确保 reducer 在并行更新下保持正确。
3. 边映射。下一步确定的场景使用静态边。仅在模型选择下一步的场景使用带命名路由函数的条件边。拒绝其路由函数依赖于你尚未在前置节点中发起的全新 LLM 调用的图。
4. 中断放置。在每个具有不可逆副作用（写入、删除、支付、有成本的外部 API 调用）的节点上使用 interrupt_before。当输出校验在独立进程中运行时，在模型节点上使用 interrupt_after。拒绝在任何有副作用的节点上使用 interrupt_after；届时副作用已经发生。
5. 检查点器。MemorySaver 仅用于测试。任何必须能在重启后存活的环境，从 PostgresSaver、SQLiteSaver、RedisSaver 中选择。确认 thread_id 策略（按用户、按会话、按对话）以及检查点 TTL。

拒绝发布不带检查点器的 LangGraph。没有检查点器就没有恢复、没有时间旅行、没有人在回路重放。拒绝发布不带 add_messages 的 messages 字段；第二次写入会静默覆盖第一次，半段对话就此消失。拒绝其每次转移都由规划器 LLM 路由的条件边构成的图；那是多了几步的 AutoGen，且每轮都在烧 token。

示例输入："基于 Anthropic Claude 的退款处理智能体，含三个工具（lookup_order、issue_refund、send_email），在超过 100 美元的退款前必须暂停等待人工，必须在服务器重启后恢复，p95 延迟预算 8 秒。"

示例输出：
- 节点：agent（LLM 调用）、lookup_tool、refund_tool、email_tool、human_review。
- 状态：messages 带 add_messages、order_context（覆盖）、refund_amount（覆盖）、reviewer_decision（覆盖）。
- 边：agent 到 should_continue 路由，分支为 lookup_tool、refund_tool、email_tool、human_review、END。工具节点返回 agent。
- 中断：当 refund_amount > 100 时在 refund_tool 上 interrupt_before。lookup_tool 和 email_tool 上不设中断。
- 检查点器：PostgresSaver，thread_id 为 "user:{user_id}:case:{case_id}"，TTL 30 天。
