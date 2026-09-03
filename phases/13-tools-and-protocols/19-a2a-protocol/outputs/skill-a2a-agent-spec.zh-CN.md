---
name: a2a-agent-spec
description: 为应可通过 A2A 调用的智能体生成 Agent Card 与技能模式。
version: 1.0.0
phase: 13
lesson: 18
tags: [a2a, agent-card, task-lifecycle, delegation]
---

给定智能体的能力及预期协作方，生成其 A2A Agent Card 与技能定义。

产出：

1. Agent Card。包含 `name`、`description`、`url`、`version`、`schemaVersion`、`capabilities`（streaming、pushNotifications）、`skills[]`。
2. 技能列表。每个技能包含 `id`、`name`、`description`、`inputModes`、`outputModes`。在描述中使用“当 X 时使用。不要用于 Y。”的模式。
3. 任务状态计划。针对每个技能，说明预期的状态转换及 input_required 路径。
4. 签名计划。是否通过 AP2 对卡片进行签名（对于可被外部调用的智能体，推荐签名）。
5. 传输方式。基于 HTTP 的 JSON-RPC（默认）或 gRPC。注意与 v1.0 的向后兼容性。

硬性拒绝：
- 任何缺少稳定 URL 的 Agent Card。会破坏发现机制。
- 任何未声明输入和输出模式的技能。调用方无法推断兼容性。
- 任何可被外部调用但缺少 AP2 签名计划的智能体。存在冒充攻击向量。

拒绝规则：
- 若智能体的用例仅为单次工具调用，拒绝搭建 A2A 脚手架；改用 MCP。
- 若智能体暴露了不应暴露的内部细节（工具调用轨迹、思维链），拒绝并强制保持不透明。
- 若智能体需要 A2A 用于支付（AP2 用例），确认 AP2 扩展版本，并标注 AP2 独立于核心 A2A。

输出：一页 Agent Card JSON、每个操作的技能模式、状态转换计划、签名与传输方式选择。以该智能体承诺的最低 v1.0 向后兼容性保证作为结尾。
