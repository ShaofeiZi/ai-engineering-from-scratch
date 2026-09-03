---
name: refine-loop
description: 根据任务、验证器的可用性以及迭代预算，配置一个评估器-优化器（Self-Refine / CRITIC）循环。
version: 1.0.0
phase: 14
lesson: 05
tags: [self-refine, critic, evaluator-optimizer, guardrails, iteration]
---

给定一个任务、一个迭代预算，以及可用的验证器类型（工具支撑的验证或仅自评估），为一个评估器-优化器循环输出提示词和停止策略。

产出内容：

1. 生成器提示词。首个输出的确定性生产者。显式声明任务、输出格式和约束条件。
2. 评估器/验证器提示词。如果工具可用（搜索、代码运行、测试、计算器、类型检查），指定如何调用这些工具，以及如何生成结构化的批评意见（JSON 格式，包含：pass/fail、violations[]、suggested_fixes[]）。如果仅有自评估可用，显式标记 Self-Refine 的橡皮图章风险，并使用结构上不同的提示词风格（例如对抗式的"至少找出一个缺陷"）。
3. 优化器提示词。必须引用先前的输出和批评意见（历史记录）。声明"不得重复先前迭代中已标记的失败模式"为强制性要求。
4. 停止策略。采用以下合取条件：验证器通过 OR（自评估认为没问题 AND 迭代次数 >= 2）OR 迭代次数 >= max_iterations。绝不可使用单一条件。
5. 可观测性钩子。将每次迭代记录为 OpenTelemetry GenAI span（evaluate、optimize），遵循第 23 课的要求，使整个优化轨迹可审计。

硬性拒绝：

- 生成器和批评者使用相同提示词。存在橡皮图章风险——模型会认同自己。
- 没有迭代上限。无限优化循环会消耗大量 token；默认始终上限设为 4。
- 验证器提示词要求自由格式的散文式反馈。仅接受结构化 JSON——pass/fail 外加分项列举的违规项。
- 从优化器提示词中丢弃历史记录。论文表明缺少历史记录会导致质量急剧下降。

拒绝规则：

- 如果任务没有验证器，也无法构建验证器，拒绝 CRITIC 并说明 Self-Refine 是可用的较弱选项——警告用户橡皮图章风险。
- 如果 max_iterations >= 10，拒绝并建议重新设计任务架构。超过 3-4 轮的优化至收敛通常意味着生成器提示词有误。
- 如果验证器调用破坏性工具（shell、git write），拒绝并要求设置沙箱边界（第 9 课）。

输出：一个完整的配置块，包含所有提示词、停止策略和工具列表，外加一条"后续阅读"提示，根据部署目标指向第 16 课（OpenAI Agents SDK guardrails）、第 12 课（Anthropic evaluator-optimizer）或第 30 课（eval-driven agent development）。
