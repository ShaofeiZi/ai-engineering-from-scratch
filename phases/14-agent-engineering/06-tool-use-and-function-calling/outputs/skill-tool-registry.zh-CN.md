---
name: tool-registry
description: 构建生产级工具目录与注册表，具备 JSON Schema 校验、并行分发与可观测性。
version: 1.0.0
phase: 14
lesson: 06
tags: [function-calling, tools, schema, validation, bfcl, parallel-tools]
---

给定一个任务领域，产出一个工具目录，使智能体能够在 BFCL V4 各个维度（agentic、multi-turn、live、non-live、hallucination）上可靠地使用它。

产出内容：

1. 工具定义。每个工具包含：`name`（snake_case）、`description`（告诉模型何时该用以及何时不该用）、带类型属性的 JSON Schema 输入、必填字段、适用的枚举、数值的最小/最大值、每工具超时、每工具沙箱策略（文件系统暴露面、网络、内存上限）。
2. 描述质量检查。对每个描述运行“这是否告诉了模型何时选择该工具而非其他工具？”如果两个工具的描述存在重叠，拒绝并重写。
3. 并行分发计划。对于每个现实任务，识别哪些工具调用是独立的（可并行化）、哪些必须串行。输出一个预期的分发图。
4. 校验策略。枚举检查、类型强制转换规则（例如“接受以字符串表示的 int，拒绝以字符串表示的 float”）、必填字段强制校验。每次失败都返回一个结构化的观察字符串，绝不向循环抛出异常。
5. 可观测性。每个工具发出一个 OpenTelemetry GenAI `tool_call` span，附带属性 `gen_ai.tool.name`、`gen_ai.tool.call.id`、`gen_ai.tool.call.arguments`、`gen_ai.tool.call.result`（当内容策略要求时，使用引用而非内联）。

硬性拒绝：

- 通用的 shell/命令执行工具。拒绝并将其拆分为具体的动词（`git_status`、`fs_read`、`npm_test`）。
- 当参数具有封闭取值集合时缺失枚举。枚举校验是捕获偏移（drift）成本最低的方式。
- 两个不同工具的描述相同。模型无法可靠地在它们之间做出选择。
- 仅命名工具的 `description`（“将两个数字相加”）。必须包含何时选择它而非替代方案。
- 没有超时。每个工具调用都必须有上限。

拒绝规则：

- 如果单个智能体的工具列表超过 30 个工具，拒绝并建议子智能体委托（Lesson 17）。
- 如果任何工具在无确认门控的情况下执行破坏性操作，拒绝并指向 Lesson 09（权限、沙箱）。
- 如果任务是 computer use（click、type、screenshot），拒绝并指向 Lesson 21——这是一种独立工具形态，使用基于视觉的操作。

输出：一个可直接粘贴到 Anthropic / OpenAI / Gemini SDK 调用中的 JSON 工具目录、一张分发图、一份校验策略文档，以及一个注册表应通过的 BFCL 风格迷你评估。

以一个“接下来读什么”指引结尾：Lesson 09（sandboxing）、Lesson 23（OTel GenAI spans）或 Lesson 30（eval-driven）。
