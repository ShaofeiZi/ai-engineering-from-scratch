---
name: provider-portability-audit
description: 针对某一提供商的函数调用集成进行审计，列出将相同逻辑移植到另外两个提供商时出现的所有字段重命名、行为差异和硬限制冲突。
version: 1.0.0
phase: 13
lesson: 02
tags: [function-calling, openai, anthropic, gemini, portability]
---

给定某一提供商（OpenAI、Anthropic 或 Gemini）上的函数调用集成，产出一份可移植性审计报告，列出将相同逻辑部署到另外两个提供商时出现的每一个字段重命名、行为差异和硬限制冲突。

产出内容：

1. 声明差异。针对集成中的每个工具，展示移植到另外两个提供商时所需的信封结构 / 字段重命名 / schema 转换。标记目标提供商不支持的任何 JSON Schema 构造（Gemini：OpenAPI 3.0 子集；OpenAI strict：不支持 `$ref`，不支持有歧义的 `oneOf`）。
2. 响应差异。记录工具调用在各提供商响应结构中的位置（`tool_calls[]` 对比 `content[]` 块 对比 `parts[]` 条目），以及由谁负责解析 `arguments`（OpenAI 上为 string，Anthropic 和 Gemini 上为 object）。
3. `tool_choice` 差异。将集成当前的选择设置（auto / forbid / force / required）映射到目标提供商的结构；标记缺失的模式。
4. 限制冲突。报告工具数量上限（128 / 64 / 64）、schema 深度（5 / 10 / 实际无限制）以及单个参数长度上限。对于超过目标提供商限制的任何集成，提升至阻断级别严重度。
5. strict 模式映射。说明 strict 模式语义在目标上是否被保留。OpenAI 的 `strict: true` 在 Anthropic 上没有完全等价物；Gemini 的 `responseSchema` 是近似方案，但位于请求级别。

硬性拒绝：
- 任何假设 `arguments` 在非 OpenAI 目标上为 string 的集成。将静默产生错误结果。
- 任何在移植到 Anthropic 或 Gemini 时工具数量超过 64 且未使用路由器的集成。
- 任何在目标为 OpenAI strict 模式时在 schema 中使用 `$ref` 的集成。

拒绝规则：
- 如果被要求移植一个依赖于无对应物的提供商特有功能的集成（例如 OpenAI Responses API 的有状态对话、Anthropic 的 computer-use blocks），则拒绝并说明哪个功能在目标上没有对应物。
- 如果被要求选出一个胜者，则拒绝。该选择取决于宿主的 strict 模式需求、成本特征和并行调用要求。

输出：一份一页审计报告，包含每个工具的差异表、限制表，以及每个目标提供商的最终"移植裁定"（ship / needs-router / blocked-by-feature）。以一句话结尾，指出影响力最大的迁移变更。
