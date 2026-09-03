---
name: skill-structured-outputs
description: 根据提供商、可靠性和复杂度选择合适结构化输出策略的决策框架
version: 1.0.0
phase: 11
lesson: 03
tags: [structured-output, json, schema, constrained-decoding, pydantic, function-calling]
---

# 结构化输出策略

在构建需要结构化数据的 LLM 应用时，应用此决策框架。

## 何时使用每种方法

**基于提示词（"Return JSON"）**：仅用于原型设计。适用于偶尔解析失败可容忍的内部工具。添加 try/except 加重试。切勿用于生产管线。

**JSON 模式（API 标志）**：你需要保证得到合法 JSON，但 schema 简单或灵活。当你能应用侧校验形状时适用。可用：OpenAI、Anthropic（通过工具使用）、Google。

**Schema 模式（受限解码）**：生产系统中每个输出都必须匹配特定 schema。零解析失败。零 schema 违规。对任何生产级抽取或分类任务默认使用。可用：OpenAI structured outputs、Outlines、Guidance。

**函数调用 / 工具使用**：模型需要选择调用哪个函数，而不仅是填充参数。你有多个 schema，由模型选择合适的一个。也适用于与现有工具/函数基础设施集成。

**Instructor 库**：你希望在任意提供商上获得 Pydantic 校验并自动重试。Python 项目中开发体验最佳。封装 OpenAI、Anthropic、Google 和开源模型。

## 针对提供商的指引

**OpenAI**：使用带 `json_schema` 类型的 `response_format`。内置受限解码。Pydantic 模型可直接使用。最可靠的结构化输出实现。

**Anthropic**：使用工具使用来实现结构化输出。定义一个具有目标 schema 的单一工具。模型返回匹配该 schema 的工具调用参数。可靠但需要走工具使用 API 模式。

**开源模型（vLLM、Ollama）**：使用 Outlines 或 Guidance 进行受限解码。这些库将 JSON Schema 编译为有限状态机，在生成期间屏蔽非法 token。需要在本地运行推理。

## Schema 设计准则

1. 尽可能保持 schema 扁平。超过 2 层的嵌套对象会增加抽取错误。
2. 对分类型字段使用 enum。不要依赖模型凭空想出正确的字符串。
3. 将歧义字段设为必填并显式支持 null，而不是设为可选。强制模型做出决定。
4. 为 schema 属性添加描述。模型会把这些当作指令来读。
5. 除非必要，避免联合类型（oneOf/anyOf）。它们会增加解码复杂度。
6. 为数字设置 minimum/maximum。捕获幻觉出的极端值。
7. 为数组使用 minItems/maxItems 以防止空数组或无界输出。

## 常见失败模式与修复

- **模型用 markdown 代码围栏包裹 JSON**：从基于提示词切换到 JSON 模式或 schema 模式
- **Schema 合法但事实上错误**：在抽取后添加 LLM-as-judge 校验步骤
- **enum 值不一致**：切换到受限解码或添加后处理归一化
- **缺失可选字段**：将它们设为必填，或在应用代码中添加默认值
- **抽取非常慢**：受限解码增加 5-15% 延迟，若延迟敏感则降低 schema 复杂度
- **大型数组且条目各异**：将输入分块并按块抽取，再合并结果

## 可靠性阶梯

| 方法 | 解析成功率 | Schema 匹配率 | 搭建工作量 |
|----------|-------------|-------------|-------------|
| 基于提示词 | ~90% | ~80% | 1 分钟 |
| JSON 模式 | 100% | ~90% | 5 分钟 |
| Schema 模式 | 100% | ~99% | 15 分钟 |
| 受限解码 | 100% | 100% | 30 分钟 |
| Instructor + 重试 | 100% | ~99.5% | 10 分钟 |
