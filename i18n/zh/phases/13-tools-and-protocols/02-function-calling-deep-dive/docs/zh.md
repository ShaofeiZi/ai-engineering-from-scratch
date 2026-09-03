# 函数调用深入解析——OpenAI、Anthropic、Gemini

> 三家前沿提供商在 2024 年收敛到了相同的工具调用循环，其他部分却各不相同。OpenAI 使用 `tools` 与 `tool_calls`；Anthropic 使用 `tool_use` 与 `tool_result` 块；Gemini 使用 `functionDeclarations` 与唯一 ID 关联。本课会并排比较三者，使在一家提供商上发布的代码迁移时不会崩溃。

**Type:** 构建
**Languages:** Python (stdlib, schema translators)
**Prerequisites:** 第 13 阶段 · 第 01 课（工具接口）
**Time:** 约 75 分钟

## 学习目标

- 说出 OpenAI、Anthropic 与 Gemini 函数调用负载在三个方面的形态差异（声明、调用、结果）。
- 在三种提供商格式之间转换同一份工具声明，并预测严格模式约束存在的差异。
- 使用每家提供商的 `tool_choice` 强制、禁止或自动选择工具调用。
- 了解各提供商的硬性限制（工具数量、Schema 深度、参数长度），以及违反限制时返回的错误特征。

## 问题

不同提供商的函数调用请求具有不同形态。以下是 2026 年生产技术栈中的三个具体示例：

**OpenAI Chat Completions / Responses API。** 你传入 `tools: [{type: "function", function: {name, description, parameters, strict}}]`。模型响应包含 `choices[0].message.tool_calls: [{id, type: "function", function: {name, arguments}}]`，其中 `arguments` 是必须自行解析的 JSON 字符串。严格模式（`strict: true`）通过受约束解码强制遵守 Schema。

**Anthropic Messages API。** 你传入 `tools: [{name, description, input_schema}]`。响应形如 `content: [{type: "text"}, {type: "tool_use", id, name, input}]`。`input` 已经解析完成（是对象，而不是字符串）。你需要回复一条新的 `user` 消息，其中包含 `{type: "tool_result", tool_use_id, content}` 块。

**Google Gemini API。** 你传入 `tools: [{functionDeclarations: [{name, description, parameters}]}]`（嵌套在 `functionDeclarations` 下）。响应形如 `candidates[0].content.parts: [{functionCall: {name, args, id}}]`，其中 `id` 在 Gemini 3 中是唯一值，可用于关联并行调用。你回复 `{functionResponse: {name, id, response}}`。

循环相同，字段名不同、嵌套方式不同、字符串与对象的约定不同，关联机制也不同。一个团队在 OpenAI 上编写天气智能体后，仅为了接线就要花两天迁移到 Anthropic，再花一天迁移到 Gemini。

本课会构建一个转换器，在代码内部统一为一份规范工具声明，只在边缘处适配不同提供商。阶段 13 · 17 会把同一模式推广为大语言模型网关。

## 概念

### 共同结构

每家提供商都需要五样东西：

1. **工具列表。** 每个工具的名称、描述与输入 Schema。
2. **工具选择。** 强制使用特定工具、禁止使用工具，或让模型自行决定。
3. **调用输出。** 指明工具与参数的结构化输出。
4. **调用 ID。** 将结果关联到正确的调用（并行调用时尤其重要）。
5. **结果注入。** 把结果与调用关联起来的消息或内容块。

### 逐字段比较形态差异

| 方面 | OpenAI | Anthropic | Gemini |
|--------|--------|-----------|--------|
| 声明信封 | `{type: "function", function: {...}}` | `{name, description, input_schema}` | `{functionDeclarations: [{...}]}` |
| Schema 字段 | `parameters` | `input_schema` | `parameters` |
| 响应容器 | 助手消息上的 `tool_calls[]` | `content[]`，类型为 `tool_use` | `parts[]`，类型为 `functionCall` |
| 参数类型 | JSON 字符串 | 已解析对象 | 已解析对象 |
| ID 格式 | `call_...`（由 OpenAI 生成） | `toolu_...`（Anthropic） | UUID（Gemini 3 以上） |
| 结果块 | `tool` 角色、`tool_call_id` | `user` 消息，带 `tool_result`、`tool_use_id` | `functionResponse`，带匹配的 `id` |
| 强制某个工具 | `tool_choice: {type: "function", function: {name}}` | `tool_choice: {type: "tool", name}` | `tool_config: {function_calling_config: {mode: "ANY"}}` |
| 禁止工具 | `tool_choice: "none"` | `tool_choice: {type: "none"}` | `mode: "NONE"` |
| 严格 Schema | `strict: true` | Schema 就是契约（始终强制） | 请求级 `responseSchema` |

### 实际会碰到的限制

- **OpenAI。** 每个请求最多 128 个工具。Schema 深度上限为 5。参数字符串不超过 8192 字节。严格模式不允许 `$ref`，不允许存在重叠的 `oneOf`/`anyOf`/`allOf`，而且每个属性都必须列入 `required`。
- **Anthropic。** 每个请求最多 64 个工具。Schema 深度在理论上不受限，但实际建议不超过 10。没有严格模式开关；Schema 是一份契约，模型通常会遵守。
- **Gemini。** 每个请求最多 64 个函数。Schema 类型使用 OpenAPI 3.0 子集（与 JSON Schema 2020-12 略有差异）。Gemini 3 起，并行调用拥有唯一 ID。

### `tool_choice` 行为

三家都支持三种模式，只是命名不同。

- **Auto。** 模型选择调用工具或输出文本，默认模式。
- **Required / Any。** 模型必须调用至少一个工具。
- **None。** 模型不得调用工具。

此外，每家提供商还有一种专用模式：

- **OpenAI。** 按名称强制使用特定工具。
- **Anthropic。** 按名称强制使用特定工具；`disable_parallel_tool_use` 标志区分单调用与多调用。
- **Gemini。** `mode: "VALIDATED"` 无论模型意图如何，都会让每个响应经过 Schema 验证器。

### 并行调用

OpenAI 的 `parallel_tool_calls: true`（默认）会在一条助手消息中输出多个调用。你运行全部调用，并回复一条批量工具角色消息，其中每个 `tool_call_id` 对应一个条目。Anthropic 过去只支持单调用；`disable_parallel_tool_use: false`（自 Claude 3.5 起默认）会启用多调用。Gemini 2 允许并行调用，却没有稳定 ID；Gemini 3 加入 UUID，使乱序返回的结果能够正确关联。

### 流式传输

三家都支持流式工具调用，但报文格式不同：

- **OpenAI。** `tool_calls[i].function.arguments` 的增量块逐步到达。你需要不断拼接，直到收到 `finish_reason: "tool_calls"`。
- **Anthropic。** 使用内容块开始 / 内容块增量 / 内容块结束事件。`input_json_delta` 块携带局部参数。
- **Gemini。** `streamFunctionCallArguments`（Gemini 3 新增）输出带 `functionCallId` 的数据块，使多个并行调用可以交错传输。

阶段 13 · 03 会深入介绍并行与流式重组。本课聚焦声明与单调用形态。

### 错误与修复

无效参数错误的形式也不相同。

- **OpenAI（非严格）。** 模型返回 `arguments: "{bad json}"`，JSON 解析失败；你注入错误消息并重新调用。
- **OpenAI（严格）。** 验证在解码期间完成，因此不可能生成无效 JSON，但可能出现 `refusal`。
- **Anthropic。** `input` 可能包含意外字段；Schema 仅起指导作用，仍需在服务器端验证。
- **Gemini。** OpenAPI 3.0 的一个特性是：对象字段上的 `enum` 可能被静默忽略；需要自行验证。

### 转换器模式

代码中的规范工具声明可以采用如下形态（由你决定具体格式）：

```python
Tool(
    name="get_weather",
    description="Use when ...",
    input_schema={"type": "object", "properties": {...}, "required": [...]},
    strict=True,
)
```

三个小函数把它转换为三家提供商的格式。`code/main.py` 中的框架正是这样做的，然后让一个模拟工具调用在各提供商响应形态中往返。不需要网络——本课教授的是数据形态，而不是 HTTP。

生产团队通常会把这种转换器包装进 `AbstractToolset`（Pydantic AI）、`UniversalToolNode`（LangGraph）或 `BaseTool`（LlamaIndex）。阶段 13 · 17 会交付一个网关，在三家任意提供商前暴露 OpenAI 形态的 API。

```figure
function-call-args
```

## 投入使用

`code/main.py` 定义一个规范 `Tool` 数据类，以及三个分别输出 OpenAI、Anthropic 与 Gemini 声明 JSON 的转换器。随后，它会把每种形态的手工构造提供商响应解析为同一个规范调用对象，证明三者表面之下的语义相同。运行后并排比较三份声明。

需要关注：

- 三个声明块只在信封与字段名上不同。
- 三个响应块的差异在于调用所在位置（顶层 `tool_calls`、`content[]` 块、`parts[]` 条目）。
- 一个 `canonical_call()` 函数从三种响应形态中提取 `{id, name, args}`。

## 交付成果

本课会产出 `outputs/skill-provider-portability-audit.md`。给定针对一家提供商的函数调用集成，它会生成可移植性审计：依赖了哪些提供商限制、哪些字段需要重命名，以及迁移到另外两家提供商时会在哪里出错。

## 练习

1. 运行 `code/main.py`，确认三家提供商的声明 JSON 都序列化自同一个底层 `Tool` 对象。为规范工具添加枚举参数，并确认只有 Gemini 转换器需要处理 OpenAPI 特性。

2. 为每家提供商添加 `ListToolsResponse` 解析器，抽取模型在 `list_tools` 或发现调用后返回的工具列表。OpenAI 没有原生接口，请记录这项不对称。

3. 实现 `tool_choice` 转换：把规范 `ToolChoice(mode="force", tool_name="x")` 映射到三种提供商形态，再映射 `mode="any"` 与 `mode="none"`。对照本课差异表检查。

4. 从三家提供商中任选一家，从头到尾阅读其函数调用指南。找出该提供商 Schema 规范中另外两家不支持的一个字段。候选项：OpenAI `strict`、Anthropic `disable_parallel_tool_use`、Gemini `function_calling_config.allowed_function_names`。

5. 编写一条测试向量：工具调用参数违反声明的 Schema。让它经过各提供商验证器（可使用第 01 课的标准库版本作为替代），记录触发的错误。说明生产环境中若追求严格性，你会选择哪家提供商。

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|----------------|------------------------|
| 函数调用 | “工具使用” | 提供商级 API，用于输出结构化工具调用 |
| 工具声明 | “工具规范” | 名称 + 描述 + JSON Schema 输入负载 |
| `tool_choice` | “强制 / 禁止” | 自动 / 必须 / 禁止 / 指定名称模式 |
| 严格模式 | “Schema 强制” | OpenAI 通过受约束解码强制模型输出符合 Schema 的标志 |
| `tool_use` 块 | “Anthropic 调用形态” | 包含 ID、名称与输入的内联内容块 |
| `functionCall` 部件 | “Gemini 调用形态” | `parts[]` 中包含名称、参数与 ID 的条目 |
| 参数字符串 | “字符串化 JSON” | OpenAI 以 JSON 字符串而不是对象返回参数 |
| 并行工具调用 | “一轮扇出” | 一条助手消息中的多个工具调用 |
| 拒绝 | “模型拒绝” | 严格模式下以拒绝块替代调用 |
| OpenAPI 3.0 子集 | “Gemini Schema 特性” | Gemini 使用的类 JSON Schema 方言，存在少量差异 |

## 延伸阅读

- [OpenAI——函数调用指南](https://platform.openai.com/docs/guides/function-calling)——包含严格模式与并行调用的权威参考
- [Anthropic——工具使用概览](https://docs.anthropic.com/en/docs/agents-and-tools/tool-use/overview)——`tool_use` 与 `tool_result` 块语义
- [Google——Gemini 函数调用](https://ai.google.dev/gemini-api/docs/function-calling)——并行调用、唯一 ID 与 OpenAPI 子集
- [Vertex AI——函数调用参考](https://docs.cloud.google.com/vertex-ai/generative-ai/docs/multimodal/function-calling)——Gemini 的企业级接口
- [OpenAI——结构化输出](https://platform.openai.com/docs/guides/structured-outputs)——严格模式 Schema 强制细节
