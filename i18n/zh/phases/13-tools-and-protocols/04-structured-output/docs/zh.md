# 结构化输出——JSON Schema、Pydantic、Zod 与受约束解码

> “礼貌地要求模型返回 JSON”即使在前沿模型上也会有 5%～15% 的失败率。结构化输出通过受约束解码弥合这道差距：模型在字面意义上无法输出任何违反 Schema 的词元。OpenAI 的严格模式、Anthropic 使用 Schema 定义类型的工具调用、Gemini 的 `responseSchema`、Pydantic AI 的 `output_type` 与 Zod 的 `.parse`，都是同一思想的五种表面形式。本课会构建 Schema 验证器，以及学习者今后可用于每条生产抽取流水线的严格模式契约。

**Type:** 构建
**Languages:** Python (stdlib, JSON Schema 2020-12 subset)
**Prerequisites:** 第 13 阶段 · 第 02 课（深入解析函数调用）
**Time:** 约 75 分钟

## 学习目标

- 使用正确的约束（enum、min/max、required、pattern），为抽取目标编写 JSON Schema 2020-12。
- 解释严格模式与受约束解码提供的保证，为何不同于“生成后再验证”。
- 区分三种失效模式：解析错误、Schema 违规、模型拒绝。
- 交付一条具备类型化修复与类型化拒绝处理的抽取流水线。

## 问题

一个读取采购订单电子邮件的智能体，需要把自由文本转换为 `{customer, line_items, total_usd}`。有三种方案。

**方案一：提示模型输出 JSON。** “使用 JSON 回答，字段为 customer、line_items、total_usd。”在前沿模型上有 85%～95% 的成功率，却会以六种方式失败：缺少大括号、尾随逗号、类型错误、编造字段、达到词元上限而截断，以及泄露“下面是你的 JSON：”之类的说明文字。

**方案二：生成后验证。** 自由生成、解析、依据 Schema 验证，失败时重试。它很可靠，但成本高——每次重试都要付费，而且每次截断都会额外消耗一轮调用。

**方案三：受约束解码。** 提供商在解码时强制执行 Schema。采样分布中，任何会违反 Schema 的词元都会被屏蔽。输出保证能够解析，也保证通过验证。失败只剩一种模式：拒绝（模型判断输入无法适配该 Schema）。

到 2026 年，每家前沿提供商都提供某种形式的方案三。

- **OpenAI。** 使用 `response_format: {type: "json_schema", strict: true}`；如果模型拒绝，响应中会包含 `refusal`。
- **Anthropic。** 对 `tool_use` 输入强制执行 Schema；不存在 `stop_reason: "refusal"`，但以 `end_turn` 结束且没有工具调用，就是拒绝信号。
- **Gemini。** 在请求级使用 `responseSchema`；到 2026 年，Gemini 已对部分类型提供词元级语法约束。
- **Pydantic AI。** `output_type=InvoiceModel` 会返回结构化 `RunResult`，其类型为 `InvoiceModel`。
- **Zod（TypeScript）。** 运行时解析器依据 Zod Schema 验证提供商输出；可与 OpenAI 的 `beta.chat.completions.parse` 配合使用。

共同主线是：Schema 只声明一次，再端到端强制执行。

## 概念

### JSON Schema 2020-12——通用语言

所有提供商都接受 JSON Schema 2020-12。最常用的结构包括：

- `type`：`object`、`array`、`string`、`number`、`integer`、`boolean`、`null` 之一。
- `properties`：字段名称到子 Schema 的映射。
- `required`：必须出现的字段名称列表。
- `enum`：允许值的封闭集合。
- `minimum` / `maximum`（数字），`minLength` / `maxLength` / `pattern`（字符串）。
- `items`：应用于每个数组元素的子 Schema。
- `additionalProperties`：`false` 会禁止额外字段（不同模式的默认值不同）。

OpenAI 严格模式还增加三项要求：每个属性都必须列入 `required`，所有层级都必须设置 `additionalProperties: false`，并且不能存在未解析的 `$ref`。违反这些要求时，API 会在请求阶段返回 400。

### Pydantic：Python 绑定

Pydantic v2 可以通过 `model_json_schema()`，从数据类形态的模型生成 JSON Schema。Pydantic AI 对其进行了封装，因此你只需编写：

```python
class Invoice(BaseModel):
    customer: str
    line_items: list[LineItem]
    total_usd: Decimal
```

智能体框架会在边缘处把 Schema 转换为 OpenAI 严格模式、Anthropic `input_schema` 或 Gemini `responseSchema`。模型输出会作为类型化 `Invoice` 实例返回。验证错误会抛出 `ValidationError`，其中包含类型化错误路径。

### Zod：TypeScript 绑定

Zod（`z.object({customer: z.string(), ...})`）是 TypeScript 中的对应方案。OpenAI 的 Node SDK 提供 `zodResponseFormat(Invoice)`，可将其转换为 API 所需的 JSON Schema 负载。

### 拒绝

严格模式无法强迫模型回答。如果输入无法适配 Schema（“这封邮件是一首诗，而不是发票”），模型会输出包含原因的 `refusal` 字段。代码必须把它当作一等结果，而不是失败。拒绝也可以作为安全信号：如果要求模型从受保护内容的电子邮件中抽取信用卡号，它会返回附带安全原因的拒绝。

### 开放权重模型中的受约束解码

开放权重实现主要使用三种技术。

1. **基于语法的解码**（`outlines`、`guidance`、`lm-format-enforcer`）：根据 Schema 构建确定有限自动机；每一步都屏蔽会违反 FSM 的词元 Logit。
2. **结合 JSON 解析器的 Logit 屏蔽**：让流式 JSON 解析器与模型同步运行；每一步计算有效的下一词元集合。
3. **带验证器的推测解码**：廉价草稿模型提出词元，验证器负责强制遵守 Schema。

商业提供商会在幕后选择其中一种。2026 年的最佳实现，对于短结构化输出比普通生成更快；对于长输出，速度大致相同。

### 三种失效模式

1. **解析错误。** 输出不是有效 JSON。在严格模式下不可能发生；在非严格提供商上仍可能发生。
2. **Schema 违规。** 输出可以解析，却违反 Schema。在严格模式下不可能发生；在严格模式之外很常见。
3. **拒绝。** 模型拒绝回答。必须将其作为类型化结果处理。

### 重试策略

如果不在严格模式中（Anthropic 工具调用、非严格 OpenAI、旧版 Gemini），恢复模式如下：

```
generate -> parse -> validate -> if fail, inject error and retry, max 3x
```

通常重试一次就够了，三次可以捕获较弱模型的偶发错误。超过三次则说明 Schema 有问题：模型无法让某些输入满足它，需要修正提示词或 Schema。

### 小模型支持

受约束解码同样适用于小模型。在结构化任务上，一个使用语法强制的 3B 开放模型，会优于仅靠原始提示的 70B 模型。这正是结构化输出对生产环境意义重大的主要原因：它把可靠性与模型规模解耦。

```figure
constrained-decoding
```

## 投入使用

`code/main.py` 使用标准库交付一个最小 JSON Schema 2020-12 验证器（类型、required、enum、min/max、pattern、items、additionalProperties）。它包装一份 `Invoice` Schema，并让模拟大语言模型输出经过验证器，演示解析错误、Schema 违规与拒绝路径。生产环境中可把模拟输出替换为任意提供商的真实响应。

需要关注：

- 验证器返回带路径与消息的类型化 `[ValidationError]` 列表。重试提示词中应暴露这种形态。
- 拒绝分支**不会**重试，而是记录日志并返回类型化拒绝。阶段 14 · 09 会把拒绝作为安全信号。
- `additionalProperties: false` 检查会在对抗测试输入上触发，展示严格模式为何能阻止模型编造字段。

## 交付成果

本课会产出 `outputs/skill-structured-output-designer.md`。给定自由文本抽取目标（发票、客服工单、简历等），它会生成兼容严格模式的 JSON Schema 2020-12 及对应 Pydantic 模型，并提供类型化拒绝与重试处理框架。

## 练习

1. 运行 `code/main.py`。添加第四个测试用例，让 `total_usd` 为负数。确认验证器通过 `minimum` 约束路径拒绝它。

2. 扩展验证器，支持带鉴别字段的 `oneOf`。常见情况是：`line_item` 可以是产品或服务，由 `kind` 标记。严格模式在这里有细微规则，请查阅 OpenAI 结构化输出指南。

3. 使用 Pydantic BaseModel 编写同一份 Invoice Schema，并将 `model_json_schema()` 输出与手写 Schema 比较。找出 Pydantic 默认设置、而手写版本遗漏的那个字段。

4. 测量拒绝率。构造十条无法抽取的输入（一段歌词、一个数学证明、一封空邮件），使用真实提供商的严格模式运行。统计拒绝与幻觉输出。这就是拒绝感知重试的真实依据。

5. 从头到尾阅读 OpenAI 结构化输出指南。找出严格模式明确禁止、但普通 JSON Schema 允许的一种结构。然后设计一个非必要地使用该结构的 Schema，并将其重构为兼容严格模式的版本。

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|----------------|------------------------|
| JSON Schema 2020-12 | “Schema 规范” | 每家现代提供商都支持的 IETF 草案 Schema 方言 |
| 严格模式 | “保证符合 Schema” | OpenAI 通过受约束解码强制 Schema 的标志 |
| 受约束解码 | “Logit 屏蔽” | 在解码时屏蔽无效下一词元的强制机制 |
| 拒绝 | “模型拒答” | 输入无法适配 Schema 时返回的类型化结果 |
| 解析错误 | “无效 JSON” | 输出无法解析为 JSON；严格模式下不可能发生 |
| Schema 违规 | “形态错误” | 输出可以解析，但违反类型 / required / enum / 范围约束 |
| `additionalProperties: false` | “禁止额外字段” | 禁止未知字段；OpenAI 严格模式要求使用 |
| Pydantic BaseModel | “类型化输出” | 生成并验证 JSON Schema 的 Python 类 |
| Zod Schema | “TypeScript 输出类型” | 用于验证提供商输出的 TypeScript 运行时 Schema |
| 语法强制 | “开放权重受约束解码” | 基于 FSM 的 Logit 屏蔽，例如 outlines / guidance |

## 延伸阅读

- [OpenAI——结构化输出](https://platform.openai.com/docs/guides/structured-outputs)——严格模式、拒绝与 Schema 要求
- [OpenAI——结构化输出简介](https://openai.com/index/introducing-structured-outputs-in-the-api/)——2024 年 8 月发布文章，解释解码保证
- [Pydantic AI——Output](https://ai.pydantic.dev/output/)——序列化到各提供商的类型化 output_type 绑定
- [JSON Schema——2020-12 发布说明](https://json-schema.org/draft/2020-12/release-notes)——权威规范
- [Microsoft——Azure OpenAI 中的结构化输出](https://learn.microsoft.com/en-us/azure/foundry/openai/how-to/structured-outputs)——企业部署说明与严格模式注意事项
