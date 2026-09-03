# 结构化输出与约束解码

> 要求大语言模型返回 JSON，它大多数时候会照做。但在生产环境中，“大多数”本身就是问题。约束解码通过在采样前修改 logits，把“大多数”变成“始终如此”。

**Type:** 构建
**Languages:** Python
**Prerequisites:** 阶段 5 · 17（聊天机器人）、阶段 5 · 19（子词分词）
**Time:** 约 60 分钟

## 问题

一个分类器提示大语言模型：“只返回 {positive, negative, neutral} 之一。”模型却返回：“情感倾向是 positive，这条评论显然非常正面，因为用户明确表示他们……”。你的解析器崩溃了，分类器的 F1 变成 0.0。

自由形式生成不是契约，只是一项建议。生产系统需要真正的契约。

2026 年有三个层次的方案。

1. **提示。** 客气地要求：“只返回 JSON 对象。”前沿模型约有 80% 的时候会遵守，小模型表现更差。
2. **原生结构化输出 API。** OpenAI `response_format`、Anthropic 工具调用、Gemini JSON 模式。对受支持的模式很可靠，但会绑定供应商。
3. **约束解码。** 在每个生成步骤修改 logits，让模型*无法*发出无效词元。从机制上保证 100% 有效，适用于任意本地模型。

本课将帮助你理解三种方案，并说明何时应该选择哪一种。

## 概念

![约束解码在每一步屏蔽无效词元](../../../../../../phases/05-nlp-foundations-to-advanced/20-structured-outputs-constrained-decoding/assets/constrained-decoding.svg)

**约束解码如何工作。** 在每个生成步骤，大语言模型都会为整个词表（约 10 万个词元）输出一个 logit 向量。一个*logit 处理器*位于模型与采样器之间。它根据当前在目标语法——JSON Schema、正则表达式、上下文无关文法——中的位置，计算哪些词元有效，并把所有无效词元的 logit 设为负无穷。对剩余 logits 执行 softmax 后，概率质量只会落在有效的后续内容上。

2026 年的实现包括：

- **Outlines。** 把 JSON Schema 或正则表达式编译成有限状态机。每个词元都能以 O(1) 查询下一个有效词元。它基于 FSM，因此递归模式需要展开。
- **XGrammar / llguidance。** 上下文无关文法引擎，可以处理递归 JSON Schema，解码开销接近零。OpenAI 在 2025 年的结构化输出实现中提到了 llguidance 的贡献。
- **vLLM 引导解码。** 通过 Outlines、XGrammar 或 lm-format-enforcer 后端，内置 `guided_json`、`guided_regex`、`guided_choice` 和 `guided_grammar`。
- **Instructor。** 基于 Pydantic、覆盖任意大语言模型的封装，在验证失败时重试。它跨供应商工作，但不会修改 logits——依赖重试与适配结构化输出的提示。

### 反直觉的结果

约束解码通常比无约束生成还要*快*，原因有二。首先，它缩小了下一个词元的搜索空间。其次，聪明的实现会完全跳过必然词元的生成过程（例如 `{"name": "` 这样的结构骨架——其中每个字节都已经确定）。

### 代价高昂的陷阱

字段顺序很重要。如果把 `answer` 放在 `reasoning` 前面，模型会在思考前就确定答案。JSON 有效，答案却错了，任何验证都发现不了。

```json
// BAD
{"answer": "yes", "reasoning": "because ..."}

// GOOD
{"reasoning": "... therefore ...", "answer": "yes"}
```

模式字段顺序是逻辑，而不是格式。

```figure
constrained-decoder
```

## 动手构建

### 第 1 步：从零实现正则约束生成

独立的 FSM 实现见 `code/main.py`，核心思想只有 30 行：

```python
def mask_logits(logits, valid_token_ids):
    mask = [float("-inf")] * len(logits)
    for tid in valid_token_ids:
        mask[tid] = logits[tid]
    return mask


def generate_constrained(model, tokenizer, prompt, fsm):
    ids = tokenizer.encode(prompt)
    state = fsm.initial_state
    while not fsm.is_accept(state):
        logits = model.next_token_logits(ids)
        valid = fsm.valid_tokens(state, tokenizer)
        logits = mask_logits(logits, valid)
        tok = sample(logits)
        ids.append(tok)
        state = fsm.transition(state, tok)
    return tokenizer.decode(ids)
```

FSM 会追踪当前已经满足了语法中的哪些部分。`valid_tokens(state, tokenizer)` 负责计算哪些词表词元可以推动 FSM 前进，同时仍保留一条通往接受状态的路径。

### 第 2 步：使用 Outlines 约束 JSON Schema

```python
from pydantic import BaseModel
from typing import Literal
import outlines


class Review(BaseModel):
    sentiment: Literal["positive", "negative", "neutral"]
    confidence: float
    evidence_span: str


model = outlines.models.transformers("meta-llama/Llama-3.2-3B-Instruct")
generator = outlines.generate.json(model, Review)

result = generator("Classify: 'The wait staff was attentive and the food arrived hot.'")
print(result)
# Review(sentiment='positive', confidence=0.93, evidence_span='attentive ... hot')
```

验证错误永远为零。FSM 让无效输出从根本上无法到达。

### 第 3 步：使用 Instructor 实现供应商无关的 Pydantic

```python
import instructor
from anthropic import Anthropic
from pydantic import BaseModel, Field


class Invoice(BaseModel):
    vendor: str
    total_usd: float = Field(ge=0)
    line_items: list[str]


client = instructor.from_anthropic(Anthropic())
invoice = client.messages.create(
    model="claude-opus-4-7",
    max_tokens=1024,
    response_model=Invoice,
    messages=[{"role": "user", "content": "Extract from: 'Acme Corp $420. Widget, Gizmo.'"}],
)
```

这里使用了不同机制。Instructor 不会接触 logits，而是把模式格式化进提示，解析输出，并在验证失败时重试（默认 3 次）。它适用于任何供应商，但重试会增加延迟与成本。跨供应商可移植性是它的核心卖点。

### 第 4 步：供应商原生 API

```python
from openai import OpenAI

client = OpenAI()
response = client.responses.create(
    model="gpt-5",
    input=[{"role": "user", "content": "Classify: 'The food was cold.'"}],
    text={"format": {"type": "json_schema", "name": "sentiment",
          "schema": {"type": "object", "required": ["sentiment"],
                     "properties": {"sentiment": {"type": "string",
                                                  "enum": ["positive", "negative", "neutral"]}}}}},
)
print(response.output_parsed)
```

这是服务端约束解码。对受支持的模式，它的可靠性与 Outlines 相当；无须管理本地模型，但会绑定供应商。

## 陷阱

- **递归模式。** Outlines 会把递归展开到固定深度。树形输出（嵌套评论、AST）需要基于 CFG 的 XGrammar 或 llguidance。
- **巨大枚举。** 包含 1 万个选项的枚举编译很慢，甚至会超时。应改用检索器：先预测前 k 个候选项，再把约束缩小到这些项。
- **语法过于严格。** 如果使用正则强制 `date: "YYYY-MM-DD"`，模型在缺少日期时便无法输出 `"unknown"`，于是会通过编造日期来满足约束。应允许 `null` 或哨兵值。
- **过早承诺。** 见上面的字段顺序陷阱。始终把推理放在前面。
- **供应商 JSON 模式没有模式约束。** 纯 JSON 模式只保证 JSON 语法有效，并不保证它*符合你的用例*。始终提供完整模式。

## 学以致用

2026 年的技术栈：

| 场景 | 选择 |
|-----------|------|
| OpenAI/Anthropic/Google 模型，简单模式 | 供应商原生结构化输出 |
| 任意供应商、Pydantic 工作流、可容忍重试 | Instructor |
| 本地模型、要求 100% 有效、扁平模式 | Outlines（FSM） |
| 本地模型、递归模式 | XGrammar 或 llguidance |
| 自托管推理服务器 | vLLM 引导解码 |
| 批处理且可以接受重试 | Instructor + 最便宜的模型 |

## 交付成果

保存为 `outputs/skill-structured-output-picker.md`：

```markdown
---
name: structured-output-picker
description: Choose a structured output approach, schema design, and validation plan.
version: 1.0.0
phase: 5
lesson: 20
tags: [nlp, llm, structured-output]
---

Given a use case (provider, latency budget, schema complexity, failure tolerance), output:

1. Mechanism. Native vendor structured output, Instructor retries, Outlines FSM, or XGrammar CFG. One-sentence reason.
2. Schema design. Field order (reasoning first, answer last), nullable fields for "unknown", enum vs regex, required fields.
3. Failure strategy. Max retries, fallback model, graceful `null` handling, out-of-distribution refusal.
4. Validation plan. Schema compliance rate (target 100%), semantic validity (LLM-judge), field-coverage rate, latency p50/p99.

Refuse any design that puts `answer` or `decision` before reasoning fields. Refuse to use bare JSON mode without a schema. Flag recursive schemas behind an FSM-only library.
```

## 练习

1. **简单。** 在不使用约束解码的情况下，提示一个小型开放权重模型（例如 Llama-3.2-3B）输出 `Review(sentiment, confidence, evidence_span)`。在 100 条评论上测量可以解析为有效 JSON 的比例。
2. **中等。** 在同一语料库上使用 Outlines JSON 模式。比较合规率、延迟和语义准确率。
3. **困难。** 从零实现一个电话号码正则约束解码器（`\d{3}-\d{3}-\d{4}`），验证 1000 次采样中没有任何无效输出。

## 关键术语

| 术语 | 人们通常怎么说 | 实际含义 |
|------|-----------------|-----------------------|
| 约束解码 | 强制输出有效 | 在每个生成步骤屏蔽无效词元的 logits。 |
| Logit 处理器 | 执行约束的组件 | 函数：`(logits, state) -> masked_logits`。 |
| FSM | 有限状态机 | 编译后的语法表示；查询下一个有效词元的复杂度为 O(1)。 |
| CFG | 上下文无关文法 | 可以处理递归的文法；比 FSM 慢，但表达能力更强。 |
| 模式字段顺序 | 它重要吗？ | 重要——第一个字段会让模型作出承诺；始终把推理放在答案前面。 |
| 引导解码 | vLLM 对它的称呼 | 同一个概念，只是集成进推理服务器。 |
| JSON 模式 | OpenAI 的早期版本 | 保证 JSON 语法有效；**不**保证符合模式。 |

## 延伸阅读

- [Willard、Louf（2023），面向大语言模型的高效引导生成](https://arxiv.org/abs/2307.09702)——Outlines 论文。
- [XGrammar 论文（2024）](https://arxiv.org/abs/2411.15100)——快速的 CFG 约束解码。
- [vLLM——结构化输出](https://docs.vllm.ai/en/latest/features/structured_outputs.html)——推理服务器集成。
- [OpenAI——结构化输出指南](https://platform.openai.com/docs/guides/structured-outputs)——API 参考与注意事项。
- [Instructor 库](https://python.useinstructor.com/)——跨供应商的 Pydantic + 重试方案。
- [JSONSchemaBench（2025）](https://arxiv.org/abs/2501.10868)——对 6 个约束解码框架的基准测试。
