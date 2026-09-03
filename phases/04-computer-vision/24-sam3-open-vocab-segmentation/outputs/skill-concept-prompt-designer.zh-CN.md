---
name: skill-concept-prompt-designer
description: 将用户话语转化为格式良好的 SAM 3 概念提示词，包含拆分、消歧和回退处理
version: 1.0.0
phase: 4
lesson: 24
tags: [sam3, open-vocab, prompt-engineering, segmentation]
---

# 概念提示词设计器

SAM 3 的准确度在很大程度上取决于概念提示词的表述方式。本技能将自由格式的用户话语规范化为 SAM 3 能够良好处理的提示词。

## 何时使用

- 构建接受自然语言对象查询的 UI。
- 通过 API 暴露 SAM 3，且上游调用方发送的是句子。
- 调试 SAM 3 匹配效果不佳的情况——通常问题出在提示词格式错误，而非模型本身。

## 输入

- `utterance`：原始用户字符串。
- `context`：可选的领域提示（例如 "surveillance"、"medical"、"retail"）。
- `max_concepts`：每条话语最多提取的概念数量；默认为 5。

## SAM 3 偏好的规则

- **简短的名词短语，而非句子。** `"cat"` 优于 `"there is a cat"`。
- **具体名词。** `"skateboard"` 优于 `"thing to ride on"`。
- **修饰语紧邻名词之前。** `"red car"` 优于 `"car that is red"`。
- **小写。** SAM 3 鲁棒性较强，但对小写输入在经验上略优。
- **单数或复数。** 两者均可；当预期存在多个实例时，复数更有帮助。

## 步骤

1. **按常见分隔符分词**——逗号、分号、"and"、"or"、"&"。
2. **丢弃填充前缀**——"find"、"show me"、"segment"、"detect"、"locate"、"a"、"an"、"the"。
3. **仅保留具有视觉性的介词修饰语**——`"striped red umbrella"` 保留，`"umbrella from yesterday"` 不保留（`"from yesterday"` 不在图像中）。
4. **使用可选的 `context` 消歧冲突项**：
   - 在 surveillance 上下文中，`"window"` -> `"building window"`。
   - 在 medical 上下文中，`"window"` -> 通常为错误；建议用户澄清。
5. **回退**至原始字符串：当拆分得到零个概念*且*话语中至少包含一个具体名词时。如果无法提取出任何具体名词，则不要生成概念——仅返回警告并要求用户澄清（参见规则）。
6. **上限为 `max_concepts`。** 如果提取的概念数量超过调用方所需，按话语顺序保留前 `max_concepts` 个概念，并将其余概念以原因 `dropped` 放入 `"exceeded max_concepts"`。这样在用户粘贴长枚举列表时可以保持延迟有界。

## 输出格式

```
[designed prompts]
  utterance:    <original>
  concepts:     ["concept_1", "concept_2", ...]
  dropped:      ["filler_1", ...]
  warnings:     ["concept too abstract", "may match many classes", ...]

[sam3 calls]
  For each concept run: sam3.detect(image, concept)
  Merge outputs with distinct concept tags per detection.
```

## 示例

```
in:  "can you find me a cat or two dogs?"
out: ["cat", "dogs"]
dropped: ["can you find me", "a", "or two", "?"]
note: "dogs" kept plural because the utterance says "two dogs" — plural hint preserved.

in:  "segment the big red truck and the blue sedan"
out: ["big red truck", "blue sedan"]
dropped: ["segment", "the", "and"]

in:  "thing near the door"
out: ["door"]
warnings: ["'thing' is too abstract for SAM 3; fell back to 'door'"]

in:  "striped red umbrella, green hat, pink balloon"
out: ["striped red umbrella", "green hat", "pink balloon"]
```

## 规则

- 切勿将超过 8 个词的句子传递给 SAM 3——超过该长度准确度会下降。
- 当话语中不包含可提取的具体名词时，不要运行 SAM 3；返回警告并要求澄清。
- 不要对引号字符串内部的标点进行拆分；如果 `"black and white cat"` 被引号括起，则将其作为一个概念保留。
- 始终记录原始话语和派生的概念，以便生产环境调试。
