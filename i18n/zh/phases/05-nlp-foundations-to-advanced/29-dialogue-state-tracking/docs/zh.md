# 对话状态跟踪

> “我想找北边的便宜餐厅……等等，改成中等价位……再加上意大利菜。”三轮对话、三次状态更新。DST 让槽位—值字典保持同步，确保预订正确完成。

**Type:** 构建
**Languages:** Python
**Prerequisites:** 阶段 5 · 17（聊天机器人）、阶段 5 · 20（结构化输出）
**Time:** 约 75 分钟

## 问题

在任务型对话系统中，用户目标被编码为一组槽位—值对：`{cuisine: italian, area: north, price: moderate}`。这组状态大致表示“意大利菜、北区、中等价位”。用户的每一轮话语都可能添加、更改或删除某个槽位。系统必须读取完整对话，并正确输出当前状态。

只要有一个槽位出错，系统就可能预订错误的餐厅、安排错误的航班或从错误的卡片扣款。DST 是连接用户话语与后端执行之间的枢纽。

即使已有大语言模型，它在 2026 年仍然重要，原因如下：

- 银行、医疗、航班预订等合规敏感领域需要确定性的槽位值，而不是自由形式生成。
- 使用工具的智能体在调用 API 前仍需完成槽位解析。
- 多轮纠错比看起来更难：“不对，改成周四吧。”

现代流水线是：经典 DST 概念 + 大语言模型抽取器 + 结构化输出护栏。

## 概念

![DST：对话历史 → 槽位—值状态](../assets/dst.svg)

**任务结构。** 模式定义领域（餐厅、酒店、出租车）及其槽位（菜系、区域、价格、人数）。每个槽位可以为空，可以从封闭集合中取值（价格：{便宜、中等、昂贵}），也可以接收自由形式的值（名称：“The Copper Kettle”）。

**DST 的两种形式。**

- **分类。** 对每个（槽位，候选值）对预测是/否。适合封闭词表槽位，是 2020 年前的标准方案。
- **生成。** 给定对话，以自由文本生成槽位值。适合开放词表槽位，是现代默认方案。

**指标。** 联合目标准确率（JGA）——*所有*槽位都正确的对话轮次比例，采取全对或全错计分。2026 年 MultiWOZ 2.4 排行榜的最好成绩约为 83%。

**架构。**

1. **基于规则（槽位正则 + 关键词）。** 适合狭窄领域的强基线，容易调试。
2. **TripPy / BERT-DST。** 使用 BERT 编码的复制式生成，是大语言模型之前的标准方案。
3. **LDST（LLaMA + LoRA）。** 使用领域—槽位提示进行指令微调的大语言模型，在 MultiWOZ 2.4 上达到 ChatGPT 水平。
4. **无本体方法（2024～2026）。** 跳过预定义模式，直接生成槽位名称和值，能够处理开放领域。
5. **提示 + 结构化输出（2024～2026）。** 大语言模型 + Pydantic 模式 + 约束解码。五行代码即可达到生产可用水平。

### 经典失败模式

- **跨轮次共指。** “我们还是选第一个吧。”需要解析“第一个选项”具体指什么。
- **覆盖还是追加。** 用户说“再加上意大利菜”，应该替换菜系还是追加一个值？
- **隐式确认。** “好，那就这样”——这是否表示用户接受了系统提供的预订？
- **纠正。** “那改成晚上 7 点吧。”必须更新时间，同时保留其他槽位。
- **共指上一条系统话语。** “对，就那个。”其中“那个”指哪一个？

```figure
n5-slot-tracker
```

## 动手构建

### 第 1 步：基于规则的槽位抽取器

实现见 `code/main.py`。在狭窄领域中，正则表达式 + 同义词字典可以覆盖 70% 的典型话语：

```python
CUISINE_SYNONYMS = {
    "italian": ["italian", "pasta", "pizza", "italy"],
    "chinese": ["chinese", "chow mein", "noodles"],
}


def extract_cuisine(utterance):
    for canonical, synonyms in CUISINE_SYNONYMS.items():
        if any(syn in utterance.lower() for syn in synonyms):
            return canonical
    return None
```

它在标准词表之外很脆弱，却适合确定性的槽位确认。

### 第 2 步：状态更新循环

```python
def update_state(state, utterance):
    new_state = dict(state)
    for slot, extractor in SLOT_EXTRACTORS.items():
        value = extractor(utterance)
        if value is not None:
            new_state[slot] = value
    for slot in NEGATION_CLEARS:
        if is_negated(utterance, slot):
            new_state[slot] = None
    return new_state
```

三个不变量：

- 绝不重置用户没有触及的槽位。
- 明确否定（“never mind the cuisine”）必须清空对应槽位。
- 用户纠正（“actually...”）必须覆盖，而不是追加。

### 第 3 步：使用结构化输出驱动大语言模型 DST

```python
from pydantic import BaseModel
from typing import Literal, Optional
import instructor

class RestaurantState(BaseModel):
    cuisine: Optional[Literal["italian", "chinese", "indian", "thai", "any"]] = None
    area: Optional[Literal["north", "south", "east", "west", "center"]] = None
    price: Optional[Literal["cheap", "moderate", "expensive"]] = None
    people: Optional[int] = None
    day: Optional[str] = None


def llm_dst(history, llm):
    prompt = f"""You track the slot values of a restaurant booking across turns.
Dialogue so far:
{render(history)}

Update the state based on the latest user turn. Output only the JSON state."""
    return llm(prompt, response_model=RestaurantState)
```

Instructor + Pydantic 可以保证得到有效的状态对象。不需要正则表达式，不会发生模式不匹配，也不会凭空产生槽位。

### 第 4 步：JGA 评估

```python
def joint_goal_accuracy(predicted_states, gold_states):
    correct = sum(1 for p, g in zip(predicted_states, gold_states) if p == g)
    return correct / len(predicted_states)
```

校准问题是：系统有多少轮把所有槽位都判断正确？在 MultiWOZ 2.4 上，2026 年最好的系统达到 80%～83%。在你的狭窄词表上，内部系统应该超过这一水平，否则大语言模型基线就能胜过你。

### 第 5 步：处理纠正

```python
CORRECTION_CUES = {"actually", "no wait", "on second thought", "change that to"}


def is_correction(utterance):
    return any(cue in utterance.lower() for cue in CORRECTION_CUES)
```

检测到纠正时，应覆盖最后更新的槽位，而不是追加。没有大语言模型帮助时，这很难正确实现。现代模式是每次都让大语言模型根据完整历史重新生成整个状态，而不是增量更新；这样可以自然处理纠正。

## 陷阱

- **重新生成完整历史的成本。** 每轮都让大语言模型重新生成状态，会产生 O(n²) 的总词元成本。应限制历史长度或总结早期轮次。
- **模式漂移。** 事后加入新槽位会破坏旧训练数据。应对模式进行版本管理。
- **大小写敏感。** “Italian”“italian”“ITALIAN”——所有位置都必须归一化。
- **隐式继承。** 如果用户之前已经指定“for 4 people”，提出新的时间要求时不应清空人数。始终传入完整历史。
- **自由形式与封闭集合。** 名称、时间和地址需要自由形式槽位；菜系和区域则是封闭集合。应在同一模式中混合二者。

## 学以致用

2026 年的技术栈：

| 场景 | 方法 |
|-----------|----------|
| 狭窄领域（一两个意图） | 基于规则 + 正则表达式 |
| 宽领域、有带标签数据 | LDST（LLaMA + 在 MultiWOZ 风格数据上训练 LoRA） |
| 宽领域、无标签、达到生产可用 | 大语言模型 + Instructor + Pydantic 模式 |
| 语音场景 | ASR + 归一化器 + 大语言模型 DST |
| 多领域预订流程 | 模式引导的大语言模型 + 逐领域 Pydantic 模型 |
| 合规敏感 | 基于规则为主，大语言模型回退并配合确认流程 |

## 交付成果

保存为 `outputs/skill-dst-designer.md`：

```markdown
---
name: dst-designer
description: Design a dialogue state tracker — schema, extractor, update policy, evaluation.
version: 1.0.0
phase: 5
lesson: 29
tags: [nlp, dialogue, task-oriented]
---

Given a use case (domain, languages, vocab openness, compliance needs), output:

1. Schema. Domain list, slots per domain, open vs closed vocabulary per slot.
2. Extractor. Rule-based / seq2seq / LLM-with-Pydantic. Reason.
3. Update policy. Regenerate-whole-state / incremental; correction handling; negation handling.
4. Evaluation. Joint Goal Accuracy on a held-out dialogue set, slot-level precision/recall, confusion on the hardest slot.
5. Confirmation flow. When to explicitly ask the user to confirm (destructive actions, low-confidence extractions).

Refuse LLM-only DST for compliance-sensitive slots without a rule-based secondary check. Refuse any DST that cannot roll back a slot on user correction. Flag schemas without version tags.
```

## 练习

1. **简单。** 为三个槽位（菜系、区域、价格）构建 `code/main.py` 中的规则式状态跟踪器。在 10 段手工编写的对话上测试并测量 JGA。
2. **中等。** 在同一数据集上使用 Instructor + Pydantic + 小型大语言模型，比较 JGA 并检查最困难的轮次。
3. **困难。** 同时实现两种方案并进行路由：默认使用规则方法；当规则方法以低置信度输出少于两个槽位时，回退到大语言模型。测量组合后的 JGA 与每轮推理成本。

## 关键术语

| 术语 | 人们通常怎么说 | 实际含义 |
|------|-----------------|-----------------------|
| DST | 对话状态跟踪 | 在多轮对话中维护槽位—值字典。 |
| 槽位 | 用户意图的单元 | 后端需要的命名参数（菜系、日期）。 |
| 领域 | 任务范围 | 餐厅、酒店、出租车——各自包含一组槽位。 |
| JGA | 联合目标准确率 | 所有槽位都正确的轮次比例，采取全对或全错计分。 |
| MultiWOZ | 基准 | 多领域 Wizard-of-Oz 数据集；标准 DST 评估。 |
| 无本体 DST | 没有模式 | 不使用固定列表，直接生成槽位名称和值。 |
| 纠正 | “Actually...” | 覆盖先前已填充槽位的一轮话语。 |

## 延伸阅读

- [Budzianowski 等（2018），MultiWOZ——大规模多领域 Wizard-of-Oz](https://arxiv.org/abs/1810.00278)——经典基准。
- [Feng 等（2023），迈向大语言模型驱动的对话状态跟踪（LDST）](https://arxiv.org/abs/2310.14970)——使用 LLaMA + LoRA 为 DST 进行指令微调。
- [Heck 等（2020），TripPy——用于与具体值无关的神经对话状态跟踪的三重复制策略](https://arxiv.org/abs/2005.02877)——复制式 DST 主力方法。
- [King、Flanigan（2024），使用大语言模型实现无监督端到端任务型对话](https://arxiv.org/abs/2404.10753)——基于 EM 的无监督 TOD。
- [MultiWOZ 排行榜](https://github.com/budzianowski/multiwoz)——标准 DST 结果。
