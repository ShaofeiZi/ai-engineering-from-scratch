# 大语言模型评估——RAGAS、DeepEval、G-Eval

> 精确匹配和 F1 无法识别语义等价，人工审阅又无法扩展。大语言模型裁判是生产方案——前提是经过充分校准，让分数值得信赖。

**Type:** 构建
**Languages:** Python
**Prerequisites:** 阶段 5 · 13（问答系统）、阶段 5 · 14（信息检索）
**Time:** 约 75 分钟

## 问题

你的 RAG 系统回答：“2007 年 6 月 29 日。”
标准答案是：“2007 年 6 月 29 日。”
精确匹配得分为 0，F1 约为 75%，而人类会给出 100%。

现在把它乘以 1 万个测试用例，再乘以检索器、分块方式、提示或模型的每一次改动。你需要一个理解语义、能够低成本规模化运行、不虚报回归，并能呈现正确失败模式的评估器。

2026 年有三个框架主导这个问题。

- **RAGAS。** Retrieval-Augmented Generation Assessment。通过 NLI 与大语言模型裁判后端提供四项 RAG 指标（忠实度、答案相关性、上下文精确率、上下文召回率）。有研究依据，且较轻量。
- **DeepEval。** 面向大语言模型的 Pytest。提供 G-Eval、任务完成度、幻觉与偏见等指标，原生适配 CI/CD。
- **G-Eval。** 一种方法（也是 DeepEval 中的指标）：带思维链、自定义标准和 0～1 分数的大语言模型裁判。

三者都依赖大语言模型裁判。本课将帮助你理解这种方法，以及围绕它建立的可信保障层。

## 概念

![四个评估维度与大语言模型裁判架构](../../../../../../phases/05-nlp-foundations-to-advanced/27-llm-evaluation-frameworks/assets/llm-evaluation.svg)

**大语言模型裁判。** 用一个依据评分标准评价输出的大语言模型取代静态指标。给定 `(query, context, answer)`，提示裁判模型：“按忠实度给出 0～1 分”，然后返回分数。

它为何有效：大语言模型能够以很低的成本近似人类判断。GPT-4o-mini 每个评分样本约 0.003 美元，因此包含 1000 个样本的回归评估成本不到 5 美元。

它为何会悄然失效：

1. **裁判偏差。** 裁判偏爱更长的答案、来自同一模型家族的答案，以及与提示风格相符的答案。
2. **JSON 解析失败。** 无效 JSON → NaN 分数 → 被悄然排除在聚合结果之外。RAGAS 用户对这种痛苦并不陌生。应使用 try/except 和显式失败模式把关。
3. **模型版本漂移。** 升级裁判会改变所有指标。必须固定裁判模型与版本。

**RAG 四项指标。**

| 指标 | 问题 | 后端 |
|--------|----------|---------|
| 忠实度 | 答案中的每项陈述都来自检索上下文吗？ | 基于 NLI 的蕴含判断 |
| 答案相关性 | 答案回应了问题吗？ | 根据答案生成假设问题，再与真实问题比较 |
| 上下文精确率 | 检索到的文本块中有多少真正相关？ | 大语言模型裁判 |
| 上下文召回率 | 检索结果返回了全部必要信息吗？ | 大语言模型裁判对照标准答案 |

**G-Eval。** 定义自定义标准，例如“答案是否引用了正确来源？”框架会自动把它扩展成思维链评估步骤，再给出 0～1 分。它适合评估 RAGAS 未覆盖的领域专用质量维度。

**校准。** 在拿到与人工标签的相关性之前，绝不要相信原始裁判分数。运行 100 个手工标注样本，绘制裁判评分与人工评分的关系，并计算 Spearman rho。如果 rho < 0.7，说明裁判的评分标准仍需改进。

```figure
n5-judge-gauge
```

## 动手构建

### 第 1 步：使用 NLI 计算忠实度（RAGAS 风格）

```python
from typing import Callable
from transformers import pipeline

nli = pipeline("text-classification",
               model="MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli",
               top_k=None)

# `llm` is any callable: prompt str -> generated str.
# Example: llm = lambda p: client.messages.create(model="claude-haiku-4-5", ...).content[0].text
LLM = Callable[[str], str]


def atomic_claims(answer: str, llm: LLM) -> list[str]:
    prompt = f"""Break this answer into simple factual claims (one per line):
{answer}
"""
    return llm(prompt).splitlines()


def faithfulness(answer: str, context: str, llm: LLM) -> float:
    claims = atomic_claims(answer, llm)
    if not claims:
        return 0.0
    supported = 0
    for claim in claims:
        result = nli({"text": context, "text_pair": claim})[0]
        entail = next((s for s in result if s["label"] == "entailment"), None)
        if entail and entail["score"] > 0.5:
            supported += 1
    return supported / len(claims)
```

把答案拆成原子陈述，逐条使用 NLI 对照检索上下文检查。忠实度就是得到支持的陈述比例。

### 第 2 步：答案相关性

```python
import numpy as np
from sentence_transformers import SentenceTransformer

# encoder: any model implementing .encode(texts, normalize_embeddings=True) -> ndarray
# e.g., encoder = SentenceTransformer("BAAI/bge-small-en-v1.5")

def answer_relevance(question: str, answer: str, encoder, llm: LLM, n: int = 3) -> float:
    prompt = f"Write {n} questions this answer could be the answer to:\n{answer}"
    generated = [line for line in llm(prompt).splitlines() if line.strip()][:n]
    if not generated:
        return 0.0
    q_emb = np.asarray(encoder.encode([question], normalize_embeddings=True)[0])
    g_embs = np.asarray(encoder.encode(generated, normalize_embeddings=True))
    sims = [float(q_emb @ g_emb) for g_emb in g_embs]
    return sum(sims) / len(sims)
```

如果答案隐含的问题不同于用户真正提出的问题，相关性就会下降。

### 第 3 步：G-Eval 自定义指标

```python
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCaseParams, LLMTestCase

metric = GEval(
    name="Correctness",
    criteria="The answer should be factually accurate and match the expected output.",
    evaluation_steps=[
        "Read the expected output.",
        "Read the actual output.",
        "List factual claims in the actual output.",
        "For each claim, mark supported or unsupported by the expected output.",
        "Return score = fraction supported.",
    ],
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
)

test = LLMTestCase(input="When was the first iPhone released?",
                   actual_output="June 29th, 2007.",
                   expected_output="June 29, 2007.")
metric.measure(test)
print(metric.score, metric.reason)
```

这些评估步骤就是评分标准。显式步骤比含糊的“给出 0～1 分”提示更稳定。

### 第 4 步：CI 门禁

```python
import deepeval
from deepeval.metrics import FaithfulnessMetric, ContextualRelevancyMetric


def test_rag_system():
    cases = load_regression_cases()
    faith = FaithfulnessMetric(threshold=0.85)
    rel = ContextualRelevancyMetric(threshold=0.7)
    for case in cases:
        faith.measure(case)
        assert faith.score >= 0.85, f"faithfulness regression on {case.id}"
        rel.measure(case)
        assert rel.score >= 0.7, f"relevancy regression on {case.id}"
```

把它作为 pytest 文件交付，在每个 PR 上运行，出现回归时阻止合并。

### 第 5 步：从零实现玩具评估

见 `code/main.py`。其中仅使用标准库近似计算忠实度（答案陈述与上下文的重叠）和相关性（答案词元与问题词元的重叠）。它不适合生产，却能展示基本形态。

## 陷阱

- **没有校准。** 与人工标签只有 0.3 相关性的裁判等同于噪声。交付前必须完成校准运行。
- **自我评估。** 使用同一个大语言模型生成并评判，会把分数夸大 10%～20%。应使用不同的模型家族担任裁判。
- **成对评判中的位置偏差。** 裁判偏爱首先呈现的选项。始终随机化顺序，并交换顺序再运行一次。
- **原始聚合值掩盖失败。** 平均分 0.85 往往隐藏了 5% 的灾难性失败。必须检查最低分位组。
- **黄金数据集腐化。** 未经版本管理、随时间漂移的评估集会破坏纵向比较。每次改动都要为数据集打标签。
- **大语言模型成本。** 大规模运行时，裁判调用会成为主要成本。应选择能够达到校准阈值的最便宜模型，例如 GPT-4o-mini、Claude Haiku、Mistral-small。

## 学以致用

2026 年的技术栈：

| 用例 | 框架 |
|---------|-----------|
| RAG 质量监控 | RAGAS（4 项指标） |
| CI/CD 回归门禁 | DeepEval + pytest |
| 自定义领域标准 | DeepEval 中的 G-Eval |
| 在线生产流量监控 | 使用无参考模式的 RAGAS |
| 人在回路中的抽查 | 带标注界面的 LangSmith 或 Phoenix |
| 红队/安全评估 | Promptfoo + DeepEval |

典型技术栈：RAGAS 负责监控，DeepEval 负责 CI，G-Eval 负责新维度。三者都运行，因为它们产生的分歧很有价值。

## 交付成果

保存为 `outputs/skill-eval-architect.md`：

```markdown
---
name: eval-architect
description: Design an LLM evaluation plan with calibrated judge and CI gates.
version: 1.0.0
phase: 5
lesson: 27
tags: [nlp, evaluation, rag]
---

Given a use case (RAG / agent / generative task), output:

1. Metrics. Faithfulness / relevance / context-precision / context-recall + any custom G-Eval metrics with criteria.
2. Judge model. Named model + version, rationale for cost vs accuracy.
3. Calibration. Hand-labeled set size, target Spearman rho vs human > 0.7.
4. Dataset versioning. Tag strategy, change log, stratification.
5. CI gate. Thresholds per metric, regression-window logic, bottom-quantile alert.

Refuse to rely on a judge untested against ≥50 human-labeled examples. Refuse self-evaluation (same model generates + judges). Refuse aggregate-only reporting without bottom-10% surfacing. Flag any pipeline where judge upgrade lands without parallel baseline eval.
```

## 练习

1. **简单。** 在 10 个已知存在幻觉的 RAG 示例上使用 RAGAS，验证忠实度指标能否捕捉每个问题。
2. **中等。** 对 50 个问答结果以 0～1 手工标注正确性，再使用 G-Eval 评分。测量裁判与人工之间的 Spearman rho。
3. **困难。** 使用 DeepEval 构建 pytest CI 门禁。故意让检索器产生回归并验证门禁失败，再通过检查最低 10% 样本是否低于阈值来增加尾部告警。

## 关键术语

| 术语 | 人们通常怎么说 | 实际含义 |
|------|-----------------|-----------------------|
| 大语言模型裁判 | 使用大语言模型评分 | 给定评分标准，提示裁判模型为输出打 0～1 分。 |
| RAGAS | RAG 指标库 | 提供 4 项无参考 RAG 指标的开源评估框架。 |
| 忠实度 | 答案有依据吗？ | 被检索上下文蕴含的答案陈述比例。 |
| 上下文精确率 | 检索块相关吗？ | 前 K 个文本块中真正有用的比例。 |
| 上下文召回率 | 检索找全了吗？ | 检索块支持的标准答案陈述比例。 |
| G-Eval | 自定义大语言模型裁判 | 自定义标准 + 思维链评估步骤 + 0～1 分。 |
| 校准 | 信任，但要验证 | 裁判分数与人工分数之间的 Spearman 相关系数。 |

## 延伸阅读

- [Es 等（2023），RAGAS：检索增强生成的自动化评估](https://arxiv.org/abs/2309.15217)——RAGAS 论文。
- [Liu 等（2023），G-Eval：使用 GPT-4 实现与人类更一致的自然语言生成评估](https://arxiv.org/abs/2303.16634)——G-Eval 论文。
- [DeepEval 文档](https://deepeval.com/docs/metrics-introduction)——开放的生产级技术栈。
- [Zheng 等（2023），使用 MT-Bench 与 Chatbot Arena 评判大语言模型裁判](https://arxiv.org/abs/2306.05685)——偏差、校准与局限。
- [MLflow 生成式 AI 评分器](https://mlflow.org/blog/third-party-scorers)——集成 RAGAS、DeepEval 与 Phoenix 的统一框架。
