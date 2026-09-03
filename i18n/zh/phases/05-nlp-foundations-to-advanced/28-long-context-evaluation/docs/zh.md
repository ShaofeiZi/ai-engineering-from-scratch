# 长上下文评估——NIAH、RULER、LongBench、MRCR

> Gemini 3 Pro 宣称拥有 1000 万词元的上下文。在 100 万词元处，8 针 MRCR 却跌至 26.3%。宣传值不等于可用值。长上下文评估会告诉你正在交付的模型究竟有多大实际容量。

**Type:** 学习
**Languages:** Python
**Prerequisites:** 阶段 5 · 13（问答系统）、阶段 5 · 23（分块策略）
**Time:** 约 60 分钟

## 问题

你有一份 200 页的合同。模型声称支持 100 万词元上下文。你把合同粘贴进去并问：“终止条款是什么？”模型给出了答案——但答案来自封面，因为终止条款位于深入上下文 12 万词元的位置，已经超出模型真正会关注的范围。

这就是 2026 年的上下文容量鸿沟。规格表写着 100 万或 1000 万，现实却是其中只有 60%～70% 可用，而且“可用”还取决于任务。

- **检索（干草堆中的单根针）：** 前沿模型在宣传上限内几乎都能完美完成。
- **多跳/聚合：** 大多数模型超过约 128k 后会急剧退化。
- **对分散事实进行推理：** 最先失效的任务。

长上下文评估会分别测量这些维度。本课将介绍各项基准、每项基准真正测量的能力，以及如何为自己的领域构建定制“大海捞针”测试。

## 概念

![NIAH 基线、RULER 多任务与 LongBench 综合评估](../../../../../../phases/05-nlp-foundations-to-advanced/28-long-context-evaluation/assets/long-context-eval.svg)

**大海捞针（NIAH，2023）。** 把一项事实（“魔法词是 pineapple”）放在长上下文中受控的深度，再要求模型把它找出来。遍历深度 × 长度组合。这是最初的长上下文基准。如今前沿模型在这项测试上已经接近饱和；它是必要基线，但远远不够。

**RULER（Nvidia，2024）。** 包含四个类别下的 13 种任务：检索（单键/多键/多值）、多跳追踪（变量追踪）、聚合（常见词频率）、问答。上下文长度可配置为 4k 到 128k 以上。它能揭示那些在 NIAH 上达到饱和、却在多跳任务上失败的模型。2024 年发布时，在 17 个宣称支持 32k 以上上下文的模型中，只有一半能在 32k 处维持质量。

**LongBench v2（2024）。** 503 道多项选择题，上下文长度从 8000 到 200 万词，覆盖六种任务：单文档问答、多文档问答、长上下文学习、长对话、代码仓库、长结构化数据。它是用于真实长上下文行为的生产基准。

**MRCR（多轮共指消解）。** 大规模多轮共指，提供 8 针、24 针和 100 针变体。它能暴露模型在同时处理多少项事实后注意力开始退化。

**NoLiMa。** “非词法针”。针与查询没有任何字面重叠，检索需要一步语义推理，因此比 NIAH 更难。

**HELMET。** 拼接许多文档，再询问来自其中任意一篇的问题，用于测试选择性注意力。

**BABILong。** 把 bAbI 推理链嵌入无关内容组成的干草堆，测试的是在干草堆中推理，而不只是检索。

### 真正应该报告什么

- **宣传的上下文窗口。** 规格表上的数字。
- **有效检索长度。** NIAH 通过某个阈值（例如 90%）时的长度。
- **有效推理长度。** 多跳或聚合任务达到该阈值时的长度。
- **退化曲线。** 按任务类型分别绘制准确率随上下文长度变化的曲线。

你的规格表应该给出两个数字：检索有效长度和推理有效长度。后者通常只有宣传窗口的 25%～50%。

```figure
gx-niah-decay
```

## 动手构建

### 第 1 步：为你的领域定制 NIAH

实现框架见 `code/main.py`：

```python
def build_haystack(filler_text, needle, depth_ratio, total_tokens):
    if not (0.0 <= depth_ratio <= 1.0):
        raise ValueError(f"depth_ratio must be in [0, 1], got {depth_ratio}")
    if total_tokens <= 0:
        raise ValueError(f"total_tokens must be positive, got {total_tokens}")

    filler_tokens = tokenize(filler_text)
    needle_tokens = tokenize(needle)
    if not filler_tokens:
        raise ValueError("filler_text produced no tokens")

    # Repeat filler until long enough to fill the haystack body.
    body_len = max(total_tokens - len(needle_tokens), 0)
    while len(filler_tokens) < body_len:
        filler_tokens = filler_tokens + filler_tokens
    filler_tokens = filler_tokens[:body_len]

    insert_at = min(int(body_len * depth_ratio), body_len)
    haystack = filler_tokens[:insert_at] + needle_tokens + filler_tokens[insert_at:]
    return " ".join(haystack)


def score_niah(model, haystack, question, expected):
    answer = model.complete(f"Context: {haystack}\nQ: {question}\nA:", max_tokens=50)
    return 1 if expected.lower() in answer.lower() else 0
```

遍历 `depth_ratio` ∈ {0, 0.25, 0.5, 0.75, 1.0} × `total_tokens` ∈ {1k, 4k, 16k, 64k}，再绘制热力图。这就是目标模型的 NIAH 能力卡。

### 第 2 步：多针变体

```python
def build_multi_needle(filler, needles, total_tokens):
    depths = [0.1, 0.4, 0.7]
    chunks = [filler[:int(total_tokens * 0.1)]]
    for depth, needle in zip(depths, needles):
        chunks.append(needle)
        next_chunk = filler[int(total_tokens * depth): int(total_tokens * (depth + 0.3))]
        chunks.append(next_chunk)
    return " ".join(chunks)
```

“三个魔法词分别是什么？”这类问题要求同时找出三项事实。单针测试成功，无法预测多针测试也会成功。

### 第 3 步：多跳变量追踪（RULER 风格）

```python
haystack = """X1 = 42. ... (filler) ... X2 = X1 + 10. ... (filler) ... X3 = X2 * 2."""
question = "What is X3?"
```

答案需要串联三次赋值。在 128k 上，前沿模型的准确率经常会降到 50%～70%。

### 第 4 步：在你的技术栈上运行 LongBench v2

```python
from datasets import load_dataset
longbench = load_dataset("THUDM/LongBench-v2")

def eval_model_on_longbench(model, subset="single-doc-qa"):
    tasks = [x for x in longbench["test"] if x["task"] == subset]
    correct = 0
    for x in tasks:
        answer = model.complete(x["context"] + "\n\nQ: " + x["question"], max_tokens=20)
        if normalize(answer) == normalize(x["answer"]):
            correct += 1
    return correct / len(tasks)
```

应按类别报告准确率，聚合分数会掩盖任务层面的巨大差异。

## 陷阱

- **只运行 NIAH 评估。** 在 100 万词元下通过 NIAH，完全不能说明模型具备多跳能力。始终运行 RULER 或自定义多跳测试。
- **均匀深度采样不足。** 许多实现只测试 depth=0.5。应测试 depth=0、0.25、0.5、0.75、1.0——“迷失在中间”效应确实存在。
- **针与填充内容存在词法重叠。** 如果二者共享关键词，检索就会变得轻而易举。应使用 NoLiMa 风格、没有字面重叠的针。
- **忽略延迟。** 100 万词元的提示需要 30～120 秒预填充。应在准确率之外，同时测量首词元时间。
- **供应商自行报告的数据。** OpenAI、Google、Anthropic 都会发布自己的分数。始终针对你的用例独立重跑。

## 学以致用

2026 年的技术栈：

| 场景 | 基准 |
|-----------|-----------|
| 快速健全性检查 | 在 3 个深度 × 3 种长度上运行自定义 NIAH |
| 生产模型选型 | 在目标长度上运行 RULER（13 项任务） |
| 真实问答质量 | LongBench v2 的单文档问答子集 |
| 多跳推理 | BABILong 或自定义变量追踪 |
| 对话 | 在目标长度上运行 MRCR 8 针测试 |
| 模型升级回归 | 固定的内部 NIAH + RULER 测试工具，每次升级模型都运行 |

生产经验法则：在目标长度上完成 NIAH 加至少一项推理任务之前，绝不要相信上下文窗口。

## 交付成果

保存为 `outputs/skill-long-context-eval.md`：

```markdown
---
name: long-context-eval
description: Design a long-context evaluation battery for a given model and use case.
version: 1.0.0
phase: 5
lesson: 28
tags: [nlp, long-context, evaluation]
---

Given a target model, target context length, and use case, output:

1. Tests. NIAH depth × length grid; RULER multi-hop; custom domain task.
2. Sampling. Depths 0, 0.25, 0.5, 0.75, 1.0 at each length.
3. Metrics. Retrieval pass rate; reasoning pass rate; time-to-first-token; cost-per-query.
4. Cutoff. Effective retrieval length (90% pass) and effective reasoning length (70% pass). Report both.
5. Regression. Fixed harness, rerun on every model upgrade, surface deltas.

Refuse to trust a context window from the model card alone. Refuse NIAH-only evaluation for any multi-hop workload. Refuse vendor self-reported long-context scores as independent evidence.
```

## 练习

1. **简单。** 构建一个包含 3 个深度（0.25、0.5、0.75）× 3 种长度（1k、4k、16k）的 NIAH，在任意模型上运行，并将通过率绘制为 3×3 热力图。
2. **中等。** 增加一个 3 针变体。测量每种长度下同时检索出全部 3 根针的能力，并与相同长度的单针通过率比较。
3. **困难。** 构建一项嵌入 64k 填充内容的变量追踪任务（X1 → X2 → X3，共 3 跳）。在三个前沿模型上测量准确率，并报告每个模型的有效推理长度。

## 关键术语

| 术语 | 人们通常怎么说 | 实际含义 |
|------|-----------------|-----------------------|
| NIAH | 大海捞针 | 在填充内容中植入一项事实，再要求模型检索它。 |
| RULER | 加强版 NIAH | 检索、多跳、聚合、问答四类下的 13 种任务。 |
| 有效上下文 | 真正的容量 | 准确率仍高于指定阈值时的上下文长度。 |
| 迷失在中间 | 深度偏差 | 模型对长输入中部内容的注意不足。 |
| 多针 | 同时处理多项事实 | 植入多项事实，测试注意力协调能力，而不只是单次检索。 |
| MRCR | 多轮共指 | 包含 8、24 或 100 根针的共指任务，可暴露注意力饱和。 |
| NoLiMa | 非词法针 | 针与查询没有共同的字面词元，需要推理。 |

## 延伸阅读

- [Kamradt（2023），大海捞针分析](https://github.com/gkamradt/LLMTest_NeedleInAHaystack)——最初的 NIAH 代码库。
- [Hsieh 等（2024），RULER：长上下文语言模型的真实上下文大小是多少？](https://arxiv.org/abs/2404.06654)——多任务基准。
- [Bai 等（2024），LongBench v2](https://arxiv.org/abs/2412.15204)——真实世界长上下文评估。
- [Modarressi 等（2024），NoLiMa：非词法针](https://arxiv.org/abs/2404.06666)——更难的针。
- [Kuratov 等（2024），BABILong](https://arxiv.org/abs/2406.10149)——干草堆中的推理。
- [Liu 等（2024），迷失在中间：语言模型如何使用长上下文](https://arxiv.org/abs/2307.03172)——深度偏差论文。
