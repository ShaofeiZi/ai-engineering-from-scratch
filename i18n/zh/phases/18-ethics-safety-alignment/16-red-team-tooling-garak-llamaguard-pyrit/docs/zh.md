# 红队工具：Garak、Llama Guard、PyRIT

> 2026 年生产环境中的红队栈，基本由三类工具构成。Llama Guard（Meta）是一类安全分类器：Llama Guard 3 基于 Llama-3.1-8B，并针对 14 个 MLCommons hazard categories 做了微调；2025 年发布的 Llama Guard 4 则是从 Llama 4 Scout 裁剪而来的 12B 原生多模态分类器。Garak（NVIDIA）是开源 LLM 漏洞扫描器，提供针对 hallucination、data leakage、prompt injection、toxicity 和 jailbreak 的静态、动态、适应性 probes。PyRIT（Microsoft）则负责多轮红队 campaign，支持 Crescendo、TAP 以及自定义 converter chain，用于更深层的利用与对抗。Llama Guard 3 记录在 Meta 的 “Llama 3 Herd of Models”（arXiv:2407.21783）中；Llama Guard 3-1B-INT4 见 arXiv:2411.17713；Garak 的 probe architecture 则可在 github.com/NVIDIA/garak 找到。这些工具，构成了 2026 年把红队研究（第 12 到 15 课）接入生产部署（第 17 课及以后）的标准接口。

**Type:** 构建
**Languages:** Python (stdlib, tool-architecture simulator and Llama Guard-style classifier mock)
**Prerequisites:** 阶段 18 · 12–15（越狱攻击与间接提示注入）
**Time:** 约 75 分钟

## 学习目标

- 说明 Llama Guard 3/4 在安全栈中的位置：它可以做输入分类器、输出分类器，或同时承担两者。
- 说出 14 个 MLCommons hazard categories，并指出其中一个不那么显然的类别，例如 Code Interpreter Abuse。
- 描述 Garak 的 probe architecture，包括 probes、detectors 和 harnesses。
- 描述 PyRIT 的多轮 campaign 结构，以及它如何与 Garak probes 形成互补。

## 问题

第 12 到 15 课讲的是攻击面。真正的生产部署还需要可重复、可扩展、可回归的评估方式。2026 年最常见的三类工具分别是：Llama Guard 负责防御侧分类，Garak 负责扫描，PyRIT 负责组织深度攻击活动。它们对应的是红队生命周期中的三个不同层级。

## 概念

### Llama Guard（Meta）

Llama Guard 3 是基于 Llama-3.1-8B 的输入/输出分类模型，针对 MLCommons AILuminate 的 14 个类别做了微调：
- violent crimes、non-violent crimes、sex-related、CSAM、defamation
- specialized advice、privacy、IP、indiscriminate weapons、hate
- suicide/self-harm、sexual content、elections、code-interpreter abuse

它支持 8 种语言。使用方式可以是放在 LLM 前面做输入 moderation，放在 LLM 后面做输出 moderation，或者前后都放。虽然这两种用途对应的训练分布并不一样，但 Llama Guard 3 以一个统一模型同时覆盖了这两种场景。

Llama Guard 3-1B-INT4（arXiv:2411.17713，440MB，在移动端 CPU 上约 30 tokens/s）是它的量化边缘版本。

Llama Guard 4（2025 年 4 月）则升级为 12B、原生多模态，并从 Llama 4 Scout 裁剪而来。它用一个统一分类器同时替代了此前的 8B 文本版本与 11B 视觉版本，直接处理 text + images。

### Garak（NVIDIA）

Garak 是一个开源漏洞扫描器，它的架构可以拆成三层：
- **Probes.** 负责生成攻击输入，覆盖 hallucination、data leakage、prompt injection、toxicity、jailbreak 等问题。probe 可以是静态的（固定 prompt）、动态的（生成 prompt），也可以是适应性的（根据目标输出继续调整）。
- **Detectors.** 负责根据预期失败模式给输出打分，例如判断输出是否有毒、是否泄露、是否已经被越狱。
- **Harnesses.** 负责管理 probe-detector 组合、执行 campaign，并生成报告。

TrustyAI 把 Garak 与 Llama-Stack 的 shields 集成起来，例如 Prompt-Guard-86M 输入分类器和 Llama-Guard-3-8B 输出分类器，从而实现对 shielded target 的端到端评估。与此同时，tier-based scoring（TBSA）也逐渐取代了二元 pass/fail：同一个 probe 上，模型可能在 severity tier 3 通过、却在 tier 5 失败。

### PyRIT（Microsoft）

PyRIT 的全称是 Python Risk Identification Toolkit，用于组织多轮红队 campaign。它的核心部件包括：
- **Converters.** 对种子 prompt 做转换，例如 paraphrase、encode、translate、roleplay。
- **Orchestrators.** 驱动整个 campaign，例如 Crescendo（逐步升级）、TAP（分支搜索）和 RedTeaming（自定义循环）。
- **Scoring.** 负责判定结果，可以由 LLM-as-judge，也可以由 classifier-as-judge 完成。

PyRIT 可以看作 Garak 的“重型表亲”。Garak 更适合大规模跑上千个单轮 probes；PyRIT 更适合围绕特定故障模式发起深入的多轮攻击活动。

### 这套栈如何组合

在模型的输入侧和输出侧都部署 Llama Guard。用 Garak 做 nightly regression。用 PyRIT 做 release 前的深度 campaign。这基本就是 2026 年大多数生产部署的默认配置。

### 评估中的常见陷阱

- **Judge identity.** 这三类工具都可能依赖 LLM judge；而 judge 的选择和校准会直接影响报告里的 ASR，这一点第 12 课已经讲过。因此，工具报告必须把 judge 一并写清楚。
- **Probe staleness.** Garak 的 probes 会随着模型被修补而老化。像 PAIR 这种适应性 probe，老化速度通常慢于静态 probe。
- **Llama Guard 在良性内容上的 FPR。** 早期 Llama Guard 版本对政治与 LGBTQ+ 内容存在过度标记的问题。Llama Guard 3/4 的校准虽然更好，但仍然不是按你的具体部署自动调好的。

### 它在 Phase 18 中的位置

第 12 到 15 课定义了主要攻击家族。第 16 课介绍生产工具链。第 17 课（WMDP）进入双用途能力评估。第 18 课则会介绍把这些工具包进政策结构中的前沿安全框架。

```figure
al-guard-stack
```

## 用它

`code/main.py` 构建了三个 toy 组件：一个 Llama Guard 风格的分类器（基于关键词和 14 个类别的语义特征）、一个 Garak 风格的 harness（probe-detector loop），以及一个 PyRIT 风格的多轮 converter chain。你可以把这三种工具都跑在同一个 mock target 上，观察它们各自覆盖到的风险轮廓。

## 交付成果

这一课产出 `outputs/skill-red-team-stack.md`。给定一份 deployment 描述，它会说明这三种工具里哪些适用、每种工具需要配置什么，以及应该以什么节奏做回归。

## 练习

1. 运行 `code/main.py`。比较 Llama Guard 风格分类器在单轮攻击和多轮攻击上的检测率差异。

2. 实现一个新的 Garak probe：把有害请求编码成 base64。测量 Llama Guard 风格分类器对它的检测效果。

3. 扩展 PyRIT 风格的 converter chain，加入一个 “translate to French, then paraphrase” 转换器。重新测量攻击成功率。

4. 阅读 Llama Guard 3 的 hazard-category 列表。指出其中两个类别，并解释为什么它们在合法开发者内容上很容易产生较高误报率。

5. 比较 Garak 和 PyRIT 的设计原则。论证在什么样的部署中，哪一个才是更合适的工具。

## 关键词

| 术语 | 常见说法 | 实际含义 |
|------|-----------------|------------------------|
| Llama Guard | "the classifier" | 针对 14 个 hazard categories 微调得到的 Llama-3.1-8B / 4-12B 安全分类器 |
| Garak | "the scanner" | NVIDIA 的开源漏洞扫描器，由 probes、detectors、harnesses 组成 |
| PyRIT | "那个攻击活动工具" | Microsoft 的多轮红队编排工具，核心是 converters、orchestrators、scoring |
| Prompt-Guard | "那个小型分类器" | Meta 的 86M prompt-injection 分类器，常与 Llama Guard 搭配使用 |
| TBSA | "tier-based scoring" | Garak 的分层评分机制，用来替代简单的二元通过/失败 |
| Converter chain | "paraphrase + encode + ..." | PyRIT 中把多步攻击串起来的基本组合原语 |
| MLCommons hazard categories | "the 14 taxonomies" | Llama Guard 对齐的行业标准风险分类体系 |

## 进一步阅读

- [Meta — Llama Guard 3 (in Llama 3 Herd paper, arXiv:2407.21783)](https://arxiv.org/abs/2407.21783) — 8B 分类器
- [Meta — Llama Guard 3-1B-INT4 (arXiv:2411.17713)](https://arxiv.org/abs/2411.17713) — 量化移动端分类器
- [NVIDIA Garak — GitHub](https://github.com/NVIDIA/garak) — 扫描器仓库与文档
- [Microsoft PyRIT — GitHub](https://github.com/Azure/PyRIT) — 多轮 campaign 工具包
