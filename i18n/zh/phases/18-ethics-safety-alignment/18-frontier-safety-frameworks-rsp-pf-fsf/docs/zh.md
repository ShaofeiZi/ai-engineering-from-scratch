# 前沿安全框架——RSP、PF、FSF

> 三家主要实验室的框架定义了 2026 年前沿能力的行业治理方式。Anthropic Responsible Scaling Policy v3.0（2026 年 2 月）引入分级的 AI Safety Levels（ASL-1 至 ASL-5+），其设计借鉴生物安全等级；其中 ASL-3 已于 2025 年 5 月针对具备 CBRN 相关能力的模型启用。OpenAI Preparedness Framework v2（2025 年 4 月）为受追踪能力定义五项标准，并将 Capabilities Reports 与 Safeguards Reports 分开。DeepMind Frontier Safety Framework v3.0（2025 年 9 月）引入 Critical Capability Levels，其中新增 Harmful Manipulation CCL。三者如今都包含竞争者调整条款：如果同行实验室在没有同等保障措施的情况下发布模型，可以推迟部分要求。跨实验室的一致性体现在结构而非术语上：“Capability Thresholds”“High Capability thresholds”和“Critical Capability Levels”表达的是相似构造。

**Type:** 学习
**Languages:** 无
**Prerequisites:** 阶段 18 · 17（WMDP 双重用途评估）、阶段 18 · 07–09（欺骗性失效相关课程）
**Time:** 约 75 分钟

## 学习目标

- 描述 Anthropic 的 ASL 分级结构，以及触发 ASL-3 的条件。
- 说出 OpenAI Preparedness Framework v2 用于追踪能力的五项标准。
- 描述 DeepMind 的 Critical Capability Level 结构与 Harmful Manipulation CCL。
- 解释竞争者调整条款，以及它们为何会影响竞赛动态。
- 定义安全论证，并描述它的三支柱结构（监控、不可理解性、无能力性）。

## 问题

第 7–17 课已经说明：欺骗可能发生，双重用途能力确实存在，评估也有局限。拥有前沿能力模型的实验室需要一套内部治理结构，用来：
- 定义何时必须启用新保障措施的阈值。
- 定义扩大规模前必须完成的评估。
- 说明一份安全论证应当具有什么形态。
- 处理竞赛动态问题（竞争者在没有保障措施的情况下发布时，该怎么办？）。

这三套 2025–2026 年框架代表当前最高水平——它们并不完美，仍在演化，但各实验室之间已有足够的结构趋同，因此治理问题如今不再是“有没有框架”，而是“框架是否充分”。

## 核心概念

### Anthropic Responsible Scaling Policy v3.0（2026 年 2 月）

ASL 结构：
- ASL-1：不是前沿模型（归入弱于前沿的基线）。
- ASL-2：当前前沿基线；采用常规保障措施进行部署。
- ASL-3：灾难性滥用风险显著提高，具有 CBRN 相关能力；已于 2025 年 5 月启用。
- ASL-4：跨过 AI R&D-2 阈值；模型能够自动化入门级 AI 研究。
- ASL-5+：高级 AI R&D；模型能够显著加速有效规模扩张。

v3.0 的新增内容：
- Frontier Safety Roadmaps（以删节形式公开）。
- Risk Reports（每季度发布，部分接受外部审查）。
- AI R&D 被拆分为 AI R&D-2 与 AI R&D-4。
- 一旦跨过 AI R&D-4，就必须给出肯定性的安全论证，识别模型追求错位目标所造成的失调风险。

### OpenAI Preparedness Framework v2（2025 年 4 月 15 日）

受追踪能力必须满足五项标准：
- **合理可能（Plausible）。** 存在合理的威胁模型。
- **可测量（Measurable）。** 可以进行实证评估。
- **严重（Severe）。** 伤害规模很大。
- **全新（Net-new）。** 不是对既有风险的简单放大。
- **即时或不可补救（Instantaneous-or-irremediable）。** 伤害发生得很快，或无法撤销。

同时满足全部五项的能力才会被追踪，其他能力不会。

PF v2 的其他结构：
- 将 Capabilities Reports（模型能做什么）与 Safeguards Reports（有哪些控制措施）分开。
- 由 Safety Advisory Group 进行审查。
- 由领导层批准，Board 的 Safety & Security Committee 负责监督。
- “调整条款”：如果另一家实验室在没有同等保障措施的情况下发布模型，OpenAI 可以降低要求。

### DeepMind Frontier Safety Framework v3.0（2025 年 9 月）

按领域划分的 Critical Capability Levels（CCL）：
- Bioweapon Uplift
- Cyber Uplift
- ML R&D Acceleration
- Harmful Manipulation（v3.0 新增）：模型能够在高风险情境下显著改变人的信念或行为。

v2.0（2025 年 2 月）新增 Deceptive Alignment 章节，并为 ML R&D CCL 增加更高安全等级。

### 跨实验室趋同

- Anthropic 使用“Capability Thresholds”。
- DeepMind 使用“Critical Capability Levels”。
- OpenAI 使用“High Capability thresholds”。

行业尚无统一术语，但结构已经趋同：都围绕带公开评估标准的三个前沿能力层级展开。三者从 2025 年起也都包含竞争者调整条款。

### 安全论证

安全论证是一份书面论证，用来说明在最坏情况假设下，某项部署仍然安全到可接受程度。标准结构围绕三个支柱：

- **监控（Monitoring）。** 不良行为发生时，我们能否检测到？
- **不可理解性（Illegibility）。** 模型是否缺乏制定并执行连贯伤害计划的能力？
- **无能力性（Incapability）。** 模型是否不具备造成相关伤害的能力？

不同安全论证会针对不同支柱。对于 ASL-3 CBRN 论证，无能力性（通过遗忘学习实现）是首要目标；对于欺骗性对齐，目标是监控和不可理解性；对于网络能力提升，三个支柱都相关。

### 竞赛动态问题

竞争者调整条款颇具争议。批评者认为它会造成逐底竞争：如果三家实验室都在竞争者违背承诺时降低要求，均衡结果就会向违背承诺倾斜。支持者则认为，如果违背承诺的实验室更不重视安全，单方面坚持保障措施反而会导致更差结果。

UK AISI、US CAISI 与 EU AI Office（第 24 课）是外部治理对应机构。实验室框架属于自愿承诺，监管框架则仍在形成。

### 本课在阶段 18 中的位置

第 17–18 课是在欺骗与红队分析之上的测量和治理层。第 19–24 课涵盖福祉、偏见、隐私、水印与监管结构。第 28 课梳理负责将这些评估落地的研究生态（MATS、Redwood、Apollo、METR）。

```figure
al-asl-ladder
```

## 实际使用

本课没有代码。请阅读三个一手来源：RSP v3.0、PF v2、FSF v3.0。将各实验室的分级结构相互映射，并找出每家实验室独有的一项阈值。

## 交付成果

本课产出 `outputs/skill-framework-diff.md`。给定一份安全框架或发布说明，它会对照 RSP v3.0、PF v2、FSF v3.0，比较其中的阈值定义、必需评估和安全论证结构，并标出跨实验室缺口。

## 练习

1. 阅读 RSP v3.0、PF v2 和 FSF v3.0。汇总一张表，列出各实验室的 CBRN 阈值、AI R&D 阈值，以及部署前必须进行的评估。

2. 三套框架从 2025 年起都包含竞争者调整条款。分别写一段支持意见和反对意见，并指出每种立场所依赖的假设。

3. 为跨过 Anthropic AI R&D-4 阈值的模型设计安全论证。说出三个支柱（监控、不可理解性、无能力性）分别需要哪些证据。

4. DeepMind FSF v3.0 引入 Harmful Manipulation CCL。提出三项实证测量，用于判断模型是否跨过这一阈值。

5. 阅读 METR 的《Common Elements of Frontier AI Safety Policies》（2025）。说出跨实验室最显著的三项趋同和最大的两项分歧。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|-----------------|------------------------|
| RSP | “Anthropic 的框架” | Responsible Scaling Policy；ASL 分级；v3.0 发布于 2026 年 2 月 |
| PF | “OpenAI 的框架” | Preparedness Framework；五项标准；v2 发布于 2025 年 4 月 |
| FSF | “DeepMind 的框架” | Frontier Safety Framework；CCL；v3.0 发布于 2025 年 9 月 |
| ASL-3 | “类似生物安全 3 级” | Anthropic 针对 CBRN 相关能力的等级；2025 年 5 月启用 |
| CCL | “关键能力等级” | DeepMind 按领域定义的阈值构造 |
| 安全论证 | “形式化论证” | 说明部署在最坏情况 U 下仍具有可接受安全性的书面论证 |
| 调整条款 | “竞争者违背承诺时的例外” | 竞争者在没有同等保障措施的情况下发布时，允许降低要求的框架条款 |

## 延伸阅读

- [Anthropic——Responsible Scaling Policy v3.0（2026 年 2 月）](https://www.anthropic.com/responsible-scaling-policy)——ASL 分级、Roadmap、AI R&D 拆分
- [OpenAI——更新 Preparedness Framework（2025 年 4 月 15 日）](https://openai.com/index/updating-our-preparedness-framework/)——五项标准、调整条款
- [DeepMind——强化 Frontier Safety Framework（2025 年 9 月）](https://deepmind.google/blog/strengthening-our-frontier-safety-framework/)——CCL v3.0、Harmful Manipulation
- [METR——前沿 AI 安全政策的共同要素（2025）](https://metr.org/blog/2025-03-26-common-elements-of-frontier-ai-safety-policies/)——跨实验室比较
