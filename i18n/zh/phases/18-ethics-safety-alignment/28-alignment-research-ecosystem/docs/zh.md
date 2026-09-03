# 对齐研究生态系统：MATS、Redwood、Apollo、METR

> 到 2026 年，实验室之外的 alignment research 生态，主要由五类机构构成。MATS（ML Alignment & Theory Scholars）自 2021 年底以来已培养 527+ 名研究者，产出 180+ 篇论文、10K+ 引用、h-index 47；2024 年夏季 cohort 以 501(c)(3) 形式独立化，当期约有 90 名 scholars 和 40 名 mentors；2025 年之前的校友中，约 80% 在安全 / 安保相关岗位继续工作，其中 200+ 人流向 Anthropic、DeepMind、OpenAI、UK AISI、RAND、Redwood、METR、Apollo。Redwood Research 是 Buck Shlegeris 创立的应用对齐实验室，提出了 AI Control（Lesson 10），并与 UK AISI 合作研究 control safety cases。Apollo Research 负责面向前沿实验室的 pre-deployment scheming evaluations，代表作包括 In-Context Scheming（Lesson 8）和 Towards Safety Cases for AI Scheming。METR（Model Evaluation and Threat Research）专注于 task-based capability evaluations 与 autonomous-task time-horizon studies，其 “Common Elements of Frontier AI Safety Policies” 比较了多家实验室的政策框架。Eleos AI Research 则专注于 model welfare 的 pre-deployment evaluations，并完成了 Claude Opus 4 的 welfare assessment。

**Type:** 学习
**Languages:** none
**Prerequisites:** 阶段 18 · 01–27（此前的第 18 阶段课程）
**Time:** 约 45 分钟

## 学习目标

- 识别实验室外 alignment research 生态中的五类关键组织，以及它们各自的核心产出。
- 描述 MATS 的规模（scholars、论文、h-index）及其作为人才管道的作用。
- 描述 Redwood 的 AI Control 议程及其与 UK AISI 的合作。
- 描述 METR 基于任务的评估方法。

## 问题

前沿实验室（见 Lesson 18）会在内部做安全评估，并选择性公开部分结果。而实验室之外的生态系统，才是这些评估被复核、全新失败模式首次被发现、以及下一代研究者被训练出来的地方。理解这层生态，有助于你判断：哪些研究结论被谁信任，为什么会被信任。

## 概念

### MATS（ML Alignment & Theory Scholars）

项目始于 2021 年底。它本质上是一个研究导师计划：scholars 会和一位资深研究者共同工作 10-12 周，围绕某个具体 alignment 问题展开研究。

到 2026 年的规模大致为：

- 自项目启动以来已覆盖 527+ 名研究者。
- 发表 180+ 篇论文。
- 获得 10K+ citations。
- h-index 为 47。
- 2024 年夏季 cohort 规模为 90 名 scholars + 40 名 mentors，并完成 501(c)(3) 独立注册。

职业流向方面，2025 年之前的校友中约 80% 继续从事 safety / security 工作。其中 200+ 人流向 Anthropic、DeepMind、OpenAI、UK AISI、RAND、Redwood、METR、Apollo 等机构。

### Redwood Research

这是一个应用型 alignment 实验室，由 Buck Shlegeris 创立。它提出了 AI Control 议程（Lesson 10），并与 UK AISI 合作研究 control safety cases，也为 DeepMind 和 Anthropic 提供评估设计方面的建议。

代表性论文包括：Greenblatt、Shlegeris 等人的 “AI Control”（arXiv:2312.06942, ICML 2024），以及与 Anthropic 合作的 Alignment Faking（Greenblatt、Denison、Wright et al., arXiv:2412.14093）。

它的方法风格很鲜明：强调具体 threat model、最坏情况对手，以及可以被压力测试的具体 protocol。

### Apollo Research

Apollo 专注于为前沿实验室做 pre-deployment scheming evaluations。它是 In-Context Scheming（Lesson 8, arXiv:2412.04984）的作者，也参与了 2025 年 OpenAI 的 anti-scheming training collaboration，并产出了 Towards Safety Cases for AI Scheming（2024）。

它的典型风格是：在可能出现欺骗的 agentic setting 中做评估，并用三支柱结构拆解问题，即 misalignment、goal-directedness 和 situational awareness。

### METR（Model Evaluation and Threat Research）

METR 主要做 task-based capability evaluations，以及 autonomous-task completion 的 time-horizon 研究。它的 “Common Elements of Frontier AI Safety Policies” （metr.org/common-elements, 2025）系统比较了多家实验室的政策框架。

它也与 Apollo 一起合作撰写过 AI Scheming 的 safety-case sketch。

它的方法论风格可以概括为：长时程任务评估、经验式能力测量、以及跨框架综合。

### Eleos AI Research

Eleos 负责 model-welfare 的 pre-deployment evaluations。它完成了 Claude Opus 4 welfare assessment，并以此为 Lesson 19 中福利相关主张提供外部方法学复核。

### 人才与研究流动

MATS 负责训练研究者。毕业生会流向 Anthropic、DeepMind、OpenAI 等实验室安全团队，也会流向 Redwood、Apollo、METR、Eleos 这样的外部评估机构。外部评估者再与实验室以及 UK AISI / CAISI 合作，产出新的评估方法与论文；这些成果又反过来喂回 MATS，影响下一批 cohort。

### 为什么这层生态重要

单一来源的评估并不可靠：实验室评估自己的模型，天然存在结构性利益冲突。外部评估者可以提出、验证并放大实验室自身可能低报的失败模式。比如 2024 年的 Sleeper Agents（Lesson 7）由 Anthropic + Redwood 合作，Alignment Faking 同样是 Anthropic + Redwood，In-Context Scheming 来自 Apollo，而 Anti-Scheming 则是 Apollo + OpenAI。多机构并存，本质上就是一种质量控制机制。

### 它在 Phase 18 中的位置

Lessons 7-11 多次引用 Redwood 与 Apollo 的工作；Lesson 18 会用到 METR 的框架比较；Lesson 19 会引用 Eleos。Lesson 28 的作用，就是把支撑整个 Phase 18 的这张组织地图明确画出来。

```figure
sae-features
```

## 用它

本课没有代码。建议直接阅读 METR 的 “Common Elements of Frontier AI Safety Policies”，把它作为“外部综合如何为实验室内部政策工作增值”的典型例子。

## 交付成果

本课产出 `outputs/skill-ecosystem-map.md`。给定一条 alignment claim 或一项 evaluation，它会识别对应组织、发布渠道和方法学风格，并与已知对口机构做交叉核对。

## 练习

1. 从 Lessons 7-15 里挑一篇论文，识别其中涉及的组织，并把作者与 MATS 校友及当前生态机构做交叉对照。

2. 阅读 METR 的 “Common Elements of Frontier AI Safety Policies”。找出他们强调的三个跨实验室收敛点，以及两个最大的分歧点。

3. MATS 的职业流向中约 80% 会进入 safety / security。讨论这种筛选压力究竟是适应性的（为领域培养人才），还是带有偏置的（过滤掉了异端立场）。

4. Redwood 和 Apollo 都在做 control / scheming 方向的工作，但方法风格不同。任选一个 failure mode，分别说明它们会如何调查。

5. Eleos AI 是唯一纯粹聚焦 model welfare 的组织。请设计一个假想中的第二家机构，专注于另一个 welfare-adjacent 问题（如 cognitive liberty、robotic embodiment 等），并明确它的方法学。

## 关键词

| 术语 | 人们通常怎么说 | 实际含义 |
|------|-----------------|------------------------|
| MATS | “那个导师项目” | 机器学习对齐与理论学者项目（ML Alignment & Theory Scholars）；自 2021 年以来已培养 527 多名研究者 |
| Redwood Research | “控制研究机构” | 从事应用型对齐研究；《AI Control》的作者团队；UK AISI 的合作方 |
| Apollo Research | “图谋评估团队” | 为前沿实验室开展部署前的图谋评估 |
| METR | “任务时程评估团队” | 开展基于任务的能力评估，并综合构建评估框架 |
| Eleos AI | “福利研究机构” | 开展模型福利相关的部署前评估 |
| 人才通道（Talent pipeline） | “MATS -> 各实验室” | MATS 毕业生会流向 Anthropic、DM、OpenAI、Redwood、Apollo 和 METR 等机构 |
| 外部评估（External evaluation） | “实验室外部复核” | 不由模型生产者自行完成的评估，因而能提高结果的可信度 |

## 进一步阅读

- [MATS（机器学习对齐与理论学者计划）](https://www.matsprogram.org/) - 导师计划主页
- [Redwood Research](https://www.redwoodresearch.org/) - AI Control 相关工作
- [Apollo Research](https://www.apolloresearch.ai/) - 图谋行为评估
- [METR — Common Elements of Frontier AI Safety Policies](https://metr.org/blog/2025-03-26-common-elements-of-frontier-ai-safety-policies/) - 框架比较
- [Eleos AI Research](https://www.eleosai.org/research) - 模型福利研究方法
