# 模型、系统与数据集卡片

> AI 透明度目前主要依赖三种文档格式。Model Cards（Mitchell et al. 2019）可以看作模型的营养标签，记录 training data、quantitative disaggregated analyses、ethical considerations、caveats 等内容；但 Oreamuno et al. 2023 的统计显示，在 Hugging Face 上只有 0.3% 的 model cards 真正记录了 ethical considerations。Datasheets for Datasets（Gebru et al. 2018，后发表于 CACM）则覆盖 motivation、composition、collection process、labeling、distribution、maintenance，直接借用了电子元件 datasheet 的类比。Data Cards（Pushkarna et al., Google 2022）进一步提出模块化、分层的信息组织方式，也就是 telescopic、periscopic、microscopic 三层，用来让不同读者把同一份文档当作 boundary object 来使用。到了 2024-2025 年，这一方向又出现了几个新发展：通过 LLM 自动生成卡片（CardGen，Liu et al. 2024）；更完整的 model card 会在 Hugging Face 上带来最高约 29% 的下载提升（Liang et al. 2024）；Laminator（Duddu et al. 2024）这类工作把 verifiable attestations 引入卡片；Jouneaux et al. 在 2025 年 7 月把 carbon / water sustainability reporting 纳入其中；EU / ISO 风格的 regulatory cards 也开始出现。与此同时，System Cards（Sidhpurwala 2024；Meta 的 system-level transparency；“Blueprints of Trust” arXiv:2509.20394）则把文档范围扩展到端到端 AI system，覆盖 security capabilities、prompt-injection protection、data-exfiltration detection 以及与 human values 的对齐情况。

**Type:** 构建
**Languages:** Python (stdlib, model-card + datasheet + system-card generator)
**Prerequisites:** 阶段 18 · 18（前沿安全框架）、阶段 18 · 24（监管框架）
**Time:** 约 60 分钟

## 学习目标

- 描述最初的 Mitchell et al. 2019 model card，以及 Gebru et al. 2018 datasheet。
- 描述 Data Cards 的 telescopic / periscopic / microscopic 三层结构。
- 描述 System Cards，以及它覆盖的端到端系统范围。
- 说出 2024-2025 年的三个发展方向，例如 automated generation、verifiable attestations、sustainability reporting。

## 问题

监管框架，也就是 Lesson 24，和实验室安全政策，也就是 Lesson 18，都要求文档化。文档格式一路从 model-specific 的 model cards，发展到 dataset-specific 的 datasheets，再发展到 system-specific 的 system cards。它们分别回答的是不同层面的透明度问题。2024-2025 年出现的自动生成与可验证证明工作，核心上是在处理这个领域长期存在的 adoption problem。

## 概念

### Model Cards（Mitchell et al. 2019）

标准部分包括：
- 模型详情。
- 预期用途。
- Factors，也就是与评估相关的人口统计或环境因素。
- 指标。
- 评估数据。
- 训练数据。
- Quantitative analyses，也就是按因素拆分的量化分析。
- 伦理考量。
- 注意事项与建议。

这一格式面临的核心难题是采用率。Oreamuno et al. 2023 对 Hugging Face model cards 的审计显示，真正记录 ethical considerations 的比例只有 0.3%。

### Datasheets for Datasets（Gebru et al. 2018）

这套格式直接借用了电子元件 datasheet 的类比。主要部分包括：
- Motivation，也就是为什么要创建这个 dataset。
- Composition，也就是里面包含什么。
- Collection process，也就是它是如何组装起来的。
- Labeling，如果适用。
- Uses，包括 intended uses、prohibited uses 和 risks。
- 分发方式。
- 维护方式。

该工作后来发表于 CACM 2021。datasheet 是上游文档，而 model card 是否可靠，很大程度上依赖 datasheet 本身是否准确。

### Data Cards（Pushkarna et al., Google 2022）

Data Cards 强调模块化与分层细节。它提出了三种 zoom level：
- **Telescopic。** 给非专家看的高层摘要。
- **Periscopic。** 给 ML practitioners 看的中层概览。
- **Microscopic。** 给 auditors 使用的细粒度 feature-level 文档。

这里的关键 framing 是 boundary object：不同类型的读者，可以从同一份文档中提取对自己有价值的信息。

### System Cards

System Card 的范围是端到端 AI system，也就是模型本体加上 safety stack，再加上 deployment context。常见部分包括：
- 安全能力。
- 提示词注入防护。
- 数据外泄检测。
- 与声明的人类价值观保持一致。
- 事件响应。

Sidhpurwala 2024 与 Meta 的 system-level transparency 都属于这一方向。“Blueprints of Trust”（arXiv:2509.20394）进一步把 System Card 形式化为与 Model Cards 相配套的 deployment-layer 文档。

### 2024-2025 年的发展

- **CardGen（Liu et al. 2024）。** 通过 LLM 自动生成 model cards；在标准化的 Mitchell 2019 字段上，它报告的客观性甚至高于许多人类手写卡片。
- **Download correlation（Liang et al. 2024）。** 更详细的 model cards 与 Hugging Face 上最高 29% 的下载增长相关。这说明 adoption pressure 现在已经不只是 compliance-driven，也开始是 market-driven。
- **Laminator（Duddu et al. 2024）。** 借助 hardware TEE 或 cryptographic signatures 提供 verifiable attestations，让 model card 携带的不只是 claim，而是 proof-of-claim。
- **Sustainability（Jouneaux et al. July 2025）。** 把 carbon、water、compute-energy footprint 纳入文档字段，并与 emerging ISO standards 接轨。
- **Regulatory cards。** 在 EU AI Act（Lesson 24）中，GPAI Code of Practice 的 Transparency 章节已经开始把 model cards 视作一种 compliance artifact。

### 它在 Phase 18 里的位置

Lessons 24-25 讲的是 regulatory 和 CVE 层。Lesson 26 对应 documentation layer。Lesson 27 是 training-data governance，也就是 datasheet 的上游。Lesson 28 则是产生这些卡片中所引用评估结果的 research ecosystem。

```figure
an-card-scopes
```

## 用它

`code/main.py` 会为一个玩具 deployment 生成最小版本的 model card、datasheet 和 system card。每一份都遵循其经典章节结构。你可以直接检查格式，并对比这三种文档各自覆盖的范围。

## 交付它

这一课会产出 `outputs/skill-card-audit.md`。给定一份 model card、datasheet 或 system card，它会审计章节覆盖情况、数值是否做了拆分分析，以及是否包含 verifiable attestations。

## 练习

1. 运行 `code/main.py`。检查生成出的卡片，找出其中仍然很弱的部分，也就是只有 placeholder 的部分，并说明需要补哪些证据才能把它们写扎实。

2. 在 model card 中加入跨两个 demographic groups 的 quantitative disaggregated analysis，对应 Lesson 20 的内容。

3. 阅读 Oreamuno et al. 2023 关于 0.3% adoption rate 的讨论。提出一个对 model card specification 的结构性改动，以提升 ethical-considerations 的采纳率。

4. Laminator（Duddu et al. 2024）使用 TEE 来提供 verifiable attestations。请设计一个 model-card field，让它能携带某项 evaluation result 的 cryptographic attestation，并说明 verifier 的角色。

5. 给你过去的某个项目，或者一个假想 deployment，写一份 System Card，注意是 System Card，不是 Model Card。指出其中对第三方审计员最有价值的那一节。

## 关键术语

| 术语 | 人们通常怎么说 | 实际含义 |
|------|-----------------|------------------------|
| 模型卡（Model Card） | “Mitchell 的模型卡” | Mitchell 等人在 2019 年提出的机器学习模型标准文档 |
| 数据表（Datasheet） | “Gebru 的数据表” | Gebru 等人在 2018 年提出的数据集标准文档 |
| 数据卡（Data Card） | “Pushkarna 的数据卡” | Google 于 2022 年提出的模块化分层数据文档 |
| 系统卡（System Card） | “部署卡” | 覆盖安全栈的端到端 AI 系统文档 |
| 边界对象（Boundary object） | “读者不同，同一文档” | Data Cards 的定位：同一份文档可以服务于多类读者 |
| 可验证证明（Verifiable attestation） | “Laminator 证明” | 附在文档声明上的密码学证明或 TEE 证明 |
| 可持续性字段（Sustainability field） | “碳足迹／水足迹” | 2025 年开始兴起、用于环境核算的文档字段 |

## 延伸阅读

- [Mitchell et al. — Model Cards for Model Reporting (arXiv:1810.03993, FAT* 2019)](https://arxiv.org/abs/1810.03993) — 模型卡的规范性论文
- [Gebru et al. — Datasheets for Datasets (CACM 2021, arXiv:1803.09010)](https://arxiv.org/abs/1803.09010) — 数据表论文
- [Pushkarna et al. — Data Cards (Google 2022)](https://arxiv.org/abs/2204.01075) — 分层数据文档
- [Sidhpurwala et al. — Blueprints of Trust (arXiv:2509.20394)](https://arxiv.org/abs/2509.20394) — 系统卡的形式化方法
