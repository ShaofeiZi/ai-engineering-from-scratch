# Anthropic 负责任扩展政策 v3.0

> RSP v3.0 于 2026 年 2 月 24 日生效，取代 2023 年版政策。它引入了双层缓解结构：一层是 Anthropic 将单方面执行的措施，另一层是面向全行业的建议，其中包括 RAND SL-4 安全标准。它把 Frontier Safety Roadmaps 和 Risk Reports 升格为常设文件，而不是一次性交付物；同时删除了 2023 年版本中的暂停承诺。新版本还提出 AI R&D-4 阈值：一旦模型越过该阈值，Anthropic 必须发布一份肯定性论证，说明失配风险是什么、缓解措施是否充分。根据公告，Claude Opus 4.6 尚未越过这一阈值，但 Anthropic 也明确表示，“要有把握地排除这种可能，正变得越来越困难”。SaferAI 对 2023 年版 RSP 的评分是 2.2，而对 v3.0 的评分下调到 1.9，使 Anthropic 与 OpenAI、DeepMind 一同落入“较弱”RSP 类别。用定性阈值替代 2023 年的定量承诺，再加上删除暂停条款，是这次回退中最尖锐的部分。

**Type:** 学习
**Languages:** Python（标准库，RSP 阈值判定引擎）
**Prerequisites:** 阶段 15 · 06（AAR），阶段 15 · 07（RSI）
**Time:** 约 45 分钟

## 问题

前沿实验室发布的扩展政策，既是技术文件，也是治理文件，同时还是向监管者和公众传递姿态的信号。Anthropic 的 RSP v3.0 就是当前这样一份文件。认真逐字读它，不是因为它具有强制法律约束力，而是因为它暴露了实验室如何界定灾难性风险，以及它准备如何向外部解释这些权衡。

真正有分析价值的，不是孤立地看 v3.0，而是把它和 v2.0 对读。新增了什么：Frontier Safety Roadmaps、Risk Reports、AI R&D-4 阈值。删除了什么：2023 年的暂停承诺。又重写了什么：把缓解安排拆成 Anthropic 单边措施与行业建议两层。外部评审机构 SaferAI 也因此把评分从 2.2（v2）下调到 1.9（v3.0）。这正说明，一份政策文件完全可能在表面上更成熟、更精致，但在约束力上反而更弱。

## 概念

### 双层缓解安排

- **Anthropic 单边措施**：无论其他实验室怎么做，Anthropic 自己都会执行的措施，例如超过阈值就停止训练、落实具体的安全控制、设置明确的部署门槛。
- **行业整体建议**：Anthropic 认为整个行业都应该共同采取的措施，其中包括 RAND SL-4 安全标准。这些不是 Anthropic 对自己的承诺，而是它对行业的倡议。

这个双层结构在 v2 中并不存在。它的直接后果是：读者必须看清每项承诺到底落在哪一列。写在“行业整体建议”那一栏的安全措施，不代表 Anthropic 已经承诺自己会做；它只代表 Anthropic 希望行业这样做。

### AI R&D-4 阈值

这是 RSP v3.0 明确点名的下一个关键能力阈值。它指的是：模型能够以具有竞争力的成本，自动化相当大一部分 AI 研究工作。一旦 Anthropic 认为某个模型越过这一能力线，在继续扩展之前，它就必须发布一份肯定性论证，说明失配风险是什么，以及缓解措施为何足够。

根据 v3.0 公告，Claude Opus 4.6 还没有跨过这个阈值。公告中还补了一句：“要有把握地排除这种可能，正变得越来越困难。”这句话很关键，因为它承认这个阈值已经不是遥远的理论边界，而是近在眼前、需要被严肃对待的现实风险点。

第 6 课（自动化对齐研究）和第 7 课（递归式自我改进）与这个阈值直接相关。如果自动化对齐研究者开始跨过研究质量门槛，就说明 AI R&D-4 正在逼近。

### Frontier Safety Roadmaps 与 Risk Reports

v3.0 把两类产物提升为常设文件：

- **Frontier Safety Roadmap**：面向未来的文件，描述计划中的安全工作、能力预期和缓解研究。
- **Risk Report**：面向已发布模型的回顾文件，总结已观测到的能力与残余风险。

这两类文件都是公开的，也都按照既定节奏更新。它们的价值在于：读者可以把 Anthropic 在 Roadmap 里承诺要做的事情，与它后来在 Risk Report 里声称实际观察到的结果逐项对照。

### 删除暂停条款

2023 年版 RSP 中有一个明确的暂停承诺：如果模型跨过某些特定能力阈值，训练就会暂停，直到缓解措施落实到位。v3.0 删除了这个明确条款，改成更柔性的表达方式：先发布一份肯定性论证，如果缓解被认为足够，就可以继续推进。SaferAI 和其他分析者都直接把这一点指出为新版本中最明显的回退。

支持修改的一方会说：2023 年写下的定量阈值，到了 2026 年，随着能力基准被重新标定，已经不再适合作为硬触发器。反对方则会说：暂停条款在 scaling policy 里本来就是一种约束承诺；把它删掉，政策的可信度就会随之下降。

### SaferAI 的降级

SaferAI 是专门给 RSP 类文件打分的独立组织。它的公开评级中，2023 年版 Anthropic RSP 得到 2.2 分；这个分数所处的量表上，4.0 代表当前最强的 RSP，1.0 则接近名义性文件。到了 v3.0，评分降为 1.9。这意味着 Anthropic 从“中等”降到了“较弱”，和 OpenAI、DeepMind 处在同一档。

SaferAI 给出的主要降级理由包括：
- 定量阈值被定性阈值替代。
- 暂停承诺被删除。
- AI R&D-4 阈值对应的缓解要求，被表述为“affirmative case”，而不是具体、可核查的措施。
- 审查机制主要依赖 Anthropic 自己的 Safety Advisory Group，独立外部监督力度有限。

### 这门课不是什么

这不是一门“如何合规”的课程。RSP v3.0 不是法规，也没有外部强制力量确保 Anthropic 一定遵守。这里要训练的是另一种能力：用足够具体、足够审慎的方式阅读这类文件。对于前沿实验室来说，扩展政策是它们向公众发出的主要灾难风险信号。能不能把这类文件读明白，是所有依赖前沿能力工作的人都需要具备的实践技能。

```figure
a5-rsp-ladder
```

## 用起来

`code/main.py` 实现了一个小型决策引擎，用来模拟 RSP 阈值评估的形状：给定一个候选模型和一组能力测量值，返回它是否越过 AI R&D-4 阈值、需要补齐哪些肯定性论证章节，以及部署是否可以继续。这个实现有意保持简单，目的不是复刻真实政策流程，而是把文件中的逻辑显式化。

## 交付物

`outputs/skill-scaling-policy-review.md` 提供了一份扩展政策审阅模板，可用于评估 Anthropic、OpenAI、DeepMind 或内部政策文件是否具备 v3.0 参考中的关键结构：双层安排、阈值设置、暂停承诺、独立审查。

## 练习

1. 运行 `code/main.py`。输入三个能力水平不同的合成模型，确认阈值评估器的行为符合预期，并且能产出正确的肯定性论证模板。

2. 通读 RSP v3.0 全文（32 页）。找出所有落在“行业整体建议”层中的承诺。哪些承诺如果放在 v2 语境下，本来会被理解为“Anthropic 单边措施”？

3. 阅读 SaferAI 的 RSP 评分方法。用它们的 rubric 重新打分，复现 v3.0 的 1.9 分。哪一行评分项最直接导致了降级？

4. 2023 年的暂停承诺已经被删除。请提出一个替代性承诺：既保留政策的可信度，又承认 2026 年基准被重标定之后，原有定量阈值不再可靠。

5. 将 RSP v3.0 与 OpenAI Preparedness Framework v2（第 20 课）进行对比。指出一个 v3.0 更强的方面，再指出一个 Preparedness Framework 更强的方面。

## 关键词

| 术语 | 常见说法 | 实际含义 |
|---|---|---|
| RSP | “Anthropic 的扩展政策” | Responsible Scaling Policy；v3.0 于 2026 年 2 月 24 日生效 |
| AI R&D-4 | “研究自动化阈值” | 以有竞争力的成本自动化相当大一部分 AI 研究工作的能力 |
| Affirmative case | “安全论证” | 一份公开发布的论证，说明风险已被识别且缓解足够充分 |
| Frontier Safety Roadmap | “前瞻性路线图” | 关于计划中的安全工作和预期能力的常设文件 |
| Risk Report | “模型风险回顾” | 模型发布后，对已观测能力与残余风险的常设回顾文件 |
| Two-tier mitigation | “双层缓解” | 将 Anthropic 自身承诺与行业建议明确分开 |
| Pause commitment | “2023 年暂停条款” | 明确承诺暂停训练；在 v3.0 中已被删除 |
| SaferAI rating | “独立 RSP 评分” | 第三方评分体系；v3.0 得分 1.9（v2 为 2.2） |

## 延伸阅读

- [Anthropic — Responsible Scaling Policy v3.0](https://anthropic.com/responsible-scaling-policy/rsp-v3-0) — 完整 32 页政策全文。
- [Anthropic — RSP v3.0 announcement](https://www.anthropic.com/news/responsible-scaling-policy-v3) — 从 v2 到 v3.0 的变更概览。
- [Anthropic — Frontier Safety Roadmap](https://www.anthropic.com/research/frontier-safety) — RSP v3.0 引出的常设路线图文件。
- [Anthropic — Risk Report: Claude Opus 4.6](https://www.anthropic.com/research/risk-report-claude-opus-4-6) — 当前前沿模型的回顾性风险报告。
- [Anthropic — Measuring agent autonomy in practice](https://www.anthropic.com/research/measuring-agent-autonomy) — 将 AI R&D-4 与实际测得的代理自主性联系起来。
