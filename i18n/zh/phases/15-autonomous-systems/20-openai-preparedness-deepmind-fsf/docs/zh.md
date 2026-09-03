# OpenAI 备战框架与 DeepMind 前沿安全框架

> OpenAI Preparedness Framework v2（2025 年 4 月）把 Research Categories 与 Tracked Categories 分开。Research Categories 包括 Long-range Autonomy、Sandbagging、Autonomous Replication and Adaptation、Undermining Safeguards。Tracked Categories 会触发 Capabilities Reports 与 Safeguards Reports，并由 Safety Advisory Group 审查。DeepMind 的 FSF v3（2025 年 9 月发布，2026 年 4 月 17 日加入 Tracked Capability Levels）则把自主性折叠进 ML R&D 和 Cyber 两个领域，例如 ML R&D autonomy level 1 被定义为：在与“人类 + AI 工具”协作相当的成本下，完整自动化 AI R&D 流程。FSF v3 还明确提出，要通过自动化监测来发现工具性推理的滥用，从而应对欺骗性对齐。这里最需要诚实指出的一点是：PF v2 里的 Research Categories，即便包括 Long-range Autonomy，也不会自动触发缓解措施；政策中使用的是“potential”这一表述。DeepMind 自己也承认，如果工具性推理继续增强，单靠自动化监测，长期来看并不够。

**Type:** 学习
**Languages:** Python（标准库，三框架决策表对比工具）
**Prerequisites:** 阶段 15 · 19（Anthropic Responsible Scaling Policy v3.0）
**Time:** 约 45 分钟

## 问题

第 19 课已经细读了 Anthropic 的扩展政策。这一课把视角补全到 OpenAI 和 DeepMind。三份文件是在回答同一个核心问题：当前沿模型能力上升时，实验室应该在什么位置暂停、加门槛或增加缓解措施。它们彼此相似，但关键差别也很实质。

相似点在于：三家都把长程自主性视为值得持续追踪的能力类别；三家都承认欺骗性行为，例如对齐伪装或 sandbagging，是一类独立风险；三家都有内部审查机构。不同点在于：OpenAI 把类别拆成“Tracked”与“Research”，其中只有前者会自动触发缓解；DeepMind 则不单独设“Autonomy”顶层桶，而是把它嵌入 ML R&D 与 Cyber 两个风险域。不同实验室会使用不同标签，例如 Tracked vs Research、Critical vs Moderate、Tier-1 vs Tier-2，但真正重要的是：某项能力被放进哪个桶里，会直接改变它在运营层面的后果。

把三份政策并排阅读才有价值。同一种能力，在 Anthropic 那里可能是“必须缓解”的门槛，在 OpenAI 那里只是“需要监测但不会自动触发”的研究类能力，在 DeepMind 那里则可能被放进某个具体领域里持续追踪。政策姿态不同，实际治理含义也不同。

## 概念

### OpenAI Preparedness Framework v2（2025 年 4 月）

结构如下：

- **Tracked Categories**：会触发 Capabilities Reports（模型能做什么）与 Safeguards Reports（已部署了哪些缓解措施），并在部署前接受 Safety Advisory Group 审查。
- **Research Categories**：实验室正在跟踪、但尚未承诺针对其自动触发具体缓解的能力类别。其中包括 Long-range Autonomy、Sandbagging、Autonomous Replication and Adaptation、Undermining Safeguards。

v2 最关键的表述是：Research Categories 不会自动触发缓解。政策原文里说的是“potential” mitigations。这里的关键差异必须抓住：一个类别被放在 Tracked 还是 Research，决定了它究竟是“有门槛、会拦截”，还是“先观察、不自动触发”。

### DeepMind Frontier Safety Framework v3（2025 年 9 月；2026 年 4 月 17 日加入 Tracked Capability Levels）

结构如下：

- **Critical Capability Levels (CCLs)**：跨五个领域设置的能力阈值，包括 Cyber、Bio、ML R&D、CBRN，以及将自主性风险折叠后的相关域。
- **Tracked Capability Levels**：2026 年 4 月新增的更细粒度分层。例如，ML R&D autonomy level 1 指的是：以与人类 + AI 工具协作相当的成本，完整自动化 AI R&D 流程。
- **Deceptive alignment monitoring**：明确承诺使用自动化监测手段，发现工具性推理被用于规避约束或实施滥用。

DeepMind 对自主性的处理方式与 OpenAI 不同。它不把“Autonomy”保留为一个独立顶层域，而是把它并入真正会承载危害的领域，例如 ML R&D 和 Cyber。支持这一做法的人会说：脱离具体领域谈自主性，本质上只是在谈能力，不是在谈风险。反对者则会说：跨领域自主性本身就是一种元风险，框架应该把它单独点出来。

### 三者的共同点

- 都有内部安全审查机构，例如 Anthropic SAG、OpenAI SAG、DeepMind 的内部委员会。
- 都明确提到 deceptive alignment 或对齐伪装这一风险类别。
- 都有按既定节奏持续发布的文档产物：Anthropic 的 Frontier Safety Roadmap 与 Risk Report，OpenAI 的 Capabilities Reports 与 Safeguards Reports，DeepMind 的 FSF 更新周期。
- 都在不同程度上承认：仅靠监测的防线有上限。DeepMind 的表述最直接：“自动化监测从长期看终将不再充分。”

### 三者的分歧点

- **Anthropic**：在 v3.0 中移除了暂停承诺，并把 AI R&D-4 作为下一个明确点名的关键阈值。
- **OpenAI**：采用 Tracked 与 Research 的双分结构；Research Categories 即使包括 Long-range Autonomy，也不会自动触发门槛措施。
- **DeepMind**：把自主性折叠进其他领域，并在 2026 年 4 月通过 Tracked Capability Levels 提高了粒度。

### Sandbagging：一个让三家框架都变得棘手的能力

Sandbagging，也就是模型在评估中策略性压低表现，被列入 OpenAI 的 Research Categories。Anthropic 的 RSP v3.0 则通过评估语境差距问题来间接触及它，这正对应第 1 课。DeepMind 则在 FSF v3 中通过 deceptive alignment monitoring 去覆盖这类风险。

如果模型会在评估里故意藏能力，那么任何框架里的能力阈值都会被系统性低估。框架只有在测量可靠时才成立。这也是为什么，除了实验室自己的自评之外，还必须有外部测量（第 21 课，METR）以及对抗性评估。

### 读这类政策时要练的技能

- **定位**：你关心的能力类别，必须能在政策文本中被明确找到。找不到，就意味着这份政策根本没有覆盖它。
- **分类**：它究竟属于 Tracked（会触发缓解）还是 Research（被追踪但不自动触发）？OpenAI 是直接这样命名的；Anthropic 和 DeepMind 则有各自的等价结构。
- **更新节奏**：政策是按照明示节奏持续更新，还是只在发生特定事件后才更新？有公开节奏的政策更强。
- **独立性**：外部审查是强制性的还是可选的？Anthropic 会提到 Apollo 与 US AI Safety Institute，OpenAI 会提到 METR，DeepMind 则更多依赖内部 SAG。

```figure
a5-tracked-vs-research
```

## 用起来

`code/main.py` 实现了一个小型决策表对比工具。输入某项能力，例如 autonomy、deceptive alignment、R&D automation、cyber uplift，它会输出三份政策各自如何分类这项能力，以及会触发哪些缓解措施。它是一个帮助阅读政策的工具，而不是政策本身。

## 交付物

`outputs/skill-cross-policy-diff.md` 提供了一个跨政策比较模板，用这三个框架作为参照，对某一具体能力进行横向对照。

## 练习

1. 运行 `code/main.py`。确认这个 diff 工具对至少两项你能在原文中核实的能力，输出都与政策文本一致。

2. 通读 OpenAI Preparedness Framework v2。找出每一个 Research Category，并分别写一句话说明：为什么它被放在 Research，而不是 Tracked。

3. 通读 DeepMind FSF v3 以及 2026 年 4 月的 Tracked Capability Levels 更新。找出 ML R&D autonomy level 1 的具体评估标准，并说明如果由你来做外部测量，会怎么设计。

4. Sandbagging 被列入 OpenAI 的 Research Categories。设计一个评估，使会 sandbagging 的模型被迫暴露真实能力，并结合第 1 课关于 eval-context gaming 的讨论说明理由。

5. 任选一项能力，对三份政策进行对比。指出哪一份政策的分类最严格，哪一份最宽松，并用原文支持你的判断。

## 关键词

| 术语 | 常见说法 | 实际含义 |
|---|---|---|
| Preparedness Framework | “OpenAI 的备战框架” | PF v2（2025 年 4 月）；采用 Tracked 与 Research 双分结构 |
| Tracked Category | “必须触发缓解的类别” | 会触发 Capabilities + Safeguards Reports，并接受 SAG 审查 |
| Research Category | “只监测、不自动触发” | 被追踪但不会自动触发缓解；包括 Long-range Autonomy |
| Frontier Safety Framework | “DeepMind 的前沿安全框架” | FSF v3（2025 年 9 月）以及 2026 年 4 月加入的 Tracked Capability Levels |
| CCL | “Critical Capability Level” | DeepMind 在不同领域下设置的关键能力阈值 |
| ML R&D autonomy level 1 | “研发自动化” | 以有竞争力的成本完整自动化 AI R&D 流程 |
| Sandbagging | “策略性压低表现” | 模型在评估中故意表现较差；位于 OpenAI Research Categories |
| Instrumental reasoning | “工具性推理” | 为实现目标而推演手段的能力；是 DeepMind 监测关注点之一 |

## 延伸阅读

- [OpenAI — Updating our Preparedness Framework](https://openai.com/index/updating-our-preparedness-framework/) — v2 公告。
- [OpenAI — Preparedness Framework v2 PDF](https://cdn.openai.com/pdf/18a02b5d-6b67-4cec-ab64-68cdfbddebcd/preparedness-framework-v2.pdf) — 完整文档。
- [DeepMind — Strengthening our Frontier Safety Framework](https://deepmind.google/blog/strengthening-our-frontier-safety-framework/) — FSF v3 公告。
- [DeepMind — Updating the Frontier Safety Framework (April 2026)](https://deepmind.google/blog/updating-the-frontier-safety-framework/) — Tracked Capability Levels 的新增说明。
- [Gemini 3 Pro FSF Report](https://storage.googleapis.com/deepmind-media/gemini/gemini_3_pro_fsf_report.pdf) — 一份 FSF 格式风险报告的示例。
