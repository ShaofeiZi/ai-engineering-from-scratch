# 监管框架：EU、US、UK、Korea

> 到了 2026 年，AI 治理格局主要由四套监管制度定义。EU AI Act 已于 2024 年 8 月 1 日生效，其中 prohibited practices 与 AI literacy 要求自 2025 年 2 月 2 日起适用；GPAI obligations 自 2025 年 8 月 2 日起适用；完整适用范围与 Article 50 transparency 自 2026 年 8 月 2 日起生效；legacy GPAI 与 embedded high-risk systems 则顺延到 2027 年 8 月 2 日。处罚最高可达 15M EUR 或全球营业额的 3%。GPAI Code of Practice 于 2025 年 7 月 10 日发布，共三章，分别是 Transparency、Copyright、Safety and Security，总计 12 项 commitments，正式执行从 2026 年 8 月开始。英国的 AISI 在 2025 年 2 月更名为 AI Security Institute，名称变化本身就反映出范围收缩。美国的 AISI 则在 2025 年 6 月改组为 NIST 旗下的 CAISI，也就是 Center for AI Standards and Innovation，整体姿态更偏向 pro-growth。韩国的 AI Framework Act 于 2024 年 12 月通过，2026 年 1 月正式生效；其第 12 条要求在 MSIT 下设立 AISI，并要求外国 AI 公司设立本地代表、进行风险评估，并为 high-impact 与 generative AI 采取安全措施。

**Type:** 学习
**Languages:** none
**Prerequisites:** 阶段 18 · 18（前沿安全框架）、阶段 18 · 27（数据来源与训练数据治理）
**Time:** 约 75 分钟

## 学习目标

- 描述 EU AI Act 的风险分层，也就是 prohibited、high-risk、general-purpose、limited-risk，以及 2025 年 8 月、2026 年 8 月、2027 年 8 月这几个关键节点。
- 描述 GPAI Code of Practice 的三章结构，并说明它分别约束哪些 provider。
- 描述 2025 年的两次更名，UK AISI -> AI Security Institute，US AISI -> CAISI，以及这些更名各自反映出的政策方向变化。
- 说出 Korea's AI Framework Act 的核心条款。

## 问题

实验室框架，也就是 Lesson 18 讨论的那类框架，本质上是 voluntary 的；监管框架则是 compulsory 的。2024 到 2026 年，是第一轮综合性 AI regulation 真正进入生效阶段的时间窗。对 deployer 来说，关键难点在于把技术控制映射到监管义务，而且这种映射会随着 jurisdiction 不同而不同。

## 概念

### EU AI Act

**2024 年 8 月 1 日生效。** 它的核心结构是风险分层：

- **Prohibited practices**（Article 5）。包括 social scoring、public 场景下的 real-time remote biometric identification（有 law-enforcement exceptions），以及对弱势群体的 exploitative manipulation。这部分自 2025 年 2 月 2 日起适用。
- **High-risk systems**（Annex III）。包括 employment、education、credit、law enforcement、justice、migration 等场景。要求进行 conformity assessment、risk management、logging、transparency。
- **General-Purpose AI (GPAI) models**。自 2025 年 8 月 2 日起适用。所有 GPAI providers 都有义务；其中 systemic-risk GPAI，也就是训练计算量超过 1e25 FLOP 的模型，还要承担额外义务。
- **Limited-risk systems**。主要对应 Article 50 下的 transparency obligations，也就是 AI-generated content labelling，自 2026 年 8 月 2 日起适用。

时间线是：
- 2025 年 2 月 2 日：prohibited practices + AI literacy。
- 2025 年 8 月 2 日：GPAI + governance。
- 2026 年 8 月 2 日：全面适用 + Article 50 transparency + 最高 15M EUR / 3% global turnover 的处罚。
- 2027 年 8 月 2 日：legacy GPAI + embedded high-risk。

欧委会在 2025 年底曾提出，把 high-risk 的时间线调整为 16 个月。

### GPAI Code of Practice

该文件于 2025 年 7 月 10 日发布，共三章：

- **Transparency。** 适用于所有 GPAI providers。
- **Copyright。** 同样适用于所有 GPAI providers。
- **Safety and Security。** 仅适用于 systemic-risk GPAI providers，估计受约束的公司数量约为 5 到 15 家。

总计 12 项 commitments。执行由 AI Office 领导下的 Signatory Taskforce 管理。正式 enforcement 从 2026 年 8 月 2 日开始；在此之前，good-faith compliance 仍会被接受。

### Article 50 的 Transparency Code

第一版草案发布于 2025 年 12 月 17 日，第二版草案在 2026 年 3 月，最终版本预期于 2026 年 6 月形成。它覆盖 AI-generated content labelling，包括 deepfakes，也就是对 Lesson 23 中 watermarking 技术提出监管要求的那一层。

### UK AI Security Institute（2025 年 2 月）

它从 AI Safety Institute 更名而来。这个更名的含义不是表面修辞，而是明显缩窄了政策重心：对 algorithmic bias 和 free-speech 这类 framing 的强调被弱化，焦点更集中到 frontier capability security。该机构曾在 2024 年 5 月开源 Inspect evaluation tool，并与 Redwood（Lesson 10）合作推进 control safety cases。

### US CAISI（2025 年 6 月）

特朗普政府把 NIST 的 AI Safety Institute 重组为 Center for AI Standards and Innovation。根据 VP Vance 在 Paris AI Action Summit 的表述，整体方向转向 “pro-growth AI policies”。这意味着对 pre-deployment evaluation 的强调减少，而对 standards 与 innovation support 的强调增强。它在政策姿态上，基本构成了 EU AI Act 式监管立场的美国国内对照面。

### Korean AI Framework Act

该法案于 2024 年 12 月通过，2025 年 1 月颁布，2026 年 1 月正式生效，并整合了原本分散的 19 项 AI 相关法案。

其 Article 12 要求在 Ministry of Science and ICT，也就是 MSIT，下设一个 AISI。主要义务包括：
- 为在韩国运营的外国 AI 公司设置 local representatives。
- 对 “high-impact” AI systems 进行风险评估。
- 为 generative AI 和 high-impact AI 采取安全措施。

这是亚洲第一个横向、综合性的 AI regulation jurisdiction。

### 跨法域动态

- EU：严格、风险分层明确、处罚重，对 privacy-adjacent regulation 具有标杆意义。
- US：更偏向 innovation，联邦层面相对分散，很多空白由州法来补，例如 California AB 2013，也就是 Lesson 27 会提到的内容。
- UK：安全焦点更窄，但 evaluation infrastructure 很强。
- Korea：由 MSIT 牵头，对外国 provider 的约束尤其明确。

这些背后其实是彼此竞争的监管哲学。对跨多个法域部署的团队来说，现实中的做法通常是按最严格的一套来设计，而在 2026 年，这往往就是 EU AI Act。

### 它在 Phase 18 里的位置

Lesson 18 讨论的是实验室层面的 voluntary governance；Lesson 24 转向 regulatory layer；Lesson 25 会进入一类新出现的 AI system CVEs；Lessons 26-27 则继续覆盖 documentation，也就是 cards，以及 training-data governance。

```figure
an-eu-act-timeline
```

## 用它

这课没有代码。直接去读 EU AI Act 的 primary sources，包括 regulation text、GPAI Code of Practice，以及 UK AISI 的 Inspect framework。然后把你自己的 deployment 映射到每个 jurisdiction 下对应的 obligations。

## 交付它

这一课会产出 `outputs/skill-regulatory-map.md`。给定一个 deployment description，它会映射出适用的 jurisdictions、各自的 tier classifications、逐法域 obligations，以及对应的 deadline structure。

## 练习

1. 阅读 EU AI Act（Regulation 2024/1689）与 GPAI Code of Practice（2025 年 7 月 10 日版）。找出三项适用于所有 GPAI providers 的 obligations，以及三项只适用于 systemic-risk GPAI 的 obligations。

2. 某项部署由一家美国公司提供，跑在欧盟基础设施上，并向韩国用户提供服务。请判断哪三个 jurisdiction 的规则会适用，以及每一个实质性问题分别由哪一套规则约束。

3. UK AI Security Institute 的更名意味着范围收缩。请分别论证支持与反对这一收缩框架的立场，并指出每种立场背后的 policy assumption。

4. CAISI 的 “pro-growth” framing 明显偏离了 2022-2024 年 AI safety institute 模型。请指出在这种 framing 下，会出现哪两种可测量的 policy shift。

5. Korea's AI Framework Act 要求外国 provider 设立 local representatives。请描述这对一家向韩国用户提供服务的 Bay Area 公司会产生哪些 operational implications。

## 关键术语

| 术语 | 人们通常怎么说 | 实际含义 |
|------|-----------------|------------------------|
| 《欧盟人工智能法案》（EU AI Act） | “那部法规” | 按风险分层、横向适用于各行业的 AI 法规；2024 年 8 月生效 |
| 通用人工智能（GPAI） | “通用人工智能” | 大型基础模型；其中具有系统性风险的子集还须履行额外义务 |
| 第 50 条（Article 50） | “透明度义务” | 对 AI 生成内容加注标签的义务；2026 年 8 月起适用 |
| UK AISI | “人工智能安全研究所” | 2025 年 2 月更名；职责范围收窄，重点转向前沿 AI 安全 |
| CAISI | “美国人工智能标准中心” | 2025 年 6 月由 AI Safety Institute 改组而来；政策立场更偏向促进增长 |
| 《韩国人工智能框架法》（Korean AI Framework Act） | “科学技术信息通信部的横向监管” | 亚洲首部综合性 AI 法；2026 年 1 月生效 |
| 系统性风险 GPAI（Systemic-risk GPAI） | “1e25 FLOP 门槛” | 须履行额外义务的层级；估计约有 5 至 15 家公司受到约束 |

## 延伸阅读

- [EU AI Act text (Regulation 2024/1689)](https://digital-strategy.ec.europa.eu/en/policies/regulatory-framework-ai) — 法规正文与实施时间线
- [GPAI Code of Practice (10 July 2025)](https://digital-strategy.ec.europa.eu/en/library/final-version-general-purpose-ai-code-practice) — 分为三章的行为准则
- [UK AI Security Institute (renamed Feb 2025)](https://www.gov.uk/government/organisations/ai-security-institute) — 官方页面
- [CSET — South Korea AI Framework Act Analysis (2025)](https://cset.georgetown.edu/publication/south-korea-ai-law-2025/) — 韩国框架法分析
