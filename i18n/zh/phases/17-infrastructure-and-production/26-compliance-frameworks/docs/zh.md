# 合规框架：SOC 2、HIPAA、GDPR、PCI-DSS、EU AI Act、ISO 42001

> 到了 2026 年，能覆盖多套合规框架已经是企业采购的基本门槛。**EU AI Act** 已于 2024 年 8 月 1 日生效，其中大多数 high-risk requirement 将在 2026 年 8 月 2 日开始执行。违反 high-risk-system obligations 的罚款上限为 €15M 或全球年营业额的 3%（Art. 99(4)）；涉及 prohibited AI practices 的罚款上限为 €35M 或 7%（Art. 99(3)）。只要你服务 EU 用户，它就可能在全球范围内适用。**Colorado AI Act** 将于 2026 年 6 月 30 日生效，原定 2026 年 2 月，后由 SB25B-004 延后；核心要求包括对 high-risk system 做 impact assessment，以及赋予用户对 AI 决策的 appeal right。Virginia 在 credit、employment、housing、education 等领域也提出了相近要求。**SOC 2 Type II** 已经成为事实上的 B2B AI 基线，尤其对 fintech 而言，采购方要的是 Type II，不是 Type I。**GDPR** 方面，当前有记录的最大 AI-specific 罚款是 Dutch DPA 于 2024 年 9 月对 Clearview AI 开出的 €30.5M；意大利 Garante 在 2024 年 12 月对 OpenAI 开出的 €15M 则是最大的 LLM-specific 罚款，不过该裁决后来在 2026 年 3 月上诉中被推翻。对 GDPR 来说，推理前的实时 PII redaction 才是可辩护标准，post-processing cleanup 不够。**HIPAA** 面向医疗场景，没有 BAA 就不能把 PHI 发送给外部 AI 服务。**PCI-DSS** 下，AI interaction layer 是否纳入覆盖，取决于配置与合同条款，不会自动成立。**ISO 42001** 是新兴的 AI governance 标准，正在与 ISO 27001 一起进入采购要求。参考轮廓方面，OpenAI 维护 SOC 2 Type 2、ISO/IEC 27001:2022、ISO/IEC 27701:2019、GDPR/CCPA/HIPAA（BAA）/FERPA，以及 ChatGPT payment component 的 PCI-DSS 覆盖。跨框架 control mapping 能显著降低审计疲劳，例如 access control 可以同时映射到 ISO 27001 A.5.15-5.18、GDPR Art. 32 与 HIPAA §164.312(a)。

**Type:** 学习
**Languages:** （Python 可选——合规属于政策与流程范畴，而不是代码问题）
**Prerequisites:** 阶段 17 · 25（安全）、阶段 17 · 13（可观测性）
**Time:** 约 60 分钟

## 学习目标

- 列出 2026 年与 LLM 产品相关的 7 套框架，并把它们分别对应到具体客户场景。
- 准确说出 EU AI Act 的执行时间线：2024 年 8 月生效，2026 年 8 月开始执行 high-risk 要求；并记住两层罚款上限：€15M / 3% 与 €35M / 7%。
- 解释为什么对 GDPR 而言，post-processing PII cleanup 不够，必须以推理前的实时 redaction 作为可辩护标准。
- 描述 cross-framework control mapping，例如 access control 同时映射到 ISO 27001 A.5.15-5.18、GDPR Art. 32 与 HIPAA §164.312(a)。

## 问题

一个企业客户的采购团队向你索要 SOC 2 Type II、GDPR、HIPAA BAA、ISO 27001，以及一份 EU AI Act compliance statement。而你们现在只有 SOC 2 Type I，距离拿到 Type II 还差 6 个月，GDPR Article 30 记录甚至还没开始做。

多框架覆盖本身并不是 LLM 独有的问题，它本质上是 enterprise SaaS 问题，只不过在 LLM 产品上又叠加了一层更具体的合规要求。到了 2026 年，采购团队想看的不是一份笼统的 PDF，而是一张矩阵：每个 framework 一行，每个 control 一列。

## 概念

### 七套框架

| 框架 | 范围 | LLM 特有要求 |
|-----------|-------|--------------------------|
| SOC 2 Type II | B2B SaaS 基线 | 控制项要在 6-12 个月窗口内接受审计验证 |
| HIPAA | 美国医疗 | 必须有 BAA；没有签署协议前，PHI 不能离开基础设施 |
| GDPR | EU 用户 | 需要实时 PII redaction、数据主体权利保障与 Article 30 records |
| PCI-DSS | 支付数据 | AI 接触支付数据时需要配置控制与合同条款 |
| EU AI Act | 服务 EU 用户 | 需要风险分级；high-risk 系统要做 conformity assessment、文档与日志 |
| Colorado AI Act | 服务科罗拉多州居民 | 需要 impact assessment 与申诉权 |
| ISO 42001 | AI governance | 新兴标准，通常与 ISO 27001 一起出现 |

### EU AI Act 时间线

- 2024 年 8 月 1 日：正式生效。
- 2025 年 2 月 2 日：prohibited AI practices 开始执行。
- 2026 年 8 月 2 日：high-risk systems 开始执行，包括 conformity assessment、documentation、logging。
- 2027 年 8 月：受协调立法约束的产品中的 high-risk system 进一步进入执行期。

风险层级分为：Unacceptable（禁止）、High-risk（需要 conformity + logging）、Limited-risk（主要是透明度要求）、Minimal-risk（基本无约束）。绝大多数 B2B LLM SaaS 属于 limited-risk；但一旦进入 employment、credit、education、law enforcement、migration 或 essential services，就可能触发 high-risk。

罚款方面（Article 99）：违反 high-risk-system obligations，最高可罚 €15M 或全球年营业额 3%（Art. 99(4)）；涉及 prohibited AI practices，最高可罚 €35M 或 7%（Art. 99(3)）；两者按更高者适用。

### GDPR：实时脱敏才是标准

后处理清理，也就是在 LLM 已经看到 PII 之后再去清洗，并不是可辩护姿态，因为模型已经接触过数据。到了 2026 年，实时的 inference-layer redaction 才是标准：

- 在调用 LLM 之前先做实体识别。
- 使用一致的 token 标记方式保留语义连贯性（例如 Mesh 方法）。
- 存储时只保留已 redaction 的 prompts，以及经过 consent 的 opt-in 原始数据。

最近的执法案例也说明了这一点：对 Clearview AI 的 €30.5M 罚款（Dutch DPA，2024 年 9 月）是目前已记录的最大 AI-specific GDPR 罚款；对 OpenAI 的 €15M 罚款（Italy's Garante，2024 年 12 月）则是最大的 LLM-specific 罚款，虽然该裁决在 2026 年 3 月上诉中被推翻，但相关结论仍在进一步审查中。靠 post-processing 自证合规，在审计中通常站不住脚。

### HIPAA：BAA 不是可选项

没有签署好的 Business Associate Agreement，你就不能把 PHI 发送给外部 AI 服务。三大 hyperscaler LLM 平台（Bedrock、Azure OpenAI、Vertex）都提供 BAA。OpenAI direct API 提供 BAA。Anthropic direct API 也提供 BAA。在发送 PHI 之前必须先确认这一点。

### SOC 2 Type II

Type I：控制设计完成并已形成文档。
Type II：控制在 6-12 个月内被证明持续有效运行。

到了 2026 年，B2B 采购默认看的就是 Type II。Type I 只是起步，Type II 才是入场券。

常见的审计驱动项包括：access logs（谁看过什么）、change management（如何部署）、risk assessments（是否按季度执行）、incident response（有没有实际演练）。Phase 17 · 25 中的 audit log 可以直接复用到这里。

### 跨框架映射

一项 access control policy 往往可以同时满足多个框架控制：

| 控制 | 框架 |
|---------|-----------|
| 访问记录 | ISO 27001 A.5.15-5.18, GDPR Art. 32, HIPAA §164.312(a) |
| 变更管理 | ISO 27001 A.8.32, PCI DSS Req. 6, HIPAA breach-notification 范围 |
| 传输中加密 | ISO 27001 A.8.24, GDPR Art. 32, HIPAA §164.312(e) |
| 密钥与 secrets 管理 | ISO 27001 A.8.19, PCI DSS Req. 8, SOC 2 CC6.1 |

像 Drata、Vanta、Secureframe 这类合规工具会自动化处理这类 mapping。在规模上来之后，这笔钱通常值得花。

### ISO 42001：正在兴起

它在 2023 年末发布，正逐渐与 ISO 27001 一起进入采购要求。它提供的是 AI governance 框架，覆盖风险管理、数据质量、透明度、人类监督等方面。

### OpenAI 的参考轮廓

OpenAI 维护 SOC 2 Type 2、ISO/IEC 27001:2022、ISO/IEC 27701:2019、GDPR/CCPA/HIPAA（BAA）/FERPA，以及 ChatGPT payment component 的 PCI-DSS 覆盖。这大致就是 2026 年 enterprise table stakes 的参考线。

### 你应该记住的数字

- EU AI Act 罚款：最高 €15M / 3%（high-risk obligations，Art. 99(4)）；最高 €35M / 7%（prohibited practices，Art. 99(3)）。
- EU AI Act high-risk enforcement：2026 年 8 月 2 日。
- 当前已记录的最大 AI-specific GDPR 罚款：€30.5M，Clearview AI（Dutch DPA，2024 年 9 月）。
- 最大 LLM-specific GDPR 罚款：€15M，OpenAI（Italy's Garante，2024 年 12 月；后于 2026 年 3 月上诉中被推翻）。
- SOC 2 Type II 窗口：6-12 个月的已运行控制。
- Colorado AI Act 生效日期：2026 年 6 月 30 日（由 SB25B-004 从 2026 年 2 月延后）。

```figure
i4-control-matrix
```

## 用起来

`code/main.py` 是一个 Python 编写的合规映射小工具：给定一个 control，列出它满足哪些框架。

## 交付物

这一课会产出 `outputs/skill-compliance-matrix.md`。它会根据客户细分和地理区域，给出所需的 framework 与 control 组合。

## 练习

1. 你的第一个企业客户要求 SOC 2 Type II、HIPAA BAA 和一份 EU AI Act statement。赢下这单所需的 minimum viable compliance posture 是什么？
2. 按照 EU AI Act 的风险等级，对三个假设中的 LLM 产品进行分类。到了 high-risk，会新增哪些要求？
3. 你不小心把 PHI 发给了一个没有 BAA 的 provider。请完整走一遍 incident response。
4. 论证 ISO 42001 对一家中端市场 AI vendor 来说，是否已经是“2026 年必需项”。
5. 把你的 LLM audit log 字段（Phase 17 · 25）映射到至少三个 framework control。

## 关键术语

| 术语 | 人们怎么说 | 它实际意味着什么 |
|------|----------------|------------------------|
| SOC 2 Type II | “审计过的控制” | 控制在 6-12 个月内持续运行，并由独立方出具证明 |
| HIPAA BAA | “医疗合同” | Business Associate Agreement；处理 PHI 的前提 |
| GDPR | “EU 隐私” | 实时 PII redaction 才是 2026 年可辩护标准 |
| EU AI Act | “EU AI 规则” | 2026 年 8 月开始执行 high-risk；€15M / 3%（high-risk obligations）与 €35M / 7%（prohibited practices） |
| Colorado AI Act | “美国州级 AI 法” | 2026 年 6 月 30 日生效（由 SB25B-004 延后）；要求 impact assessments |
| ISO 42001 | “AI 治理” | 面向 AI 风险与透明度的新兴框架 |
| ISO 27001 | “安全 ISMS” | 信息安全管理体系的基线 |
| Conformity assessment | “EU AI 文件包” | high-risk 所需的文档、测试与日志 |
| 跨框架映射 | “一个控制，多套框架” | 单一政策可同时满足多个框架控制 |

## 延伸阅读

- [OpenAI 安全与隐私](https://openai.com/security-and-privacy/)：OpenAI 的参考合规轮廓。
- [GuardionAI — LLM Compliance 2026: ISO 42001, EU AI Act, SOC 2, GDPR](https://guardion.ai/blog/llm-compliance-guide-iso-42001-eu-ai-act-soc2-gdpr-2026)
- [Dsalta — 2026 SOC 2 Type 2 审计指南：10 项 AI 控制](https://www.dsalta.com/resources/ai-compliance/soc-2-type-2-audit-guide-2026-10-ai-powered-controls-every-saas-team-needs)
- [EU AI Act 官方文本](https://eur-lex.europa.eu/eli/reg/2024/1689/oj)：主要来源。
- [Colorado AI Act](https://leg.colorado.gov/bills/sb24-205)：主要来源。
- [ISO/IEC 42001:2023](https://www.iso.org/standard/81230.html)：AI management system standard。
