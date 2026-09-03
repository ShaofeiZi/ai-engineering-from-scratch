---
name: obs-platform-wiring
description: 选择一个可观测性平台（Langfuse、Phoenix、Opik、Datadog），并将 trace、评估和提示词版本接入到已有的智能体中。
version: 1.0.0
phase: 14
lesson: 24
tags: [observability, langfuse, phoenix, opik, datadog, tracing]
---

给定一个智能体运行时和产品需求，选择一个可观测性平台并搭建接入配置。

决策：

1. 需要集中化的提示词管理 + 会话回放 -> **Langfuse**。
2. 需要深度 RAG 相关性 + 漂移/异常检测 -> **Phoenix**。
3. 需要自动化提示词优化 + PII 防护栏 -> **Opik**。
4. 已在使用 Datadog -> **Datadog LLM Observability**（从 v1.37+ 原生支持 GenAI 映射）。
5. 需要 ELv2 免授权 -> **Langfuse**（MIT）或 **Opik**（Apache 2.0）；纯 OSS 分发场景下避免使用 Phoenix。

产出：

1. OTel GenAI instrumentation（Lesson 23）——这是公共底层基础。
2. 平台专属 SDK 或 OTel exporter 配置。
3. 面向你所在领域的 LLM-judge 评分细则（事实正确性、范围、语气、拒答质量）。
4. 与 trace 关联的提示词版本管理（Langfuse），或 trace 聚类配置（Phoenix），或实验定义（Opik）。
5. 对已记录内容的防护栏：PII 脱敏、密钥擦除。
6. 仪表盘：会话健康度、失败分类、延迟分布、单次会话成本。

硬性拒绝：

- 不带评估就上线。仅有 trace 不过是昂贵的日志。
- 使用自行编写的 LLM-judge 且无外部验证。CRITIC 模式（Lesson 05）：评判者需要外部工具来提供事实依据。
- 在 span 体中存储 PII。始终使用外部存储 + 引用 ID。

拒绝规则：

- 如果用户要求"一个平台解决一切"，拒绝并给出上述决策建议。没有任何单一平台能在三个维度上全面占优。
- 如果产品对每个智能体任务没有验收标准，拒绝交付评估。LLM-judge 需要评分细则；评分细则需要产品决策。
- 如果用户希望"不采样，捕获一切"，拒绝。Trace 体量随流量线性增长；在规模化场景下必须采样（基于头部或基于尾部）。

输出：`instrumentation.py`、`judge.py`、`dashboards.md`、`README.md`，说明平台选择、评分细则、采样策略以及事故响应。以"延伸阅读"结尾，指向 Lesson 30（评估驱动开发）或 Lesson 26（失败模式分类）。
