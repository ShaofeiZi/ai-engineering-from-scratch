---
name: llm-observability
description: 构建一个自托管的 LLM 可观测性仪表板，接收 OpenTelemetry GenAI span，运行评估，并在五分钟内捕获注入的回归。
version: 1.0.0
phase: 19
lesson: 11
tags: [capstone, observability, otel, langfuse, phoenix, evals, drift, clickhouse]
---

给定跨越至少六个 SDK 家族（OpenAI、Anthropic、Google GenAI、LangChain、LlamaIndex、vLLM）的生产 LLM 流量，部署一个自托管的可观测性平面，接收 OTLP GenAI-semconv span，运行评估，检测漂移并告警。

构建计划：

1. OpenTelemetry Collector，带 OTLP HTTP 接收器、尾部采样处理器（保留 100% 错误、10% 成功、100% 高毒性/PII）、导出器到 ClickHouse + S3。
2. ClickHouse span 模式镜像 GenAI semconv：gen_ai.system、gen_ai.request.model、usage.input/output_tokens、latency_ms、user_id、app_id，以及用于 prompt/completion 的 JSON 包。
3. Postgres 元数据存储，用于应用、用户、会话、标注队列。
4. 每个 SDK 家族在客户端应用上进行 OpenLLMetry 自动埋点；验证规范化 span 成功落地。
5. DeepEval + RAGAS + Phoenix 评估器包，按采样轨迹调度运行；自定义 LLM 判官用于 PII 和违规检测。
6. 每周在汇总的 prompt 嵌入上运行 PSI / KL 漂移检测器；告警阈值 0.2。
7. Prometheus 导出器用于评估分数聚合和延迟百分位；Alertmanager 到 Slack（警告）+ PagerDuty（严重）。
8. Next.js 15 App Router 仪表板：概览、轨迹搜索 + 瀑布图、评估趋势、漂移图表、告警。
9. 回归探针：注入一种 1% 概率泄露假 SSN 的响应模式；测量 MTTR（告警触发时间）。

评估标准：

| 权重 | 标准 | 度量 |
|:-:|---|---|
| 25 | 轨迹模式覆盖 | 产生规范化 GenAI span 的 SDK 家族数量（目标 6+） |
| 20 | 评估正确性 | DeepEval / RAGAS 分数 vs 人工标注集 |
| 20 | 仪表板用户体验 | 注入回归的 MTTR（目标低于 5 分钟） |
| 20 | 成本 / 规模 | 持续 1k span/秒摄入无积压 |
| 15 | 告警 + 漂移检测 | 端到端演练 Prometheus/Alertmanager 链路 |

硬性拒绝：

- Span 模式发明不在 OpenTelemetry GenAI semconv 中的属性名。
- 尾部采样策略丢弃错误（一种众所周知的反模式）。
- 评估以摄入速率运行而不采样（不可接受的成本）。
- 仪表板显示"延迟"而不区分 p50/p95/p99。

拒绝规则：

- 拒绝在没有 PII 脱敏策略的情况下持久化 prompt 或 completion。
- 拒绝在没有每个 SDK 规范化 span 回归测试的情况下声称"多 SDK 支持"。
- 拒绝在没有基线窗口的情况下发布漂移检测；零样本漂移检测毫无用处。

产出：一个包含 collector 配置、ClickHouse 模式、Next.js 15 仪表板、评估作业、漂移检测器、告警链、带标注回归的 10k 条轨迹演示数据集的仓库，以及一份文档，记录注入 PII 回归的 MTTR 以及迭代过程中降低 MTTR 的前三项仪表板用户体验改进。
