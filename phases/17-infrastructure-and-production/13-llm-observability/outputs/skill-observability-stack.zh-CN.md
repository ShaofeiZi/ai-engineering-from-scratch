---
name: observability-stack
description: 根据技术栈、规模、预算和许可姿态，选择 LLM 可观测性技术栈（开发平台 + 网关 + 可选扩展层），并定义 OpenTelemetry GenAI 属性集。
version: 1.0.0
phase: 17
lesson: 13
tags: [observability, langfuse, langsmith, phoenix, arize, helicone, opik, opentelemetry, genai-conventions]
---

给定技术栈（LangChain / DSPy / 原始 SDK）、规模（traces/day）、预算、许可姿态（仅 MIT vs 可接受商业许可）及自托管要求，产出一份可观测性方案。

产出内容：

1. 开发平台选型。Langfuse（OSS）、LangSmith（LangChain 优先的商业方案）、Opik（Comet OSS）或不用。以技术栈和许可来论证。
2. 网关/遥测选型。Helicone（代理 + 网关）、SigNoz（全 APM）、OpenLLMetry（纯 OTel）。如果已使用 AI 网关（Phase 17 · 19），说明集成方式。
3. 扩展/数据湖层。可选；Arize AX 或原始 Iceberg 用于长期分析，Phoenix 用于 RAG 漂移检测。
4. OTel GenAI 约定。指定最小属性集：`gen_ai.system`、`gen_ai.request.model`、`gen_ai.usage.input_tokens`、`gen_ai.usage.output_tokens`、`gen_ai.request.temperature`、`gen_ai.response.finish_reasons`，加上组织专属属性（tenant_id、user_id、task）。
5. 采样策略。100% 错误、100% 高成本（>$0.10/调用）、N% 成功采样率。原始数据保留窗口（14d / 30d / 90d）。聚合数据保留更久。
6. 告警。必须设置告警的五个指标：错误率、P99 TTFT、成本/请求、提示词缓存命中率、拒绝率。

硬性拒绝：
- 在框架专属 SDK 内插桩而没有 OTel 降级方案。拒绝——框架锁定。
- 在 Datadog 级别定价下为非合规工作负载保留 100% 链路，月费 >$500。拒绝——建议采样。
- 忽略 OpenTelemetry GenAI 约定。拒绝——2026 年互操作性需要它们。

拒绝规则：
- 如果 traces/day > 5M 且团队坚持完整 Datadog 保留，在没有成本预测的情况下拒绝。
- 如果团队仅接受 MIT 许可却选择了 LangSmith，拒绝——Langfuse 是 MIT 等价方案。
- 如果团队没有 AI 网关，选择 Helicone 同时作为网关和可观测性方案，接受——该代理在 ~500 RPS 以内可兼作网关（Phase 17 · 19 涵盖网关扩展）。

输出：一页方案，命名开发平台、网关、扩展层（如有）、OTel 属性集、采样规则、五项告警。以单一指标结尾，该指标标识技术栈漂移：过去 7 天内具有完整 OTel GenAI 属性的 LLM 调用百分比。
