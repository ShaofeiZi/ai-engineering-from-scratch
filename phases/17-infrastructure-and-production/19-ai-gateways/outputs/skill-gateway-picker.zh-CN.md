---
name: gateway-picker
description: 根据规模、延迟预算、合规要求、运维姿态和定价容忍度，选择 AI 网关（LiteLLM、Portkey、Kong AI、Cloudflare/Vercel）。
version: 1.0.0
phase: 17
lesson: 19
tags: [ai-gateway, litellm, portkey, kong, cloudflare, vercel, bifrost, fallback, rate-limit, guardrails]
---

给定 RPS（当前及未来 12 个月预测）、延迟预算、合规要求（是否必须自托管？）、护栏需求（PII 脱敏、越狱检测、审计）以及定价容忍度，输出一份网关推荐方案。

产出内容：

1. 主网关。指明工具名称。以 RPS 上限、开销和功能匹配度作为理由。
2. 降级链。按顺序列出三家提供商；OpenAI → Anthropic → 自托管为经典方案。计算预期可用性。
3. 限流策略。>500 RPS 时推荐滑动窗口；否则令牌桶可接受。按租户分层。
4. 护栏。若需 PII/越狱检测则选 Portkey；若需大规模 + 护栏则选 Kong；若仅为开发层级则选 LiteLLM。
5. 可观测性交接。指向第 17 阶段 · 13 的选择；确认 OTel GenAI 约定贯穿流转。
6. 迁移方案。若从应用层集成迁移，分阶段推出（网关上 1% 灰度，成功后逐步扩大）。

硬性拒绝条件：
- LiteLLM 用于 >2000 RPS。拒绝——Kong 基准测试显示级联故障；应先迁移。
- Portkey 用于 TTFT P99 < 100 ms 的 SLA。拒绝——30 ms 开销占用过多预算。
- Cloudflare AI Gateway 用于受监管的本地部署客户。拒绝——仅提供托管版；无自托管。

拒绝规则：
- 若规模不确定性较大（当前 100 RPS，计划 6 个月内达 2K+），在确定 LiteLLM 前要求先提供迁移方案。
- 若合规要求 SOC 2 Type II 且所选网关为纯 OSS 无托管 SLA，要求客户提供自身的 SOC 2 鉴证报告。
- 若团队无 Kubernetes 却选择 Kong 自托管，拒绝——推荐托管 Kong 或 Portkey 托管版。

输出：一页决策文档，包含网关、降级链、限流策略、护栏姿态、可观测性流向、迁移方案。以唯一指标结尾：过去一小时的网关延迟 P99；超限即告警。
