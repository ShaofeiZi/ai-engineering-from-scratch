# AI 网关：LiteLLM、Portkey、Kong AI Gateway、Bifrost

> Gateway 位于你的应用与模型提供方之间。它的核心职责包括 provider routing、fallback、retries、rate limiting、secret references、observability 和 guardrails。到 2026 年，这个市场已经明显分层：**LiteLLM** 是 MIT 协议开源，支持 100+ provider，兼容 OpenAI API，但在约 2000 RPS 左右开始失稳，公开 benchmark 中会出现 8 GB 内存占用和 cascading failure，更适合 Python 场景、<500 RPS、开发与原型阶段。**Portkey** 的定位更偏控制平面，强调 guardrails、PII redaction、jailbreak detection 与 audit trails，2026 年 3 月转为 Apache 2.0 开源，额外延迟大约 20–40 ms，production tier 价格为 $49/月。**Kong AI Gateway** 构建在 Kong Gateway 之上，Kong 自己在同样 12 CPU 条件下的 benchmark 声称：相对 Portkey 快 228%，相对 LiteLLM 快 859%；定价是 $100/model/month，Plus 档最多 5 个模型。如果你的基础设施本来就在 Kong 上，它通常是更契合企业场景的选择。**Bifrost**（Maxim AI）主打自动重试与可配置 backoff，典型用法是在 OpenAI 429 时自动 fallback 到 Anthropic。**Cloudflare / Vercel AI Gateway** 则是托管型、零运维、带基础 retry 的方案。最终是否自托管，往往不是由“功能多不多”决定，而是由 data residency 要求决定；Portkey 和 Kong 刚好位于 OSS 与托管之间的中间带。

**Type:** 学习
**Languages:** Python（标准库，玩具级网关路由模拟器）
**Prerequisites:** 阶段 17 · 01（托管 LLM 平台）、阶段 17 · 16（模型路由）
**Time:** 约 60 分钟

## 学习目标

- 枚举 gateway 的六类核心能力：routing、fallback、retries、rate limits、secrets、observability、guardrails。
- 将 2026 年四类主流 gateway（LiteLLM、Portkey、Kong AI、Bifrost）映射到各自的规模上限与使用场景。
- 准确引用 Kong 的 benchmark 数字（相对 Portkey 228%，相对 LiteLLM 859%），并说明为什么这对 >500 RPS 的系统重要。
- 在 data residency 与 ops budget 的约束下，判断该选 self-hosted 还是 managed。

## 问题

你的产品同时调用 OpenAI、Anthropic 和一个 self-hosted Llama。每个 provider 都有自己的 SDK、错误模型、限流方式和鉴权方案。你希望在 OpenAI 429 时自动 failover 到 Anthropic，同时还想有统一的凭据管理、统一的可观测性，以及按租户维度的限流。

如果这些逻辑都堆在应用层，结果就是每个服务都得自己对接每个 provider，耦合度会迅速失控。Gateway 层的作用，就是把这些差异统一收口到一个进程里，对外提供一个 API，通常还是 OpenAI-compatible，再由它向后分发到不同 provider。

## 核心概念

### 七类核心能力

1. **Provider routing**：把 OpenAI、Anthropic、Gemini、self-hosted 等 provider 收敛到一个 API 后面。
2. **Fallback**：遇到 429、5xx 或质量失败时，自动切到别的 provider。
3. **Retries**：指数退避、带上限的重试策略。
4. **Rate limits**：支持按租户、按 key、按 model 限流。
5. **Secret references**：运行时从 vault 拉凭据，不把密钥写进应用。
6. **Observability**：接 OTel、GenAI attributes（Phase 17 · 13）以及成本归因。
7. **Guardrails**：例如 PII redaction、jailbreak detection、allowed-topics filter。

### LiteLLM：MIT OSS，Python 友好

- 支持 100+ provider，兼容 OpenAI，提供 router config、fallback 和基础 observability。
- 在 Kong benchmark 中，大约到 2000 RPS 左右会开始失稳；内存占用约 8 GB，持续负载下可能出现 cascading failure。
- 最适合：Python 应用、<500 RPS、开发/测试环境、实验性 routing。
- 成本：开源版为 $0，也有 cloud free tier。

### Portkey：控制平面定位

- 截至 2026 年 3 月，Portkey 以 Apache 2.0 OSS 提供。
- 功能侧强调 guardrails、PII redaction、jailbreak detection、audit trails。
- 单请求额外延迟大约 20–40 ms。
- production tier 价格约为 $49/月。
- 更适合合规要求高、希望把 guardrails 与 observability 一次打包解决的行业，例如金融、医疗、企业软件。

### Kong AI Gateway：为规模而生

- 建立在 Kong Gateway 之上，而后者本身就是成熟的 API gateway 产品，技术栈以 lua + OpenResty 为主。
- Kong 自己在等价 12 CPU 条件下的 benchmark 声称：相对 Portkey 快 228%，相对 LiteLLM 快 859%。
- 定价为 $100/model/month，Plus tier 最多 5 个模型。
- 如果你的组织已经在用 Kong，或系统目标规模 >1000 RPS，并且能接受商业授权，它通常是更合理的选择。

### Bifrost（Maxim AI）

- 提供自动 retries，并支持可配置的 backoff 策略。
- “在 OpenAI 429 时 fallback 到 Anthropic”是它的典型 recipe。
- 相对更新，也更偏商业产品。

### Cloudflare AI Gateway 与 Vercel AI Gateway

- 托管式、零运维，内建基础 retry 与 observability。
- 更适合运行在 Cloudflare / Vercel 边缘环境中的 JavaScript 应用。
- 与 Kong 或 Portkey 相比，在 guardrails 与 rate limits 的能力上通常更基础。

### 自托管还是托管

真正的分界线往往是 data residency。医疗和金融这类行业，通常默认自托管，因此更容易落在 LiteLLM、Portkey OSS 或 Kong 这类方案上。面向大众消费者的产品则更容易接受 managed，例如 Cloudflare AI Gateway，或者 Portkey 的托管版本。混合部署也很常见：受监管租户走 self-hosted，其余租户走 managed。

### 延迟预算

- LiteLLM：常见额外开销约 5–15 ms。
- Portkey：约 20–40 ms。
- Kong：约 3–8 ms。
- Cloudflare/Vercel：约 1–3 ms，主要靠 edge 带来的路径优势。

Gateway 的延迟会直接加到 TTFT 上。如果你的 SLA 是 TTFT P99 < 100 ms，通常就更偏向 Kong 或 Cloudflare。若 P99 只要求 < 500 ms，则几乎所有方案都能接受。

### 限流语义真的很重要

简单 token-bucket 足以覆盖中等规模系统。但如果是多租户 SaaS，你通常需要 sliding-window、burst allowance 和按租户分层。LiteLLM 默认更偏 token-bucket；Kong 提供 sliding-window；Portkey 则更强调 tiered policy。

### 网关、可观测性和路由本来就是同一层

Phase 17 · 13（observability）+ 16（model routing）+ 19（gateways），在生产里其实就是同一层系统。你可以选一个同时覆盖三者的产品，也可以自己拼装，但拼装时必须非常小心。到 2026 年，很多真实部署会把 Helicone 这类 observability 工具，或 Portkey 这类 guardrails 工具，与 Kong 这种高吞吐 gateway 组合使用，形成职责拆分。

### 你需要记住的数字

- LiteLLM：大约在 2000 RPS 左右失稳，内存占用约 8 GB。
- Portkey：额外延迟约 20–40 ms；自 2026 年 3 月起为 Apache 2.0。
- Kong：相对 Portkey 快 228%，相对 LiteLLM 快 859%。
- Kong 定价：$100/model/month，Plus tier 最多 5 个模型。
- Cloudflare/Vercel：边缘侧额外开销约 1–3 ms。

```figure
mx-gateway-fallback
```

## 用起来

`code/main.py` 会在注入 429/5xx 故障的条件下，模拟 3 个 provider 之间的 gateway routing 与 fallback，并报告延迟、重试率和 fallback hit rate。

## 交付物

本课产出 `outputs/skill-gateway-picker.md`。它会根据系统规模、运维姿态、合规要求和延迟预算，帮你挑选合适的 gateway。

## 练习

1. 运行 `code/main.py`。把 fallback 配成 OpenAI→Anthropic→self-hosted。在 provider error rate 为 5% 时，预期 hit rate 是多少？
2. 你的 SLA 是 TTFT P99 < 200 ms，而基线本身已经有 300 ms。哪些 gateway 还能留在预算内？
3. 某医疗客户要求 self-hosted + PII redaction + audit。你会在 Portkey OSS 和 Kong 之间怎么选？
4. 比较 LiteLLM 与 Kong：团队大概应在什么 RPS 天花板附近开始迁移？
5. 给一个多租户 SaaS 设计限流策略：free tier、trial tier、paid tier 各用什么规则？该用 token-bucket 还是 sliding-window？

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------------|------------------------|
| Gateway | “API 代理” | 位于应用与 provider 之间的中间层进程 |
| LiteLLM | “MIT 开源方案” | Python 开源 gateway，100+ provider，约 2K RPS 开始失稳 |
| Portkey | “护栏网关” | 带控制平面与 observability 的 gateway，Apache 2.0 |
| Kong AI Gateway | “规模化方案” | 构建在 Kong Gateway 之上，以吞吐见长 |
| Bifrost | “Maxim 的网关” | 主打 retries 与 Anthropic fallback recipe |
| Cloudflare AI Gateway | “边缘托管方案” | 部署在边缘的托管 gateway，零运维 |
| PII redaction | “数据脱敏” | 在请求发给模型前，先做正则与 NER 脱敏 |
| Jailbreak detection | “提示词注入防护” | 针对用户输入做分类检测的防护层 |
| Audit trail | “合规日志” | 对每一次 LLM 调用保留不可变记录 |
| Token-bucket | “简单限流” | 基于 refill 机制的限流器 |
| Sliding-window | “精确限流” | 基于时间窗口的限流器，公平性更好 |

## 延伸阅读

- [Kong AI Gateway 基准测试](https://konghq.com/blog/engineering/ai-gateway-benchmark-kong-ai-gateway-portkey-litellm)
- [TrueFoundry — 2026 AI Gateway 对比](https://www.truefoundry.com/blog/a-definitive-guide-to-ai-gateways-in-2026-competitive-landscape-comparison)
- [Techsy — 2026 顶级 LLM Gateway 工具](https://techsy.io/en/blog/best-llm-gateway-tools)
- [LiteLLM GitHub](https://github.com/BerriAI/litellm)
- [Portkey GitHub](https://github.com/Portkey-AI/gateway)
- [Kong AI Gateway 文档](https://docs.konghq.com/gateway/latest/ai-gateway/)
