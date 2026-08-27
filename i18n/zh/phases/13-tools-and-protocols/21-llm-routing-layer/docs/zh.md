# LLM 路由层——LiteLLM、OpenRouter 与 Portkey

> 被单一提供商锁定的代价很高。不同的工具调用负载适合不同的模型。路由网关提供统一的 API 接口，并集中处理重试、故障转移、成本追踪和防护规则。到 2026 年，三种主要形态是：LiteLLM（开源、自托管）、OpenRouter（托管 SaaS）以及 Portkey（生产级，并于 2026 年 3 月开源）。本课将说明选型标准，并带你实现一个仅依赖标准库的路由网关。

**Type:** 学习
**Languages:** Python (stdlib, routing + failover + cost tracker)
**Prerequisites:** 第 13 阶段 · 第 02 课（函数调用）、第 13 阶段 · 第 17 课（网关）
**Time:** 约 45 分钟

## 学习目标

- 区分自托管、托管服务和生产级路由方案。
- 实现一条回退链，在提供商失败时按明确的优先顺序重试。
- 跨提供商追踪每次请求的成本与 token 用量。
- 根据具体生产约束，在 LiteLLM、OpenRouter 和 Portkey 之间做出选择。

## 问题

以下场景都需要提供商路由：

1. **成本。** Claude Sonnet 的成本是 Haiku 的三倍。对分诊任务来说 Haiku 已经足够；对综合分析任务来说 Sonnet 值得更高成本。应按请求路由。

2. **故障转移。** OpenAI 某个时段发生故障，所有请求都失败。你希望无需重新部署，就能自动回退到 Anthropic。

3. **延迟。** 实时聊天界面要求更短的首 token 时间，批量摘要器则没有这项要求。应根据延迟 SLA 路由。

4. **合规。** 欧盟用户的数据必须留在欧盟区域内。应按地域路由。

5. **实验。** 在同一工作负载上对两个模型做 A/B 测试。应按测试分桶路由。

为每个集成分别手写这些逻辑会造成大量重复。路由网关提供一个兼容 OpenAI 的统一 API，并替你处理其余工作。

## 概念

### 兼容 OpenAI 的代理形态

所有后端都通过 OpenAI 风格的接口通信。路由网关暴露 `/v1/chat/completions`，接受 OpenAI schema，再在内部把请求代理到 Anthropic / Gemini / Cohere / Ollama / 任意其他后端。客户端无需关心实际提供商。

### 模型别名

代码不必写死某个快照 ID，而是使用 `our_smart_model`。网关负责把别名映射到真实模型。提供商发布新一代模型时，只需在服务端修改别名映射，业务代码完全不用变。

### 回退链

```
primary: openai/gpt-4o
on 5xx: anthropic/claude-3-5-sonnet
on 5xx: google/gemini-1.5-pro
on 5xx: refuse
```

网关通过配置定义这条链。重试会消耗预设预算，避免连续回退让成本失控。

### 语义缓存

完全相同或近似相同的 prompt 会命中缓存，不再请求提供商。对于反复执行的智能体循环，成本可能降低 30% 到 60%。缓存键基于 embedding，因此近似 prompt 可以共享一个缓存槽位。

### 防护规则

网关层可以提供：

- **PII 脱敏。** 在发送 prompt 前通过正则表达式或机器学习进行处理。
- **策略违规检测。** 拒绝含有禁止内容的 prompt。
- **输出过滤。** 清理 completion 中的泄漏信息。

Portkey 和 Kong 都内置了一套具有明确取向的防护规则。LiteLLM 则将其作为可选能力。

### 按密钥限流

一个 API key 对应一个团队。按密钥设置预算，可以避免某个团队耗尽共享配额。多数网关都支持这一能力。

### 自托管与托管服务的取舍

| 因素 | LiteLLM（自托管） | OpenRouter（托管） | Portkey（生产级） |
|--------|----------------------|----------------------|----------------------|
| 代码 | 开源，Python | 托管 SaaS | 开源（2026 年 3 月）+ 托管服务 |
| 部署 | 部署一个代理 | 注册即可 | 两种方式均可 |
| 提供商 | 100+ | 300+ | 100+ |
| 计费 | 使用自己的密钥 | 使用 OpenRouter 额度 | 使用自己的密钥 |
| 可观测性 | OpenTelemetry | 仪表盘 | 完整 OTel + PII 脱敏 |
| 最适合 | 有 SRE 团队且需要完全控制的团队 | 快速原型开发 | 有合规要求的生产环境 |

如果你拥有 SRE 团队并看重数据主权，LiteLLM 更合适。如果你想要单一订阅且不想维护基础设施，OpenRouter 更合适。如果你开箱即用就需要防护规则和合规能力，Portkey 更合适。

### 成本追踪

每次请求都会记录 `provider`、`model`、`input_tokens` 和 `output_tokens`。用各模型的每 token 价格（来自网关维护的价格表）乘以用量，再按用户、团队或项目汇总。

### MCP 与路由结合

网关既可以路由 LLM 调用，也可以路由 MCP sampling 请求。当 sampling 请求的 modelPreferences 偏好某个特定模型时，网关将请求转换并发送到正确后端。这也是阶段 13 · 17（MCP 网关）与本课路由网关有时会合并为同一个服务的原因。

### 路由策略

- **静态优先级。** 先尝试列表首项；出错后回退。
- **负载均衡。** 轮询或加权分配。
- **成本感知。** 选择满足延迟和质量要求的最便宜模型。
- **延迟感知。** 选择过去 N 分钟内最快的模型。
- **任务感知。** 使用 prompt 分类器，把编码任务路由到一种模型，把摘要任务路由到另一种模型。

```figure
tp-router-failover
```

## 使用它

`code/main.py` 用约 150 行代码实现了一个路由网关：接收 OpenAI 风格的请求，将其转换给各提供商 stub，运行按优先级排列的回退链，追踪每次请求的成本，并对输入执行 PII 脱敏。运行程序可以看到三个场景：正常请求、主提供商宕机后触发回退，以及由脱敏器捕获 PII 泄漏。

阅读代码时请重点观察：

- `ROUTES` 字典：别名 -> 按优先级排序的具体提供商列表。
- 回退循环会在 5xx 错误时重试。
- 成本追踪器用 token 用量乘以各模型费率。
- PII 脱敏器会在转发前清理形似 SSN 的模式。

## 交付它

本课产出 `outputs/skill-routing-config-designer.md`。给定一份工作负载特征（延迟、成本、合规），该技能会在 LiteLLM / OpenRouter / Portkey 中做出选择，并生成路由配置。

## 练习

1. 运行 `code/main.py`。触发宕机场景；确认请求回退到第二个提供商，并且成本被正确归属。

2. 添加语义缓存：以 prompt 的 SHA256 作为查询键；缓存命中时立即返回。测量重复调用节省的成本。

3. 添加 prompt 分类器，把“code ...”类 prompt 路由到偏重智能水平的别名，把“summarize ...”类 prompt 路由到偏重速度的别名。

4. 设计按团队分配的预算：每个团队都有月度支出上限；达到上限后网关拒绝请求。选择一种执行粒度（逐请求或按时间窗口）。

5. 对照阅读 LiteLLM、OpenRouter 和 Portkey 的文档。分别指出每个产品独有、另外两个没有提供的一项功能。

## 关键术语

| 术语 | 人们通常怎么说 | 它的实际含义 |
|------|----------------|------------------------|
| Routing gateway | “LLM 代理” | 位于多个提供商前方、提供统一 API 接口的一层 |
| OpenAI-compatible | “使用 OpenAI schema” | 接受 `/v1/chat/completions` 结构，并转换给任意后端 |
| Model alias | “our_smart_model” | 代码使用的名称，由网关映射到具体模型 |
| Fallback chain | “重试列表” | 失败时按顺序尝试的提供商列表 |
| Semantic caching | “prompt embedding 缓存” | 以 prompt 的 embedding 为键；近似请求共享缓存命中 |
| Guardrails | “输入/输出过滤器” | 脱敏 PII、拒绝策略违规内容 |
| Per-key rate limit | “团队预算” | 作用域限定到一个 API key 的配额 |
| Cost tracking | “单次请求支出” | 汇总 token 用量 × 对应模型单价 |
| LiteLLM | “开放代理” | 可自托管的开源路由网关 |
| OpenRouter | “托管 SaaS” | 使用额度计费的托管网关 |
| Portkey | “生产方案” | 开源 + 托管服务，内置防护规则 |

## 延伸阅读

- [LiteLLM — 文档](https://docs.litellm.ai/)——自托管路由网关
- [OpenRouter — 快速入门](https://openrouter.ai/docs/quickstart)——托管路由 SaaS
- [Portkey — 文档](https://portkey.ai/docs)——带防护规则的生产级路由
- [TrueFoundry — LiteLLM 与 OpenRouter 对比](https://www.truefoundry.com/blog/litellm-vs-openrouter)——选型指南
- [Relayplane — 2026 年 LLM 网关比较](https://relayplane.com/blog/llm-gateway-comparison-2026)——供应商调研
