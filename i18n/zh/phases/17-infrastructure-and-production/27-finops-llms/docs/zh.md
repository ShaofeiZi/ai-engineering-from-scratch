# LLM 的 FinOps：单位经济与多租户归因

> 传统 FinOps 放到 LLM 支出上会失效。成本不是资源运行时长，而是 token transaction。标签也不会天然映射，因为一次 API call 是交易，不是资产。工程决策，例如 prompt 设计、context window、输出长度，本质上也是财务决策。2026 年的实践里，有三个必须从第一天就埋点的归因维度：按用户（`user_id`），用于 seat pricing 与客户增购；按任务（`task_id` + `route`），用于衡量各产品功能的成本与确定功能优先级；按租户（`tenant_id`），用于单位经济和续约决策。Token 至少要拆成四层：prompt、tool、memory、response。全部塞进一个桶里，你就看不见钱花在哪。对多租户产品，常见的 enforcement ladder 是：按租户 rate limit（设为预期峰值的 2-3 倍，并返回清晰的 429 + retry-after）；每日 spend cap（合同上限的 1.5-3 倍，触发限流收紧与告警）；当 spend z-score > 4 时触发 kill switch（自动暂停并 page on-call）。常见归因模式包括：tag-and-aggregate、telemetry-joiner（trace-ID → billing，精度最高）、sampling-and-extrapolation、model-based allocation、event-sourced、real-time streaming。单位指标应该是每个 resolved query 或每个生成 artifact 的成本，而不是 $/M tokens。事后补标签永远会漏，必须在 request 创建时就埋点。

**Type:** 学习
**Languages:** Python（标准库，带 kill switch 的玩具级成本归因模拟器）
**Prerequisites:** 阶段 17 · 13（可观测性）、阶段 17 · 14（缓存）
**Time:** 约 60 分钟

## 学习目标

- 解释为什么传统 FinOps（标签 + 层级）在 LLM 支出场景下会失效，并说出三个新的归因维度。
- 列出四层 token 成本（prompt、tool、memory、response），并解释为什么单桶计费会遮蔽成本结构。
- 为多租户产品设计一套 enforcement ladder（rate → spend cap → kill switch）。
- 选择 cost per resolved query / artifact 这样的单位指标，而不是只看 $/M tokens。

## 问题

你的账单写着 $40,000，但你不知道：

- 哪个租户花掉了这笔钱。
- 哪个产品功能把成本推高了。
- 是否有某个单独用户在滥用系统。
- 真正的问题出在 prompt 膨胀、tool 调用，还是 memory 放大。

在 provider 侧做 tag-and-aggregate，对云资源（例如 EC2、S3）是有效的，因为标签会传递到 line item。可 LLM API 调用不会自动带标签，你必须在调用点把 user/task/tenant 打上去并一路传下去。事后再做归因，边角情况一定会漏。

## 概念

### 三个归因维度

**按用户**（`user_id`）：谁在花钱。它驱动 seat pricing、客户增购沟通，也能识别 power users。

**按任务**（`task_id` + `route`）：哪个产品功能在花钱。它驱动功能优先级，以及是否需要砍掉昂贵功能。

**按租户**（`tenant_id`）：哪个客户真正赚钱。它驱动单位经济、续约报价与层级阈值。

这三个维度都要在调用点从第一天开始埋。事后补做只会更差。

### 四层 token

| 层级 | 示例 | 常见占比 |
|-------|---------|---------------------|
| 提示词 | 系统提示词 + 用户输入 | 40-60% |
| 工具 | 回填的工具调用结果 | 20-40%（智能体工作负载） |
| 记忆 | 先前对话 / 检索到的文档 | 10-30% |
| 响应 | 模型输出 | 10-30% |

把这四层全部并到一个桶里，你就无法做有效优化。归因 schema 里必须把它们拆开。

### 分级管控措施

1. **Rate limit**，按租户限流。阈值设在预期峰值的 2-3 倍。返回 429 和 `Retry-After`。租户感受到的是摩擦，而不是一张意外账单。

2. **Daily spend cap**，按租户设每日支出上限。通常是合同上限的 1.5-3 倍。触发后收紧 rate limit，并通知 customer-success。

3. **Kill switch**，当租户支出相对自身基线的 z-score > 4 时触发。自动暂停该租户，page on-call，并升级给 ops 与 CS。

### 归因模式

- **Tag-and-aggregate**：请求上打元数据，后续聚合。最简单，但比较粗糙。
- **Telemetry joiner**：通过 trace ID 把 trace 和账单关联起来。精度最高，成熟团队通常用这个。
- **Sampling + extrapolation**：采样 5-10%，再做外推。适合粗略估算，但会漏掉长尾。
- **Model-based allocation**：用回归或其他模型推断成本驱动因素。适合没有标签的遗留数据。
- **Event-sourced**：把成本作为流里的事件（Kafka / Kinesis）。适合实时场景。
- **Real-time streaming**：亚秒级更新 dashboard。

### 每个 X 的成本才是单位指标

$/M tokens 是 vendor 视角。产品真正关心的是：

- 每个已解决 support ticket 的成本。
- 每篇生成文章的成本。
- 每个成功 agent task 的成本。
- 每个用户会话分钟的成本。

必须把成本绑到产品结果上，否则优化就没有锚点。

### 成本归因 trace 形状

```
trace_id: abc123
  user_id: u_42
  tenant_id: t_7
  task_id: task_classify_doc
  route: model_haiku
  layers:
    prompt_tokens: 1800
    tool_tokens: 600
    memory_tokens: 400
    response_tokens: 150
  cost_usd: 0.0135
  cached_input: true
  batch: false
```

每次调用都要发出这一类事件，存进 data lake，再按维度聚合。Phase 17 · 13 的 observability stack 就是它落地的地方。

### 复合节省栈

栈的组成是：cache + batch + route + gateway。四者都启用时：

- Cache L2（Phase 17 · 14）：输入成本大约可降到原来的 1/10。
- Batch（Phase 17 · 15）：打 5 折。
- Route 到便宜模型（Phase 17 · 16）：成本再降约 60%。
- Gateway efficiency（Phase 17 · 19）：提供冗余与重试能力。

理想情况下，叠加后总成本可能只剩 naive baseline 的约 5-10%。大多数团队只用到 2-3 个杠杆，四个都叠满的很少。

### 你应该记住的数字

- 归因维度：按用户、按任务、按租户。
- 四层 token：prompt、tool、memory、response。
- Kill switch：spend z-score > 4。
- 单位指标：cost per resolved query，而不是 $/M tokens。
- 叠加优化后：理论上可降到 baseline 的约 5-10%。

```figure
i4-spend-ladder
```

## 用起来

`code/main.py` 会模拟一个多租户 LLM 服务，并实现三层 enforcement ladder。它会注入一个滥用型租户，演示 kill switch 被触发的过程。

## 交付物

这一课会产出 `outputs/skill-finops-plan.md`。它会根据产品形态和规模，设计归因 schema 与 enforcement ladder。

## 练习

1. 运行 `code/main.py`。kill switch 在什么 z-score 上触发？这个阈值应该如何选择？
2. 设计一个按租户、按任务的成本 dashboard。你最先会做哪 5 个视图？
3. 你最大的租户已经是单位经济负值。请按客户影响从低到高，提出三个干预方案。
4. 为一个客服产品计算每张已解决工单的成本：3M tokens/ticket、约 800 tickets/day、GPT-5 cached rate。
5. 论证事后补标签是否真的可行。什么情况下它还能被接受？

## 关键术语

| 术语 | 人们怎么说 | 它实际意味着什么 |
|------|----------------|------------------------|
| 按用户归因 | “用户级成本” | 每次调用都要带上 `user_id` |
| 按任务归因 | “功能成本” | 用 `task_id` + `route` 标识产品功能 |
| 按租户归因 | “客户成本” | 用 `tenant_id` 驱动单位经济分析 |
| 四层 token | “成本层” | prompt + tool + memory + response |
| Rate limit | “429 护栏” | 在 gateway 上按租户强制上限 |
| Daily spend cap | “每日天花板” | 以租户为范围的预算与告警 |
| Kill switch | “自动暂停” | spend z-score > 4 时自动停用 |
| 每次解决成本 | “产品单位指标” | 成本必须绑定到产品结果，而不是 token 数 |
| Telemetry joiner | “trace 对账单” | 精度最高的归因模式 |
| Stacked optimization | “cache+batch+route+gateway” | 组合后可把成本压到 baseline 的约 5-10% |

## 延伸阅读

- [FinOps Foundation — AI FinOps 概览](https://www.finops.org/wg/finops-for-ai-overview/)
- [FinOps School — 2026 单位成本指南](https://finopsschool.com/blog/cost-per-unit/)
- [Digital Applied — 2026 LLM 智能体成本归因](https://www.digitalapplied.com/blog/llm-agent-cost-attribution-guide-production-2026)
- [PointFive — Managed LLMs in Azure OpenAI](https://www.pointfive.co/blog/finops-for-ai-economics-of-managed-llms-in-azure-open-ai)
