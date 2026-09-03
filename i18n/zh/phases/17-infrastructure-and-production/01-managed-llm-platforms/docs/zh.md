# 托管 LLM 平台：Bedrock、Vertex AI、Azure OpenAI

> 三家超大云厂商，三种完全不同的策略。AWS Bedrock 是模型市场模式：Claude、Llama、Titan、Stability、Cohere 共用一套 API。Azure OpenAI 是与 OpenAI 的排他性合作，再叠加 Provisioned Throughput Units（PTUs）来购买专属容量。Vertex AI 则是 Gemini 优先，在长上下文和多模态方面的方案最完整。到 2026 年，Artificial Analysis 测得 Azure OpenAI 在 Llama 3.1 405B 同等级部署上的中位延迟约为 50 ms，Bedrock 约为 75 ms；差距背后的核心原因是 PTU，因为专属容量天然优于共享按需容量。真正该问的问题不是“谁最快”，而是“谁的模型目录和 FinOps 能力最适合我的产品”。这一课要教你把这些取舍写清楚，而不是凭感觉拍板。

**Type:** 学习
**Languages:** Python（标准库，玩具级成本与延迟比较器）
**Prerequisites:** 阶段 11（LLM 工程），阶段 13（工具与协议）
**Time:** 约 60 分钟

## 学习目标

- 说出三种平台策略（marketplace、exclusive、Gemini-first），并把它们分别映射到合适的产品场景。
- 解释 Azure OpenAI 中的 Provisioned Throughput Units（PTUs）到底买来了什么，以及为什么按需 Bedrock 在 405B 级别上通常会慢大约 25 ms。
- 画出三家平台的 FinOps 归因体系：Bedrock 的 Application Inference Profiles、Vertex 的 team-per-project、Azure 的 scopes + PTU reservations。
- 写下一条 “two-provider minimum” 政策，并解释为什么在 2026 年，单供应商锁定是一种昂贵错误。

## 问题

你已经为产品选好了 Claude 3.7 Sonnet。现在你要决定怎么托管它。你可以直接调用 Anthropic API，也可以走 AWS Bedrock，或者再加一层网关。直接 API 最简单；Bedrock 会带来 BAA、VPC endpoints、IAM 和 CloudWatch attribution；网关则带来故障切换、统一账单以及跨供应商限流。

更深一层的问题是模型目录。如果你的同一个产品同时需要 Claude、Llama 和 Gemini，那你不可能从一个单一入口把它们全买齐，除非那个入口同时就是 Bedrock、Vertex 和 Azure OpenAI。三家超大云并不是可互换的，它们分别押注在“谁拥有模型层”这件事上的答案不同。

这一课就是把这三种下注方式、延迟差异、FinOps 差异，以及供应商锁定风险，放在同一张图上讲清楚。

## 概念

### 三种策略

**AWS Bedrock**：市场模式。Claude（Anthropic）、Llama（Meta）、Titan（AWS 第一方）、Stability（图像）、Cohere（嵌入）、Mistral，以及图像和嵌入子目录，都走同一套 API、同一层 IAM、同一份 CloudWatch 导出。Bedrock 的下注是：客户真正想要的是可选性，而不是单一模型。

**Azure OpenAI**：排他合作模式。你能拿到 GPT-4 / 4o / 5 / o-series、DALL·E、Whisper，以及运行在 Azure 数据中心里的 OpenAI 模型微调能力。“Azure OpenAI Service” 目录里没有非 OpenAI 模型；那些模型归 Azure AI Foundry（另一条产品线）。Azure 的下注是：OpenAI 会持续站在前沿，而企业客户想要的是围绕这段关系的企业级控制。

**Vertex AI**：Gemini 优先，其余其次。Gemini 1.5 / 2.0 / 2.5 Flash 和 Pro，再加上 Model Garden（第三方）。Vertex 的下注是长上下文多模态，1M-token 的 Gemini 上下文窗口就是它最关键的差异点。

### 规模化部署下的延迟差异

Artificial Analysis 持续跑跨平台基准。在与 Llama 3.1 405B 等价的部署上（共享按需容量），Azure OpenAI 的中位首 token 延迟大约是 50 ms，Bedrock 大约是 75 ms。这个差距不代表 AWS “做得差”，而是容量模型不同。Azure 会卖 PTUs（Provisioned Throughput Units），也就是为你的租户保留 GPU 推理容量。Bedrock 也有对应能力（Provisioned Throughput），但它的价格通常从约 $21/hour 每单位起步，因此大多数客户仍然停留在共享按需层。

共享按需容量要和其他租户争抢流量；专属容量不用。如果你的产品 SLA 要求 TTFT < 100 ms at P99，那你要么买 Azure 的 PTUs，要么买 Bedrock Provisioned Throughput，要么接受共享容量带来的默认波动。

### Provisioned Throughput 的经济学

Azure PTUs：一块预留的推理算力。对于稳定可预测工作负载，最多可比按需便宜大约 70%。成本按小时固定，无论有没有流量都照样计费。通常在 40-60% 的持续利用率附近达到 break-even。

Bedrock Provisioned Throughput：根据模型和区域不同，大约在 $21-$50 每小时。经济学逻辑类似，通常在峰值利用率的一半左右打平。需要月度承诺。

Vertex 的 provisioned capacity 按 Gemini SKU 销售；价格随模型和区域变化，而且公开透明度相对低一些。

### FinOps 能力：真正的分水岭

**Bedrock 的 Application Inference Profiles** 是“市场模式”下最干净的原生归因能力。你可以给一个 profile 打上 `team`、`product`、`feature` 标签，把所有模型调用都从这个 profile 走过去，CloudWatch 就能直接按 profile 拆出成本，不用后处理。这个能力在 2025 年加入，到 2026 年仍然是三大云厂商里颗粒度最细、最原生的一种。

**Vertex** 的归因方式是 team-per-project，再加上 labels-everywhere。你通常把每个团队建成一个 GCP project，所有资源统一打标签，然后用 BigQuery Billing Export + DataStudio 做汇总。工作量更大，但 BigQuery 给你任意 SQL 的自由度。

**Azure** 则更依赖 subscription / resource-group scopes 和 tags，PTU reservations 被当成一类一等成本对象。问题在于，标签是从 resource group 继承的，而不是按请求打上的，所以如果你想做 per-request attribution，往往还需要 Application Insights custom metrics，或者一个能给请求盖 header 的网关。

总结下来：Bedrock 的原生归因最干净，Vertex 借助 BigQuery 最灵活，Azure 如果不自己补一层埋点，原生表面最不透明。

### 2026 年真正的风险是锁定

当某一个模型长期一家独大时，押一个超大云并不算糟糕。但 2026 年的前沿模型月月在变：这个季度也许是 Claude 3.7，下个季度可能是 Gemini 2.5，再下个季度又轮到 GPT-5。把自己锁在单一平台上，本质上就是把自己锁在前沿能力的三分之一里。

成熟团队现在采用的模式是：任何产品关键 LLM 路径，都至少有两家供应商。最常见的组合是 Bedrock + Azure OpenAI，一边接 Claude，一边接 GPT，中间加同一个网关做故障切换。成本提升通常很小，因为网关会把流量路由到最优位置；但一旦发生供应商故障，比如 Azure OpenAI 2025 年 1 月事故或 AWS us-east-1 故障，这种冗余的可用性收益就是决定性的。

### 数据驻留、BAA 与强监管行业

Bedrock：多数区域支持 BAA、VPC endpoints 和 guardrails，是很多 fintech 团队的默认选择。
Azure OpenAI：HIPAA、SOC 2、ISO 27001、EU data residency，常常是企业监管场景的默认选项。
Vertex：HIPAA、GDPR，以及按区域提供的数据驻留能力，继承 Google Cloud 的合规栈。

三家都能满足基础合规勾选项。真正的差异在于数据保留策略、日志怎么处理，以及 abuse monitoring 是否会读取你的流量（大多数默认开启，企业版通常提供 opt-out）。

### 你应该记住的数字

- Azure OpenAI 在 Llama 3.1 405B 等价部署上的中位 TTFT：~50 ms（使用 PTUs）。
- Bedrock 按需中位 TTFT：~75 ms。
- Bedrock Provisioned Throughput：$21-$50/hr 每单位。
- Azure PTU break-even：持续利用率约 40-60%。
- 高利用率场景下，PTU 相比按需最多可省约 70%。

```figure
i4-platform-lanes
```

## 用起来

`code/main.py` 会在一个合成工作负载上比较这三个平台，模拟按需与 PTU 的经济学、TTFT 波动，以及成本归因清晰度。运行它，你会看到 PTU 在什么情况下开始划算，也会看到“模型目录更丰富”在什么地方足以抵消 TTFT 的差距。

## 交付成果

这一课会产出 `outputs/skill-managed-platform-picker.md`。给它一个工作负载画像（需要哪些模型、TTFT SLA、多大日调用量、合规要求），它会给出主平台、兜底平台，以及一份 FinOps 埋点方案。

## 练习

1. 运行 `code/main.py`。对一个 70B 级别模型来说，在什么持续利用率下 Azure PTU 会优于按需模式？自己算出 break-even，再与宣传里的 40-60% 区间对照。
2. 你的产品同时需要 Claude 3.7 Sonnet 和 GPT-4o。设计一套 two-provider deployment：哪一个放在哪个 hyperscaler，前面接什么 gateway，故障切换策略如何写？
3. 一个受监管的医疗客户要求 BAAs、US-East data residency，以及 sub-100ms P99 TTFT。选一个平台，并用三个明确特性来论证。
4. 你发现这个月 Bedrock 账单涨了 4 倍，但流量没有变化。如果没有 Application Inference Profiles，你会怎么定位元凶？如果有 profiles，这件事会快到什么程度？
5. 去读 Azure OpenAI 和 Bedrock 的 pricing 页面。对于一个 100M-token/month 的 Claude 工作负载，哪种更便宜：直接 Anthropic API、Bedrock on-demand，还是 Bedrock Provisioned Throughput？

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------|----------|
| Bedrock | “AWS LLM 服务” | 覆盖 Claude、Llama、Titan、Mistral、Cohere 的模型市场 |
| Azure OpenAI | “Azure 的 ChatGPT” | 运行在 Azure 数据中心、带企业控制的 OpenAI 独家模型服务 |
| Vertex AI | “Google 的 LLM 平台” | Gemini 优先，并通过 Model Garden 接入第三方模型 |
| PTU | “专属容量” | Provisioned Throughput Unit，即按小时计费的预留推理 GPU 容量 |
| Bedrock Application Inference Profile | “Bedrock 打标签” | 带标签的按产品成本/用量 profile，原生接入 CloudWatch |
| Model Garden | “Vertex 模型目录” | Vertex AI 中承载第三方模型的独立区域，与 Gemini 分开 |
| Two-provider minimum | “LLM 冗余” | 所有关键 LLM 路径至少跑在两家超大云上的策略 |
| BAA | “HIPAA 文书” | Business Associate Agreement，处理 PHI 所需，三家都提供 |
| Abuse monitoring | “日志审查器” | 供应商侧对提示词与输出做的安全扫描，企业版通常可 opt-out |

## 延伸阅读

- [AWS Bedrock Pricing](https://aws.amazon.com/bedrock/pricing/) — 权威价格表与 Provisioned Throughput 定价。
- [Azure OpenAI Service Pricing](https://azure.microsoft.com/en-us/pricing/details/azure-openai/) — PTU 经济学与费率说明。
- [Vertex AI Generative AI Pricing](https://cloud.google.com/vertex-ai/generative-ai/pricing) — Gemini 分层与 Model Garden 附加费用。
- [Artificial Analysis LLM Leaderboard](https://artificialanalysis.ai/) — 跨供应商的持续延迟与吞吐基准。
- [The AI Journal — AWS Bedrock vs Azure OpenAI CTO Guide 2026](https://theaijournal.co/2026/03/aws-bedrock-vs-azure-openai/) — 面向企业的决策框架。
- [Finout — Bedrock、Vertex 与 Azure 的 FinOps 对比](https://www.finout.io/blog/bedrock-vs.-vertex-vs.-azure-cognitive-a-finops-comparison-for-ai-spend) — 并排比较三家的归因机制。
