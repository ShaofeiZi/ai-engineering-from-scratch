# 作为降本原语的模型路由

> 一个动态 broker 会评估每个请求的特征，例如任务类型、token 长度、embedding 相似度和置信度，然后把简单查询发给便宜模型，把复杂查询升级到 frontier model。这也常被称为 model cascading。生产案例表明，在美国、英国和欧盟部署中，只要质量保持不变，路由通常可以带来 20-60% 的成本下降；而对高流量 SaaS 来说，30% 的 routing efficiency 改进往往就意味着每年六位数的节省。到 2026 年，LLM inference 价格背景也发生了根本变化：从 2022 年末到 2026 年，GPT-4 级别 token 的价格大约从 $20/M 降到 ~$0.40/M，年均约 10 倍下降。这里面大部分降幅来自更好的 serving stack（Phase 17 · 04-09），而不是硬件本身。Routing 的价值在于：它让你能在不牺牲产品体验的前提下，把这波价格下跌真正转成利润。最典型的失败模式是 cheap-model drift：router 把 40% 流量推给弱模型，结果推理任务质量悄悄掉了 3-5%，一个季度都没人察觉。真正要守住路线，靠的是 online quality metrics，而不是离线 eval set。

**Type:** 学习
**Languages:** Python（标准库，玩具级级联路由模拟器）
**Prerequisites:** 阶段 17 · 01（托管 LLM 平台）、阶段 17 · 19（AI 网关）
**Time:** 约 60 分钟

## 学习目标

- 解释 model cascading：先走便宜模型，再根据低置信度决定是否升级。
- 列出四类 routing signals：task classification、prompt length、embedding similarity 到已知 hard set、cheap model 的 first-pass self-confidence。
- 在给定 routing split 与质量损失容忍度时，计算 blended cost。
- 说出能发现 cheap-model creep 的 drift 监控指标：online quality gate。

## 问题

你的服务目前在 GPT-5 上每月花 $80k。分析数据后发现，70% 的查询其实非常简单，例如“巴黎现在几点？”或“换一种说法改写这句话。”这类请求用 Haiku 级别模型就能几乎无损处理，而成本只有 GPT-5 的 3%。剩下 30% 才真正需要 GPT-5 的推理能力，例如代码生成、数学和多步规划。

如果把 70% 路由到便宜模型、30% 路由到贵模型，那么账单在产品质量不变的前提下大约可以下降 65%。这就是 routing。难点不在“算省了多少钱”，而在于怎么把 broker 搭出来，同时不造成质量回退。

## 概念

### 四类路由信号

1. **Task classification**：把请求分成 simple / complex / codegen / math / chat。可以用规则分类器、一个小模型（例如 Haiku 级别，$0.25/M），或者基于 embedding 的已标注桶匹配。输出通常是：cheap / balanced / frontier。
2. **Prompt length**：长度超过 4K tokens 的 prompt，通常更容易需要 frontier model 来维持一致性；短于 500 tokens 的 prompt 往往不需要。
3. **Embedding similarity to known-hard set**：如果某个查询和已知困难桶的 embedding 距离足够近，例如 cosine > 0.88，就可以直接升级到 frontier。
4. **Self-confidence from first-pass**：先送给便宜模型，如果它的 logprobs 显示低置信度、直接 refuse，或者输出里充满 hedging 语言，就在 frontier model 上重试。这样会给大约 10% 的流量增加 P95 latency，但能为剩余 90% 节省 50% 以上成本。

### 三种典型模式

**Pre-route**：先跑分类器，再选模型。通常只额外增加 ~5-10ms 延迟，因此整体速度最快。

**Cascade**：先跑便宜模型，再根据低置信度升级。中位延迟大约会增加到 ~1.2x，而被升级的那部分请求可能接近 ~2x。它的优势是质量下限最好。

**Ensemble route**：便宜模型和 frontier model 并行跑，再让 reward-model 从中选一个。质量最高、成本也最高，一般只在关键 A/B 中使用。

### 落地实现

AI gateways（Phase 17 · 19）通常直接暴露 routing 能力。LiteLLM 有 `router` 配置，支持 fallback 和 cost-routing；Portkey 提供 guards + routing；Kong AI Gateway 通过插件做 routing；OpenRouter 的模型市场则暴露 recommendation API。

开源生态方面，比较常见的是 RouteLLM（LMSYS）、Not Diamond（商业）和 Prompt Mule。

### 2026 年的价格曲线

| 模型级别 | 2022 年末 | 2026 年 | 变化 |
|-------------|-----------|------|--------|
| GPT-4 级质量 | ~$20/M | ~$0.40/M | 便宜 50 倍 |
| 前沿模型（GPT-5、Claude 4） | — | ~$3-10/M | 新层级 |

这波成本下降的大部分来源，其实就是 serving efficiency 的提升，也就是 Phase 17 · 04-09 所讲的那一整套基础设施优化。Routing 让你能在应用层立刻吃到这些收益，而不是等所有用户自己迁移去廉价层。

### 真正的风险在 drift

一开始 router 把 40% 流量送到 cheap model。六个月后，用户的问题逐渐变长、变复杂，但 router 还在按 Q1 的分类器工作，于是质量悄悄下降。没有人立刻投诉，直到你在一次竞品 benchmark 里发现自己输了。

因此要用 online quality metrics 去守 gate：

- 每条 route 的用户 thumbs-up / thumbs-down
- 每条 route 抽样 5% 做 automated LLM-judge
- escalation rate：如果 cascade 的 uproute 比例超过 30%，说明 cheap model 被过度分流
- 每条 route 的 refusal rate

### 你应该记住的数字

- 2026 年 routing savings：20-60% 的案例研究范围。
- 2022-2026 的 LLM 价格下降：整体上约每年 10x。
- GPT-4 级别价格：从 ~$20/M 降到 ~$0.40/M。
- cascade 的延迟影响：中位数大约 ~1.2x，被升级流量约 ~2x（通常占 10%）。

```figure
model-cascade-router
```

## 用起来

`code/main.py` 会模拟 pre-route、cascade 和 ensemble 三种策略在混合工作负载下的表现，并报告 blended cost、quality loss 与 escalation rate。

## 交付物

本课产出 `outputs/skill-router-plan.md`。它会根据工作负载与质量预算，为你选择 routing pattern 以及具体 signals。

## 练习

1. 运行 `code/main.py`。在哪个 accuracy floor 上，cascade 会开始优于 pre-route？
2. 假设你的用户群有 30% enterprise（复杂查询）和 70% free tier（简单查询），你会如何设计 routing split？用什么 online metric 做 gate？
3. 某条 route 让质量下降 2%，但节省了 40% 成本。这能不能 ship？取决于产品类型，请分别为两边论证。
4. 用 OpenAI / Anthropic API 的 logprobs 实现一个 confidence check。你会从什么 threshold 开始？
5. 六个月内 escalation rate 从 8% 涨到 22%。请给出三个可能原因，以及每个原因的修复方案。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------------|------------------------|
| Model routing | “成本代理” | 针对每个请求动态选择模型 |
| Model cascade | “便宜模型优先、按需升级” | 先跑便宜模型，低置信度时再升级到 frontier |
| Pre-route | “先分类” | 先分类再路由，不会重跑 |
| Ensemble route | “并行择优” | 多个模型并行运行，再由 reward-model 选择最好结果 |
| Escalation rate | “升级比例” | cascade 请求中被升级的比例 |
| RouteLLM | “LMSYS 路由器” | 开源 routing library |
| Not Diamond | “商业路由器” | SaaS 形式的 model-routing 产品 |
| Drift | “廉价模型侵蚀” | 分布已经变化，但 router 没察觉 |
| 在线质量闸门 | “线上检查” | 对线上流量做 automated LLM-judge 抽样 |

## 延伸阅读

- [AbhyashSuchi — 2026 年 LLM 模型路由最佳实践](https://abhyashsuchi.in/model-routing-llm-2026-best-practices/)
- [Lukas Brunner — 2026 推理优化的崛起](https://dev.to/lukas_brunner/the-rise-of-inference-optimization-the-real-llm-infra-trend-shaping-2026-4e4o)
- [RouteLLM paper / code](https://github.com/lm-sys/RouteLLM)
- [Not Diamond — 模型路由](https://www.notdiamond.ai/)
- [OpenRouter](https://openrouter.ai/) — 多模型网关，带 routing primitives
