# 多区域 LLM 服务与 KV 缓存局部性

> 对带缓存的 LLM 推理来说，轮询负载均衡不是次优，而是有害。请求如果没有落到持有对应前缀缓存的节点上，就要支付完整的 prefill 成本。在长提示下，P50 大约是 800 ms；而命中缓存时约为 80 ms。到 2026 年，生产环境的主流模式已经是缓存感知路由器，例如用 Rust 实现的 vLLM Router 和 llm-d router：它们消费 KV 缓存事件，并按 prefix-hash 匹配来路由。最新研究（GORGO）还把跨区域网络延迟显式纳入路由目标。商业化的“跨区域推理”产品（Bedrock cross-region inference、GKE multi-cluster gateways）把推理当成黑盒处理，解决的是可用性，不是 TTFT。JPMorgan 与 Mayo Clinic 在 2024 年 11 月完成 us-east-1 故障切换，恢复时间约 22 分钟。真正的 DR 现实是：32% 的 LLM 灾备失败，原因不是权重没备份，而是团队漏掉了 tokenizer 文件或量化配置。

**Type:** 学习
**Languages:** Python（标准库，玩具级前缀缓存感知路由模拟器）
**Prerequisites:** 第 17 阶段 · 04（vLLM 服务），第 17 阶段 · 06（SGLang RadixAttention）
**Time:** 约 60 分钟

## 学习目标

- 解释为什么轮询负载均衡会破坏缓存推理，并量化它对 TTFT 的惩罚。
- 画出一个缓存感知路由器：输入是什么（KV-cache events）、算法是什么（prefix-hash match）、在无命中时如何做 tie-breaker（GPU utilization）。
- 说出导致 32% LLM DR 失败的原因（缺失 tokenizer 文件或量化配置），并给出一个三项 DR 检查清单。
- 区分商业跨区域能力（Bedrock CRI、GKE Multi-Cluster Gateway）与 KV 感知路由。

## 问题

你的服务部署在 us-east-1、us-west-2 和 eu-west-1。前面挂了一个 ALB，采用 round-robin。结果生产环境里的前缀缓存命中率掉到 8%，TTFT P50 变成原来的三倍。vLLM 日志显示，每个请求都在支付完整的 prefill 成本。

轮询对于无状态服务是最优策略。LLM 推理则天生是有状态的，因为 KV 缓存编码了模型已经看过的全部上下文。盲目路由，本质上就是把请求送进错误的缓存。

与此同时，你的团队还准备了一个 DR 方案。模型权重已经跨区域备份到 S3。某个区域真的故障后，你尝试故障切换，副本却拒绝启动。原因是 tokenizer.json、量化配置以及 RoPE scaling 配置放在另一个没同步的 bucket 里。

多区域 LLM 服务首先是缓存问题、路由问题和 DR 卫生问题，而不是负载均衡器问题。

## 概念

### 缓存感知路由

请求带着一个 prompt 进入。路由器对前缀做哈希，比如取前 512 个 token；然后问每个副本：“这个前缀你缓存了吗？”各个副本会在 pub/sub 通道里发布 KV 缓存事件，报告自己何时分配和驱逐 block。路由器优先选中有匹配前缀的副本；如果没人命中，再退回到基于 GPU 利用率的 tie-breaker。

**vLLM Router**：2026 年生产栈里的 Rust 组件。它订阅 `kv.cache.block_added` 事件，维护一个 prefix-hash → replica 索引，并用 O(1) 查找完成路由。如果没有匹配，就退回到最小队列深度策略。

**llm-d router**：模式相同，但原生面向 Kubernetes。它通过 ControlPlane API 发布事件。

**SGLang RadixAttention**（Phase 17 · 06）可以看作单副本内部的等价机制。跨副本路由则严格发生在它的上游。

### 数字

在 2K-token prompt、Llama 3.3 70B FP8、H100 这一条件下：
- 缓存命中（同一副本，且前缀仍驻留）：约 80 ms。
- 缓存未命中（冷启动 prefill）：约 800 ms。

差距达到 10 倍。如果你的路由器能在多个副本之间把前缀缓存命中率打到 60% 到 80%，就能在 N 副本容量下逼近单副本性能；如果只能做到 10%，那基本就接近朴素扩容的效果。

### 跨区域多了一个新约束：网络延迟

区域间 RTT：
- us-east-1 ↔ us-west-2: 约 65 ms。
- us-east-1 ↔ eu-west-1: 约 75 ms。
- us-east-1 ↔ ap-southeast-1: 约 220 ms。

如果路由把一个来自 us-east-1 的请求发到 ap-southeast-1，只因为那里有一个热前缀，那么省下来的 prefill 时间（800 → 80 ms）很可能会被 440 ms 的往返网络延迟吃掉。GORGO（2026 年研究）明确把这一点写进目标函数里，即同时最小化 `prefill_time + network_latency`，而不只是最小化 prefill。很多时候，正确答案是默认保持区域内路由，只有在前缀极大、达到多 MB 级别、prefill 成本明显主导时，才值得跨区。

### 商业“跨区域推理”在这里帮不上忙

AWS Bedrock cross-region inference 会在容量紧张时把请求自动转发到其他区域。它优化的是可用性，而不是 TTFT，而且把推理当成黑盒。GKE Multi-Cluster Gateway 也是同一类能力：做服务级故障切换，不理解 KV 缓存。

即便你用了这些产品，仍然需要应用层的缓存感知路由器。前者处理的是“us-east-1 着火了怎么办”，后者处理的是“如何把 TTFT 压下来”。

### 灾备文件完整性：导致 32% 失败的缺文件问题

2026 年被广泛引用的一项统计是：32% 的 LLM DR 失败，原因是团队备份了权重，却漏掉了下面这些文件：

- `tokenizer.json` 或 `tokenizer.model`
- 量化配置（`quantize_config.json`、AWQ scales、GPTQ zero-points）
- 模型专属配置（RoPE scaling、attention masks、chat templates）
- 引擎配置（`vllm_config.yaml`、sampling defaults、LoRA adapter manifests）

修复方式是准备一个最低限度的三项 DR manifest：

1. HF model repo 下的全部文件，包括权重、配置和 tokenizer。
2. 与引擎绑定的 serving 配置。
3. 部署清单，包括 K8s YAML、Dockerfile 和依赖锁定文件。

此外，每个季度至少做一次 DR drill。JPMorgan 在 2024 年 11 月的 us-east-1 演练之所以能在 22 分钟内恢复，是因为他们真的排练过这套 playbook。

### 数据驻留是另一条约束轴

欧盟客户的 PHI 不能离开欧盟。如果你的缓存感知路由器为了命中前缀，把一个来自巴黎的请求送到 us-east-1，那么不管 TTFT 提升多少，你都已经违反了 GDPR。先按数据驻留边界切分路由域，再去优化缓存命中。

### 你需要记住的数字

- 缓存命中与未命中的 TTFT 差距：约 10 倍（2K prompt 下 80 ms 对 800 ms）。
- 美欧之间的区域 RTT：约 75 ms。
- DR 失败中有 32% 是因为漏掉 tokenizer 或量化配置。
- JPMorgan 在 2024 年 11 月完成 us-east-1 故障切换：22 分钟（SLA 为 30 分钟）。

```figure
cache-aware-router
```

## 用起来

`code/main.py` 会在一个多区域工作负载上模拟三种路由策略：round-robin、cache-aware regional、cache-aware global。它会输出缓存命中率、TTFT P50/P99，以及跨区域流量账单。

## 产出

这一课会产出 `outputs/skill-multi-region-router.md`。给定区域分布、数据驻留约束和 SLA，它会设计一份路由方案。

## 练习

1. 运行 `code/main.py`。在 RTT 为 75 ms 时，prompt 长度达到多少后，跨区域路由会优于仅本地路由？
2. 你的缓存命中率从 70% 掉到 12%。给出三种可能原因，以及各自对应的可观测证据。
3. 为一个在 vLLM 中部署、采用 70B AWQ 量化并带有 5 个 LoRA adapter 的模型设计 DR manifest。把所有文件和配置列完整。
4. 论证 Bedrock cross-region inference 对一个有严格 TTFT SLO 的金融科技公司来说是否“足够”。请引用具体行为。
5. 一个来自巴黎的请求，其前缀在 us-east-1 命中了。你会路由过去吗？把策略写出来。

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|----------|----------|
| Cache-aware routing | “智能负载均衡” | 基于 prefix-hash 命中把请求路由到持有 KV cache 的副本 |
| KV-cache events | “缓存发布/订阅” | 副本发布 block add/evict 事件，路由器据此建立索引 |
| Prefix hash | “缓存键” | 对前 N 个 token 做哈希，作为路由查找键 |
| GORGO | “跨区域路由研究” | arXiv 2602.11688；把网络延迟作为显式项纳入目标函数 |
| Cross-region inference | “Bedrock CRI” | AWS 产品；处理可用性故障切换，不感知 TTFT |
| DR manifest | “备份清单” | 恢复服务所需的全部文件清单，不只是权重 |
| Data residency | “GDPR 边界” | 哪些区域可以接触用户数据的法律边界 |
| RTT | “往返时间” | 网络往返时延；美欧约 75 ms，美亚太约 220 ms |
| LLM-aware LB | “缓存命中型负载均衡” | 以缓存感知路由器为代表的一类产品 |

## 延伸阅读

- [BentoML — 多云与跨区域推理](https://bentoml.com/llm/infrastructure-and-operations/multi-cloud-and-cross-region-inference)
- [arXiv — GORGO (2602.11688)](https://arxiv.org/html/2602.11688v1) — 讨论带网络延迟项的跨区域 KV-cache 复用。
- [TianPan — 多区域 LLM 服务的缓存局部性](https://tianpan.co/blog/2026-04-17-multi-region-llm-serving-data-residency-routing)
- [AWS Bedrock 跨区域推理](https://docs.aws.amazon.com/bedrock/latest/userguide/cross-region-inference.html) — 官方可用性故障切换文档。
- [vLLM 生产栈路由器](https://github.com/vllm-project/production-stack) — vLLM 生产栈路由器源码。
