# 生产级服务栈：KV 卸载与缓存感知路由

> 生产级服务栈会把 router、engine 和 observability 串成一个完整的 Kubernetes 部署，并把 KV cache 当成一种可以离开 GPU 的资源来看待。KV offloading 的核心，就是把 KV cache 从 GPU 显存里抽出来，在查询之间、甚至引擎之间复用，先落到 CPU DRAM，再视情况继续下沉到磁盘或 Ceph。vLLM 的 production-stack 是参考部署；LMCache 则是主要的 offloading 层。vLLM 0.11.0 的 KV Offloading Connector（2026 年 1 月）通过 Connector API（v0.9.0+）把这条链路做成了异步、可插拔的能力。多数情况下，卸载路径会被隐藏在请求主路径之外，但缓存未命中与缓存上移依然可能拉高端到端延迟。即便没有共享前缀，LMCache 依旧有价值：当 GPU 的 KV 槽位用尽，被抢占的请求可以直接从 CPU 恢复，而不必重做预填充。公开基准测试在 16x H100（80GB HBM）、分布于 4 台 a3-highgpu-4g 机器上显示：当 KV 缓存超过 HBM 容量时，native CPU offload 与 LMCache 都能明显提升吞吐；当 KV 占用较小时，所有配置都与基线接近，仅有少量额外开销。

**Type:** 学习
**Languages:** Python（标准库，玩具级 KV 溢写模拟器）
**Prerequisites:** 阶段 17 · 04（服务引擎内部原理），阶段 17 · 06（SGLang / RadixAttention）
**Time:** 约 60 分钟

## 学习目标

- 画出 vLLM production-stack 的层次结构：router、engine、KV offload、observability。
- 解释 KV Offloading Connector API（v0.9.0+）是什么，以及 0.11.0 的异步路径如何隐藏卸载延迟。
- 量化 LMCache 的 CPU DRAM 何时有帮助（KV > HBM），何时只会增加开销（KV 足够小，本来就放得下 HBM）。
- 在真实部署约束下，判断该选 native vLLM CPU offload，还是 LMCache connector。

## 问题

你的 vLLM 服务一到并发升高，GPU 的 HBM 就飙到 100%，同时不断出现抢占事件。请求被驱逐、重新排队，同一个 2K-token prompt 在一分钟内被重复预填充四次。GPU 计算资源被浪费在重复预填充上，goodput 明显低于原始 throughput。

继续加 GPU，成本会线性上升。增加 HBM 容量则几乎做不到。但 CPU DRAM 很便宜，一个 socket 往往就有 512 GB 以上，虽然延迟比 HBM 高几个数量级，但对于“暂时仍然热着”的 KV cache 来说，已经够用。

LMCache 的价值就在这里：它把 KV cache 提取到 CPU DRAM，让被抢占的请求可以更快恢复；同时，如果多个 engine 遇到重复前缀，也能共享缓存，而不是每个 engine 都重新预填充一遍。

## 核心概念

### vLLM production-stack

`github.com/vllm-project/production-stack` 是 vLLM 官方给出的参考 Kubernetes 部署：

- **Router**：具备 cache-aware 能力，对应 Phase 17 · 11，并消费 KV 事件。
- **Engines**：vLLM worker，通常按单 GPU 或 TP/PP group 部署。
- **KV cache offload**：可以是 LMCache deployment，也可以是 native connector。
- **Observability**：包括 Prometheus 抓取、Grafana dashboard、OTel trace。
- **Control plane**：负责 service discovery、配置管理和 rolling update。

这套东西通常以 Helm chart + operator 的形式交付。

### KV Offloading Connector API（v0.9.0+）

vLLM 0.9.0 引入了 Connector API，用来支持可插拔的 KV cache backend。engine 会把 KV block 交给 connector；connector 再负责把它们存到 RAM、disk、object storage，或者 LMCache 这类后端中。当请求再次需要某个 block 时，connector 再把它取回。

vLLM 0.11.0（2026 年 1 月）又增加了异步卸载路径。这样一来，在常见场景下，卸载可以在后台发生，engine 不必在主路径上同步等待。但端到端延迟和吞吐仍然取决于负载形态、KV cache hit rate 和系统压力。vLLM 自己的说明也特别提到：custom-kernel offload 在低 hit rate 场景下可能拉低吞吐，而 async scheduling 与 speculative decoding 之间也存在已知交互问题。

### 原生 CPU 卸载与 LMCache 的区别

**Native vLLM CPU offload**：以单 engine 为范围，把 KV block 存在本机 host RAM。实现快，没有网络跳数，但无法跨 engine 共享。

**LMCache connector**：面向集群级别，把 KV block 存在共享的 LMCache server 中，后端可以是 CPU DRAM，再叠加 Ceph/S3 等分层存储。任意 engine 都能访问这些 block。公开的 16x H100 benchmark 也是基于这一路径发布的。

如果只是单个 engine 遇到 HBM 压力，优先考虑 native。若多个 engine 之间本来就共享前缀，例如带公共 system prompt 的 RAG，或多租户共享模板的场景，LMCache 更合适。

### 基准表现

16x H100（80 GB HBM）、分布在 4 台 a3-highgpu-4g 上的测试，大致呈现以下规律：

- 低 KV footprint：提示词短、并发低时，所有配置都接近基线，LMCache 会多出约 3–5% 的开销。
- 中等 KV footprint：如果存在跨 engine 的前缀复用，LMCache 开始体现价值。
- KV 超过 HBM：native CPU offload 与 LMCache 都会显著提升吞吐；LMCache 的提升往往更大，因为它还能利用跨 engine 共享。

### LMCache 真正关键的场景

- 多租户 serving，并且 system prompt 在多个租户之间重复。
- RAG 场景中，文档 chunk 会在查询之间重复出现。
- 在同一个 base model 上挂多个 LoRA 变体时，base-model KV 的复用可以减少重复工作。
- 抢占很重的工作负载：从 CPU 恢复要比重做预填充便宜得多。

### 什么情况下不要启用

- HBM 压力并不高：这时只会引入额外开销，没有对应收益。
- 上下文很短（<1K tokens）：传输时间可能比重新预填充还长。
- 单租户、单 prompt 型工作负载：几乎没有可捕获的复用。

### 与解耦式 serving 的联动

Phase 17 · 17 的 disaggregated serving 和 LMCache 是叠加关系：KV 从 prefill pool 传到 decode pool 后，如果没有立即用完，可以落进 LMCache；后续查询再从 LMCache 取回。Phase 17 · 11 的 cache-aware router 也可以进一步把请求路由到“本地 cache 或 LMCache 共享 cache 最匹配”的 engine 上。

### 需要记住的数字

- vLLM 0.9.0：Connector API 正式提供。
- vLLM 0.11.0（2026 年 1 月）：增加异步 offload 路径；但端到端延迟影响仍取决于工作负载、KV hit rate 和系统压力，并不是“绝对免费”。
- 16x H100 benchmark：当 KV footprint 超过 HBM 时，LMCache 会明显有帮助。
- HBM 压力较小时：通常会多出 3–5% 开销，但没有明显收益。

```figure
zero-sharding
```

## 实际使用

`code/main.py` 会模拟一个抢占很重的工作负载，对比开启与不开启 LMCache 的差别，并报告避免了多少次重复预填充、吞吐提升，以及 HBM 利用率的 break-even 点。

## 交付成果

本课产出 `outputs/skill-vllm-stack-decider.md`。它会根据工作负载形态与 vLLM 部署方式，判断该选 native、LMCache，还是两者都不启用。

## 练习

1. 运行 `code/main.py`。在什么 HBM 利用率之后，LMCache 开始划算？
2. 某租户共享一个 6K-token system prompt，每小时发起 200 次查询。估算该租户在 LMCache 下的收益。
3. LMCache server 是单点故障。请设计一套 HA 策略，包括 replicas 与退回 native 的方案。
4. 如果 LMCache 把数据落到 Ceph 的 spinning disk 上，对于 70B FP8 模型的 4K-token KV（约 500 MB），读取时间相对 re-prefill 是什么量级？
5. 论证 vLLM 0.11.0 的异步路径是否真的“免费”。它把开销藏在了哪里？

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------------|------------------------|
| Production-stack | “参考部署” | vLLM 的 Kubernetes Helm chart 加 operator |
| Connector API | “KV 后端接口” | vLLM 0.9.0+ 的可插拔 KV store 接口 |
| Native CPU offload | “引擎本地溢写” | 把 KV 存到同一 engine 所在主机的 RAM |
| LMCache | “集群级 KV 缓存” | 运行在 CPU DRAM 加磁盘之上的跨 engine KV cache 服务 |
| 0.11.0 async | “非阻塞卸载” | 将 offload 隐藏在 engine stream 背后的异步路径 |
| Preemption | “驱逐腾位置” | HBM 满时对 KV cache 进行驱逐与腾挪 |
| Prefix reuse | “相同 system prompt” | 多个查询共享开头部分，因此能够命中缓存 |
| Ceph tier | “磁盘层” | 位于 DRAM 之下的持久化缓存层 |

## 延伸阅读

- [vLLM Blog — KV Offloading Connector（2026 年 1 月）](https://blog.vllm.ai/2026/01/08/kv-offloading-connector.html)
- [vLLM Production Stack GitHub](https://github.com/vllm-project/production-stack) — Helm chart 与 operator
- [面向企业规模的 LMCache LLM 推理（arXiv:2510.09665）](https://arxiv.org/html/2510.09665v2)
- [LMCache GitHub](https://github.com/LMCache/LMCache) — Connector 实现
- [vLLM 0.11.0 release notes](https://github.com/vllm-project/vllm/releases) — 异步路径细节
