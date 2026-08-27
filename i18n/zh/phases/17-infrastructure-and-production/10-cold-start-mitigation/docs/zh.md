# 无服务器 LLM 的冷启动缓解

> 一个 20 GB 的模型镜像，从冷状态到可服务通常需要 5-10 分钟（7B），到 20 分钟以上（70B）。在真正的 serverless 世界里，这不是热身，而是故障。缓解手段横跨五层：预置节点镜像（AWS 上的 Bottlerocket、双卷架构）、模型流式加载（NVIDIA Run:ai Model Streamer，vLLM 已原生支持）、GPU 内存快照（Modal checkpoints，可带来最高 10x 的更快重启）、warm pools（`min_workers=1`）、分层加载（ServerlessLLM 的 NVMe→DRAM→HBM 管道，可把延迟降 10-200 倍），以及 live migration，把输入 token（KB）而不是 KV cache（GB）迁走。Modal 公布的冷启动下限是 2-4 秒；Baseten 默认是 5-10 秒，预热后可到亚秒级。本课会教你如何测量、预算并组合这五层手段。

**Type:** 学习
**Languages:** Python（标准库，玩具级冷启动路径模拟器）
**Prerequisites:** 第 17 阶段 · 02（推理平台经济学），第 17 阶段 · 03（GPU 自动扩缩容）
**Time:** 约 60 分钟

## 学习目标

- 列出冷启动缓解的五层，并为每一层说出一种工具或模式。
- 对一个 70B 模型，计算总冷启动时间，即（节点供给）+（权重下载）+（权重载入 HBM）+（引擎初始化）。
- 解释为什么 live migration 迁的是输入 token（KB）而不是 KV cache（GB），以及这样做的代价是什么（recomputation）。
- 说出 warm-pool 的权衡点（为闲置 GPU 付费，或接受冷启动尾部延迟），以及在哪个 SLA 阈值下 `min_workers > 0` 会成为必选项。

## 问题

你的 serverless LLM endpoint 夜里缩到零。早上 8 点流量冲上来。第一条请求会卡在以下几个阶段：

1. Karpenter provision 一台 GPU 节点：45-60s。
2. 容器拉一个带权重的 30 GB 镜像：120-300s。
3. 引擎把权重装入 HBM：45-120s，取决于模型大小与存储速度。
4. vLLM 或 TRT-LLM 初始化 CUDA graphs、KV cache pool、tokenizer：10-30s。

总计：220-510s，也就是大约 3-8 分钟，用户才看到第一个 token。你的 SLA 却是 2s。于是你上了一个 warm pool（`min_workers=1`），问题似乎消失了，但代价是你要为一张闲置 GPU 24x7 付费。如果你的服务有 5 个产品，每个产品都保一个热副本，那就是 5 × 24 × 30 = 3,600 GPU-hours/月，无论是否真的有一个用户调用。

冷启动缓解的本质，就是在尽量保住 serverless 经济性的同时，把延迟逼近 always-on 服务。

## 概念

### 第 1 层：预置节点镜像（Bottlerocket）

在 AWS 上，Bottlerocket 的双卷架构把 OS 与数据分开。你可以把已经预拉取好容器镜像的数据卷做成 snapshot，再把这个 snapshot ID 填进 `EC2NodeClass`。这样一来，新节点启动时，本地 NVMe 上就已经有权重了，于是第 2 步和第 3 步的一部分会直接消失。它与 Karpenter 可以原生配合。典型收益是：对大模型单次冷启动节省 2-4 分钟。

在 GCP 上，对应模式是预烘焙容器层的自定义 VM 镜像。在 Azure 上，则是同样思路的 managed disk snapshot。

### 第 2 层：模型流式加载（Run:ai Model Streamer）

不要等完整文件加载完再响应第一个请求，而是把权重一层层流入 GPU 内存，只要第一个 transformer block 到位就开始处理。NVIDIA Run:ai Model Streamer 在 2026 年已被 vLLM 原生支持。它可配合 S3、GCS 与本地 NVMe 使用。对于大模型，它通过把 I/O 与计算初始化重叠，通常能把权重加载时间削掉约一半。

### 第 3 层：GPU 内存快照（Modal）

Modal 会在首次加载完成后，把 GPU 状态（权重、CUDA graphs、KV cache 区域）做成 checkpoint。后续重启时可以直接反序列化回 HBM，比重新初始化快约 10 倍。这基本就是“在 2 秒内启动一张热 GPU”最接近现实的做法。代价是：snapshot 会绑定 GPU 拓扑，如果 Karpenter 把你迁到另一种 SKU，就得重新做 checkpoint。

### 第 4 层：warm pool（min_workers=1）

最直接的缓解方法：始终保留一个 ready replica。成本就是一张 GPU 的小时单价乘以 24x7。对小模型来说，这笔账很残酷（每小时付 $0.85-$1.50，只是为了避免 30s 冷启动）；对大模型来说则更划算一些（每小时付 $4，可避免 5 分钟冷启动）。warm pool 变成必选项的 SLA 阈值，通常是 70B+ 模型上要求 TTFT P99 < 60s。

### 第 5 层：分层加载（ServerlessLLM）

ServerlessLLM 把存储看作分层体系：NVMe（速度快、容量大）、DRAM（速度与容量居中）、HBM（容量小但最快）。权重预先装到 DRAM，再按需载入 HBM。论文报告称，相比朴素的 disk-to-HBM 冷加载，这一设计能把延迟降 10-200 倍。生产落地还在早期，但已存在与 vLLM 的集成。

### 第 6 层：实时迁移（额外模式）

当一个节点将不可用时（spot eviction、node drain），传统做法是冷启动另一个副本，然后清空请求队列。live migration 的做法是，把输入 token（KB 级）迁到另一个已经加载好模型的目的地，并在目的地重新计算 KV cache。因为跨网络传 GB 级 KV cache 太贵，recomputation 反而更便宜。这种模式适合 disaggregated deployment。

### warm-pool 的数学问题

对于一个 P99 TTFT SLA 为 2s 的服务，你真正的问题不是“要不要 warm pool”，而是“要配多少个 warm replicas，以及哪些路径必须配”。

- 高价值交互路径（实时聊天、语音 agent）：`min_workers=1-2`。
- 后台批处理路径（夜间分类任务）：可以接受 scale-to-zero，5-10 分钟冷启动也能容忍。
- 高级付费层：按租户设置 `min_workers`，提供专属容量。

### 优化前先测量

一个 70B 模型在全新节点上的冷启动解剖图如下（示意值）：

| 阶段 | 时间 | 缓解手段 |
|------|------|----------|
| 节点供给 | 50s | Bottlerocket + pre-seeded image，warm pool |
| 拉取镜像 | 180s | 预置数据卷（消除） |
| 权重载入 HBM | 75s | Model streamer（减半）；GPU snapshot（消除） |
| 引擎初始化 | 20s | 持久化 CUDA graph cache |
| 首次前向 | 3s | 固有最小延迟 |
| **冷启动总计** | **328s** | |
| **加缓解后的总计** | **~15s** | 22x 降幅 |

### 你该记住的数字

- Modal cold start: 2-4s（带 GPU snapshots）。
- Baseten default cold start: 5-10s；预热后可到亚秒级。
- Raw 70B cold start: 3-8 分钟。
- Run:ai Model Streamer: ~2x weight-load speedup。
- ServerlessLLM tiered loading: 10-200x latency reduction（论文数字）。

```figure
cold-start-pipeline
```

## 动手用

`code/main.py` 会模拟带或不带各类缓解手段的冷启动路径，输出总冷启动时间、warm-pool 成本，以及请求率达到多高后，warm pool 会开始回本。

## 交付物

本课产出 `outputs/skill-cold-start-planner.md`。给定 SLA、模型大小与流量形态，它会帮你选出该叠哪些缓解手段。

## 练习

1. 运行 `code/main.py`。计算一个 warm replica 在什么请求率以上，会比因为冷启动导致额外请求跌出 SLO 更便宜。
2. 你要部署一个 13B 模型，P99 TTFT SLA 为 3s。选出达标所需的最小缓解组合，也就是层数最少的 stack。
3. Bottlerocket 预置会消除 image pull，但权重仍需从 snapshot 载入 HBM。若 snapshot-backed NVMe 读取速度为 7 GB/s，计算一个 70B 模型的 wall-clock 时间。
4. 你的 serverless provider 提供 GPU snapshots，但团队拒绝使用，因为“snapshots 会泄漏 PII”。分别站在支持和反对方论证：真实风险是什么？缓解手段又是什么（ephemeral snapshots、encryption、namespace isolation）？
5. 设计一个分层 warm-pool 策略：付费用户、试用用户、批处理 workload 分别保留多少 warm replicas？把账算出来。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------|----------|
| Cold start | “大停顿” | 新副本从收到请求到返回第一个 token 的时间 |
| Warm pool | “始终在线的最小值” | `min_workers >= 1`，保证至少一个副本始终 ready |
| Pre-seeded image | “烘焙好的 AMI” | 容器权重已预驻留的节点镜像 |
| Bottlerocket | “AWS 节点操作系统” | AWS 的容器优化 OS，支持双卷 snapshot |
| Model streamer | “流式加载” | 把权重 I/O 与计算初始化重叠 |
| GPU snapshot | “写进 HBM 的 checkpoint” | 序列化加载后的 GPU 状态；重启时再反序列化 |
| Tiered loading | “NVMe + DRAM + HBM” | 存储分层体系；按需加载 |
| Live migration | “迁 token” | 迁输入（KB），在目标端重算 KV |
| `min_workers` | “热副本数” | serverless 保活的最小副本数 |
| Scale-to-zero | “完全 serverless” | 空闲时零成本；接受完整冷启动税 |

## 延伸阅读

- [Modal — 冷启动性能](https://modal.com/docs/guide/cold-start) — Modal 公布的 benchmark 与 checkpoint 架构。
- [AWS Bottlerocket](https://github.com/bottlerocket-os/bottlerocket) — 预置数据卷 snapshot 模式。
- [NVIDIA Run:ai 模型流式加载器](https://github.com/run-ai/runai-model-streamer) — 用计算初始化重叠权重加载。
- [Baseten — 冷启动缓解](https://www.baseten.co/blog/cold-start-mitigation/) — 预热实践手册。
- [ServerlessLLM paper (USENIX OSDI'24)](https://www.usenix.org/conference/osdi24/presentation/fu) — 分层加载设计。
- [NVIDIA — Disaggregated LLM Inference on Kubernetes](https://developer.nvidia.com/blog/deploying-disaggregated-llm-inference-workloads-on-kubernetes/) — 面向分离式部署的 live migration。
