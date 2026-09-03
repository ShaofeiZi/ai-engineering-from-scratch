# Kubernetes 上的 GPU 自动扩缩容：Karpenter、KAI Scheduler 与 Gang Scheduling

> 这不是一层自动扩缩，而是三层协同。Karpenter 负责动态供给节点，通常不到一分钟就能拉起新节点，比 Cluster Autoscaler 快约 40%。KAI Scheduler 负责 gang scheduling、拓扑感知和分层队列，避免“8 张卡只凑齐 7 张”的部分分配陷阱，也就是 7 个节点空等最后 1 张 GPU、持续烧钱的局面。应用层自动扩缩器，例如 NVIDIA Dynamo Planner 和 llm-d Workload Variant Autoscaler，则根据推理专用信号扩缩副本，比如队列深度和 KV cache 利用率，而不是 CPU 或 DCGM duty cycle。经典 HPA 的误区在于，`DCGM_FI_DEV_GPU_UTIL` 本质上是占空比指标：100% 利用率既可能对应 10 个请求，也可能对应 100 个请求。vLLM 还会预分配 KV cache 内存，所以显存指标几乎不会触发缩容。本课会教你把这三层组合起来，并避开 Karpenter 默认的 `WhenEmptyOrUnderutilized` 策略，因为它会在推理进行到一半时直接终止还在运行的 GPU 作业。

**Type:** 学习
**Languages:** Python（标准库，玩具级队列深度自动扩缩模拟器）
**Prerequisites:** 第 17 阶段 · 02（推理平台经济学），第 17 阶段 · 04（服务引擎内部原理）
**Time:** 约 75 分钟

## 学习目标

- 画出三层自动扩缩架构图，说明节点供给、gang scheduling、应用层扩缩分别对应什么工具。
- 解释为什么 `DCGM_FI_DEV_GPU_UTIL` 不适合拿来给 vLLM 做 HPA 信号，并说出两个替代指标，例如队列深度和 KV cache 利用率。
- 描述 gang scheduling 的含义，以及 KAI Scheduler 避免的部分分配故障模式，也就是“8 张卡只到位 7 张”时整批任务空等。
- 说出 Karpenter 会误杀运行中 GPU 作业的 consolidation policy `WhenEmptyOrUnderutilized`，并指出 2026 年更安全的替代配置。

## 问题

你的团队在 Kubernetes 上部署了一个 LLM serving 服务。你把 `DCGM_FI_DEV_GPU_UTIL` 作为 HPA 的扩缩信号。业务高峰期间，服务的 GPU utilization 一直卡在 100%。HPA 没有扩容，因为它认为系统已经“满载”。你手动多加了一个副本，TTFT 立刻下降。可 HPA 还是没有动。问题不在容量，而在信号本身误导了你。

另一边，你用 Cluster Autoscaler 来扩节点。凌晨 2 点来了一个 1M-token 的长提示词，集群花了 3 分钟才供给出新节点，结果请求直接超时。

再另一边，你上线了一个需要跨 2 个节点、总计 8 张 GPU 的 70B 模型。集群当前空闲了 7 张 GPU，剩下那第 8 张零散分布在 3 个节点上。Cluster Autoscaler 为最后那 1 张 GPU 新拉了一个节点。结果其余 7 个节点上的资源干等了 4 分钟，一边烧钱，一边等 Kubernetes 把最后那张卡补齐。

三层架构，三种完全不同的失败模式。到了 2026 年，GPU-aware autoscaling 早就不是“把 HPA 打开”这么简单，而是要把节点供给、gang scheduling 和应用信号扩缩容组合起来。

## 概念

### 第 1 层：节点供给（Karpenter）

Karpenter 监控 pending pods，并在大约 45 到 60 秒内供给新节点。相比之下，Cluster Autoscaler 给 GPU 节点扩容通常要 90 到 120 秒。它会按照 `NodePool` 约束动态选择实例类型。如果你的 pod 需要 8 张 H100，而集群里没有匹配节点，Karpenter 会直接创建合适的新节点，而不是去扩一个现有节点组。

**consolidation trap**：Karpenter 默认的 `consolidationPolicy: WhenEmptyOrUnderutilized` 对 GPU 池非常危险。它会为了把 pod 迁移到更便宜、规格更“合适”的实例上，直接终止一台还在运行的 GPU 节点。对推理工作负载来说，这意味着正在处理的请求会被驱逐，70B 模型还得在新节点上重新加载。代价通常是数分钟容量损失，再加上一批请求失败。

适用于 GPU 池的安全配置如下：

```yaml
disruption:
  consolidationPolicy: WhenEmpty
  consolidateAfter: 1h
```

这表示 Karpenter 只会在节点真正空闲并持续一小时后再做整合，而不会驱逐仍在运行的 GPU 作业。

### 第 2 层：gang scheduling（KAI Scheduler）

KAI Scheduler 负责默认 kube-scheduler 做不好的那一部分。这个项目早期代号叫 “Karp”，后来改名。

**Gang scheduling**：要么全部一起调度成功，要么一个都不启动。一个分布式推理工作负载如果需要 8 张 GPU，那就必须 8 张一次到位；否则就全部等待。没有这一层，就会落入部分分配陷阱：8 个 pod 里先起来 7 个，然后无限等待最后 1 个，白白占资源、持续烧钱。

**拓扑感知**：它知道哪些 GPU 共享 NVLink，哪些 GPU 位于同一机架，哪些节点之间有 InfiniBand，可以据此做合理放置。例如 DeepSeek-V3 67B 的 tensor-parallel 工作负载必须放在同一个 NVLink 域内，KAI Scheduler 会尊重这一点。

**分层队列**：多个团队共享同一个 GPU 池时，KAI 能根据优先级和配额来调度。团队 A 的线上高优先级任务，是否能被团队 B 的训练作业抢占，取决于队列中的优先级规则。

KAI 通常作为 kube-scheduler 旁边的辅助调度器部署，你通过给工作负载加注解来指定它。Ray 和 vLLM 的生产栈都已经可以和它集成。

### 第 3 层：应用层信号

**HPA 的经典误区**：`DCGM_FI_DEV_GPU_UTIL` 是一个 duty-cycle 指标，它只衡量 GPU 在采样周期内是否一直有活干。100% utilization 既可能意味着只有 10 个并发请求，也可能意味着已经积压到 100 个请求。GPU 都是满忙状态，但系统拥塞程度完全不同。用 duty cycle 来扩缩容，本质上是在盲扩。

更麻烦的是，vLLM 这类引擎会预先占用 KV cache 内存，最高可以顶到 `--gpu-memory-utilization` 所设的上限。即使系统里只有 1 个请求，显存使用率也可能接近 90%。所以基于内存的 HPA 根本不会触发缩容。

**2026 年更合适的替代信号**：

- 队列深度，也就是等待进入 prefill 的请求数量。
- KV cache 利用率，也就是活跃序列实际占用的 block 比例。
- 单副本的 P99 TTFT，也就是你的 SLA 信号。
- Goodput，也就是每秒真正满足全部 SLO 的请求数。

NVIDIA Dynamo Planner 和 llm-d Workload Variant Autoscaler 会消费这些信号来调整副本数。在 LLM serving 场景里，它们往往是对传统 HPA 的直接替代。

### 什么时候用什么

| 扩缩决策 | 工具 |
|----------------|------|
| 增减节点 | Karpenter |
| 调度多 GPU 作业 | KAI Scheduler |
| 增减副本 | Dynamo Planner / llm-d WVA（或基于队列深度自定义 HPA） |
| 选择 GPU 类型 | Karpenter NodePool |
| 抢占低优先级任务 | KAI Scheduler queues |

### 解耦 prefill / decode 会让问题更复杂

如果你采用了解耦的 prefill / decode 架构（Phase 17 · 17），那就会同时存在两类 pod，而且它们的扩缩信号不同：prefill pod 应该按队列深度扩缩，decode pod 应该按 KV cache 压力扩缩。llm-d 会把它们暴露成不同的 `Services`，并分别配置各自的 HPA。不要试图在这两类流量前面共用一个 HPA。

### 这里同样要考虑 cold start

冷启动缓解（Phase 17 · 10）在这里也很关键，因为节点供给时间会直接暴露给用户。Karpenter 需要 45 到 60 秒来拉起新节点，再加上 20GB 模型加载和引擎初始化，从零启动的一次请求很容易就要等 2 到 5 分钟。对于 SLO 敏感路径，你应该保留一个 warm pool，也就是 `min_workers=1`；或者在应用层使用类似 Modal 的 checkpointing 方案。

### 这些数字你应该记住

- Karpenter 供给节点大约需要 45 到 60 秒，而 Cluster Autoscaler 给 GPU 节点扩容通常要 90 到 120 秒。
- KAI Scheduler 能避免部分分配浪费，也就是典型的 “7-of-8” 陷阱。
- `DCGM_FI_DEV_GPU_UTIL` 作为 HPA 信号是错误的；应该改用队列深度或 KV 利用率。
- Karpenter 的 `WhenEmptyOrUnderutilized` 会终止仍在运行的 GPU 作业。做推理时，更安全的配置是 `WhenEmpty + consolidateAfter: 1h`。

```figure
autoscaling
```

## 用起来

`code/main.py` 会模拟一个面向突发 GPU 工作负载的三层自动扩缩系统。它比较三种方案：天真的 duty-cycle HPA、基于队列深度的 HPA，以及带有 KAI gang scheduling 的扩缩策略。输出结果包括未满足请求数、GPU 空转分钟数，以及一个综合评分。

## 交付物

本课会产出 `outputs/skill-gpu-autoscaler-plan.md`。给定集群拓扑、工作负载形态和 SLO，它会设计一份三层自动扩缩方案。

## 练习

1. 运行 `code/main.py`。在突发工作负载下，天真的 duty-cycle HPA 会漏掉多少本应由队列深度 HPA 捕获的请求？差异来自哪里？
2. 为一个在 H100 SXM5 上服务 Llama 3.3 70B FP8 的集群设计 Karpenter NodePool。明确写出 `capacity-type`、`disruption.consolidationPolicy`、`consolidateAfter`，以及一个能把非 GPU 工作负载挡在外面的 taint。
3. 你的团队反馈部署一直卡在 Pending，报错是“集群里明明有 GPU，但 pod 就是调度不上去”。请诊断问题在 Karpenter、kube-scheduler 还是 KAI Scheduler，并说明要看哪些指标来确认。
4. 为解耦的 prefill pod 选一个自动扩缩信号，再为 decode pod 选另一个不同的信号，并分别说明理由。
5. 计算 `WhenEmptyOrUnderutilized` 这个 consolidation trap 在一个 24x7 线上服务中的成本。已知它平均每天触发 60 次导致请求掉落的事件，并且 P99 TTFT 都高于 10 秒。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------------|------------------------|
| Karpenter | “节点供给器” | Kubernetes 的节点自动扩缩器，能做到亚分钟级供给 |
| Cluster Autoscaler | “老一代扩缩器” | Karpenter 之前的 Kubernetes 节点自动扩缩方案，速度更慢，基于节点组 |
| KAI Scheduler | “GPU 调度器” | 提供 gang scheduling、拓扑感知和队列管理的辅助调度器 |
| Gang scheduling | “要么全上，要么全等” | N 个 pod 只有在能整体调度成功时才会一起启动，否则全部延后 |
| Topology awareness | “感知机架与互联拓扑” | 基于 NVLink、InfiniBand、机架位置等信息来放置 pod |
| `DCGM_FI_DEV_GPU_UTIL` | “GPU 利用率” | Duty-cycle 指标，不适合作为 LLM 的扩缩信号 |
| Queue depth | “排队请求数” | 适合 prefill 扩缩的 HPA 信号 |
| KV cache utilization | “KV cache 压力” | 适合 decode 扩缩的 HPA 信号 |
| Consolidation | “Karpenter 整合” | 为了迁移到更便宜实例类型而终止节点 |
| `WhenEmpty + 1h` | “安全整合策略” | 不会驱逐运行中 GPU 作业的整合配置 |

## 延伸阅读

- [KAI Scheduler GitHub](https://github.com/kai-scheduler/KAI-Scheduler) — 设计文档与配置示例。
- [Karpenter 中断控制](https://karpenter.sh/docs/concepts/disruption/) — 节点整合策略的语义，以及适合 GPU 的安全默认值。
- [NVIDIA — Disaggregated LLM Inference on Kubernetes](https://developer.nvidia.com/blog/deploying-disaggregated-llm-inference-workloads-on-kubernetes/) — Dynamo Planner 所使用的扩缩信号。
- [Ray docs — KAI Scheduler for RayClusters](https://docs.ray.io/en/latest/cluster/kubernetes/k8s-ecosystem/kai-scheduler.html) — Ray 的集成方式。
- [AWS EKS 计算与自动扩缩最佳实践](https://docs.aws.amazon.com/eks/latest/best-practices/aiml-compute.html) — EKS 上计算与自动扩缩的最佳实践。
- [llm-d GitHub](https://github.com/llm-d/llm-d) — Workload Variant Autoscaler 的设计。
