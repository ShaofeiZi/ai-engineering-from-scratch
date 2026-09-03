# 解耦式 Prefill/Decode：NVIDIA Dynamo 与 llm-d

> Prefill 是计算受限型工作，decode 是显存与带宽受限型工作。把两者放在同一张 GPU 上，会天然浪费其中一种资源。解耦式服务把它们拆到不同资源池，并通过 NIXL 在两池之间传输 KV cache，底层可走 RDMA/InfiniBand，也可回退到 TCP。NVIDIA Dynamo 在 GTC 2025 宣布、1.0 GA 后，位于 vLLM/SGLang/TRT-LLM 之上，利用 Planner Profiler 与 SLA Planner 自动匹配 prefill:decode 的容量比例来满足 SLO。NVIDIA 公布过这一区间内的吞吐提升：developer.nvidia.com（2025-06）展示 DeepSeek-R1 MoE 在 GB200 NVL72 + Dynamo、medium-latency 区间下约有 6x 提升；Dynamo 产品页（developer.nvidia.com，无日期）则宣传 GB300 NVL72 + Dynamo 相对 Hopper 可达最高 50x MoE 吞吐。至于“30x”这个数字，更像社区对 Blackwell + Dynamo + DeepSeek-R1 整栈结果的汇总说法；我们没有找到单一一手来源明确写出 30x，因此只能把它当作方向性信号。llm-d（Red Hat + AWS）走的是 Kubernetes 原生路线：把 prefill、decode、router 拆成独立 Service，并按角色各自使用 HPA。llm-d 0.5 又加入了分层 KV offloading、cache-aware LoRA routing、UCCL networking、scale-to-zero。经济性方面：内部汇总多个客户披露后，推断当年推理成本在 $2M 量级时，从 colocated serving 切到基于 Dynamo 的 disaggregated serving，在 SLA 不变的前提下，常见节省区间约为 30–40%（即 $600K–$800K/年）。这里的 $2M→$600-800K 是内部复合估算，不是公开案例研究，适合当数量级锚点，不适合当正式引文。对于短提示词（<512 tokens）和短输出，KV 传输成本通常不值得。

**Type:** 学习
**Languages:** Python（标准库，玩具级解耦式与共置式服务模拟器）
**Prerequisites:** 阶段 17 · 04（服务引擎内部原理）、阶段 17 · 08（推理指标）
**Time:** 约 75 分钟

## 学习目标

- 解释为什么 prefill 和 decode 对 GPU 的最优配置不同，并量化 colocated serving 下的浪费。
- 画出解耦式架构图：prefill pool、decode pool、通过 NIXL 传输的 KV cache，以及 router。
- 说清楚在什么条件下 disaggregation 不划算，例如短提示词、短输出。
- 区分 NVIDIA Dynamo 这种“位于推理引擎之上的编排层”和 llm-d 这种“Kubernetes 原生方案”，并能将其对应到合适的运维场景。

## 问题

假设你在 8 张 H100 上运行 Llama 3.3 70B。面对混合工作负载时，如果提示词很长、输出很短，GPU 在 decode 阶段会出现空闲，因为大部分计算已经在 prefill 阶段完成。反过来，如果提示词很短、输出很长，那么 prefill 很快结束，瓶颈就转移到 decode。

colocated prefill + decode 的问题在于：你必须同时为两种瓶颈过度配置资源。

预算上的直接影响是，20–40% 的 GPU 时间会浪费在“不匹配的资源类型”上。你可能正在买 H100 的计算能力来跑内存受限的 decode，也可能正在买 H100 的 HBM 带宽来跑计算受限的 prefill。两种情况都很昂贵，而且都不是高效利用。

disaggregation 的做法是，把 prefill 和 decode 拆到不同资源池，让每个池按照自己的瓶颈单独定容。KV cache 再通过高带宽互联，从 prefill pool 传给 decode pool。

## 核心概念

### 为什么瓶颈不同

**Prefill**：在一次 forward 中把整段输入 prompt 跑完。主导成本是矩阵乘法，因此更偏计算受限。H100 FP8 能提供大约 2000 TFLOPS 的有效吞吐。批处理效率也比较高，一次 forward 可以覆盖很多 token。

**Decode**：一次只生成一个 token，但每一步都要重新读取整套权重，因此更偏显存带宽受限。HBM3 大约提供 3 TB/s 带宽。只有在高并发下，decode 的 batch 效率才会明显提升，因为权重读取成本才能被摊薄。

把两者 colocate 在一起，等于要求同一批 GPU 同时兼顾两类优化目标。H100 两边都能做，但无论做哪边成本都一样。到了更大规模时，更合理的思路通常是：把 prefill pool 放在 H100 这类偏计算型设备上，把 decode pool 放在 H200 这类偏显存与带宽型设备上，或者叠加强量化策略。

### 架构图

```
            ┌──────────────┐
  Request → │    Router    │ ───────────────────────┐
            └──────┬───────┘                        │
                   │                                │
                   ▼ (prompt only)                  │
            ┌──────────────┐    KV cache    ┌───────▼──────┐
            │ Prefill pool │ ─── NIXL ────► │ Decode pool  │
            │  (compute)   │                │  (memory)    │
            └──────────────┘                └──────┬───────┘
                                                   │ tokens
                                                   ▼
                                                 Client
```

NIXL 是 NVIDIA 的跨节点传输机制。优先走 RDMA/InfiniBand，拿不到时再走 TCP。KV 传输延迟是现实存在的：对于 70B FP8 模型、4K-token prompt 的 KV cache，常见延迟大约在 20–80 ms。这也是为什么短 prompt 不适合做 disaggregation：传输税往往比节省下来的资源更贵。

### Dynamo 与 llm-d 的区别

**NVIDIA Dynamo**（GTC 2025 发布，1.0 GA）：
- 作为编排层，位于 vLLM、SGLang、TRT-LLM 之上。
- Planner Profiler 用来测量工作负载，SLA Planner 自动配置 prefill:decode 的容量比例。
- 核心实现用 Rust，扩展能力用 Python。
- 吞吐增益方面：NVIDIA 报告 DeepSeek-R1 MoE 在 GB200 NVL72 + Dynamo、medium-latency 区间可达到 6x（developer.nvidia.com, 2025-06）；社区常见的“up to 30x”更像全 Blackwell + Dynamo + DeepSeek-R1 方案的方向性汇总，没有单一一手来源支撑。
- GB300 NVL72 + Dynamo：根据 Dynamo 产品页（developer.nvidia.com，无日期），MoE 吞吐相对 Hopper 最高可达 50x。

**llm-d**（Red Hat + AWS，Kubernetes 原生）：
- prefill、decode、router 都是独立的 Kubernetes Service。
- 每个角色都能单独配 HPA，信号来源分别可以是 queue depth（prefill）和 KV utilization（decode）。
- `topologyConstraint packDomain: rack` 用来把 prefill+decode clique 尽量打包在同一个机架内，以保证 KV 传输有足够带宽。
- llm-d 0.5（2026）加入了 hierarchical KV offloading、cache-aware LoRA routing、UCCL networking、scale-to-zero。

如果你想要一个“位于推理引擎之上的托管编排层”，就偏向 Dynamo；如果你想要 Kubernetes 原语优先、并且团队本身已经深度投入 CNCF 生态，那就偏向 llm-d。

### 经济性

内部复合估算（不是单一公开案例研究，只能作为数量级锚点）：

- colocated serving 的年推理开销约为 $2M。
- 切换到基于 Dynamo 的 disaggregated serving。
- 请求量不变，P99 latency SLA 也不变。
- 常见节省区间约为 $600K–$800K/年，也就是 30–40%。
- 不需要新增硬件。

这个数字来自多个客户披露的综合推断，而不是某一篇可直接引用的案例研究。公开材料里，最接近的参考点包括：Baseten 在 2025-10 披露通过 Dynamo KV routing 获得 2x 更快的 TTFT 和 61% 更高吞吐；VAST + CoreWeave 在 2025-12 预测，当 KV hit rate 在 40–60% 时，tokens/$ 可提升 60–130%。节省的本质来自按角色正确配池；如果你的业务是 prefill 很重的工作负载，例如带 8K+ 前缀的 RAG，收益通常会比负载更均衡的系统更明显。

### 什么时候不该解耦

- prompts < 512 tokens 且 outputs < 200 tokens：KV 传输税大于收益。
- 小集群（< 4 GPUs）：池化弹性不足，分拆意义不大。
- 团队没有能力同时运维两个独立 GPU 资源池并按角色扩缩容：即便 Dynamo 能降低复杂度，也并不等于“没有复杂度”。
- 没有 RDMA 网络：如果只能走 TCP，传输税会更重。

### Router 会与 Phase 17 · 11 形成联动

解耦式 router 本质上也是 KV-cache-aware 的，这一点和 Phase 17 · 11 是连起来的。请求应尽量落到已经持有对应 prefix 的 decode pool 上；如果没有命中，才走 prefill → decode。也就是说，cache hit rate 与 disaggregation 的收益是叠加关系，而 cache-aware router 决定了你是否还需要重新做一次 prefill。

### 真正夸张的数字通常出现在 Blackwell 上的 MoE

GB300 NVL72 + Dynamo 给出的 50x MoE 吞吐，对比基线是 Hopper。MoE expert routing 在 prefill 阶段偏计算重，在 decode 阶段又偏显存与缓存重，因此 disaggregation 会同时在两边受益。到 2026 年，前沿模型服务已经明显往 MoE 主导方向移动，例如 DeepSeek-V3，以及未来一些 GPT-5 变体。

### 你需要记住的数字

基准会漂移。NVIDIA 和推理栈供应商几乎每个季度都会刷新结果，正式引用前应重新核对。

- DeepSeek-R1 on GB200 NVL72 + Dynamo：在 medium-latency 区间下，相对基线约 6x 吞吐（developer.nvidia.com, 2025-06）；社区里“up to 30x”的说法没有单一一手来源，适合当方向性信息，不适合当精确事实。
- GB300 NVL72 + Dynamo：相对 Hopper，MoE 吞吐最高 50x（developer.nvidia.com，无日期）。
- 节省锚点：在年成本 $2M、SLA 不变前提下，内部复合估算显示可节省 $600K–$800K/年。
- 经验阈值：prompts > 512 tokens 且 outputs > 200 tokens，disaggregation 才更容易成立。
- NIXL 的 KV 传输延迟：70B FP8、4K prompt 的 KV 大约需要 20–80 ms。

```figure
prefill-decode-split
```

## 用起来

`code/main.py` 会模拟 colocated serving 与 disaggregated serving，输出吞吐、单请求成本，以及 prompt 长度的盈亏交叉点。

## 交付物

本课产出 `outputs/skill-disaggregation-decider.md`。它会根据工作负载形态和集群条件，判断你是否应该采用 disaggregation。

## 练习

1. 运行 `code/main.py`。在哪个 prompt 长度之后，disaggregation 开始优于 colocation？
2. 为一个 RAG 服务设计 prefill pool 和 decode pool：P99 prefix length 为 8K，输出长度为 300。
3. Dynamo vs llm-d：如果团队是纯 Kubernetes shop，而且对 Python runtime 没有偏好，你会选哪个？
4. 计算 KV 传输成本：70B FP8 模型上，4K prefill 大约对应 500 MB KV。若 RDMA 为 100 GB/s，则传输约 5 ms；若 TCP 为 10 GB/s，则约 50 ms。你的 SLA 更在意哪一个？
5. MoE expert routing 会改变 KV 的访问模式。当每个 token 激活的专家都不同，disaggregation 会呈现怎样的行为特征？

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------------|------------------------|
| Disaggregated serving | “拆分 prefill/decode” | 把两个阶段拆到不同 GPU 资源池中 |
| NIXL | “NVIDIA 传输层” | Dynamo 使用的跨节点 KV 传输层（RDMA/TCP） |
| NVIDIA Dynamo | “编排器” | 位于 vLLM/SGLang/TRT-LLM 之上的协调编排层 |
| llm-d | “Kubernetes 原生” | Red Hat + AWS 提供的 K8s 解耦式推理栈 |
| Planner Profiler | “Dynamo 自动配置” | 负责测量工作负载并给出池比例配置 |
| SLA Planner | “Dynamo 策略” | 自动匹配 prefill:decode 比例以满足 SLO |
| `packDomain: rack` | “llm-d 拓扑约束” | 把 prefill+decode 尽量放在同一机架内以加速 KV |
| UCCL | “统一集合通信” | llm-d 0.5 中面向 scale-to-zero 的网络层 |
| MoE expert routing | “每 token 选择专家” | 类似 DeepSeek-V3 的专家路由模式，解耦更受益 |

## 延伸阅读

- [NVIDIA — Introducing Dynamo](https://developer.nvidia.com/blog/introducing-nvidia-dynamo-a-low-latency-distributed-inference-framework-for-scaling-reasoning-ai-models/)
- [NVIDIA — Kubernetes 上的解耦式 LLM 推理](https://developer.nvidia.com/blog/deploying-disaggregated-llm-inference-workloads-on-kubernetes/)
- [TensorRT-LLM 解耦式服务博客](https://nvidia.github.io/TensorRT-LLM/blogs/tech_blog/blog5_Disaggregated_Serving_in_TensorRT-LLM.html)
- [llm-d GitHub](https://github.com/llm-d/llm-d)
- [llm-d 0.5 发布说明](https://github.com/llm-d/llm-d/releases)
