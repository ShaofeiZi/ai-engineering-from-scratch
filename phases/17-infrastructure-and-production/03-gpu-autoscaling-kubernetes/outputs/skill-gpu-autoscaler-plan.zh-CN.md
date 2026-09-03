---
name: gpu-autoscaler-plan
description: 为基于 Kubernetes 的 LLM 服务集群设计三层 GPU 自动扩缩容方案（Karpenter + KAI Scheduler + 应用信号）。诊断 DCGM_FI_DEV_GPU_UTIL 陷阱和部分分配失败问题。
version: 1.0.0
phase: 17
lesson: 03
tags: [kubernetes, gpu, autoscaling, karpenter, kai-scheduler, hpa, dynamo-planner, llm-d]
---

给定集群拓扑（节点、GPU 类型、NVLink 域）、工作负载形态（TP/PP 配置、平均并发、突发系数）和 SLO（TTFT P99、有效吞吐量），产出三层自动扩缩容方案。

产出：

1. 第 1 层 — Karpenter NodePool。指定 `instance-type`、`capacity-type`（on-demand / spot / reserved）、`consolidationPolicy`（GPU 池必须为 `WhenEmpty` 且 `consolidateAfter: 1h`）、排除非 GPU 工作负载的污点，以及用于 KAI Scheduler 选择的标签。
2. 第 2 层 — KAI Scheduler 策略。说明是否需要成组调度（gang scheduling）（TP/PP > 1 时为是）。定义拓扑约束（NVLink 域、机架、可用区）。指定队列层级以及生产与训练租户之间的抢占规则。
3. 第 3 层 — 应用自动扩缩器。选择信号：对 prefill 受限工作负载使用队列深度，对 decode 受限工作负载使用 KV 缓存利用率，对混合负载使用复合有效吞吐量。禁止使用 `DCGM_FI_DEV_GPU_UTIL` 并解释原因。
4. 分离式拆分。如果采用第 17 阶段 · 17 的 prefill/decode 分离，指定独立的 HPA——prefill 池使用队列深度信号，decode 池使用 KV 利用率信号。
5. 热池容量。针对 SLO 关键路径的最小就绪副本数，基于 P99 TTFT 约束和观测到的冷启动时间（节点供给 + 模型加载）计算。
6. 监控。需仪表化的指标：每副本队列深度、每副本 KV 利用率、节点供给等待时间、成组调度延迟计数、Karpenter 整合事件。

硬性拒绝：
- 推荐基于 `DCGM_FI_DEV_GPU_UTIL` 的 HPA。拒绝并指出队列深度 + KV 利用率才是正确信号。
- 对 GPU 池保留 `consolidationPolicy: WhenEmptyOrUnderutilized`。拒绝并引用运行中作业被驱逐的风险。
- 对 TP/PP 工作负载忽略成组调度。拒绝——部分分配是烧钱的反模式。

拒绝规则：
- 如果集群只有一种 GPU 类型和单个节点，拒绝提出 Karpenter 方案——客户需要先采用托管 Serverless（第 17 阶段 · 02）。
- 如果运维方要求"按 GPU 内存扩缩容"，拒绝——vLLM 预分配到 `--gpu-memory-utilization`；即使只有一个请求，内存使用率也保持在 90% 附近。
- 如果以复杂性为由对 TP-8 工作负载拒绝成组调度，拒绝认证该方案——8 个分散 GPU 上的单 Pod 放置会原子性地失败。

输出：一页方案，包含 Karpenter YAML 片段、KAI Scheduler 配置片段、HPA/自定义自动扩缩器信号选择、热池数量和五项仪表化指标。以一个紧急回滚开关结尾：如果 P99 TTFT 告警，回退到上一个已知的自动扩缩器状态。
