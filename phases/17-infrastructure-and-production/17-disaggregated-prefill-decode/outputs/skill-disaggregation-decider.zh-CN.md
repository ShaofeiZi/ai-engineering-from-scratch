---
name: disaggregation-decider
description: 针对给定工作负载和集群，决定是否采用 Prefill/Decode 分离架构（Dynamo 或 llm-d）。量化 prefill:decode 比例、KV 传输开销以及预期节省。
version: 1.0.0
phase: 17
lesson: 17
tags: [disaggregated-serving, dynamo, llm-d, nixl, kv-transfer, prefill-decode]
---

给定工作负载画像（提示词/输出长度分布、模型、并发度）、集群拓扑（GPU、网络互联、RDMA 可用性）以及当前服务成本，输出一份分离架构决策。

产出内容：

1. 是否分离？是 / 否，并附编号理由。基线条件：提示词 > 512 且输出 > 200。网络互联：有 RDMA 则有利；仅 TCP 则推高盈亏平衡点。
2. 技术栈选择。NVIDIA Dynamo（在 vLLM/SGLang/TRT-LLM 之上的托管编排器）或 llm-d（Kubernetes 原生 Service）。需匹配运维场景。
3. Prefill:decode 比例。使用 Dynamo Planner Profiler 的读取结果，或根据工作负载形态计算（prefill TFLOPS 对比 decode bytes/sec）。示例：RAG 密集型为 2 prefill : 1 decode；输出密集型为 1:2。
4. KV 传输方案。指明传输方式（NIXL over InfiniBand / RDMA / TCP 回退）。针对你的提示词 P99 计算每请求传输税。
5. 路由集成。缓存感知路由器（第 17 阶段 · 11）必须在前端——分离架构若不做前缀匹配则失去缓存收益。
6. 预期节省。与共置基线对比计算；引用已发表案例（相同 SLA 下节省 30-40%）。

硬性拒绝条件：
- 对短提示词工作负载（<512 token）做分离。拒绝——传输税将占主导。
- 在没有缓存感知路由器的情况下部署。拒绝——盲路由会使 KV 局部性失效。
- 忽略拓扑（机柜打包）。拒绝——跨机柜跳数的 KV 传输成本高于同机柜 RDMA。

拒绝规则：
- 若集群 GPU 数 < 4，拒绝——池多样性不足以让分离架构回本。
- 若无 RDMA/InfiniBand 且无规划，需说明 TCP 会将盈亏平衡点推高至提示词 >2K；建议重新评估。
- 若团队无法运营两个具有按角色独立扩缩容的 GPU 池，拒绝 llm-d，要求改用 Dynamo 作为托管替代方案。

输出：一页决策文档，包含分离 Y/N、技术栈选择、比例、传输方式、路由器、预期节省。以需验证的唯一指标结尾：KV 传输 P99 延迟；超过计划指定阈值即触发熔断。
