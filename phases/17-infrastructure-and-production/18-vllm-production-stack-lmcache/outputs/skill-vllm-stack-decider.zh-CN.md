---
name: vllm-stack-decider
description: 根据工作负载和集群规模，决定 vLLM 部署布局——production-stack Helm chart、KV 卸载（原生 CPU 或 LMCache）、路由器/可观测性集成。
version: 1.0.0
phase: 17
lesson: 18
tags: [vllm, production-stack, lmcache, kv-offload, connector-api]
---

给定工作负载（提示词形态、并发度、前缀复用模式）、集群（引擎数、GPU 型号）和运维场景（Kubernetes 原生、多租户、预算），输出一份 vLLM 技术栈方案。

产出内容：

1. 技术栈。使用 vLLM production-stack Helm chart（推荐用于新部署）或自行构建。说明适用的 operator/CRD。
2. KV 卸载。选择：
   - None（短提示词、低并发——开销大于收益）。
   - 原生 vLLM CPU 卸载（单引擎 HBM 压力大、方案简单）。
   - LMCache connector（多引擎前缀复用、抢占频繁或多租户共享提示词）。
3. HBM 利用率监控。设置 `--gpu-memory-utilization` 时预留余量；持续超过 92% 时告警，作为抢占前置信号。
4. 路由集成。缓存感知路由器（第 17 阶段 · 11）。确认 KV-event 通道已配置。
5. 可观测性。每引擎 Prometheus 采集，OTel GenAI 属性（第 17 阶段 · 13），使用 production-stack 的 Grafana 仪表盘模板。
6. 预期影响。量化预期吞吐提升并与当前对比——参考 16x H100 基准测试形态（当 KV 占用超过 HBM 时 LMCache 才有显著效果）。

硬性拒绝条件：
- 在无共享前缀或无抢占的场景部署 LMCache。拒绝——只有开销，没有收益。
- 在无 HBM 压力监控的情况下运行 vLLM。拒绝——首次抢占将是意外事件。
- 当 Helm chart 已覆盖使用场景时仍自行搭建 production-stack。拒绝——重复造轮子的成本。

拒绝规则：
- 若集群引擎数 <2，拒绝 LMCache——跨引擎复用才是其意义；单引擎应使用原生卸载。
- 若工作负载提示词 < 1K token 且并发 < 100，拒绝任何形式的卸载——HBM 余量足够。
- 若团队不具备 K8s 能力，拒绝 production-stack——从单引擎 vLLM + 简单代理开始。

输出：一页方案，包含技术栈、KV 卸载选择、HBM 监控、路由集成、可观测性、预期影响。以唯一门控指标结尾：过去 24 小时 HBM 利用率 P99。
