# 面向 LLM 生产环境的混沌工程

> 到了 2026 年，面向 LLM 的混沌工程已经是一门独立学科。想在生产环境里做实验，至少要先满足几个前置条件：定义好的 SLI/SLO、完整的 trace + metric + log 可观测性、自动回滚、runbook，以及 on-call 值守。整套架构可以拆成四个平面：control（实验调度器）、target（服务、基础设施、数据存储）、safety（护栏、abort、流量过滤）、observability（指标、trace、日志），外加一个把发现回灌到 SLO 调整里的 feedback loop。护栏不是可选项，而是硬要求：如果 daily error-budget burn rate 超过预期的 2 倍，就必须暂停实验；suppression windows 和 trace-ID correlation 用来去重告警噪音。推荐节奏也已经比较固定：每周做小范围 canary + SLO review；每月做一次 game day + postmortem；每季度做跨团队韧性审计和依赖图梳理。LLM 特有的实验包括：内存过载、网络故障、供应商故障、畸形提示词、KV cache 驱逐风暴。常见工具则有 Harness 混沌工程（支持 LLM 生成建议、blast radius 下调、MCP 工具集成）、LitmusChaos（CNCF）和 Chaos Mesh（CNCF Kubernetes-native）。

**Type:** 学习
**Languages:** Python（标准库，玩具级混沌实验运行器）
**Prerequisites:** 阶段 17 · 23（AI SRE）、阶段 17 · 13（可观测性）
**Time:** 约 60 分钟

## 学习目标

- 说出混沌工程的五个前置条件（SLI/SLO、observability、rollback、runbooks、on-call），并解释为什么缺任何一个都会把演练变成真实事故。
- 画出四个平面（control、target、safety、observability）以及回灌到 SLO 的 feedback loop。
- 列举五类 LLM 特有实验（memory overload、network failure、provider outage、malformed prompt、KV eviction storm）。
- 根据技术栈选择合适工具：Harness、LitmusChaos 或 Chaos Mesh。

## 问题

传统系统里的 chaos testing 已经比较成熟，但 LLM 栈带来了新的故障模式。一个带毒字符的 4K-token prompt，可能把 tokenizer 卡住 12 秒。上游 provider 开始返回 429，你的 gateway 触发重试，结果并发被重试放大，服务直接 OOM。突发负载下的 KV cache eviction storm，会引发 re-prefill 级联，把计算资源瞬间打满。

这些问题在单元测试里几乎不会暴露。混沌工程的意义，就是在用户先踩中之前，你先把它们炸出来。

## 概念

### 前置条件

不要在生产环境里做 chaos experiment，除非你已经具备下面这些条件：

1. **SLI/SLO**：已经定义好服务级指标和目标。
2. **Observability**：trace、metrics、logs 都已经接入 dashboard。
3. **Automated rollback**：例如 Phase 17 · 20 里那种基于 policy flag 的自动回滚。
4. **Runbooks**：已经结构化，参考 Phase 17 · 23。
5. **On-call**：实验期间真的有人能响应。

缺任何一个，混沌工程都不是“演练”，而是“人为制造事故”。

### 四个平面 + 反馈回路

**Control plane**：实验调度层，例如 Litmus workflow、Chaos Mesh schedule、Harness UI。

**Target plane**：被攻击的对象层，包括服务、pods、nodes、load balancers、data stores。

**Safety plane**：护栏层，包括 kill switch、suppression windows、blast-radius 限制和 error-budget gates。

**Observability plane**：观测层，除了常规指标外，还要通过 trace-ID correlation 区分“混沌实验导致的故障”和“自然发生的故障”。

**Feedback loop**：把实验发现回灌到 SLO 调整、runbook 更新和代码修复中。

### 护栏是硬要求

- **Burn-rate alert**：如果 daily error-budget burn 超过预期的 2 倍，就暂停实验。
- **Suppression windows**：实验期间，在 blast radius 内压制与实验无关的告警。
- **Trace-ID correlation**：所有实验引发的错误都带上标记，方便 on-call 去重和归因。

### 五类 LLM 特有实验

1. **Memory overload**：通过高并发长上下文请求强行制造 KV cache preemption storm。观察服务会优雅限流，还是直接崩掉。

2. **Network failure**：切断 inference gateway 和 provider 之间的连接。观察 fallback 能否在 SLA 内接管。（Phase 17 · 19）

3. **供应商故障模拟**：让 OpenAI 100% 返回 429。观察路由是否会 failover 到 Anthropic。（Phase 17 · 16, 19）

4. **Malformed prompt**：注入会卡 tokenizer 的 payload，例如深层嵌套 unicode 或超大 UTF-8 codepoint。观察一个请求是否就能卡死一个 worker。

5. **KV eviction storm**：通过打满 vLLM 的 block budget 触发大量 eviction。观察 LMCache 是能恢复，还是整个服务持续劣化。

### 执行节奏

- **每周**：小范围 canary experiment，通常先在 staging，必要时最多带 5% prod 流量。
- **每月**：固定场景的 game day，跨团队参加，并在结束后写 postmortem。
- **每季度**：跨团队 resilience audit，同时更新依赖图谱。

### 工具选择

- **Harness 混沌工程**：商业产品，支持 AI 生成实验建议、自动收缩 blast radius，并可接 MCP 工具。
- **LitmusChaos**：CNCF 毕业项目，偏 Kubernetes workflow 风格。
- **Chaos Mesh**：CNCF sandbox，偏 Kubernetes-native CRD 风格。
- **Gremlin**：商业方案，覆盖面广。
- **AWS FIS** / **Azure 混沌工作室**：云厂商托管方案。

### 从小处开始

第一个实验，通常应该非常保守：在稳定流量下杀掉一个 decode replica，观察流量重路由和恢复时间。如果这一步都安全且可控，再升级到 network chaos。

对 LLM 团队来说，第一个真正有价值的特有实验通常是：给某个 provider 注入 5 分钟的 429。大多数团队都会在这里第一次发现，他们自以为“已经测过”的 fallback 其实并没有真正打通过。

### 你应该记住的数字

- 四个平面：control、target、safety、observability。
- Burn-rate pause 阈值：预期 daily budget burn 的 2 倍。
- 节奏：weekly canary、monthly game day、quarterly audit。
- 五类 LLM 实验：memory、network、provider、malformed prompt、KV storm。

```figure
i4-chaos-guard
```

## 用起来

`code/main.py` 会模拟三种带 safety-plane 闸门的 chaos experiments，并报告哪些实验会触发 burn-rate abort。

## 交付物

这一课会产出 `outputs/skill-chaos-plan.md`。给定技术栈和成熟度后，它会帮你选前三个实验，并配好工具方案。

## 练习

1. 运行 `code/main.py`。是哪一个实验触发了 burn-rate gate？为什么？
2. 为一个基于 vLLM 的 RAG 服务设计前五个 chaos experiments，并写出 success criteria。
3. 你的 burn-rate alert 暂停了实验。你该如何判断根因来自 chaos 还是自然事故？
4. 论证 chaos 应不应该在生产环境里跑，还是只该停留在 staging。什么情况下 production 才是正确答案？
5. 说出三种 generic network chaos 无法复现的 LLM 特有故障模式。

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|----------------|------------------------|
| SLI / SLO | “服务目标” | 指标 + 目标，是开展 chaos 的前置条件 |
| Blast radius | “影响范围” | 实验会影响到的服务或用户集合 |
| Burn-rate alert | “预算闸门” | 当 error-budget burn rate 超过预期 2 倍时触发 |
| Game day | “每月演练” | 跨团队的定期混沌演练日 |
| LitmusChaos | “CNCF workflow” | CNCF 毕业的 Kubernetes chaos 工具 |
| Chaos Mesh | “CNCF CRD” | CNCF sandbox 的 Kubernetes-native chaos 工具 |
| Harness CE | “商业 AI 辅助方案” | 带 AI 建议的 Harness chaos 产品 |
| Malformed prompt | “tokenizer bomb” | 会让 tokenizer 卡顿的异常输入 |
| KV eviction storm | “驱逐风暴” | 大量 KV eviction 触发 re-prefill 级联 |

## 延伸阅读

- [DevSecOps School — 2026 混沌工程指南](https://devsecopsschool.com/blog/chaos-engineering/)
- [Ankush Sharma — 面向 LLM 的可观测性（书籍）](https://www.amazon.com/Observability-Large-Language-Models-Engineering-ebook/dp/B0DJSR65TR)
- [LitmusChaos (CNCF)](https://litmuschaos.io/)
- [Chaos Mesh (CNCF)](https://chaos-mesh.org/)
- [Harness 混沌工程](https://www.harness.io/products/chaos-engineering)
- [AWS FIS](https://aws.amazon.com/fis/)
