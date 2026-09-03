---
name: hybrid-picker
description: 针对给定工作负载，在纯 Transformer、Jamba 风格混合架构和纯 SSM 之间做出选择。
version: 1.0.0
phase: 10
lesson: 21
tags: [jamba, mamba, ssm, hybrid, long-context, memory-budget, architecture]
---

给定一份工作负载规格（上下文长度 profile 的 p50/p99、任务类型构成、每 GPU 显存预算、目标吞吐、质量与速度的优先级权衡），在纯 Transformer（+MoE +MLA）、Jamba 风格混合架构、以及纯 Mamba 模型之间给出推荐。

产出内容：

1. 上下文长度分桶。短（16k 以下）、中（16k-64k）、长（64k-256k）、超长（256k 以上）。决定第一轮筛选。
2. 架构推荐。从纯 Transformer、1:7 混合、1:3 混合、1:15 混合、或纯 Mamba 中选一。基于上下文分桶与任务的上下文内召回需求进行论证。
3. 显存预算检查。计算目标上下文下的 KV cache + SSM state。确认在计入权重和激活显存（通常在权重与 KV cache 之上再额外 10-20 GB）后，仍能装上目标加速卡。
4. 质量代价披露。记录所选稀疏度水平带来的质量损失。低于 1:7 比例的混合架构在上下文内检索上会有可观测的退化；纯 Mamba 在某些状态跟踪任务上会失败。
5. 推理栈兼容性。确认所选架构被目标栈（vLLM、TensorRT-LLM、SGLang、llama.cpp）支持。混合架构的工具链覆盖比纯 Transformer 更薄。

硬性拒绝：
- 对 16k 以下上下文使用 Jamba 风格混合架构。其架构开销并不划算。
- 对重推理或多文档交叉引用任务使用纯 Mamba。状态跟踪的局限会反咬一口。
- 低于 1:15 的混合比例。在此之下，上下文内召回不可靠。
- 任何在指定加速卡上超出所算显存预算的推荐。

拒绝规则：
- 若工作负载是真正意义上的短长上下文混合，拒绝混合架构推荐，改为推荐纯 Transformer（如可能则带 MLA）——混合架构恰恰在长上下文工作负载上才大放异彩。
- 若加速卡为消费级（24GB 或更少），拒绝 hybrid 尺寸模型，推荐蒸馏的小型混合模型或量化的纯 Transformer。
- 若工作负载是延迟敏感的 batch-1 生成且模型较新（尚无现成部署路径），则拒绝，改为推荐有良好支持的纯 Transformer 加投机解码（Phase 10 · 15）这一更简单的路径。

输出：一份一页推荐，列出上下文分桶、架构选择、目标上下文下的 KV cache、质量代价披露、以及推理栈兼容性。结尾附一段“监控什么”，指出在前 1 万条生产请求中能印证该推荐的某个具体长上下文评测（RULER、LongBench、needle-in-haystack）。
