---
name: eagle3-tuner
description: 为新的推理工作负载选择并调优投机解码策略（vanilla / Medusa / EAGLE-1/2/3 / lookahead）。
version: 1.0.0
phase: 10
lesson: 15
tags: [speculative-decoding, eagle, eagle-3, medusa, inference, vllm, sglang, tensorrt-llm]
---

给定一个生产推理目标（验证器模型、batch size、序列长度画像、目标 p50/p99 decode 延迟、加速器、来自遥测的期望 alpha 范围、任务组合），推荐一种投机解码策略和调优参数。该推荐必须精确保持验证器的输出分布——若没有明确签字，不接受任何质量权衡。

产出：

1. Draft 族。从 vanilla、Medusa、EAGLE-1、EAGLE-2、EAGLE-3 或 lookahead 中选择。依据 alpha 遥测（或一个校准过的估计）、可用训练成本（无、小型 SFT、完整的 60B+ token 训练）、以及验证器是否附带已发布的 draft（EAGLE-3 检查点存在于 Llama 3.1/3.3、DeepSeek-V3、Qwen 2.5、Qwen 3）来论证。
2. Draft 长度 N。选择整数 N，使其在给定 alpha 和 draft 相对验证器成本比 c 的条件下最小化每 token 的期望墙上时间：minimize (1 + N*c) / ((1 - alpha^(N+1)) / (1 - alpha))。对最优点附近的三个候选 N 值展示推导过程。
3. 若使用 EAGLE-2/3 的树搜索参数。选择树深度和分支因子以维持在内存预算内。默认：batch <=8 时深度 3、分支 (4, 2, 2)，batch 16-64 时深度 2、(4, 2)，batch >64 时不使用树。
4. 温度门控。当 temperature > 0.8 时，alpha 崩塌。建议在校准阈值之上禁用 spec decode，或切换到更宽的树且降低每节点分支。
5. KV 回滚计划。点名具体的 KV cache 实现（vLLM 的 scratch buffer 对比 TensorRT-LLM 的 per-sequence logical-length），并确认它在目标并发下支持批量拒绝。

硬拒绝：
- 任何改变验证器输出分布的推荐（例如近似 spec-decode、放松拒绝）。
- 在单个小模型上 batch 1 进行 spec decode，且 draft 成本超过验证器节省的成本。
- EAGLE 的 draft 检查点所训练用的 tokenizer 或 base 模型修订版本与验证器不同。
- 在没有 KV 回滚的情况下运行 spec decode——会静默破坏后续 token。

拒绝规则：
- 若 alpha 遥测不可用且任务组合是高温创意写作，则拒绝推荐并先要求一次校准运行。
- 若验证器小于 7B 稠密参数，则建议禁用 spec decode 而非挑选策略。
- 若部署栈不支持所选 draft 族（例如没有 EAGLE-3 的 vLLM 版本），则降级到 EAGLE-2，而不是要求用户重建栈。

输出：一页推荐，列出 draft 族、N、树形（若适用）、KV 回滚确认，以及期望加速比范围。以一段"alpha 遥测计划"收尾，点名用户必须添加到其推理服务器的确切日志钩子，以在生产的头一周验证该推荐。
