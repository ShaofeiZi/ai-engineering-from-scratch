# 生产环境中的 EAGLE-3 推测解码

> Speculative decoding 会把一个快速 draft model 和目标模型配对使用。draft 先提出 K 个 token，target 再用一次 forward 统一验证；被接受的 token 相当于“白拿”。到了 2026 年，EAGLE-3 已经是生产级实现：它不是在原始 token 上训练一个小草稿模型，而是在目标模型的 hidden states 上训练 draft head，因此在通用对话流量上，acceptance rate alpha 可以稳定进入 0.6 到 0.8 这一档。真正该问的问题不是“draft 有多快”，而是“在我的真实流量上 alpha 到底是多少”。如果 alpha 掉到大约 0.55 以下，那么在高并发场景里，speculative decoding 往往会变成净负收益，因为每次 draft 被拒都会逼出额外的一次 target forward。本课会先教你测 alpha，再教你决定要不要开这个开关。

**Type:** 学习
**Languages:** Python（标准库，玩具级接受率模拟器）
**Prerequisites:** 第 17 阶段 · 04（服务引擎内部原理），第 10 阶段 · 18（多词元预测）
**Time:** 约 60 分钟

## 学习目标

- 说出 speculative decoding 的三代演进，并解释 EAGLE-3 相比 EAGLE-2 和经典草稿模型方案到底改了什么。
- 定义 acceptance rate alpha，能够从 alpha 和 K，也就是 draft length，计算期望加速比，并识别你所在并发点下的 break-even alpha。
- 解释为什么 speculative decoding 在 2026 年的 vLLM 里是 opt-in，而不是默认开启，以及为什么不测 alpha 就直接上线是一种生产反模式。
- 写出一份测量计划，明确该用什么 benchmark、什么 prompt 分布、什么并发点，以及最终要用哪个指标做 gate。

## 问题

Decode 是 memory-bound 的。在一张 H100 上运行 Llama 3.3 70B FP8 时，每生成一个 token，大约都要读取 140 GB/s 的权重，并只产出 1 个 token。也就是说，decode 阶段 GPU 的算力其实几乎没被吃满，真正的瓶颈是 HBM 带宽，而不是 matmul 吞吐。

Speculative decoding 正是利用了这个空档。它先让一个便宜的 draft model 生成 K 个候选 token，再让目标模型用一次 forward 去验证这 K 个候选。每一个通过验证的 token，本质上都相当于“免费”，因为它们被摊进了 target 本来就必须做的那次 batch-of-K forward 里。

经典 draft-model 路线会使用同家族的更小模型，比如用 Llama 3.2 1B 给 Llama 3.3 70B 打草稿。它确实能工作，但 acceptance rate 往往不够高，因为小模型分布和 target 模型的分布偏差比较大。EAGLE，再到 EAGLE-2，再到 EAGLE-3，走的是另一条路：直接在目标模型的内部状态上训练一个轻量 draft head，让草稿分布更贴近 target。也正因如此，alpha 才会从经典 draft model 的 0.4 左右，提升到 EAGLE-3 在通用聊天里的 0.6 到 0.8。

但有个前提：EAGLE-3 在 2026 年的 vLLM 里不是默认打开的，必须显式设置 `speculative_config`。不开这个配置，就没有任何加速。很多团队的问题恰恰出在这里：他们没有先测真实流量上的 alpha，就直接把 spec decode 打开，结果看到的不是收益，而是尾延迟恶化。

## 概念

### 推测解码到底带来了什么

不开 spec decode 时，每个 token 的成本就是一次 target forward。打开 spec decode 后，如果 draft length 是 K，acceptance rate 是 alpha，那么每次 target forward 理论上能摊出 `1 + K * alpha` 个 token。于是加速比大致是 `(1 + K * alpha) / (1 + epsilon)`，其中 epsilon 是 draft 和 verify 带来的额外开销。比如 K=5、alpha=0.7 时：`(1 + 5*0.7) / (1 + 0.1) = 4.5 / 1.1 = 4.1x`。不过真实世界里更常见的是 2 到 3 倍，因为生产流量上的 alpha 很少这么理想，而且高 batch 下 epsilon 也会变大。

### 为什么 alpha 才是唯一真正重要的指标

被拒掉的 token 不会凭空消失，它们会迫使 target 针对第一个被拒 token 再做一次 forward。假如某类工作负载上的 alpha 降到了 0.4，那你就同时承担了 draft 开销、verify 开销和 reroll 开销。到了高并发，比如 256 并发时，decode batch 本身已经足够大，target-only 和 target-plus-verify 之间的带宽摊薄差距会缩小。于是低 alpha 场景下，spec decode 很容易从正收益变成负收益。对 2026 年的大多数硬件来说，alpha 低于 0.55 时通常就已经危险了。

而 alpha 又高度依赖流量分布。在 ShareGPT 风格的通用聊天上，用 ShareGPT 训练出来的 EAGLE-3 往往能到 0.6 到 0.8。可一旦切到代码、医疗、法律这类垂直领域，基于通用数据训练的 draft head 就可能掉到 0.4 到 0.6。此时的正确方向通常不是硬扛，而是为这个领域再训练一个专用 draft head。相比 target finetuning，这仍然是一项相对轻量、快速的训练任务。

### 几代 EAGLE 的区别

- **经典草稿模型**：同家族的小模型做草稿。alpha 常见在 0.3 到 0.5。基础设施相对简单，你只需要同时加载两个模型，然后让 draft 在每次 target forward 之前先跑 K 次。
- **EAGLE-1 (2024)**：在 target 的 hidden states 上训练单个 draft head，通常基于最后一层。alpha 大致在 0.5 到 0.6。相对 target 本体，只增加少量参数。
- **EAGLE-2 (2025)**：引入自适应 draft length 和基于树的草稿结构，允许 target 一次验证多个分支。alpha 大致在 0.6 到 0.7，但 draft scheduler 更复杂。
- **EAGLE-3 (2025-2026)**：draft head 不只看最后一层，而是对多层 target states 做训练，对齐效果更好。在通用聊天场景下，alpha 常见于 0.6 到 0.8。

### 2026 年的生产配方

1. 先用纯 target 上线，测出目标并发点下的基线 TTFT、ITL 和吞吐。
2. 通过 vLLM 的 `speculative_config` 打开 EAGLE-3 draft，然后完整重跑 benchmark。
3. 记录 acceptance rate alpha。vLLM V1 会通过 `spec_decode_metrics.accepted_tokens_per_request` 暴露这个值。用它除以请求中的 draft length，就能得到 alpha。
4. 如果生产流量分布上的 alpha 低于 0.55，就关掉 spec decode，或者为该流量训练领域专用的 EAGLE-3 draft。
5. 在生产并发点重新复测，确认 P99 ITL 没有变差。

### 生产里的坑：P99 尾延迟

Spec decode 往往会拉低平均 ITL，但如果你不做调优，P99 可能反而会变坏。原因在于，被拒的 draft 会触发两段式流程：draft、verify-fail、再 reroll。满 batch 时，这两段流程会串行堆叠。因此你不能只看 P50，也不能只看平均值，必须盯住 P99 ITL。

### EAGLE-3 已经在哪里落地

Google 在 2025 年已经把 speculative decoding 用在 AI Overviews 里，实现了同样质量、但响应更快的效果。vLLM V1 也已经把 `speculative_config` 作为正式接口暴露出来；其中和 chunked prefill 兼容的，是 V1 里的 N-gram GPU speculative decoding。SGLang 则把 EAGLE-3 作为 prefix-heavy 工作负载下推荐的 draft 路径。

### 一行看懂 break-even 公式

期望加速比是 `S(alpha, K) = (1 + K*alpha) / (1 + verify_overhead)`。令 `S = 1`，就能解出 `alpha_breakeven = verify_overhead / K`。如果 verify_overhead ~0.15 且 K=5，那么原始公式会给出 `alpha_breakeven = 0.03`。但这只是理想化的 decode 数学。在高并发下，verify overhead 会上升，而 decode batch 本身也已经在多个序列之间摊薄了内存读取成本，所以真正有效的 alpha_breakeven 在实践里会升到大约 0.45 到 0.55。

### 什么时候不要用推测解码

- Batch-1 的离线生成，延迟并不重要，这时直接用纯 target 更简单。
- 输出特别短的场景，比如不到 50 tokens，draft 与 verify 的固定开销会占主导。
- 专业领域流量但没有对应领域训练过的 draft head，这时 alpha 往往太低。
- vLLM v0.18.0 下，把 draft-model speculative decoding 和 `--enable-chunked-prefill` 一起打开。这种组合跑不起来。文档里写明的例外是 V1 里的 N-gram GPU spec decode。

```figure
mx-speculative-tree
```

## 用起来

`code/main.py` 会模拟一条 decode loop，比较开启和不开启 speculative decoding 时，在不同 alpha 值和不同 draft length K 下的表现。它会打印 break-even alpha、实测加速比，以及尾延迟行为。你可以手动尝试几组 (alpha, K) 组合，直观看到 speculative decoding 从“划算”变成“不划算”的分界点。

## 交付物

本课会产出 `outputs/skill-eagle3-rollout.md`。给定目标模型、流量分布描述和目标并发，它会生成一份分阶段的 EAGLE-3 rollout 方案：先测 baseline，再打开配置，再测 alpha，以 alpha >= 0.55 作为是否继续推进的门槛，同时监控 P99 ITL。

## 练习

1. 运行 `code/main.py`。当 K=5 时，要实现 2x 加速需要多高的 alpha？3x 又需要多少？这个结果对 verify_overhead 有多敏感？
2. 假设生产流量由 70% 的通用聊天和 30% 的代码场景构成。通用聊天在 ShareGPT 训练的 EAGLE-3 下 alpha 能到 0.7，而代码流量只有 0.4。混合之后的 alpha 是多少？此时 spec decode 还是净正收益吗？
3. 阅读 vLLM 的 `speculative_config` 文档。说出三种模式，也就是 draft model、EAGLE 和 N-gram，并指出哪一种与 chunked prefill 兼容。
4. 你打开 EAGLE-3 之后，平均 ITL 下降了 25%，但 P99 ITL 却上升了 15%。请诊断原因，并提出一种缓解办法。
5. 计算 Llama 3.3 70B 上 EAGLE-3 draft head 的显存成本。它和把 Llama 3.2 1B 当作经典 draft model 跑起来相比，代价差多少？

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------------|------------------------|
| 推测解码 | “草稿加验证” | 先用便宜模型提出 K 个 token，再用一次 target forward 验证全部 K 个 |
| 接受率 alpha | “spec 接受率” | 草稿 token 中被 target 接受的比例；这是最关键的指标 |
| Draft length K | “spec k” | 每次 target forward 之前，draft 会先提多少个 token；常见范围是 4 到 8 |
| 验证开销 epsilon | “spec 额外开销” | 相比纯 target forward，多出来的 verify 与 reroll 成本；会随 batch 增长 |
| EAGLE-3 | “最新一代 EAGLE” | 2025 到 2026 年的版本；在多个 target layer 上训练 draft head；通用聊天 alpha 可达 0.6 到 0.8 |
| `speculative_config` | “vLLM 的 spec 配置” | vLLM V1 中显式 opt-in 的接口；不配置就没有加速 |
| N-gram 推测解码 | “N-gram 草稿” | 在 GPU 侧基于 prompt 内 N-gram 查找来起草；兼容 chunked prefill |
| Break-even alpha | “不亏不赚的 alpha” | 使 spec decode 刚好没有加速收益的 alpha；生产并发下必须重点盯这个值 |
| Rejected-draft two-pass | “reroll 成本” | draft 被拒后需要两次 target 级处理；它是 P99 尾延迟的重要来源 |

## 延伸阅读

- [vLLM — Speculative Decoding 文档](https://docs.vllm.ai/en/latest/features/spec_decode/) — 关于 `speculative_config` 与 V1 下 chunked-prefill 兼容性的权威文档。
- [vLLM Speculative Config API](https://docs.vllm.ai/en/latest/api/vllm/config/speculative/) — 具体字段定义。
- [EAGLE paper (arXiv:2401.15077)](https://arxiv.org/abs/2401.15077) — EAGLE draft-head 方案的原始论文。
- [EAGLE-2 paper (arXiv:2406.16858)](https://arxiv.org/abs/2406.16858) — 关于自适应 draft 与树结构的论文。
- [UC Berkeley EECS-2025-224](https://www2.eecs.berkeley.edu/Pubs/TechRpts/2025/EECS-2025-224.html) — 讨论带 speculative decoding 的高效 LLM 系统。
- [BentoML — Speculative Decoding](https://bentoml.com/llm/inference-optimization/speculative-decoding) — 面向生产 rollout 的检查清单。
