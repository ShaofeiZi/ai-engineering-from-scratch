---
name: speculative-tuning
description: 分析解码工作负载，并为投机解码选择 draft model、draft length K、temperature gate 和回退策略。
version: 1.0.0
phase: 10
lesson: 25
tags: [speculative-decoding, draft-model, alpha, throughput, inference, decode-latency]
---

给定目标模型（规模、家族、tokenizer）、工作负载遥测（任务构成、prompt 与 decode token 比例、p50/p99 decode 延迟、加速卡及 HBM 余量、平均 batch size、采样温度分布），以及可用的 draft 检查点，输出：

1. Draft 选择。从以下中选一：同家族小模型（Llama-70B 用 Llama-3.2-1B）、蒸馏 draft（Qwen3-0.6B-spec）、挂载在目标模型上的 Medusa heads、或在 FLOP 成本比差距大于 30% 时选择“不做 spec decode”。与目标逐字节确认 tokenizer 匹配；tokenizer 不匹配则拒绝。
2. Draft 长度 K。取 E[tokens] / (1 + K x c) 的 argmax，其中 c 为 draft 与目标的成本比。展示在 5_000 token 分布内数据上的校准运行所得的 measured alpha 下，K 为 2、3、4、5、6 时的推导过程。默认 K=4 用于对话、K=6 用于代码、K=2 用于高温创意写作。
3. 温度门控。设定一个温度阈值，超过则禁用 spec decode。默认 0.8；若校准显示 alpha 更早崩塌则下调至 0.6。拒绝任何依赖逐请求检查且增加超过 50 微秒的温度门控。
4. Tree 预算。若 serving 栈支持 tree drafting，batch 8 以下选一棵小型固定 tree（深度 2、分支 3-2）；batch 32 以上选扁平链。以字节给出 verifier 的 KV scratch 大小，并确认其能放入 HBM 余量。
5. Fallback 策略。指明指标（最近 1_000 次 verify 的滑动窗口 measured alpha）与阈值（alpha 低于 0.4），达到时服务器对该请求流退回普通自回归解码。并给出该 fallback 决策的逐请求有效期。

在 batch size 超过 verifier 计算受限的拐点时拒绝 spec decode。超过该拐点，speculator 本应吸收的空闲 FLOPs 已不复存在；吞吐反而下降。对任何 measured alpha 低于 0.4 的任务族拒绝 spec decode；draft 开销会喧宾夺主，wall-clock 延迟更差。拒绝任何未在留出的 1_000 token 样本上针对目标验证过的 draft：未经验证的 draft 是一次悄无声息的 KL 漂移。

示例输入："Llama-3.3-70B on 8xH100, chat workload, batch 16, p50 decode 28 ms, p99 60 ms, temperature distribution mean 0.4 / max 1.2, calibration shows alpha 0.78 on chat, 0.61 on code."

示例输出：
- Draft：Llama-3.2-1B-Instruct-spec。同 tokenizer、同家族，比率 c 约 0.03。
- K：4。E[tokens/verify] = 3.4 chat、2.5 code。K=5 在 chat 上仅多赚 0.1 token 却多付 0.03 c；拒绝。
- 温度门控：0.8。超过 0.8 时 alpha 在校准集上跌破 0.45。
- Tree 预算：深度 2 分支 (3, 2)。batch 16 下 KV scratch 480 MB，放得下。
- Fallback：最近 1_000 次 verify 的滑动窗口 alpha 低于 0.40 时，对该流禁用 spec decode 30 秒，随后再次探测。
