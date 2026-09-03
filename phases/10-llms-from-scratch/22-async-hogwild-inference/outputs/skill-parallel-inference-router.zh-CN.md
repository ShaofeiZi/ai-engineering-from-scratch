---
name: parallel-inference-router
description: 在 voting、tree-of-thought、multi-agent、Hogwild! 和投机解码策略之间路由推理工作负载。
version: 1.0.0
phase: 10
lesson: 22
tags: [parallel-inference, hogwild, speculative-decoding, tree-of-thought, multi-agent, reasoning]
---

给定一份推理工作负载 profile（每任务 token 预算、任务并行特征、模型家族、部署目标、延迟预算），推荐一种或一组并行推理策略。

产出内容：

1. 任务分类。长推理（5k+ token）、中等思维链（1k-5k）、短对话（1k 以下）、或分类。决定第一轮筛选。
2. 并行轴。序列内（投机解码）vs 跨序列（voting、Hogwild!、multi-agent）。多数工作负载优先从序列内轴获益。
3. 策略推荐。从以下中选择：仅投机解码（任何 100 token 以上工作负载的安全默认）、投机解码 + Hogwild!（具有可并行结构的长期推理）、tree-of-thought（显式分支剪枝问题）、multi-agent（角色可专责化的问题）、voting ensemble（高风险分类）。
4. 参数设置。对投机解码：draft 家族（默认 EAGLE-3）与 `N`（Phase 10 · 15 skill）。对 Hogwild!：worker 数 N（2 到 4，极少更多）、协调 prompt 模板、单节点部署确认。
5. 组合加速估算。若将投机解码与 Hogwild! 组合，报告乘性加速（典型范围：3x spec * 1.5-2x Hogwild! = 4.5-6x）。

硬性拒绝：
- 对任何 2000 token 以下工作负载使用 Hogwild!。协调开销会喧宾夺主。
- 在非推理模型上使用 Hogwild!（无涌现式协调）。
- 对没有自然角色分解的问题使用 multi-agent 框架。
- 在没有显式分支剪枝逻辑时使用 tree-of-thought（否则策略退化为线性 CoT）。
- 跨节点运行 Hogwild!（跨节点 cache 同步太慢）。

拒绝规则：
- 若工作负载属于实验性研究，推荐将 Hogwild! 作为实验而非生产押注。其加速因任务而异，截至 2026 年 4 月真实部署仍属罕见。
- 若用户要求保证加速，则拒绝并说明：只有投机解码具有强保证性质（输出分布保持不变）。Hogwild! 是经验性的。
- 若用户 VRAM 有限，拒绝 Hogwild! N>2——每个 worker 即便共享 cache，仍需各自的激活显存。

输出：一份一页推荐，列出任务分类、并行轴、策略、参数、以及组合加速估算。结尾附一段“回滚触发器”，指出在前 100 条生产请求中，若 Hogwild! 未兑现收益，哪项具体延迟或精度指标足以构成退回仅投机解码的理由。
