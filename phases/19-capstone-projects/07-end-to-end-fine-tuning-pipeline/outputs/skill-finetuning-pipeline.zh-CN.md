---
name: finetuning-pipeline
description: 运行一条可复现的从数据到 SFT 到 DPO 到部署的微调流水线，包含消融实验、量化，以及 2026 模型开放性框架模型卡。
version: 1.0.0
phase: 19
lesson: 07
tags: [capstone, fine-tuning, axolotl, trl, dpo, grpo, vllm, eagle-3, mof]
---

给定一个基础模型（Llama 3.3 8B、Qwen3 14B 或 Gemma 3 12B）和一个特定任务数据集，构建一条单命令流水线，产出已部署的推理端点和可复现的模型卡。

构建计划：

1. 数据阶段：Datatrove 去重、Nemotron-CC 风格质量过滤、Presidio PII 清洗、带种子的训练/验证集划分。
2. 污染检查：使用 MinHashLSH 对比 MMLU-Pro、MT-Bench-v2、RewardBench-2。发现重叠则拒绝。
3. SFT：Axolotl v0.8，ZeRO-3，Flash Attention 3，序列打包，8xH100 上训练 2-3 个 epoch。
4. 偏好调优：TRL 0.15 DPO（或使用可验证奖励的 GRPO），1 个 epoch，beta 参数扫描。
5. 量化：GPTQ-INT4-Marlin + AWQ-INT4 + GGUF-Q4_K_M。
6. 部署：vLLM 0.7 搭配 EAGLE-3 投机解码（草稿头通过 Red Hat Speculators 或 SGLang SpecForge 提供）。K8s 部署，基于队列等待时间配置 HPA。
7. 评估：lm-evaluation-harness、RewardBench-2、MT-Bench-v2、MMLU-Pro，覆盖 base/SFT-only/SFT+DPO/SFT+GRPO。
8. 安全：Llama Guard 4 通过率，ShieldGemma-2 输出过滤。
9. 按 2026 模型开放性框架编写模型卡，包含数据、训练、评估、安全、可复现性等章节。

评分标准：

| 权重 | 评估维度 | 测量方式 |
|:-:|---|---|
| 25 | 相对基线的评估增益 | 在 MMLU-Pro、MT-Bench-v2、任务特定基准上的实测提升 |
| 20 | 流水线可复现性 | 相同种子下单命令重跑产出匹配的哈希值 |
| 20 | 数据卫生 | 去重率、PII 清洗覆盖率、污染检查通过 |
| 20 | 部署效率 | batch 1/8/32 下的 tokens/s、EAGLE-3 接受率、$/1M tokens |
| 15 | 模型卡 + 安全评估 | 2026 MOF 完整性 + Llama Guard 4 通过率 |

一票否决项：

- 跳过 MinHash 污染检查的流水线。将 MMLU-Pro 泄漏到训练集中是经典的评估作弊失败模式。
- 未附带种子或 YAML 的训练运行。可复现性是硬性要求。
- 未使用 EAGLE-3 或等效投机解码配置的部署。基线 tokens/s 不满足 2026 年的标准。
- 缺失安全评估。每个微调模型都必须附带 Llama Guard 4 通过率。

拒绝规则：

- 拒绝发布在未附带 lm-eval-harness commit SHA 的情况下声称基准得分的模型卡。
- 拒绝在许可证禁止衍生模型的数据上进行微调。MOF 对数据许可进行评级。
- 拒绝在未于评估矩阵上测量质量损失的情况下发布量化模型。

交付物：一个仓库，包含流水线编排器、Llama 3.3 8B 及一个备选基座的 YAMLs、SFT 和 DPO 的 W&B 运行日志、量化产物、已部署的推理端点、三基准评估矩阵、安全评估、2026 MOF 模型卡，以及一篇关于你发现并修复的三个最大数据卫生问题的报告。
