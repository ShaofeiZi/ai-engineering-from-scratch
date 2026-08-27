# 综合项目 07——端到端微调流水线（从数据到 SFT，再到 DPO 与上线服务）

> 用自己的数据训练一个 8B 模型，再以自己的偏好数据完成 DPO 对齐，随后量化模型、启用推测解码，并以可衡量的 $/1M tokens（每百万词元成本）对外提供服务。2026 年的主流开源技术栈包括 Axolotl v0.8、TRL 0.15、用于快速迭代的 Unsloth、用于量化的 GPTQ/AWQ/GGUF，以及搭载 EAGLE-3 的 vLLM 0.7。本综合项目要求你可复现地跑通整条流水线：输入 YAML 配置，输出服务端点，并依据 2026 Model Openness Framework 发布模型卡。

**Type:** 综合项目
**Languages:** Python（流水线）、YAML（配置）、Bash（脚本）
**Prerequisites:** 第 2 阶段（机器学习）、第 3 阶段（深度学习）、第 7 阶段（Transformer）、第 10 阶段（从零构建 LLM）、第 11 阶段（LLM 工程）、第 17 阶段（基础设施）、第 18 阶段（安全）
**Phases exercised:** P2 · P3 · P7 · P10 · P11 · P17 · P18
**Time:** 35 小时

## 问题

到 2026 年，任何成熟的 AI 团队都会准备一条随时可用的微调流水线。这并不是因为他们要发布前沿基础模型，而是因为可衡量的收益通常来自后续适配：面向领域的 SFT、基于已标注偏好数据的 DPO、为推测解码蒸馏草稿模型，以及通过 EAGLE-3 提高服务吞吐量。Axolotl v0.8 负责多 GPU SFT 配置，TRL 0.15 支持 DPO 和 GRPO，Unsloth 适合单 GPU 快速迭代，而 vLLM 0.7 配合 EAGLE-3 可在不损失质量的前提下将解码吞吐量提升至原来的 2～3 倍。工具本身已经可用，真正考验工程能力的是 YAML 配置、数据卫生和严谨的评估流程。

你将选择一个基础模型（Llama 3.3 8B、Qwen3 14B 或 Gemma 3 12B），先用任务专用数据进行 SFT，再完成 DPO，然后为部署进行量化，并通过 lm-evaluation-harness、RewardBench-2、MT-Bench-v2 和 MMLU-Pro 衡量改进。你还要依据 2026 Model Openness Framework 编写模型卡。项目的核心是可复现性：只需一条命令，就能从头到尾重新运行整条流水线。

## 概念

整条流水线分为五个阶段。**数据阶段**：使用 MinHash / Datatrove 去重，使用 Nemotron-CC 风格分类器过滤低质量数据，清除 PII，并检查训练集与公开基准测试集之间是否存在数据污染。**SFT 阶段**：使用 Axolotl YAML，在 8xH100 上启用 ZeRO-3、余弦学习率调度和序列打包，训练 2～3 个 epoch。**DPO 或 GRPO 阶段**：使用 TRL 配置训练 1 个 epoch；偏好样本对可以由人工标注或模型评判，并需要调整 beta。**量化阶段**：同时生成 GPTQ、AWQ 和 GGUF 版本，以便灵活部署。**服务阶段**：使用带 EAGLE-3 推测头的 vLLM 0.7（也可选择 SGLang + SpecForge），部署到 Kubernetes，并根据 queue-wait 指标进行 HPA 自动扩缩容。

消融实验才是关键交付物：比较 SFT-only、SFT+DPO 和 SFT+GRPO 在三个任务专用基准上的表现。服务侧需要报告批大小为 1、8、32 时的 tokens/s、EAGLE-3 接受率，以及 $/1M tokens（每百万 token 成本）；安全评估需要报告 Llama Guard 4 通过率；模型卡则应记录偏差评估、复现用随机种子和数据许可证。

## 架构

```
raw data (HF datasets + internal)
    |
    v
Datatrove dedup + Nemotron-CC quality filter + PII scrub
    |
    v
split hygiene (MMLU-Pro contamination check)
    |
    v
Axolotl SFT config (YAML)  ---> 8xH100, ZeRO-3
    |
    v
TRL DPO / GRPO config       ---> 4xH100, 1 epoch
    |
    v
GPTQ + AWQ + GGUF quantize
    |
    v
vLLM 0.7 + EAGLE-3 speculative decoding
    |
    v
K8s deployment, HPA on queue-wait
    |
    v
lm-eval-harness + RewardBench-2 + MT-Bench-v2 + MMLU-Pro
    |
    v
model card (2026 MOF) + safety eval (Llama Guard 4)
```

## 技术栈

- 数据：Datatrove 用于去重，Nemotron-CC 分类器用于质量过滤，Presidio 用于 PII 清洗
- 基础模型：Llama 3.3 8B、Qwen3 14B 或 Gemma 3 12B
- SFT：Axolotl v0.8，配合 ZeRO-3、Flash Attention 3 和序列打包
- 偏好调优：TRL 0.15 用于 DPO 或 GRPO；Unsloth 用于单 GPU 快速迭代
- 量化：GPTQ（Marlin）、AWQ，以及通过 llama.cpp 生成 GGUF
- 服务：vLLM 0.7 + EAGLE-3 推测解码，或 SGLang 0.4 + SpecForge
- 评估：lm-evaluation-harness、RewardBench-2、MT-Bench-v2、MMLU-Pro
- 安全评估：Llama Guard 4、ShieldGemma-2
- 基础设施：Kubernetes + NVIDIA 设备插件，基于 queue-wait 指标进行 HPA 自动扩缩容
- 可观测性：训练阶段用 W&B，推理阶段用 Langfuse

```figure
ce-finetune-stages
```

## 动手构建

1. **数据流水线。** 对原始语料运行 Datatrove 去重，应用 Nemotron-CC 风格的质量分类器，用 Presidio 清除 PII，并使用明确的随机种子划分训练集和验证集。

2. **污染检查。** 对每个验证集切分，使用 MinHash 检查它与 MMLU-Pro、MT-Bench-v2、RewardBench-2 测试集是否重叠；发现任何重叠都必须拒绝该切分。

3. **Axolotl SFT。** 编写启用 ZeRO-3、FA3 和序列打包的 YAML，在 8xH100 上训练 2～3 个 epoch，并将训练日志记录到 W&B。

4. **TRL DPO / GRPO。** 从 SFT 检查点开始，使用偏好样本对训练 1 个 epoch 的 DPO；对于数学或代码任务，也可以改用带可验证奖励的 GRPO。需要对 beta 进行参数扫描。

5. **量化。** 同时产出三种量化版本：GPTQ-INT4-Marlin、AWQ-INT4，以及供 llama.cpp 使用的 GGUF-Q4_K_M。记录各版本的大小和标称吞吐量。

6. **使用推测解码提供服务。** 配置 vLLM 0.7，接入由 Red Hat Speculators 训练的 EAGLE-3 草稿头。测量批大小为 1、8、32 时的接受率和尾延迟，并在同一评估任务上比较其 $/1M tokens（每百万 token 成本）与 Anthropic / OpenAI 的成本。

7. **评估矩阵。** 分别在基础模型、SFT-only、SFT+DPO 和 SFT+GRPO 四个版本上运行 lm-evaluation-harness、RewardBench-2、MT-Bench-v2 与 MMLU-Pro，并将结果整理成表格。

8. **安全评估。** 统计开发集上的 Llama Guard 4 通过率，并使用 ShieldGemma-2 过滤输出。

9. **模型卡。** 按照 2026 MOF 模板生成模型卡，覆盖数据、训练、评估、安全和许可证，并在可复现性部分提供 YAML 与 commit SHA。

## 实际运行

```
$ ./pipeline.sh config/llama3.3-8b-domainX.yaml
[data]    300k deduped, 12k filtered, 280k accepted (seed=7)
[SFT]     3 epochs, 8xH100, 6h12m, val loss 1.42 -> 1.03
[DPO]     1 epoch, beta=0.08, 4xH100, 1h40m
[quant]   GPTQ-INT4 4.6 GB, AWQ-INT4 4.8 GB, GGUF-Q4_K_M 5.1 GB
[serve]   vLLM 0.7, EAGLE-3 acceptance 0.74, p99 126ms @ bs=8
[eval]    MMLU-Pro +3.2, MT-Bench-v2 +0.41, RewardBench-2 +0.08
[card]    model-card.md generated under 2026 MOF
```

## 交付成果

`outputs/skill-finetuning-pipeline.md` 描述了最终交付物。一条命令即可让数据依次经过 SFT、DPO、量化、服务部署和评估，并产出模型卡与可调用的服务端点。

| 权重 | 评分标准 | 衡量方式 |
|:-:|---|---|
| 25 | 相对基础模型的评估增益 | 在目标任务上带来的可测提升（MMLU-Pro、MT-Bench-v2、任务专项基准） |
| 20 | 流水线可复现性 | 使用相同的随机种子，一条命令即可端到端重跑 |
| 20 | 数据卫生 | 去重率、PII 清洗覆盖率、污染检查全部通过 |
| 20 | 服务效率 | bs=1/8/32 时的 tokens/s、EAGLE-3 接受率、$/1M tokens |
| 15 | 模型卡与安全评估 | 2026 MOF 完整度与 Llama Guard 4 通过率 |
| **100** | | |

## 练习

1. 在同一个任务专项基准上，对比 SFT-only、SFT+DPO、SFT+GRPO。报告哪种偏好优化方法最好，以及领先了多少。

2. 把 Llama 3.3 8B 换成 Qwen3 14B，在质量相当的前提下测量每百万 token 的成本。

3. 对比 EAGLE-3 在领域数据与通用 ShareGPT 数据上的接受率。报告差值，并解释它对延迟预算意味着什么。

4. 人为向训练数据注入 1% 的污染，例如泄漏 MMLU-Pro 的答案，然后重新运行评估。观察 MMLU-Pro 准确率是否异常跃升，并构建一道能检出此问题的数据污染 CI 门禁。

5. 增加 LoRA SFT 作为全量微调的替代方案，在显存用量降至约十分之一时测量质量差距。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|-----------------|------------------------|
| Axolotl | “SFT 训练器” | 由 YAML 统一配置的训练器，支持 SFT、DPO 和蒸馏 |
| TRL | “偏好调优库” | Hugging Face 提供的 LLM 偏好优化库，支持 DPO、GRPO、PPO |
| GRPO | “群体相对策略优化” | DeepSeek R1 使用的、依赖可验证奖励的 RL 配方 |
| EAGLE-3 | “推测解码草稿头” | 草稿头提前预测 N 个 token，再由目标模型验证并决定是否接受 |
| MOF | “Model Openness Framework” | 2026 年依据数据、代码和许可证评定模型发布开放程度的标准 |
| 污染检查（Contamination check） | “切分卫生检查” | 基于 MinHash 的测试集泄漏检测机制 |
| 接受率（Acceptance rate） | “EAGLE / MTP 指标” | 草稿模型提出的词元中，被目标模型接受的比例 |

## 延伸阅读

- [Axolotl documentation](https://axolotl-ai-cloud.github.io/axolotl/) — SFT / DPO 训练器参考文档
- [TRL documentation](https://huggingface.co/docs/trl) — DPO 与 GRPO 的官方参考实现
- [Unsloth](https://github.com/unslothai/unsloth) — 单 GPU 快速迭代参考
- [DeepSeek R1 paper (arXiv:2501.12948)](https://arxiv.org/abs/2501.12948) — GRPO 方法论来源
- [vLLM + EAGLE-3 documentation](https://docs.vllm.ai) — 参考级服务栈文档
- [SGLang SpecForge](https://github.com/sgl-project/SpecForge) — 另一套推测解码训练方案
- [Model Openness Framework 2026](https://isocpp.org/) — 开源模型发布评分标准
- [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) — 标准评估运行器
