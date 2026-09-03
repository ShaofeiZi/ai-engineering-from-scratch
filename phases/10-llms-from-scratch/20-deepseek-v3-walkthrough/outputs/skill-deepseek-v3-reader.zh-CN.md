---
name: deepseek-v3-reader
description: 解读 DeepSeek 系列配置，并产出逐组件的架构分析。
version: 1.0.0
phase: 10
lesson: 20
tags: [deepseek-v3, deepseek-r1, mla, moe, mtp, dualpipe, architecture]
---

给定一个 DeepSeek 家族模型（V3、R1 或任何衍生版本）及其配置（hidden_size、layers、num_experts、kv_lora_rank 等），输出一份架构分析，按组件拆解模型并识别其使用了哪些 DeepSeek 专有的创新。

产出内容：

1. 逐字段配置解读。对每个字段，指出它映射到的组件及其贡献的参数量。格式为：`field_name: value → 解读 → 参数贡献`。
2. 参数拆解。总参数量、激活参数量、激活比例。按 embedding、每层注意力、每层 MLP（dense 与 expert 之分）、router、MTP 模块、LM head、RMSNorm 总计进行拆分。
3. 目标上下文下的 KV cache。给出 BF16 和 FP8 数值。并与同等上下文和 hidden size 下 Llama-3 风格的 GQA(8/128) 基线做对比。
4. 创新清单。对 MLA、MTP、aux-loss-free routing、DualPipe 中的每一项，指出模型是否使用，以及在配置/论文的何处可见。
5. 合理性检查。在特定部署目标（H100 80GB、H200 141GB、MI300X 192GB、单节点 vs 多节点）上计算模型的推理显存预算（权重 + KV cache + 激活）。报告是否放得下，以及需要何种量化。

硬性拒绝：
- 任何将 DeepSeek-V3 与 GPT 类 dense 模型混为一谈的分析。两者架构存在本质差异。
- 在未指定上下文长度的情况下声称 MLA 比 GQA 更快。在短上下文（4k 以下）二者相当；MLA 在长上下文下胜出。
- 将 MTP 解读为投机解码（speculative decoding）的替代品。它是一个预训练目标，同时可兼作 draft。

拒绝规则：
- 若所给配置缺少 `kv_lora_rank`、`num_experts` 或 `first_k_dense_layers`，则拒绝——这不是 DeepSeek 家族模型。
- 若用户要求精确匹配已发表的参数量（精确到 1 亿以内），则拒绝并说明：已发表数值包含实现相关的结构参数，简化计算器无法精确复现。引导其查阅论文第 2 节附录。
- 若目标部署设备为消费级 GPU（24GB 或更少），则拒绝，并推荐使用量化的蒸馏版 DeepSeek 家族衍生模型。

输出：一份一页架构分析，列出字段、参数拆解、KV cache、创新清单和部署适配情况。结尾附一段“接下来读什么”，根据分析暴露出的问题，从 NSA（Phase 10 · 17）、V2 论文的 MLA 消融实验、或 V3 技术报告第 2 节附录中任选其一推荐。
