---
name: skill-fine-tuning-guide
description: 判断何时以及如何使用 LoRA 和 QLoRA 微调 LLM 的决策树
version: 1.0.0
phase: 11
lesson: 8
tags: [fine-tuning, lora, qlora, peft, llm-engineering]
---

# 微调决策指南

在微调之前，按以下顺序依次尝试：

```
1. Prompt engineering (minutes, $0)
2. Few-shot examples in prompt (minutes, $0)
3. RAG for knowledge retrieval (days, $10-100/month)
4. Fine-tuning with LoRA/QLoRA (days, $5-50 per experiment)
5. Full fine-tuning (weeks, $100-10,000 per run)
```

仅当前一步实测不足时，才进入下一步。

## 何时微调

- 模型需要一种提示词无法实现的、一致的输出风格或格式
- 你在蒸馏更大的模型（从 8B 模型得到 GPT-4 级别质量）
- 延迟敏感，且 few-shot 示例增加了过多 token
- 你需要模型可靠地遵循某种复杂推理模式
- 你拥有 1,000 条以上高质量的输入-输出行为样本

## 何时不微调

- 用合适的提示词，模型已经能做到你想要的效果
- 你需要模型掌握事实（应使用 RAG）
- 训练样本少于 500 条（容易过拟合）
- 任务频繁变化（重新训练成本高）
- 你需要审计某条特定输出受哪些数据影响（微调是黑盒）

## 方法选择

| GPU VRAM | 7B 模型 | 13B 模型 | 70B 模型 |
|----------|----------|-----------|-----------|
| 16GB (T4) | QLoRA | 不可行 | 不可行 |
| 24GB (3090/4090) | QLoRA 或 LoRA | QLoRA | 不可行 |
| 40GB (A100) | LoRA 或 Full | QLoRA 或 LoRA | QLoRA |
| 80GB (A100/H100) | Full | LoRA 或 Full | QLoRA 或 LoRA |

## LoRA 配置清单

1. 从 r=16, alpha=32 开始（大多数任务的安全默认值）
2. 先以 q_proj 和 v_proj 为目标（最小可用 LoRA）
3. QLoRA 使用学习率 2e-4，LoRA fp16 使用 5e-5
4. 设置 lora_dropout=0.05
5. 训练 1-3 个 epoch（更多则有 过拟合风险）
6. 每 100 步在留出集上评估一次
7. 保存检查点，按评估损失选取最佳

## 常见错误

- 训练轮数过多（小数据集在 2-3 个 epoch 后即过拟合）
- 使用与全量微调相同的学习率（LoRA 需要更高的学习率）
- 忘记设置 pad token（在 Llama 系列模型上会导致 NaN 损失）
- 未冻结基座模型（违背 LoRA 的初衷）
- 只在训练数据上评估（应始终留出 10-20% 用于评估）
- 跳过提示词工程基线（去微调一个提示词本已解决的问题）

## 质量验证

训练完成后，在 200 条以上留出样本上对比：
1. 带最佳提示词的基座模型（基线）
2. 带有 LoRA 适配器的基座模型（你微调后的模型）
3. 带相同提示词的 GPT-4 或 Claude（上限）

如果 LoRA 模型未能超过提示词基线，需要改进的是你的训练数据或配置，而不是更多算力。

## 适配器管理

- 多任务服务时保持适配器独立（按请求切换适配器）
- 单任务部署时将适配器合并进基座权重
- 将适配器存放在 Hugging Face Hub（10-100MB，便于版本管理与共享）
- 部署前测试合并后模型的输出与未合并时一致
- 使用 TIES-Merging 或 DARE 将多个适配器合并为一个

## 训练调试

如果损失不下降：
1. 检查学习率（对 LoRA 而言可能过低，尝试 2e-4）
2. 确认 LoRA 层确实在接收梯度
3. 确认基座模型权重已冻结
4. 检查数据格式（分词器必须与模型期望的格式匹配）

如果损失下降但评估质量差：
1. 训练数据质量问题（垃圾进，垃圾出）
2. 过拟合（减少 epoch，增大 dropout，增加数据）
3. 目标模块选错（复杂任务可加入 MLP 层）
4. 秩过低（尝试 r=32 或 r=64）
