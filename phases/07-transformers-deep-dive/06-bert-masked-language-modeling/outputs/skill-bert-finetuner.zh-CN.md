---
name: bert-finetuner
description: 为新的分类、抽取或检索任务规划 BERT 微调方案。
version: 1.0.0
phase: 7
lesson: 6
tags: [bert, fine-tuning, nlp]
---

给定一个下游任务（分类 / NER / 检索 / 重排 / NLI）、标注数据规模以及部署约束（延迟、设备），输出：

1. 主干网络选择。模型名称（ModernBERT-base / large、DeBERTa-v3、multilingual-e5 等），并附一句理由。对于需要 ≤8K 上下文的英文任务，优先选择 ModernBERT。
2. 头部设计。分类：`[CLS]` → dropout → linear(num_classes)。NER：逐 token 的 linear，可选 CRF。检索：mean-pool + 对比损失。
3. 训练配方。优化器（AdamW，lr 通常为 2e-5）、warmup 比例（6–10%）、epoch 数（3–5）、batch size、fp16/bf16。
4. 评估方案。与任务匹配的指标（分类用准确率 + F1，NER 用实体级 F1，检索用 MRR/NDCG）。留出验证集规模。
5. 失败模式检查。指出一个具名风险：标签泄露、类别不平衡、上下文截断、预训练与微调语料之间的分词器不匹配。

拒绝针对生成式输出（文本生成）微调 BERT——应建议改用 decoder-only 模型。当少数类占比低于 10% 时，拒绝在没有按类别分层的评估下上线微调模型。对于标注样本少于 1,000 却解冻整个主干网络的微调，标记为可能过拟合。
