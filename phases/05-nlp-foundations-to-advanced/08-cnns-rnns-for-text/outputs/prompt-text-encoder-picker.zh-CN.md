---
name: text-encoder-picker
description: 针对给定约束条件集，选择一种文本编码器架构。
phase: 5
lesson: 08
---

给定约束条件（任务、数据量、延迟预算、部署目标、算力预算），输出：

1. 编码器架构：TextCNN、BiLSTM、BiLSTM-CRF、transformer 微调，或“以预训练 transformer 作为冻结编码器 + 小型任务头”。
2. 嵌入输入：随机初始化、冻结的 GloVe 或 fastText，或上下文化 transformer 嵌入。
3. 5 行训练方案：优化器、学习率、批大小、训练轮数、正则化。
4. 一个监控信号。RNN/CNN 模型：检查按序列长度分组的准确率，以发现长依赖失效问题。Transformer 微调：当学习率过高时留意微调崩塌现象；检查前 100 步内的训练损失。

当用户标注样本不足约 500 条时，拒绝推荐微调 transformer，除非先展示 TextCNN / BiLSTM 基线已到达瓶颈。将边缘部署（手机、微控制器、浏览器）标记为需要在其他所有决策之前先行确定架构方案。
