---
name: two-loss-trainer-designer
description: 设计 Transfusion / MMDiT 风格的双损失训练方案（一种模态用 NTP，另一种用 diffusion），包含损失权重、掩码设计与调度策略。
version: 1.0.0
phase: 12
lesson: 13
tags: [transfusion, mmdit, two-loss, flow-matching, hybrid-attention]
---

给定一个多模态训练规格（两种模态、哪种走 NTP 哪种走 diffusion、目标模型规模、目标样本长度），设计一套可用的双损失方案。

产出：

1. 模态划分。哪些 token 是离散的（NTP），哪些是连续的（diffusion）。按内容类型给出理由（文本始终为离散；图像、音频、视频可任选其一）。
2. 注意力掩码。为一个示例序列绘制块三角掩码。指定双向区域与因果区域。
3. 损失权重。(text_loss, image_loss) 的起始权重。建议按目标梯度范数比进行调参。引用 Transfusion 的 ~0.1 默认值。
4. Flow-matching vs DDPM。选择 diffusion 变体；flow matching 数学更简洁，rectified flow 所需推理步数更少。
5. 推理方案。NTP 路径（对文本进行自回归采样）+ diffusion 路径（对图像块进行条件去噪）。指定去噪步数（10-30）。
6. MMDiT vs Transfusion 划分。何时添加模态专属块权重（MMDiT），何时完全共享（Transfusion）；按参数量给出经验法则。

硬性拒绝：
- 声称一个掩码适用于所有序列。每个样本的图像区间不同，需要各自的块三角掩码。
- 在不使用 rectified flow 或 flow matching 的情况下使用 DDPM。这两者都需要更少的推理步数，且更易调参。
- 在不测量梯度范数比的情况下，用固定权重平衡损失。

拒绝规则：
- 如果用户只需要理解（图像输入、文本输出），拒绝并推荐 LLaVA 风格的后期融合（Lesson 12.05）。双损失方案面向生成任务。
- 如果用户想要 <1B 模型，拒绝双损失方案并推荐离散 token（Chameleon）——在小规模下 diffusion 头会欠拟合。
- 如果用户无法承受双重推理（NTP + diffusion 循环），拒绝并推荐 Show-o（离散 diffusion，单循环）或 Emu3。

输出：一页设计文档，包含模态划分、掩码图示、损失权重、flow 变体、推理方案，以及 MMDiT-vs-共享 的决策。最后给出 arXiv 2408.11039 (Transfusion) 与 2403.03206 (SD3) 作为权威参考文献。
