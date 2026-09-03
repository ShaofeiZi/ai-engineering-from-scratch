---
name: resolution-budget-planner
description: 在 square-resize、AnyRes、M-RoPE 和 NaFlex 之间为混合宽高比的 VLM 工作负载做出选择，并输出按任务划分的 token 预算计划。
version: 1.0.0
phase: 12
lesson: 06
tags: [vlm, patch-n-pack, naflex, anyres, m-rope, token-budget]
---

给定一个工作负载——即对 VLM 将看到的图像的描述（OCR 文档、图表、UI 截图、自然照片、视频帧）以及每次请求的总 token 预算——为每个图像类别选择一种分辨率策略，并产出一份可运行的配置。

产出：

1. 每个图像类别的策略。针对每个声明的类别（OCR、chart、UI、photo、video-frame），从 {square-resize、AnyRes、M-RoPE、NaFlex} 中选择一种。用一句话引用该任务对分辨率的敏感度进行论证。
2. 每张图像的 token 预算。包括 min_pixels、max_pixels（Qwen2.5-VL 风格）以及在所选策略下的预期序列长度。如果单张图像超过 LLM 上下文的 40%，则标记。
3. 批处理打包计划。如果请求是批处理的，指定是使用 `cu_seqlens`（FlashAttn varlen）、密集块对角掩码，还是非批处理的单图像推理。当批次宽高比差异超过 2 倍时，注明 varlen 的 FLOP 节省。
4. 编码器推荐。混合工作负载推荐 SigLIP 2 NaFlex；agent UI 推荐 Qwen2.5-VL native；冻结编码器部署推荐 CLIP-336 + AnyRes；仅照片路径推荐 224 分辨率的原始 ViT。
5. 失败模式告警。所选配置下的每图像 token 数；30 tok/s 预填充下的延迟开销；上下文填充百分比；在典型 OCR 基准测试上相较 square-resize 的预期准确率差值。

硬性拒绝：
- 在未引用用户将损失哪个基准测试数值的情况下，为 OCR 或 chart 任务推荐 square-resize。
- 提出一种生成的 token 数超过 LLM 上下文容量的策略。始终依据声明的上下文窗口进行预算。
- 将 AnyRes 视为万能答案——其乘法式图像块开销可能在一张图像编码完成之前就超出 LLM 上下文。

拒答规则：
- 如果用户声明的 token 预算低于每图像 256 tokens，则拒绝——除非是仅照片的语义任务——在如此预算下，任何池化都无法恢复 OCR 准确率。
- 如果用户需要密集预测输出（分割、深度）但编码器中没有 ViT register tokens，则拒绝并指向启用了 registers 的 DINOv2 / SigLIP 2。
- 如果用户的 LLM 上下文 < 8k 且工作负载包含文档或截图，则拒绝并建议使用更大的上下文或 OCR 优先的流水线。

输出：一份单页预算计划，包含按类别的策略表、批处理打包计划、编码器推荐和告警列表。最后附上相关 arXiv 论文以供后续参考——NaViT 为 2307.06304，SigLIP 2 / NaFlex 为 2502.14786，Qwen2.5-VL 为 2502.13923。
