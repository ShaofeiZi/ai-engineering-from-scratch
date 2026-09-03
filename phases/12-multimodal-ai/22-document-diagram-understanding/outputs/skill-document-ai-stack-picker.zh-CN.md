---
name: document-ai-stack-picker
description: 根据领域、规模和合规需求，在 OCR 流水线、无 OCR 专用模型和 VLM 原生方案之间为文档 AI 项目做出选择。
version: 1.0.0
phase: 12
lesson: 22
tags: [document-ai, ocr, donut, nougat, paligemma, vlm-native]
---

给定一个文档 AI 项目（领域：发票 / 科学论文 / 表单 / 混合；规模：每日页数；质量门槛；合规需求），选择一套技术栈并产出参考配置。

产出：

1. 技术栈选择。时代 1（OCR 流水线 + LayoutLMv3）、时代 2（Donut / Nougat 无 OCR）、时代 3（VLM 原生），或混合方案。
2. 每页成本估算。所选技术栈的 token 数量与延迟。
3. 精度预期。DocVQA + ChartQA + 领域专属基准。
4. 手写策略。成本不敏感时用 VLM 原生；大规模时用专用 TrOCR + 路由。
5. 数学 / LaTeX 输出。科学论文用 Nougat；其他用 VLM。
6. 合规回退。混合方案，附交叉审计日志。

硬性拒绝：
- 在未做成本分析的情况下为 >1M 页/天推荐 VLM 原生方案。每页 2576px 的 token 成本不可忽视。
- 针对受监管工作流，在无审计路径的情况下推荐单模型方案。
- 声称 Nougat 能处理扫描发票。它做不到——它是科学论文专用模型。

拒绝规则：
- 若规模 >10M 页/天，则拒绝时代 3，推荐时代 1 并以时代 3 作为抽样校验器。
- 若领域以手写为主，则拒绝 OCR 流水线，推荐 VLM 原生 + 手写专用模型（TrOCR）。
- 若公式需要 LaTeX 保真度，则要求 Nougat 纳入流程。

输出：一页方案，包含技术栈、成本、精度、手写、数学、合规。以 arXiv 2308.13418（Nougat）、2204.08387（LayoutLMv3）、2111.15664（Donut）结尾。
