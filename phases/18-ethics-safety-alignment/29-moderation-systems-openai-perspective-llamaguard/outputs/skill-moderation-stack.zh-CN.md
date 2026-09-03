---
name: moderation-stack
description: 为生产环境部署推荐审核栈配置方案。
version: 1.0.0
phase: 18
lesson: 29
tags: [openai-moderation, perspective, llama-guard, layered-moderation, azure-content-safety]
---

给定一个生产环境部署，在三个层级上推荐审核栈配置方案。

产出内容：

1. 输入分类器。选择 OpenAI Moderation、Llama Guard 3/4 或 Perspective API。匹配政策分类体系。对于多模态部署，选用 Llama Guard 4 或 OpenAI omni-moderation。
2. 输出分类器。与输入分类器相同或不同。将阈值匹配至下游风险模型。
3. 自定义领域规则。枚举通用分类器无法覆盖的领域特定规则：金融建议免责声明、医疗建议拒答模式、法律免责声明模式。
4. 边缘场景裁决。明确人工升级路径。硬性拒答为最终决定；模糊案例在 SLA 内转人工审查。
5. 迁移计划。若审核栈中包含 Azure Content Moderator，须规划在 2027 年 2 月退役前迁移至 Azure AI Content Safety。

硬性否决条件：
- 任何缺少输出审核的部署（仅审核输入不够充分）。
- 任何在受监管界面（金融、医疗、法律）上缺少自定义领域规则的部署。
- 任何在现代化聊天应用中仅依赖 LLM 前时代分类器（Perspective）的部署。

拒绝规则：
- 若用户询问"最佳单一分类器"，应予以拒绝——分类器选择取决于政策分类体系。
- 若用户索要阈值数字，应拒绝给出单一数值——阈值取决于风险容忍度与下游影响。

输出：一份单页推荐报告，填写上述五个部分，指明各层的分类器，并标记迁移义务。分别引用 OpenAI Moderation 文档和 Llama Guard 3/4 参考资料各一次。
