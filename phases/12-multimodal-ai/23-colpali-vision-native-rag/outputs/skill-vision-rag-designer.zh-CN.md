---
name: vision-rag-designer
description: 设计基于 ColPali / ColQwen2 / VisRAG 的视觉原生文档 RAG，包含存储估算与生成器选型。
version: 1.0.0
phase: 12
lesson: 23
tags: [colpali, colqwen2, visrag, late-interaction, vidore]
---

给定一个文档 RAG 项目（语料规模、查询延迟目标、存储预算、单次查询成本），输出一份视觉原生 RAG 配置。

产出：

1. 检索器选型。ColPali（基于 PaliGemma）、ColQwen2（基于 Qwen2-VL，质量更优）、ColSmol（1B，适用于边缘端）或 VisRAG（双编码器，存储成本更低）。
2. 存储估算。N_docs * N_p_per_doc * D * 4 字节为原始大小；使用 PQ 压缩则除以 8。
3. 延迟估算。
   - 检索 SLA：约 10ms 查询嵌入 + top-k 检索（MaxSim 或 ANN），取决于索引规模。
   - 完整回答 SLA：检索延迟 + 200-500ms 生成器（取决于模型和硬件）。
4. 生成器选型。开源选 Qwen2.5-VL-72B，前沿选 Claude Opus 4.7。
5. 压缩方案。PQ / OPQ 压缩比目标为 8-16 倍；使用 HNSW 索引实现快速 ANN。
6. 从文本 RAG 迁移路径。如何进行 A/B 测试，何时全面切换。

硬性拒绝：
- 在 >10k 页的语料上使用 ColPali 且不进行 PQ 压缩。存储会爆炸式增长。
- 声称双编码器检索在文档召回上能匹敌 ColBERT MaxSim。在 ViDoRe 上并不能。
- 针对图表 + 表格类工作负载推荐文本 RAG。文本 RAG 会丢失大部分信号。

拒绝规则：
- 如果语料为纯文本（wiki、聊天日志），拒绝视觉原生 RAG 并推荐标准文本 RAG。
- 如果检索 SLA <100ms，优先选择 VisRAG（双编码器）而非 ColPali MaxSim。
- 如果完整回答 SLA <100ms，完全拒绝生成式 RAG，并推荐仅检索的 UX 或缓存答案。
- 如果存储预算 <1 GB 且语料 >100k 页，拒绝全精度 ColPali；提议激进 PQ 压缩或 VisRAG。

输出：一页 RAG 设计，包含检索器选型、存储估算、延迟、生成器、压缩、迁移。结尾附 arXiv 2407.01449（ColPali）、2410.10594（VisRAG）。
