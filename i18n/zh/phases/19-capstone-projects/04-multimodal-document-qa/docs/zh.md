# 综合项目 04——多模态文档问答（视觉优先的 PDF、表格与图表）

> 到 2026 年，文档问答已经从“先做 OCR、再处理文本”（OCR-then-text）转向视觉优先的后期交互（vision-first late interaction）。ColPali、ColQwen2.5 和 ColQwen3-omni 把 PDF 的每一页都当作图像，以多向量后期交互生成嵌入，让查询直接关注页面中的图像块。处理金融 10-K 年报、科学论文和手写笔记时，这套方法明显优于以 OCR 为先的方案。请在一万页文档上从头搭建整条流水线，并发布一份与 OCR-then-text 方案的并排对比报告。

**Type:** 综合项目
**Languages:** Python（流水线）、TypeScript（查看器界面）
**Prerequisites:** 第 4 阶段（计算机视觉）、第 5 阶段（自然语言处理）、第 7 阶段（Transformer）、第 11 阶段（LLM 工程）、第 12 阶段（多模态）、第 17 阶段（基础设施）
**Phases exercised:** P4 · P5 · P7 · P11 · P12 · P17
**Time:** 30 小时

## 问题

企业积压着大量 OCR 流水线难以正确处理的 PDF：表格发生旋转的扫描版 10-K 年报、公式密集的科学论文、必须看图才能理解的图表，以及带有手写批注的页面。如果一开始就把这些材料当作文本处理，许多视觉信息会直接丢失。目前的解决办法是在原始页面图像上进行后期交互式多向量检索。ColPali（Illuin Tech）率先引入了这种方法，ColQwen2.5-v0.2 和 ColQwen3-omni 随后进一步提高了准确率。在 ViDoRe v3 上，视觉优先检索明显优于 OCR-then-text；遇到图表、表格和手写内容时，两者的差距更大。

代价在于存储空间和延迟。ColQwen 为每页生成的不是一个 1024 维向量，而是约 2048 个图像块向量，未经处理的存储量会迅速膨胀。DocPruner（2026）可以剪掉 50% 的向量，而且不会造成可测量的准确率损失。你需要为一万页文档建立索引，测量 ViDoRe v3 的 nDCG@5，将回答时间控制在 2 秒以内，并与 OCR-then-text 基线直接比较。

## 核心概念

后期交互会让每个查询词元分别与每个图像块词元计算得分，再取每个查询词元的最高分并求和。这样既能保留细粒度匹配，也不必把整页压缩成一个池化向量。多向量索引（如 Vespa、Qdrant multi-vector 或 AstraDB）负责存储各图像块的嵌入，并在检索时执行 MaxSim。

回答器采用视觉语言模型（VLM）。它接收查询和检索排名前 k 的页面图像，生成答案时附上证据区域，例如边界框或页码引用。Qwen3-VL-30B、Gemini 2.5 Pro 和 InternVL3 都是 2026 年可选的先进模型。遇到公式和科学记号时，还可以接入 OCR 回退方案（Nougat 或 dots.ocr），增加一条可选的文本通道。

评估结果用二维矩阵呈现。一个维度是内容类型：纯文本段落、密集表格、柱状图或折线图、手写笔记以及公式；另一个维度是检索方式：视觉优先的后期交互、OCR-then-text 和混合检索。每个单元格都要给出 nDCG@5 和回答准确率。最终交付的就是这份报告。

## 架构

```
PDFs -> page renderer (PyMuPDF, 180 DPI)
           |
           v
  ColQwen2.5-v0.2 embed (multi-vector per page, ~2048 patches)
           |
           +------> DocPruner 50% compression
           |
           v
   multi-vector index (Vespa or Qdrant multi-vector)
           |
query ----+----> retrieve top-k pages (MaxSim)
           |
           v
  VLM answerer: Qwen3-VL-30B | Gemini 2.5 Pro | InternVL3
    inputs: query + top-k page images + optional OCR text
           |
           v
  answer with cited page numbers + evidence regions
           |
           v
  Streamlit / Next.js viewer: highlighted boxes on source page
```

## 技术栈

- 页面渲染：PyMuPDF（fitz），180 DPI，统一调整为纵向
- 后期交互模型：ColQwen2.5-v0.2 或 ColQwen3-omni（Hugging Face 上的 ViDoRe 团队模型）
- 索引：使用带多向量字段的 Vespa、Qdrant multi-vector，或支持 MaxSim 的 AstraDB
- 剪枝：DocPruner 2026 策略（保留高方差图像块，以低于 0.5% 的准确率损失实现 50% 压缩）
- OCR 回退通道（公式 / 稠密表格）：dots.ocr 或 Nougat
- VLM 回答器：自托管 Qwen3-VL-30B 或托管 Gemini 2.5 Pro；InternVL3 作为回退
- 评测：ViDoRe v3 基准，以及用于多页推理的 M3DocVQA
- 查看器界面：Next.js 15，使用 canvas 叠加层渲染证据区域

```figure
ce-late-interaction
```

## 动手构建

1. **导入。** 收集一万页 PDF 语料，涵盖 10-K 年报、科学论文和扫描文档。将每页渲染为 1536x2048 的 PNG，并持久化 `{doc_id, page_num, image_path}`。

2. **嵌入。** 对每张页面图像运行 ColQwen2.5-v0.2，得到约 2048 个 128 维的图像块嵌入。再应用 DocPruner，只保留信号最强的一半，最后写入 Vespa 多向量字段或 Qdrant multi-vector。

3. **查询。** 每次收到查询后，先用查询编码塔生成词元级嵌入，再对索引执行 MaxSim：针对每个查询词元，在页面图像块嵌入中取点积最大值，然后求和。返回排名前 k 的页面。

4. **生成答案。** 将查询和排名前 5 的页面图像交给 Qwen3-VL-30B。提示词写成：“只能根据提供的页面回答。每项论断都要以 (doc_id, page) 标明出处，并指出证据所在区域（插图、表格或段落）。”

5. **证据区域。** 对答案做后处理，提取其中引用的证据区域。如果 VLM 能直接输出边界框（Qwen3-VL 可以），就在查看器中以叠加层显示这些边界框。

6. **OCR 回退。** 对公式密集的页面运行 Nougat 或 dots.ocr；这类页面可以用图像方差等启发式规则识别。将得到的 OCR 文本作为图像之外的附加输入通道。

7. **评测。** 运行 ViDoRe v3（检索 nDCG@5）和 M3DocVQA（多页问答准确率）。同时在同一语料上使用相同的答案生成器运行 OCR-then-text 流水线，输出“内容类型 × 检索方法”矩阵。

8. **界面。** 先用 Streamlit 制作原型，再用 Next.js 15 构建正式查看器，支持逐页叠加显示证据区域。

## 实际使用

```
$ doc-qa ask "what was the 2024 operating margin change for segment EMEA?"
[retrieve]   top-5 pages in 320ms (ColQwen2.5, MaxSim, Vespa)
[synth]      qwen3-vl-30b, 1.4s, cited (form-10k-2024, p. 88) + (..., p. 92)
answer:
  EMEA operating margin moved from 18.2% to 16.8%, a 140bp decline.
  cited: 10-K-2024.pdf p.88 (Table 4, Segment Operating Margin)
         10-K-2024.pdf p.92 (MD&A, Operating Performance)
[viewer]     open with highlighted bounding boxes overlaid on p.88 Table 4
```

## 交付成果

`outputs/skill-doc-qa.md` 描述最终交付物：一套针对特定语料调优的视觉优先多模态文档问答系统，并在 ViDoRe v3 上与 OCR-then-text 基线进行对比评测。

| 权重 | 标准 | 测量方式 |
|:-:|---|---|
| 25 | ViDoRe v3 / M3DocVQA 准确率 | 与 OCR 文本基线及公开排行榜对比的基准成绩 |
| 20 | 证据区域定位 | 所引区域中实际包含答案片段的比例 |
| 20 | 存储与延迟工程 | DocPruner 压缩比、索引 p95、答案 p95 |
| 20 | 多页推理 | 手工标注的 100 题多页集合上的准确率 |
| 15 | 源文档查阅体验 | 查看器清晰度、叠加区域准确度和并排比较工具 |
| **100** | | |

## 练习

1. 在同一语料上比较 ColQwen2.5-v0.2 与 ColQwen3-omni。找出哪些页面只有一个模型能够正确检索。给索引增加“内容类别（content class）”标签，再按类别选择模型。

2. 尝试更激进的嵌入剪枝比例（75%、90%）。找出压缩临界点（compression cliff），即 ViDoRe nDCG@5 开始低于 OCR 基线的位置。

3. 构建混合方案：并行运行 OCR-then-text 和 ColQwen，用 RRF 融合结果，再由交叉编码器重排。它能否胜过任一单独方案？对哪类页面帮助最大？

4. 将 Qwen3-VL-30B 换成较小的 VLM（Qwen2.5-VL-7B），测量单位成本准确率曲线。

5. 增加手写笔记支持。渲染手写语料，用 ColQwen 生成嵌入并测量检索效果，再与手写 OCR 流水线比较。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|-----------------|------------------------|
| 后期交互（Late interaction） | “ColPali 式检索” | 查询词元分别与页面图像块计算得分，再由 MaxSim 聚合 |
| 多向量（Multi-vector） | “逐图像块嵌入” | 每份文档包含多个向量，而不是只有一个池化向量 |
| MaxSim | “后期交互评分” | 针对每个查询词元，在文档向量中取最大相似度后求和 |
| DocPruner | “图像块压缩” | 2026 年提出的剪枝方法，只保留 50% 的图像块，准确率几乎不受影响 |
| ViDoRe v3 | “文档检索基准” | 2026 年用于衡量视觉文档检索效果的标准基准 |
| 证据区域（Evidence region） | “引用边界框” | 在源页面上框定答案片段的边界框 |
| OCR 回退（OCR fallback） | “公式通道” | 与视觉通道配合使用的文本流水线，适合公式或表格密集的页面 |

## 延伸阅读

- [ColPali（Illuin Tech）代码仓库](https://github.com/illuin-tech/colpali) — 后期交互式文档检索的参考实现
- [ColPali 论文（arXiv:2407.01449）](https://arxiv.org/abs/2407.01449) — 最初提出该方法的论文
- [Hugging Face 上的 ColQwen 系列](https://huggingface.co/vidore) — 可用于生产的模型检查点
- [M3DocRAG (Adobe)](https://arxiv.org/abs/2411.04952) — 多页多模态 RAG 基线
- [Vespa 多向量教程](https://docs.vespa.ai/en/colpali.html) — 可供参考的服务部署方案
- [Qdrant 多向量支持](https://qdrant.tech/documentation/concepts/vectors/#multivectors) — 另一种索引方案
- [AstraDB 多向量](https://docs.datastax.com/en/astra-db-serverless/databases/vector-search.html) — 另一种托管索引方案
- [Nougat OCR](https://github.com/facebookresearch/nougat) — 能够处理公式的 OCR 回退方案
