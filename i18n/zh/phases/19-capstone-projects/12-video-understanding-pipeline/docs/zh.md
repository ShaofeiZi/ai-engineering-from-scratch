# 综合项目 12——视频理解流水线（场景、问答、搜索）

> Twelve Labs 已将 Marengo + Pegasus 产品化，VideoDB 推出了面向视频的 CRUD API，AI2 的 Molmo 2 发布了开放的 VLM 检查点，Gemini 的长上下文则能原生处理数小时视频。TimeLens-100K 为大规模时序定位建立了基准。到 2026 年，这条流水线的标准形态已经明确：场景分割、逐场景描述与向量嵌入、转录对齐、多向量索引，以及能返回 (start, end) 时间戳和帧预览的查询。本综合项目要摄取 100 小时视频，在公开基准上评测，并衡量计数题和动作题中的幻觉。

**Type:** 综合项目
**Languages:** Python（流水线）、TypeScript（UI）
**Prerequisites:** 第 4 阶段（计算机视觉）、第 6 阶段（语音）、第 7 阶段（Transformer）、第 11 阶段（LLM 工程）、第 12 阶段（多模态）、第 17 阶段（基础设施）
**Phases exercised:** P4 · P6 · P7 · P11 · P12 · P17
**Time:** 30 小时

## 问题

以 2026 年的处理规模来看，长视频问答是带宽需求最高的多模态任务。Gemini 2.5 Pro 可以原生读取两小时的视频，但要把 100 小时视频摄取成可查询的语料库，仍然需要场景级索引。生产系统通常会组合场景分割（TransNetV2 或 PySceneDetect）、由 VLM 为每个场景生成描述并从关键帧提取向量嵌入（Gemini 2.5、Qwen3-VL-Max 或 Molmo 2）、转录对齐（带词级时间戳的 Whisper-v3-turbo），以及并排存储场景描述、画面和转录向量的多向量索引。查询流水线最终返回 (start, end) 时间戳和帧预览。

评估既采用 ActivityNet-QA 和 NeXT-GQA 等公开基准，也使用一套自行构建的 100 题数据集。计数题和动作类问题中的幻觉是公认的棘手失败类型，本综合项目会单独衡量这两类问题。

## 概念

摄取阶段有三条并行流水线。**场景分割**把视频切成多个场景；**VLM 场景描述**为每个场景生成文字描述，并从关键帧提取画面向量；**ASR 对齐**产生词级时间戳。三条数据流按 (scene_id, time range) 汇合。多向量索引 Qdrant 为每个场景存储三种向量：场景描述向量、关键帧向量和转录向量。

查询时，自然语言问题会同时检索三种向量，结果通过倒数排名融合（RRF）合并；TimeLens 风格的时序定位适配器再在排名最高的场景内细化 (start, end) 时间窗口。VLM 答案合成器使用 Gemini 2.5 Pro 或 Qwen3-VL-Max，接收查询、排名靠前的场景和裁剪帧，返回带时间戳引用与帧预览的答案。

幻觉测量非常重要。计数问题（“有多少人进入房间？”）和动作类问题（“厨师是在搅拌前倒入液体吗？”）尤其不可靠。应将它们的准确率与描述类问题分开报告。

## 架构

```
video file / URL
      |
      v
PySceneDetect / TransNetV2  (scene segmentation)
      |
      +--- per-scene keyframe --- VLM caption + frame embedding
      |                            (Gemini 2.5 Pro / Qwen3-VL-Max / Molmo 2)
      |
      +--- audio channel --- Whisper-v3-turbo ASR + word timestamps
      |
      v
multi-vector Qdrant: {caption_emb, keyframe_emb, transcript_emb}
      |
query:
  dense queries against all three -> RRF merge -> top-k scenes
      |
      v
TimeLens / VideoITG temporal grounding (refine start/end within scene)
      |
      v
VLM synth: query + top scenes + frame previews
      |
      v
answer + (start, end) timestamps + frame thumbs + citations
```

## 技术栈

- 场景分割：TransNetV2（2024–2026 年的前沿方案）或 PySceneDetect
- ASR：通过 faster-whisper 使用 Whisper-v3-turbo，并生成词级时间戳
- VLM 场景描述与回答：Gemini 2.5 Pro、Qwen3-VL-Max 或 Molmo 2
- 时序定位：基于 TimeLens-100K 训练的适配器或 VideoITG
- 索引：支持场景描述、画面和转录多向量的 Qdrant
- 界面：Next.js 15，配备 HTML5 视频播放器和场景缩略图
- 评估：ActivityNet-QA、NeXT-GQA，以及自行构建并人工标注的 100 题数据集
- 幻觉基准：带人工标签的计数与动作类子集

```figure
cf-scene-index
```

## 动手构建

1. **摄取遍历器。** 接受 YouTube URL 或本地 MP4 文件，必要时将分辨率降至 720p，并持久化 `{video_id, file_path}`。

2. **场景分割。** 运行 TransNetV2 或 PySceneDetect，产出 `[{scene_id, start_ms, end_ms, keyframe_path}]`。100 小时的目标规模约为 6,000–8,000 个场景。

3. **ASR 处理。** 在音频上运行 Whisper-v3-turbo，导出词级时间戳，再按场景切分转录片段。

4. **VLM 场景描述。** 针对每个场景，将关键帧和简短的描述模板交给 Gemini 2.5 Pro（或 Qwen3-VL-Max），生成场景描述与画面向量。

5. **多向量索引。** 创建一个包含三个命名向量的 Qdrant 集合，其载荷为 `{video_id, scene_id, start_ms, end_ms, keyframe_url}`。

6. **查询。** 用自然语言问题分别发起三次稠密检索，再通过倒数排名融合（Reciprocal Rank Fusion）合并结果，最终取 top-k=5 个场景。

7. **时序定位。** 对排名最高的场景运行 TimeLens 风格适配器，细化场景内的 (start, end) 时间窗口。

8. **VLM 答案合成。** 将查询、排名前三的场景片段（图像或短视频）和转录一并交给 Gemini 2.5 Pro，并强制要求返回 `(video_id, start_ms, end_ms)` 引用。

9. **评估。** 运行 ActivityNet-QA 与 NeXT-GQA，并构建一套 100 题自定义数据集。报告总体准确率与逐类别明细（计数、动作、描述）。

## 运行示例

```
$ video-qa ask --url=https://youtube.com/watch?v=X "how many cars pass the intersection in the first minute?"
[scene]    23 scenes detected
[asr]      transcript complete, 4m12s
[index]    69 vectors written (23 scenes x 3)
[query]    top scene: scene 3 [01:32-01:54], confidence 0.84
[ground]   refined window: [00:12-00:58]
[synth]    gemini 2.5 pro, 1.4s
answer:    5 cars pass the intersection between 00:12 and 00:58.
citations: [scene 3: 00:12-00:58]
          [frame preview at 00:14, 00:27, 00:44, 00:51, 00:57]
```

## 交付成果

`outputs/skill-video-qa.md` 是最终交付物。给定 YouTube URL 或上传的视频后，流水线会为场景建立索引，回答问题并附上时间戳引用。

| 权重 | 标准 | 测量方式 |
|:-:|---|---|
| 25 | 时序定位 IoU | 留出定位集上的交并比 |
| 20 | QA 准确率 | NeXT-GQA 与自定义 100 题数据集 |
| 20 | 摄取吞吐量 | 每美元可处理的视频小时数 |
| 20 | 界面与引用体验 | 时间戳链接、缩略图条、跳转到指定帧 |
| 15 | 幻觉率 | 分别统计计数题与动作题准确率 |
| **100** | | |

## 练习

1. 在场景描述生成步骤中，将 Gemini 2.5 Pro 替换为 Qwen3-VL-Max。使用 50 个场景的人工评分样本，报告描述质量差异。

2. 把逐场景画面向量从多向量表示简化为一个池化向量，测量检索效果的下降幅度。

3. 构建“严格计数”模式：答案合成器为每个计数对象提取时间戳，用户点击即可核验。测量这种用户核验机制是否能降低幻觉率。

4. 对摄取成本进行基准测试：比较三种 VLM 的每美元视频处理小时数，并选择最佳平衡点。

5. 增加带说话人分离的转录：在音频上运行 pyannote 说话人分离，并为每位说话人的转录分别生成向量。演示“关于 X，Alice 说了什么？”这类查询。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|-----------------|------------------------|
| 场景分割 | “镜头检测” | 在镜头边界处将视频切分为场景 |
| 多向量索引 | “场景描述 + 帧 + 转录” | 在一个 Qdrant 集合中为每种表征配置命名向量 |
| 时序定位 | “它究竟何时发生” | 为查询答案细化 (start, end) 时间窗口 |
| 帧向量（Frame embedding） | “视觉表征” | 关键帧的向量表示，用于计算场景之间的视觉相似度 |
| RRF 融合 | “Reciprocal Rank Fusion” | 合并多个排序列表的策略，是经典的混合检索方法 |
| 计数幻觉 | “计数错误” | VLM 在“有多少个 X”问题上的已知失败模式 |
| ActivityNet-QA | “视频问答基准” | 长视频问答准确率基准 |

## 延伸阅读

- [AI2 Molmo 2](https://allenai.org/blog/molmo2)——开放 VLM 检查点
- [TimeLens (CVPR 2026)](https://github.com/TencentARC/TimeLens)——大规模时序定位
- [Gemini Video long-context](https://deepmind.google/technologies/gemini)——托管式参考实现
- [VideoDB](https://videodb.io)——面向视频的 CRUD API 参考
- [Twelve Labs Marengo + Pegasus](https://www.twelvelabs.io)——商业参考实现
- [TransNetV2](https://github.com/soCzech/TransNetV2)——场景分割模型
- [PySceneDetect](https://github.com/Breakthrough/PySceneDetect)——经典开放替代方案
- [ActivityNet-QA](https://arxiv.org/abs/1906.02467)——参考评估基准
