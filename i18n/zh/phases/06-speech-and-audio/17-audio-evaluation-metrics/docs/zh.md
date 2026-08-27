# 音频评估——WER、MOS、UTMOS、MMAU、FAD 与开放排行榜

> 无法衡量，就无法交付。本课列出各类音频任务在 2026 年使用的指标：ASR（WER、CER、RTFx）、TTS（MOS、UTMOS、SECS、ASR 往返 WER）、音频语言（MMAU、LongAudioBench）、音乐（FAD、CLAP）和说话人（EER），以及用于横向比较的排行榜。

**Type:** 学习
**Languages:** Python
**Prerequisites:** 阶段 6 · 04、06、07、09、10；阶段 2 · 09（模型评估）
**Time:** 约 60 分钟

## 问题

每一种音频任务都有多个指标，分别衡量不同维度。使用错误指标，会让你交付一个在仪表盘上看起来很棒、在生产环境中却表现糟糕的模型。2026 年的标准清单如下：

| 任务 | 主要指标 | 次要指标 |
|------|---------|-----------|
| ASR | WER | CER · RTFx · 首词元延迟 |
| TTS | MOS / UTMOS | SECS · ASR 往返 WER · CER · TTFA |
| 声音克隆 | SECS（ECAPA 余弦相似度） | MOS · CER |
| 说话人验证 | EER | minDCF · 工作点上的 FAR / FRR |
| 说话人分离 | DER | JER · 说话人混淆 |
| 音频分类 | top-1 · mAP | Macro-F1 · 逐类别召回率 |
| 音乐生成 | FAD | CLAP · 听评小组 MOS |
| 音频语言模型 | MMAU-Pro | LongAudioBench · AudioCaps FENSE |
| 流式语音到语音 | 延迟 P50/P95 | WER · MOS |

## 概念

![音频评估矩阵——任务、指标与 2026 年排行榜](../assets/eval-landscape.svg)

### ASR 指标

**WER（词错误率）。** `(S + D + I) / N`。评分前应转为小写、移除标点并规范数字。使用 `jiwer` 或 OpenAI 的 `whisper_normalizer`。低于 5% 表示在朗读语音上达到人类水平。

**CER（字符错误率）。** 公式相同，但在字符级计算。用于普通话、粤语等分词存在歧义的声调语言。

**RTFx（实时因子的倒数）。** 每墙钟秒可以处理的音频秒数，越高越好。Parakeet-TDT 可达到 3380×，Whisper-large-v3 约为 30×。

**首词元延迟。** 从音频输入到首个转写词元的墙钟时间，对流式处理至关重要。Deepgram Nova-3 约为 150 毫秒。

### TTS 指标

**MOS（平均意见分）。** 人工给出 1～5 分，是黄金标准，但速度很慢。每个样本应收集至少 20 位听众的评价，每个模型至少评估 100 个样本。

**UTMOS（2022～2026）。** 学习式 MOS 预测器。在标准基准上与人工 MOS 的相关性约为 0.9。F5-TTS 的 UTMOS 为 3.95，真实语音为 4.08。

**SECS（说话人编码器余弦相似度）。** 用于声音克隆。计算参考语音与克隆输出的 ECAPA 嵌入余弦相似度。高于 0.75 通常表示克隆可以辨认。

**ASR 往返 WER。** 用 Whisper 转写 TTS 输出，再对照输入文本计算 WER，用于发现可懂度回归。2026 年顶尖水平为 CER 低于 2%。

**TTFA（首音频时间）。** 墙钟延迟。Kokoro-82M 约为 100 毫秒，F5-TTS 约为 1 秒。

### 声音克隆专用指标

同时使用 **SECS + MOS + CER**。SECS 高而 MOS 低，意味着音色正确但听起来不自然；反过来则意味着声音自然，但说话人不对。

### 说话人验证

**EER（等错误率）。** 错误接受率等于错误拒绝率时的阈值。ECAPA 在 VoxCeleb1-O 上为 0.87%。

**minDCF（最小检测代价）。** 在选定工作点（通常 FAR=0.01）上的加权成本，比 EER 更贴近生产需求。

### 说话人分离

**DER（说话人分离错误率）。** `(FA + Miss + Confusion) / total_speaker_time`。漏掉的语音 + 误报的语音 + 说话人混淆，各自占总说话时间的比例。在 AMI 会议数据上，10%～20% DER 是现实水平；pyannote 3.1 + 商业版 Precision-2 在录音良好的音频上可低于 10%。

**JER（Jaccard 错误率）。** DER 的替代指标，对短片段偏差更稳健。

### 音频分类

多标签任务：在所有类别上计算 **mAP（平均精确率均值）**。AudioSet 上，BEATs-iter3 达到 0.548 mAP。

互斥多分类任务：**top-1、top-5 准确率**。Speech Commands v2 上，Audio-MAE 的 top-1 为 99.0%。

不平衡任务：**Macro-F1** + **逐类别召回率**。必须逐类报告——聚合准确率会掩盖失败的类别。

### 音乐生成

**FAD（Fréchet 音频距离）。** 真实音频与生成音频的 VGGish 嵌入分布之间的距离。MusicGen-small 在 MusicCaps 上为 4.5，MusicLM 为 4.0，越低越好。

**CLAP 分数。** 使用 CLAP 嵌入衡量文本—音频对齐。高于 0.3 表示对齐尚可。

**听评小组 MOS。** 对消费级音乐而言，它仍是最终标准。根据成对人工偏好，Suno v5 在 TTS Arena 上的 ELO 为 1293。

### 音频语言基准

**MMAU（海量多音频理解）。** 1 万个音频问答对。

**MMAU-Pro。** 1800 个难题，分为语音、声音、音乐和多音频四类。四选一的随机水平为 25%。Gemini 2.5 Pro 总体约为 60%；所有模型在多音频类别上都只有约 22%。

**LongAudioBench。** 带语义查询的多分钟音频片段。Audio Flamingo Next 胜过 Gemini 2.5 Pro。

**AudioCaps / Clotho。** 音频描述基准，使用 SPICE、CIDEr 和 FENSE 指标。

### 流式语音到语音

**延迟 P50 / P95 / P99。** 从用户结束说话到首个可听响应的墙钟时间。Moshi 为 200 毫秒，GPT-4o Realtime 为 300 毫秒。

输出上的 **WER / MOS**。

**打断响应时间。** 从用户插话到助手静音所需的时间，目标低于 150 毫秒。

### 2026 年排行榜

| 排行榜 | 赛道 | URL |
|------------|--------|-----|
| 开放 ASR 排行榜（HF） | 英语 + 多语言 + 长音频 | `huggingface.co/spaces/hf-audio/open_asr_leaderboard` |
| TTS Arena（HF） | 英语 TTS | `huggingface.co/spaces/TTS-AGI/TTS-Arena` |
| Artificial Analysis Speech | TTS + STT，基于成对投票计算 ELO | `artificialanalysis.ai/speech` |
| MMAU-Pro | LALM 推理 | `mmaubenchmark.github.io` |
| SpeakerBench / VoxSRC | 说话人识别 | `voxsrc.github.io` |
| MMAU 音乐子集 | 音乐 LALM | （MMAU 内部） |
| HEAR 基准 | 自监督音频 | `hearbenchmark.com` |

```figure
sp-wer-align
```

## 动手构建

### 第 1 步：经过规范化的 WER

```python
from jiwer import wer, Compose, ToLowerCase, RemovePunctuation, Strip

transform = Compose([ToLowerCase(), RemovePunctuation(), Strip()])
score = wer(
    truth="Please turn on the lights.",
    hypothesis="please turn on the light",
    truth_transform=transform,
    hypothesis_transform=transform,
)
# ~0.17
```

### 第 2 步：TTS 往返 WER

```python
def ttr_wer(tts_model, asr_model, texts):
    errors = []
    for txt in texts:
        audio = tts_model.synthesize(txt)
        recog = asr_model.transcribe(audio)
        errors.append(wer(truth=txt, hypothesis=recog))
    return sum(errors) / len(errors)
```

### 第 3 步：声音克隆的 SECS

```python
from speechbrain.inference.speaker import EncoderClassifier
sv = EncoderClassifier.from_hparams("speechbrain/spkrec-ecapa-voxceleb")

emb_ref = sv.encode_batch(load_wav("reference.wav"))
emb_clone = sv.encode_batch(load_wav("cloned.wav"))
secs = torch.nn.functional.cosine_similarity(emb_ref, emb_clone, dim=-1).item()
```

### 第 4 步：音乐生成的 FAD

```python
from frechet_audio_distance import FrechetAudioDistance
fad = FrechetAudioDistance()
score = fad.get_fad_score("generated_folder/", "reference_folder/")
```

### 第 5 步：说话人验证的 EER（与第 6 课代码相同）

```python
def eer(same_scores, diff_scores):
    thresholds = sorted(set(same_scores + diff_scores))
    best = (1.0, 0.0)
    for t in thresholds:
        far = sum(1 for s in diff_scores if s >= t) / len(diff_scores)
        frr = sum(1 for s in same_scores if s < t) / len(same_scores)
        if abs(far - frr) < best[0]:
            best = (abs(far - frr), (far + frr) / 2)
    return best[1]
```

## 学以致用

每次部署都应配备固定的评估工具，并在每次模型更新时运行。三条基本规则：

1. **评分前先规范化。** 转小写、移除标点、展开数字，并报告所用的规范化规则。
2. **报告分布，而不只是平均值。** 延迟报告 P50/P95/P99，分类报告逐类别召回率，MMAU 报告逐类别结果。
3. **运行一项标准公开基准。** 即使生产数据不同，在 Open ASR / TTS Arena / MMAU 上报告结果，也能让评审者进行同类比较。

## 陷阱

- **外推 UTMOS。** 它在 VCTK 风格的干净语音上训练，对嘈杂、克隆或带情绪的音频评分不佳。
- **MOS 评审组偏差。** 20 名 Amazon Mechanical Turk 工作人员不等于 20 名目标用户。如果风险较高，应付费组织领域用户评审。
- **FAD 依赖参考集。** 比较不同模型时必须使用同一个参考分布。
- **聚合 WER。** 总体 5% 的 WER 可能掩盖口音语音上 30% 的 WER。应按人口统计切片报告。
- **公开基准饱和。** 大多数前沿模型在标准基准上已接近上限。应建立反映真实流量的内部留出集。

## 交付成果

保存为 `outputs/skill-audio-evaluator.md`。为任意音频模型发布选择指标、基准与报告格式。

## 练习

1. **简单。** 运行 `code/main.py`。在玩具输入上计算 WER / CER / EER / SECS / 类 FAD / 类 MMAU 指标。
2. **中等。** 构建 TTS 往返 WER 测试工具。把 Kokoro 或 F5-TTS 输出送入 Whisper，在 50 个提示上计算 WER，并标记 WER 大于 10% 的提示。
3. **困难。** 在 MMAU-Pro 的语音与多音频子集上评测第 10 课选择的 LALM（每类 50 项）。报告逐类别准确率，并与公开数字比较。

## 关键术语

| 术语 | 人们通常怎么说 | 实际含义 |
|------|-----------------|-----------------------|
| WER | ASR 分数 | 规范化后按词计算 `(S+D+I)/N`。 |
| CER | 字符 WER | 用于声调语言或字符级系统。 |
| MOS | 人类意见 | 1～5 分；至少 20 位听众 × 100 个样本。 |
| UTMOS | 机器学习 MOS 预测器 | 学习式模型，与人工 MOS 的相关性约为 0.9。 |
| SECS | 声音克隆相似度 | 参考声音与克隆声音之间的 ECAPA 余弦相似度。 |
| EER | 说话人验证分数 | FAR 等于 FRR 时的阈值。 |
| DER | 说话人分离分数 | （误报 + 漏检 + 混淆）/ 总时长。 |
| FAD | 音乐生成质量 | VGGish 嵌入上的 Fréchet 距离。 |
| RTFx | 吞吐量 | 每墙钟秒处理的音频秒数。 |

## 延伸阅读

- [jiwer](https://github.com/jitsi/jiwer)——带规范化工具的 WER/CER 库。
- [UTMOS（Saeki 等，2022）](https://arxiv.org/abs/2204.02152)——学习式 MOS 预测器。
- [Fréchet 音频距离（Kilgour 等，2019）](https://arxiv.org/abs/1812.08466)——音乐生成的标准指标。
- [Hugging Face 开放 ASR 排行榜](https://huggingface.co/spaces/hf-audio/open_asr_leaderboard)——2026 年实时排名。
- [TTS Arena](https://huggingface.co/spaces/TTS-AGI/TTS-Arena)——由人类投票的 TTS 排行榜。
- [MMAU-Pro 基准](https://mmaubenchmark.github.io/)——LALM 推理排行榜。
- [HEAR 基准](https://hearbenchmark.com/)——音频自监督学习基准。
