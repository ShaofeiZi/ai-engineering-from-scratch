# 语音识别（ASR）——CTC、RNN-T 与注意力

> 语音识别是在每个时间步执行音频分类，再由理解语言与静音的序列模型把结果连接起来。CTC、RNN-T 和注意力是三种实现方式。选择一种，并理解原因。

**Type:** 构建
**Languages:** Python
**Prerequisites:** 阶段 6 · 02（频谱图与梅尔特征）、阶段 5 · 08（用于文本的 CNN 与 RNN）、阶段 5 · 10（注意力）
**Time:** 约 45 分钟

## 问题

你有一段 16 kHz、10 秒的音频，希望得到字符串：“turn on the kitchen lights”。难点在于结构：音频帧与字符并非一一对应。“okay”可能持续 200 毫秒，也可能持续 1200 毫秒。话语由静音分隔，不同音素的持续时间也不一样，输出词元数量无法预先确定。

三种形式可以解决这个问题：

1. **CTC（连接时序分类）。** 逐帧输出词元概率，其中包括一个特殊的*空白*。解码时折叠重复项并移除空白。非自回归，速度快。wav2vec 2.0 与 MMS 使用它。
2. **RNN-T（循环神经网络转导器）。** 联合网络根据编码器帧和先前词元预测下一个词元。支持流式处理。Google 的设备端 ASR 与 NVIDIA Parakeet 使用它。
3. **注意力编码器—解码器。** 编码器把音频压缩成隐藏状态，解码器通过交叉注意力自回归地生成词元。Whisper 与 SeamlessM4T 使用它。

2026 年，LibriSpeech test-clean 上的顶尖 WER 分别为 1.4%（NVIDIA Parakeet-TDT-1.1B）和 1.58%（Whisper-Large-v3-turbo）。质量差异很小，部署差异却很大。

## 概念

![三种 ASR 形式：CTC、RNN-T、注意力编码器—解码器](../assets/asr-formulations.svg)

**CTC 直觉。** 让编码器输出 `T` 个帧级概率分布，每个分布覆盖 `V+1` 个词元（V 个字符 + 空白）。对于目标字符串 `y`，若其长度 `U < T`，任何折叠后等于 `y` 的帧对齐都计入结果。CTC 损失会对所有这类对齐求和。推理时：逐帧取 argmax、折叠重复项、移除空白。

优点：非自回归、支持流式处理、零前瞻。缺点是*条件独立假设*——每帧预测都彼此独立，因而不存在内部语言模型。可以在束搜索中通过外部语言模型或浅层融合补救。

**RNN-T 直觉。** 增加一个嵌入词元历史的*预测器*网络，以及一个把预测器状态与编码器帧组合成 `V+1` 类联合分布的*联合器*（`+1` 是空值/不发射）。它显式建模了 CTC 所忽略的条件依赖。每一步只依赖过去的帧和过去的词元，因此可以流式处理。

优点：支持流式处理 + 内置语言模型。缺点：训练更复杂、更耗内存（三维损失网格）；RNN-T 损失内核本身就形成了一个完整的库类别。

**注意力编码器—解码器。** 编码器包含 6～32 层 Transformer，处理对数梅尔帧；解码器也包含 6～32 层 Transformer，通过交叉注意力访问编码器输出，并自回归地生成词元。不存在对齐约束——注意力可以查看音频中的任意位置。除非限制注意力（2024 年的分块式 Whisper-Streaming），否则它无法流式处理。

优点：离线 ASR 质量最高，使用标准序列到序列工具即可轻松训练。缺点：自回归延迟与输出长度成正比；若无额外工程，无法流式处理。

### WER：最重要的一个数字

**词错误率** = `(S + D + I) / N`，其中 S=替换、D=删除、I=插入、N=参考文本词数。它等于词级 Levenshtein 编辑距离。越低越好。WER 高于 20% 通常无法使用；在朗读语音上低于 5% 则达到人类水平。2026 年标准基准上的结果如下：

| 模型 | LibriSpeech test-clean | LibriSpeech test-other | 大小 |
|-------|------------------------|------------------------|------|
| Parakeet-TDT-1.1B | 1.40% | 2.78% | 1.1B 参数 |
| Whisper-Large-v3-turbo | 1.58% | 3.03% | 809M |
| Canary-1B Flash | 1.48% | 2.87% | 1B |
| Seamless M4T v2 | 1.7% | 3.5% | 2.3B |

这些模型都基于编码器—解码器或 RNN-T。纯 CTC 系统（wav2vec 2.0）在 test-clean 上约为 1.8%～2.1%。

```figure
ctc-collapse
```

## 动手构建

### 第 1 步：CTC 贪心解码

```python
def ctc_greedy(frame_logits, blank=0, vocab=None):
    # frame_logits: list of per-frame probability vectors
    preds = [max(range(len(p)), key=lambda i: p[i]) for p in frame_logits]
    out = []
    prev = -1
    for p in preds:
        if p != prev and p != blank:
            out.append(p)
        prev = p
    return "".join(vocab[i] for i in out) if vocab else out
```

两条规则：折叠连续重复项，删除空白。例如：`a a _ _ a b b _ c` → `a a b c`。

### 第 2 步：CTC 束搜索

```python
def ctc_beam(frame_logits, beam=8, blank=0):
    import math
    beams = [([], 0.0)]  # (tokens, log_prob)
    for p in frame_logits:
        log_p = [math.log(max(pi, 1e-10)) for pi in p]
        candidates = []
        for seq, lp in beams:
            for t, lpt in enumerate(log_p):
                new = seq[:] if t == blank else (seq + [t] if not seq or seq[-1] != t else seq)
                candidates.append((new, lp + lpt))
        candidates.sort(key=lambda x: -x[1])
        beams = candidates[:beam]
    return beams[0][0]
```

生产系统使用带语言模型融合的前缀树束搜索；这里只展示概念骨架。

### 第 3 步：WER

```python
def wer(ref, hyp):
    r, h = ref.split(), hyp.split()
    dp = [[0] * (len(h) + 1) for _ in range(len(r) + 1)]
    for i in range(len(r) + 1):
        dp[i][0] = i
    for j in range(len(h) + 1):
        dp[0][j] = j
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            cost = 0 if r[i - 1] == h[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost,
            )
    return dp[len(r)][len(h)] / max(1, len(r))
```

### 第 4 步：使用 Whisper 推理

```python
import whisper
model = whisper.load_model("large-v3-turbo")
result = model.transcribe("clip.wav")
print(result["text"])
```

一行代码即可使用 2026 年最强的通用 ASR。在 24 GB GPU 上可以达到约 20 倍实时速度。

### 第 5 步：使用 Parakeet 或 wav2vec 2.0 流式处理

```python
from transformers import pipeline
asr = pipeline("automatic-speech-recognition", model="nvidia/parakeet-tdt-1.1b")
for chunk in streaming_audio():
    print(asr(chunk, return_timestamps=True))
```

流式 ASR 需要分块编码器注意力与延续状态；应使用支持这些功能的库（Parakeet 使用 NeMo，`transformers` 流水线使用 `chunk_length_s`）。

## 学以致用

2026 年的技术栈：

| 场景 | 选择 |
|-----------|------|
| 英语、离线、追求最高质量 | Whisper-large-v3-turbo |
| 多语言、强调稳健性 | SeamlessM4T v2 |
| 流式、低延迟 | Parakeet-TDT-1.1B 或 Riva |
| 边缘端、移动端、延迟低于 500 毫秒 | 量化后的 Whisper-Tiny 或 Moonshine（2024） |
| 长音频 | 使用基于 VAD 分块的 Whisper（WhisperX） |
| 领域专用（医学、法律） | 微调 wav2vec 2.0 + 领域语言模型融合 |

## 2026 年仍会进入生产的陷阱

- **没有 VAD。** 在静音上运行 Whisper 会产生幻觉（“Thanks for watching!”）。始终使用 VAD 把关。
- **字符、单词与子词 WER 混淆。** 应在规范化之后（转小写、移除标点）报告词级 WER。
- **语言识别漂移。** Whisper 自动语言识别会把嘈杂音频误路由到日语或威尔士语；如果已知语言，请强制设置 `language="en"`。
- **长音频不分块。** Whisper 的窗口为 30 秒。更长内容应使用 `chunk_length_s=30, stride=5`。

## 交付成果

保存为 `outputs/skill-asr-picker.md`。根据部署目标选择模型、解码策略、分块与语言模型融合。

## 练习

1. **简单。** 运行 `code/main.py`。它会对手工构造的 CTC 输出进行贪心解码，并对照参考文本计算 WER。
2. **中等。** 正确实现第 2 步中的前缀树束搜索（处理空白合并规则）。在包含 10 个样本的合成数据集上与贪心解码比较。
3. **困难。** 在 [LibriSpeech test-clean](https://www.openslr.org/12) 上使用 `whisper-large-v3-turbo`，计算前 100 条话语的 WER，并与公开数据比较。

## 关键术语

| 术语 | 人们通常怎么说 | 实际含义 |
|------|-----------------|-----------------------|
| CTC | 空白词元损失 | 对所有帧—词元对齐求边缘概率；非自回归。 |
| RNN-T | 流式损失 | CTC + 下一词元预测器；可以处理词序。 |
| 注意力编码器—解码器 | Whisper 风格 | 编码器 + 交叉注意力解码器；离线质量最佳。 |
| WER | 应报告的数字 | 词级 `(S+D+I)/N`。 |
| 空白 | 什么都不发射 | CTC 中表示“本帧不输出”的特殊词元。 |
| 语言模型融合 | 外部语言模型 | 束搜索期间加入加权的语言模型对数概率。 |
| VAD | 静音门 | 语音活动检测器，用于裁掉非语音片段。 |

## 延伸阅读

- [Graves 等（2006），连接时序分类](https://www.cs.toronto.edu/~graves/icml_2006.pdf)——CTC 论文。
- [Graves（2012），使用 RNN 进行序列转导](https://arxiv.org/abs/1211.3711)——RNN-T 论文。
- [Radford 等 / OpenAI（2022），Whisper：通过大规模弱监督实现稳健语音识别](https://arxiv.org/abs/2212.04356)——2022 年经典论文；v3-turbo 扩展发布于 2024 年。
- [NVIDIA NeMo——Parakeet-TDT 模型卡](https://huggingface.co/nvidia/parakeet-tdt-1.1b)——2026 年开放 ASR 排行榜领先模型。
- [Hugging Face——开放 ASR 排行榜](https://huggingface.co/spaces/hf-audio/open_asr_leaderboard)——覆盖 25 个以上模型的实时基准。
