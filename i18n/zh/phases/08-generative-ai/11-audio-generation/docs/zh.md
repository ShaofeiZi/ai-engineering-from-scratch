# 音频生成

> 音频是采样率为 16～48 kHz 的一维信号。五秒音频就有 8 万～24 万个采样点，没有 Transformer 会直接关注这么长的序列。2026 年所有生产音频模型都采用同一种解决方案：先由神经编解码器（Encodec、SoundStream、DAC）把音频压缩成频率为 50～75 Hz 的离散词元，再由 Transformer 或扩散模型生成这些词元。

**Type:** 构建
**Languages:** Python
**Prerequisites:** 阶段 6 · 02（音频特征）、阶段 6 · 04（自动语音识别）、阶段 8 · 06（DDPM）
**Time:** 约 45 分钟

## 问题

音频生成有三类任务：

1. **文本转语音。** 给定文本，生成语音。干净语音频带较窄，且具有很强的语音结构——在词元上运行 Transformer 已能很好地解决这一问题。代表系统有 VALL-E（Microsoft）、NaturalSpeech 3、ElevenLabs、OpenAI TTS。
2. **音乐生成。** 给定提示词（文本、旋律、和弦进行、流派），生成音乐。其分布宽广得多。代表系统有 MusicGen（Meta）、Stable Audio 2.5、Suno v4、Udio、Riffusion。
3. **音效/声音设计。** 给定提示词，生成环境音或拟音。代表系统有 AudioGen、AudioLDM 2、Stable Audio Open。

三类任务都建立在同一种底座上：神经音频编解码器 + 词元自回归生成器或扩散生成器。

## 概念

![音频生成：编解码器词元 + Transformer 或扩散模型](../../../../../../phases/08-generative-ai/11-audio-generation/assets/audio-generation.svg)

### 神经音频编解码器

代表模型包括 Encodec（Meta，2022）、SoundStream（Google，2021）和 Descript Audio Codec（DAC，2023）。卷积编码器把波形压缩成每个时间步的向量；残差向量量化（RVQ）把每个向量转换为 K 个码本索引组成的级联；解码器再逆转这一过程。24 kHz 音频若以 2 kbps 编码，并在 75 Hz 下使用 8 个 RVQ 码本，每秒会产生 600 个词元。

```
waveform (16000 samples/sec)
    └─ encoder conv ─┐
                     ├─ RVQ layer 1 → indices at 75 Hz
                     ├─ RVQ layer 2 → indices at 75 Hz
                     ├─ ...
                     └─ RVQ layer 8
```

### 上层的两种生成范式

**词元自回归。** 将 RVQ 词元展平为序列，再运行仅解码器 Transformer。MusicGen 采用“延迟并行”机制，以各不相同的偏移量并行输出 K 条码本流。VALL-E 根据文本提示 + 3 秒语音样本生成语音词元。

**潜在扩散。** 把编解码器词元打包成连续潜变量，或使用类别扩散为其建模。Stable Audio 2.5 在连续音频潜变量上使用流匹配；AudioLDM 2 则使用文本到梅尔频谱再到音频的扩散过程。

2024～2026 年的趋势是：流匹配凭借更快的推理和更干净的样本，在音乐领域胜出；词元自回归仍然主导语音领域，因为它天然具有因果性，并且易于流式传输。

## 生产格局

| 系统 | 任务 | 骨干网络 | 延迟 |
|--------|------|----------|---------|
| ElevenLabs V3 | TTS | 词元自回归 + 神经声码器 | 首个词元约 300ms |
| OpenAI GPT-4o audio | 全双工语音 | 端到端多模态自回归 | 约 200ms |
| NaturalSpeech 3 | TTS | 潜在流匹配 | 非流式 |
| Stable Audio 2.5 | 音乐/音效 | 音频潜变量上的 DiT + 流匹配 | 1 分钟片段约 10s |
| Suno v4 | 完整歌曲 | 未公开；疑似词元自回归 | 每首歌约 30s |
| Udio v1.5 | 完整歌曲 | 未公开 | 每首歌约 30s |
| MusicGen 3.3B | 音乐 | Encodec 32kHz 上的词元自回归 | 实时 |
| AudioCraft 2 | 音乐 + 音效 | 流匹配 | 5 秒片段约需 5s |
| Riffusion v2 | 音乐 | 频谱图扩散 | 约 10s |

```figure
score-matching
```

## 动手构建

`code/main.py` 模拟核心思路：在由两种不同“风格”生成的合成“音频词元”序列上，训练一个微型下一词元 Transformer（风格 A 是高低词元交替，风格 B 是单调递增序列）。然后以风格为条件进行采样。

### 第 1 步：合成音频词元

```python
def make_tokens(style, length, vocab_size, rng):
    if style == 0:  # "speech-like": alternating
        return [i % vocab_size for i in range(length)]
    # "music-like": ramp
    return [(i * 3) % vocab_size for i in range(length)]
```

### 第 2 步：训练微型词元预测器

使用以风格为条件的二元语法式预测器。重点在于这套模式：编解码器词元 → 交叉熵训练 → 自回归采样。

### 第 3 步：条件采样

给定风格词元和起始词元，从预测分布中采样下一个词元，持续生成 20～40 个词元。

## 陷阱

- **编解码器质量决定输出质量上限。** 如果编解码器无法忠实表示某种声音，再优秀的生成器也无济于事。DAC 是当前最佳的开放方案。
- **RVQ 误差累积。** 每个 RVQ 层都对前一层的残差建模，第一层的误差会向后传播。在更高层使用温度 0 采样会有所帮助。
- **音乐结构。** 以 75 Hz 编码时，30 秒音频包含超过 2 万个词元，对 Transformer 而言十分困难。MusicGen 使用滑动窗口 + 提示续写；Stable Audio 使用更短片段 + 交叉淡化。
- **边界伪影。** 在生成片段之间进行交叉淡化时，必须谨慎处理重叠相加。
- **对干净数据的巨大需求。** 音乐生成器需要数万小时获得许可的音乐。Suno / Udio 与 RIAA 在 2024 年的诉讼让这个问题浮出水面。
- **语音克隆伦理。** 3 秒样本加一段文本提示，就足以让 VALL-E / XTTS / ElevenLabs 克隆声音。每个生产模型都需要滥用检测 + 退出名单。

## 学以致用

| 任务 | 2026 年技术栈 |
|------|------------|
| 商用 TTS | ElevenLabs、OpenAI TTS 或 Azure Neural |
| 语音克隆（已验证同意） | XTTS v2（开放）或 ElevenLabs Pro |
| 快速生成背景音乐 | Stable Audio 2.5 API、Suno 或 Udio |
| 带歌词的音乐 | Suno v4 或 Udio v1.5 |
| 音效/拟音 | AudioCraft 2、ElevenLabs SFX 或 Stable Audio Open |
| 实时语音智能体 | GPT-4o realtime 或 Gemini Live |
| 开放权重音乐研究 | MusicGen 3.3B、Stable Audio Open 1.0、AudioLDM 2 |
| 配音/翻译 | HeyGen、ElevenLabs Dubbing |

## 交付成果

保存 `outputs/skill-audio-brief.md`。该技能接收音频需求说明（任务、时长、风格、声音、许可），并输出：模型与托管方式、提示词格式（流派标签、风格描述词、结构标记）、编解码器 + 生成器 + 声码器链路、随机种子协议，以及评估方案（MOS / CLAP 分数 / TTS 的 CER / 用户 A/B 测试）。

## 练习

1. **简单。** 运行 `code/main.py` 并显式设置风格，确认生成序列符合该风格的模式。
2. **中等。** 添加延迟并行解码：模拟两条必须相差 1 个时间步的词元流，并训练联合预测器。
3. **困难。** 使用 HuggingFace transformers 在本地运行 MusicGen-small。用三个不同的提示词各生成一段 10 秒音频，再通过 A/B 测试比较风格遵循程度。

## 关键术语

| 术语 | 人们通常怎么说 | 实际含义 |
|------|-----------------|-----------------------|
| 编解码器 | “神经压缩” | 音频的编码器/解码器；典型输出是频率为 50～75 Hz 的词元。 |
| RVQ | “残差 VQ” | K 个量化器的级联；每个量化器对前一个的残差建模。 |
| 词元 | “一个编解码器符号” | 码本中的离散索引；通常有 1024 或 2048 个取值。 |
| 延迟并行 | “错开码本” | 以错开的偏移量输出 K 条词元流，从而缩短序列。 |
| 流匹配 | “音频领域 2024 年的赢家” | 路径更直的扩散替代方法；采样更快。 |
| 语音提示 | “3 秒样本” | 用来引导克隆声音的说话者嵌入或词元前缀。 |
| 梅尔频谱图 | “那张可视化图” | 对数幅度感知频谱图；许多 TTS 系统都会使用。 |
| 声码器 | “梅尔频谱转波形” | 将梅尔频谱图转回音频的神经网络组件。 |

## 生产说明：音频是一个流式传输问题

音频是唯一一种用户期望*边生成边到达*、而非一次性返回的输出模态。用生产术语来说，这意味着 TPOT（每个输出词元耗时）非常重要，因为目标吞吐率由用户的收听速度决定，而不是阅读速度。对于以约 75 词元/秒（Encodec）进行词元化的 16kHz 音频，服务器必须为每位用户生成 ≥75 词元/秒，才能保证播放流畅。

这会带来两个架构后果：

- **流匹配音频模型无法轻易流式传输。** Stable Audio 2.5 与 AudioCraft 2 会一次性渲染固定长度的片段。若要流式传输，就必须切分片段并重叠边界——可以把它理解为滑动窗口扩散；与编解码器自回归模型相比，这会增加 100～300ms 的延迟开销。

如果产品是“实时语音聊天”或“实时续写音乐”，应选择编解码器自回归路线。如果产品是“提交后渲染 30 秒片段”，流匹配在质量和总延迟上更胜一筹。

## 延伸阅读

- [Défossez 等（2022），Encodec：高保真神经音频压缩](https://arxiv.org/abs/2210.13438)——编解码器标准。
- [Zeghidour 等（2021），SoundStream](https://arxiv.org/abs/2107.03312)——首个得到广泛使用的神经音频编解码器。
- [Kumar 等（2023），采用改进 RVQGAN 的高保真音频压缩（DAC）](https://arxiv.org/abs/2306.06546)——DAC。
- [Wang 等（2023），神经编解码器语言模型是零样本文本转语音合成器（VALL-E）](https://arxiv.org/abs/2301.02111)——VALL-E。
- [Copet 等（2023），简单且可控的音乐生成（MusicGen）](https://arxiv.org/abs/2306.05284)——MusicGen。
- [Liu 等（2023），AudioLDM 2：通过自监督预训练学习整体音频生成](https://arxiv.org/abs/2308.05734)——AudioLDM 2。
- [Stability AI（2024），Stable Audio 2.5](https://stability.ai/news/introducing-stable-audio-2-5)——2025 年采用流匹配的文生音乐模型。
