# 音乐生成——MusicGen、Stable Audio、Suno 与许可证巨震

> 2026 年的音乐生成领域：Suno v5 与 Udio v4 主导商业市场；MusicGen、Stable Audio Open 和 ACE-Step 领跑开源领域。技术问题大体已经解决，法律问题（Warner Music 5 亿美元和解、UMG 和解）则在 2025～2026 年重塑了整个行业。

**Type:** 构建
**Languages:** Python
**Prerequisites:** 阶段 6 · 02（频谱图）、阶段 4 · 10（扩散模型）
**Time:** 约 75 分钟

## 问题

文本 → 一段 30 秒至 4 分钟、包含歌词、人声和曲式结构的音乐。其中有三个子问题：

1. **纯音乐生成。** “lo-fi hip-hop drums with warm keys”等文本 → 音频。代表模型有 MusicGen、Stable Audio、AudioLDM。
2. **歌曲生成（含人声 + 歌词）。** “Country song about rainy Texas nights”→ 完整歌曲。代表模型有 Suno、Udio、YuE、ACE-Step。
3. **条件式/可控生成。** 延长已有片段、重新生成桥段、更换曲风、分离音轨或局部重绘。Udio 的局部重绘 + 音轨分离是 2026 年需要看齐的功能。

## 概念

![音乐生成：基于词元的语言模型与扩散模型，以及 2026 年模型版图](../assets/music-generation.svg)

### 在神经编解码器词元上运行语言模型

Meta 的 **MusicGen**（2023，MIT 许可证）及其许多衍生模型：以文本/旋律嵌入为条件，自回归预测 EnCodec 词元（32 kHz，4 个码本），再用 EnCodec 解码。参数量从 3 亿到 33 亿不等。它是强大的基线，但在超过 30 秒后表现吃力。

**ACE-Step**（开源，4B XL 于 2026 年 4 月发布）把这一方法扩展到以歌词为条件生成完整歌曲，是开源社区最接近 Suno 的方案。

### 在梅尔特征或潜变量上进行扩散

**Stable Audio（2023）**与 **Stable Audio Open（2024）**：在压缩音频上执行潜在扩散。擅长循环乐段、声音设计和环境纹理，不擅长结构完整的整首歌曲。

**AudioLDM / AudioLDM2**：采用类似文生图的潜在扩散实现文本生成音频，并扩展到音乐、音效和语音。

### 混合架构（生产）——Suno、Udio、Lyria

权重封闭。很可能采用自回归编解码器语言模型 + 基于扩散的声码器，并配备专用的人声、鼓组与旋律头。Suno v5（2026）以 ELO 1293 领跑质量榜。Udio v4 则增加了局部重绘 + 音轨分离功能（可分别下载贝斯、鼓、人声）。

### 评估

- **FAD（Fréchet 音频距离）。** 使用 VGGish 或 PANNs 特征，计算生成音频与真实音频分布之间的嵌入级距离。越低越好。MusicGen small 在 MusicCaps 上的 FAD 为 4.5，顶尖水平约为 3.0。
- **音乐性（主观）。** 人类偏好。Suno v5 以 ELO 1293 领先。
- **文本—音频对齐。** 计算提示与输出之间的 CLAP 分数。
- **音乐性瑕疵。** 节拍错位的转场、人声短语漂移、30 秒后丢失结构。

## 2026 年模型版图

| 模型 | 参数量 | 长度 | 人声 | 许可证 |
|-------|--------|--------|--------|---------|
| MusicGen-large | 3.3B | 30 秒 | 无 | MIT |
| Stable Audio Open | 1.2B | 47 秒 | 无 | Stability 非商业许可 |
| ACE-Step XL（2026 年 4 月） | 4B | &gt; 2 分钟 | 有 | Apache-2.0 |
| YuE | 7B | &gt; 2 分钟 | 有，多语言 | Apache-2.0 |
| Suno v5（封闭） | ? | 4 分钟 | 有，ELO 1293 | 商业许可 |
| Udio v4（封闭） | ? | 4 分钟 | 有 + 分轨 | 商业许可 |
| Google Lyria 3（封闭） | ? | 实时 | 有 | 商业许可 |
| MiniMax Music 2.5 | ? | 4 分钟 | 有 | 商业 API |

## 法律格局（2025～2026）

- **Warner Music 与 Suno 和解。** 金额 5 亿美元。如今 WMG 对 Suno 上的 AI 相似性、音乐权利与用户生成曲目拥有监督权。UMG 与 Udio 也达成了类似和解。
- **欧盟《人工智能法案》** + **加利福尼亚州 SB 942**：必须披露由 AI 生成的音乐。
- 采用 MIT 许可证的 **Riffusion / MusicGen** 没有合规包袱，但也无法提供商业级人声。

可以安全交付的模式：

1. 只生成纯音乐（MusicGen、Stable Audio Open，输出采用 MIT/CC0）。
2. 使用商业 API（Suno、Udio、ElevenLabs Music），并为每次生成取得许可。
3. 在自有或已授权的曲库上训练（大多数企业最终都会走这条路）。
4. 为生成内容添加水印与元数据。

```figure
sp-codec-tokens
```

## 动手构建

### 第 1 步：使用 MusicGen 生成

```python
from audiocraft.models import MusicGen
import torchaudio

model = MusicGen.get_pretrained("facebook/musicgen-small")
model.set_generation_params(duration=10)
wav = model.generate(["upbeat synthwave with driving drums, 128 BPM"])
torchaudio.save("out.wav", wav[0].cpu(), 32000)
```

共有三种大小：`small`（300M，快速）、`medium`（1.5B）、`large`（3.3B）。要判断“这个想法能否成立”，小模型已经足够。

### 第 2 步：旋律条件

```python
melody, sr = torchaudio.load("humming.wav")
wav = model.generate_with_chroma(
    ["jazz piano cover"],
    melody.squeeze(),
    sr,
)
```

MusicGen-melody 接收色度图，在替换音色的同时保留旋律。适合“把这段旋律改成弦乐四重奏”一类请求。

### 第 3 步：FAD 评估

```python
from frechet_audio_distance import FrechetAudioDistance
fad = FrechetAudioDistance()

fad.get_fad_score("generated_folder/", "reference_folder/")
```

它计算 VGGish 嵌入分布之间的距离，适合按流派进行回归测试，但不能代替人类听众。

### 第 4 步：加入大语言模型—音乐工作流

结合第 7～8 课的思想：

```python
prompt = "Write a 30-second jazz loop. Describe the drums, bass, and piano voicing."
description = llm.complete(prompt)
music = musicgen.generate([description], duration=30)
```

## 学以致用

| 目标 | 技术栈 |
|------|-------|
| 纯音乐声音设计 | Stable Audio Open |
| 游戏/自适应音乐 | Google Lyria RealTime（封闭） |
| 带人声的完整歌曲（商业） | Suno v5 或 Udio v4，并取得明确许可 |
| 带人声的完整歌曲（开放） | ACE-Step XL 或 YuE |
| 简短广告歌曲 | 使用哼唱参考为 MusicGen 提供旋律条件 |
| 音乐视频背景 | MusicGen + Stable Video Diffusion |

## 2026 年仍会进入生产的陷阱

- **版权洗白提示。** “Song in the style of Taylor Swift”——商业版 Suno/Udio 如今会过滤此类提示，开放模型不会。应添加自己的过滤列表。
- **超过 30 秒后的重复/漂移。** 自回归模型会陷入循环。可以交叉淡化多次生成结果，或使用 ACE-Step 保持结构连贯。
- **速度漂移。** 模型会偏离指定 BPM。在提示中加入 BPM 标签，并使用 librosa 的 `beat_track` 进行后过滤。
- **人声可懂度。** Suno 表现极佳；开放模型的歌词往往含混不清。如果歌词很重要，应使用商业 API 或微调。
- **单声道输出。** 开放模型生成单声道或伪立体声。应使用适当的立体声重建方法升级（ezst、Cartesia 的立体声扩散）。

## 交付成果

保存为 `outputs/skill-music-designer.md`。为音乐生成部署选择模型、许可策略、时长/结构方案和披露元数据。

## 练习

1. **简单。** 运行 `code/main.py`。它会用 ASCII 符号生成一段“生成式”和弦进行与鼓点模式——音乐生成的卡通版本。你也可以使用任意 MIDI 渲染器播放它。
2. **中等。** 安装 `audiocraft`，使用 MusicGen-small 为四种流派提示生成 10 秒片段，并对照参考流派集合测量 FAD。
3. **困难。** 使用 ACE-Step（或 MusicGen-melody）为同一旋律生成三种不同音色提示的变体。计算它们与提示的 CLAP 相似度，以验证对齐程度。

## 关键术语

| 术语 | 人们通常怎么说 | 实际含义 |
|------|-----------------|-----------------------|
| FAD | 音频 FID | 真实音频与生成音频的嵌入分布之间的 Fréchet 距离。 |
| 色度图 | 用音高表示旋律 | 每帧一个 12 维向量；用作旋律条件输入。 |
| 分轨 | 乐器音轨 | 分离为贝斯、鼓、人声、旋律等 WAV 文件。 |
| 局部重绘 | 重新生成一个区段 | 遮蔽一个时间窗口，只让模型重新生成该部分。 |
| CLAP | 文本—音频 CLIP | 对比式音频—文本嵌入，用于评估文本—音频对齐。 |
| EnCodec | 音乐编解码器 | MusicGen 使用的 Meta 神经编解码器；32 kHz，4 个码本。 |

## 延伸阅读

- [Copet 等（2023），MusicGen](https://arxiv.org/abs/2306.05284)——开放自回归基准。
- [Evans 等（2024），Stable Audio Open](https://arxiv.org/abs/2407.14358)——声音设计的默认方案。
- [ACE-Step](https://github.com/ace-step/ACE-Step)——2026 年 4 月发布的开放 4B 完整歌曲生成器。
- [Suno v5 平台文档](https://suno.com)——商业质量领先者。
- [AudioLDM2](https://arxiv.org/abs/2308.05734)——用于音乐与音效的潜在扩散。
- [WMG—Suno 和解报道](https://www.musicbusinessworldwide.com/suno-warner-music-settlement/)——2025 年 11 月的先例。
