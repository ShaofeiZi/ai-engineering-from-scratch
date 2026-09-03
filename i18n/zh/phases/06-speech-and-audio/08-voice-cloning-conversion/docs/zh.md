# 声音克隆与声音转换

> 声音克隆让别人的声音读出你的文本。声音转换则在保留你所说内容的同时，把你的声音变成另一个人的声音。二者都依赖同一种分解：把说话人身份与内容分离。

**Type:** 构建
**Languages:** Python
**Prerequisites:** 阶段 6 · 06（说话人识别）、阶段 6 · 07（TTS）
**Time:** 约 75 分钟

## 问题

到 2026 年，只需一段 5 秒音频和消费级 GPU，就足以高质量克隆任何人的声音。ElevenLabs、F5-TTS、OpenVoice v2、VoiceBox 都提供零样本或少样本克隆。这项技术既是福音（无障碍 TTS、配音、辅助语音），也是武器（诈骗电话、政治深度伪造、知识产权盗用）。

两个密切相关的任务：

- **声音克隆（TTS 侧）：** 文本 + 5 秒参考声音 → 使用该声音生成音频。
- **声音转换（语音侧）：** 源音频（A 说了内容 X）+ B 的参考声音 → 生成 B 说出 X 的音频。

二者都会把波形分解为（内容、说话人、韵律），再把一个来源的内容与另一个来源的说话人重新组合。

到 2026 年，交付时必须遵守一项关键约束：**欧盟（《人工智能法案》，2026 年 8 月起可执行）和加利福尼亚州（AB 2905，2025 年生效）都依法要求水印与同意门禁**。流水线必须嵌入听不见的水印，并拒绝未经同意的克隆。

## 概念

![声音克隆与转换：分解、替换说话人、重新组合](../../../../../../phases/06-speech-and-audio/08-voice-cloning-conversion/assets/voice-cloning.svg)

**零样本克隆。** 向一个已在数千名说话人上训练的模型提供 5 秒音频。说话人编码器把音频映射为说话人嵌入，TTS 解码器再以该嵌入和文本作为条件生成语音。

采用这种方法的模型包括：F5-TTS（2024）、YourTTS（2022）、XTTS v2（2024）、OpenVoice v2（2024）。

**少样本微调。** 录制目标声音的 5～30 分钟音频，用一小时对基础模型进行 LoRA 微调。质量会从“尚可”跃升至“难以分辨”。Coqui 与 ElevenLabs 都支持这种模式，社区也将它用于 F5-TTS。

**声音转换（VC）。** 分为两个家族：

- **识别—合成。** 运行类似 ASR 的模型提取内容表示（例如软音素后验概率、PPG），再使用目标说话人嵌入重新合成。对语言和口音稳健。KNN-VC（2023）、Diff-HierVC（2023）使用这种方法。
- **解耦表示。** 训练一个自动编码器，在瓶颈潜在空间中分离内容、说话人和韵律；推理时替换说话人嵌入。质量较低，但速度更快。AutoVC（2019）和 VITS-VC 变体采用这种方法。

**基于神经编解码器的克隆（2024+）。** VALL-E、VALL-E 2、NaturalSpeech 3、VoiceBox 把音频视作 SoundStream / EnCodec 生成的离散词元，并在编解码器词元上训练大型自回归或流匹配模型。对于短提示，其质量可与 ElevenLabs 相当。

### 伦理不是附加功能

**水印。** PerTh（Perth）和 SilentCipher（2024）会在音频中嵌入人耳无法察觉、约 16～32 比特的 ID。它能经受重新编码、流式传输和常见编辑，且已有可用于生产的开源实现。

**同意门禁。** 每个克隆输出都必须关联一条可验证的同意记录。“I, Rohit, on 2026-04-22, authorize this voice for X purpose.”将其存入防篡改日志。

**检测。** AASIST、RawNet2 和 Wav2Vec2-AASIST 都提供检测器。ASVspoof 2025 挑战赛公布的结果显示，面对 ElevenLabs、VALL-E 2 和 Bark 输出，顶尖检测器的 EER 为 0.8%～2.3%。

### 数据（2026）

| 模型 | 零样本？ | SECS（目标相似度） | WER（可懂度） | 参数量 |
|-------|-----------|--------------------|--------------|--------|
| F5-TTS | 是 | 0.72 | 2.1% | 335M |
| XTTS v2 | 是 | 0.65 | 3.5% | 470M |
| OpenVoice v2 | 是 | 0.70 | 2.8% | 220M |
| VALL-E 2 | 是 | 0.77 | 2.4% | 370M |
| VoiceBox | 是 | 0.78 | 2.1% | 330M |

对大多数听众而言，SECS > 0.70 通常已经无法与目标声音区分。

```figure
sp-voice-factorize
```

## 动手构建

### 第 1 步：使用识别—合成进行分解（仅在 main.py 中演示代码）

```python
def clone_pipeline(ref_audio, text, target_embedder, tts_model):
    speaker_emb = target_embedder.encode(ref_audio)
    mel = tts_model(text, speaker=speaker_emb)
    return vocoder(mel)
```

概念很简单，主要实现工作都集中在 `tts_model` 与说话人编码器中。

### 第 2 步：使用 F5-TTS 进行零样本克隆

```python
from f5_tts.api import F5TTS
tts = F5TTS()
wav = tts.infer(
    ref_file="rohit_5s.wav",
    ref_text="The quick brown fox jumps over the lazy dog.",
    gen_text="Please add milk and bread to my list.",
)
```

参考转写必须与参考音频完全一致；不匹配会破坏对齐。

### 第 3 步：使用 KNN-VC 进行声音转换

```python
import torch
from knnvc import KNNVC  # 2023 model, https://github.com/bshall/knn-vc
vc = KNNVC.load("wavlm-base-plus")
out_wav = vc.convert(source="my_voice.wav", target_pool=["alice_1.wav", "alice_2.wav"])
```

KNN-VC 使用 WavLM 提取源音频与目标池的逐帧嵌入，再把每个源帧替换为目标池中的最近邻。它是非参数方法，只需一分钟目标语音即可工作。

### 第 4 步：嵌入水印

```python
from silentcipher import SilentCipher
sc = SilentCipher(model="2024-06-01")
payload = b"consent_id:abc123;ts:1745353200"
watermarked = sc.embed(wav, sr=24000, message=payload)
detected = sc.detect(watermarked, sr=24000)   # returns payload bytes
```

载荷约为 32 比特，经 MP3 重新编码和轻微噪声后仍能检测。

### 第 5 步：同意门禁

```python
def cloned_inference(text, ref_audio, consent_record):
    assert verify_signature(consent_record), "Signed consent required"
    assert consent_record["speaker_id"] == hash_speaker(ref_audio)
    wav = tts.infer(ref_file=ref_audio, gen_text=text)
    wav = watermark(wav, payload=consent_record["id"])
    return wav
```

## 学以致用

2026 年的技术栈：

| 场景 | 选择 |
|-----------|------|
| 5 秒零样本克隆、开源 | F5-TTS 或 OpenVoice v2 |
| 商业生产级克隆 | ElevenLabs Instant Voice Clone v2.5 |
| 声音转换（改写声音） | KNN-VC 或 Diff-HierVC |
| 多说话人微调 | StyleTTS 2 + 说话人适配器 |
| 跨语言克隆 | XTTS v2 或 VALL-E X |
| 深度伪造检测 | Wav2Vec2-AASIST |

## 陷阱

- **参考转写未对齐。** F5-TTS 等模型要求参考文本与参考音频完全一致，标点也不能不同。
- **参考音频有混响。** 回声会毁掉克隆效果。应在近距离使用麦克风录制干声。
- **情绪不匹配。** “欢快”的训练参考会让所有克隆语音都很欢快。参考音频的情绪应与目标用途匹配。
- **语言泄漏。** 克隆英语说话人的声音后要求模型说法语，往往仍会带着英语口音。应使用跨语言模型（XTTS、VALL-E X）。
- **没有水印。** 自 2026 年 8 月起，在欧盟将无法合法交付。

## 交付成果

保存为 `outputs/skill-voice-cloner.md`。设计带同意门禁、水印和质量目标的声音克隆或转换流水线。

## 练习

1. **简单。** 运行 `code/main.py`。它通过计算两个“说话人”在交换前后的余弦相似度，演示说话人嵌入替换。
2. **中等。** 使用 OpenVoice v2 克隆自己的声音。测量参考语音与克隆语音之间的 SECS，并通过 Whisper 测量 CER。
3. **困难。** 为 20 段克隆音频添加 SilentCipher 水印，再以 128 kbps MP3 编码并解码，检测载荷并报告比特准确率。

## 关键术语

| 术语 | 人们通常怎么说 | 实际含义 |
|------|-----------------|-----------------------|
| 零样本克隆 | 5 秒就够了 | 预训练模型 + 说话人嵌入，无须训练。 |
| PPG | 音素后验概率图 | 逐帧 ASR 后验概率，用作与语言无关的内容表示。 |
| KNN-VC | 最近邻转换 | 用目标池中最近的帧替换每个源帧。 |
| 神经编解码器 TTS | VALL-E 风格 | 在 EnCodec/SoundStream 词元上运行的自回归模型。 |
| 水印 | 听不见的签名 | 嵌入音频中的比特，可以经受重新编码。 |
| SECS | 克隆保真度 | 目标声音与克隆声音的说话人嵌入余弦相似度。 |
| AASIST | 深度伪造检测器 | 检测合成语音的反欺骗模型。 |

## 延伸阅读

- [Chen 等（2024），F5-TTS](https://arxiv.org/abs/2410.06885)——开源零样本克隆顶尖方案。
- [Baevski 等 / Microsoft（2023），VALL-E](https://arxiv.org/abs/2301.02111)与 [VALL-E 2（2024）](https://arxiv.org/abs/2406.05370)——神经编解码器 TTS。
- [Qian 等（2019），AutoVC](https://arxiv.org/abs/1905.05879)——基于解耦表示的声音转换。
- [Baas、Waubert de Puiseau、Kamper（2023），KNN-VC](https://arxiv.org/abs/2305.18975)——基于检索的声音转换。
- [SilentCipher（2024）——音频水印](https://github.com/sony/silentcipher)——可用于生产的 32 位音频水印。
- [ASVspoof 2025 结果](https://www.asvspoof.org/)——检测器与合成器之间的攻防竞赛，更新至 2026 年。
