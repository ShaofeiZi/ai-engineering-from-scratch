# 语音反欺骗与音频水印——ASVspoof 5、AudioSeal、WaveVerify

> 声音克隆的发展速度超过了防御技术。2026 年的生产语音系统需要两样东西：一个区分真实与伪造语音的检测器（AASIST、RawNet2），以及一个能够经受压缩与编辑的水印（AudioSeal）。两者缺一，就不应交付声音克隆。

**Type:** 构建
**Languages:** Python
**Prerequisites:** 阶段 6 · 06（说话人识别）、阶段 6 · 08（声音克隆）
**Time:** 约 75 分钟

## 问题

三种相互关联的防御：

1. **反欺骗/深度伪造检测。** 给定一段音频，判断它是合成的还是真实的。ASVspoof 基准（ASVspoof 2019 → 2021 → 5）是黄金标准。
2. **音频水印。** 在生成音频中嵌入人耳无法察觉、之后可由检测器提取的信号。AudioSeal（Meta）和 WavMark 是开放方案。
3. **认证溯源。** 对音频文件与元数据进行加密签名。C2PA / Content Authenticity Initiative。

检测用于应对不配合的攻击者，水印用于满足合规要求——AI 生成的音频应该可以被识别。2026 年两者都不可或缺。

## 概念

![反欺骗、音频水印与溯源——三层防御](../../../../../../phases/06-speech-and-audio/16-anti-spoofing-audio-watermarking/assets/spoofing-watermark.svg)

### ASVspoof 5——2024～2025 年基准

与之前版本相比，最大的变化包括：

- **众包数据**（而不是干净的录音室数据）——更接近真实条件。
- **约 2000 位说话人**（此前约 100 位）。
- **32 种攻击算法。** TTS + 声音转换 + 对抗扰动。
- **两条赛道。** 独立检测的对抗措施（CM）；面向生物识别系统的抗欺骗 ASV（SASV）。

ASVspoof 5 上的顶尖水平约为 7.23% EER；在较早的 ASVspoof 2019 LA 上则为 0.42% EER。部署到真实环境时，对自然采集音频应预期 5%～10% EER。

### AASIST 与 RawNet2——检测模型家族

**AASIST**（2021，持续更新至 2026）。在频谱特征上使用图注意力，是当前 ASVspoof 5 对抗措施任务的顶尖方案。

**RawNet2。** 在原始波形上使用卷积前端，再接 TDNN 骨干网络。它是更简单的基线，经过微调后仍有竞争力。

**NeXt-TDNN + 自监督学习特征。** 2025 年的变体：ECAPA 风格架构 + WavLM 特征 + 焦点损失。在 ASVspoof 2019 LA 上达到 0.42% EER。

### AudioSeal——2024 年的默认水印

Meta 的 **AudioSeal**（2024 年 1 月发布，2024 年 12 月推出 v0.2）。关键设计包括：

- **可定位。** 以 16 kHz 的采样分辨率逐帧检测水印（1/16000 秒）。
- **联合训练生成器与检测器。** 生成器学习嵌入不可听信号，检测器学习在经过数据增强后找出它。
- **稳健。** 可以经受 MP3/AAC 压缩、均衡、±10% 变速，以及 +10 dB SNR 的噪声混合。
- **快速。** 检测器达到 485 倍实时速度，比 WavMark 快 1000 倍。
- **容量。** 每段话语可嵌入 16 比特载荷（可编码模型 ID、生成时间戳、用户 ID）。

### WavMark

AudioSeal 之前的开放基线。采用可逆神经网络，每秒 32 比特。问题包括：

- 通过暴力搜索进行同步，速度很慢。
- 高斯噪声或 MP3 压缩可以将其移除。
- 不适合实时处理。

### WaveVerify（2025 年 7 月）

它针对 AudioSeal 的弱点，尤其是反转、变速等时间变换。使用基于 FiLM 的生成器与混合专家检测器。在标准攻击上的表现可与 AudioSeal 竞争，并能处理时间编辑。

### 攻击者利用的缺口

AudioMarkBench 指出：“在音高偏移下，所有水印的比特恢复准确率都低于 0.6，表明水印近乎完全移除。”**音高偏移是通用攻击。** 2026 年没有任何水印能完全抵抗激进的音高修改。因此，除水印外还必须部署检测器（AASIST）。

### C2PA / 内容真实性倡议

它不是机器学习技术，而是一种清单格式。音频文件携带关于创建工具、作者和日期的加密签名元数据。Audobox / Seamless 使用它。它有助于溯源，但恶意行为者只需重新编码并删除元数据即可绕过。

```figure
v4-audio-watermark
```

## 动手构建

### 第 1 步：简单的频谱特征检测器（玩具实现）

```python
def spectral_rolloff(spec, percentile=0.85):
    cum = 0
    total = sum(spec)
    if total == 0:
        return 0
    threshold = total * percentile
    for k, v in enumerate(spec):
        cum += v
        if cum >= threshold:
            return k
    return len(spec) - 1

def is_suspicious(audio):
    spec = magnitude_spectrum(audio)
    rolloff = spectral_rolloff(spec)
    return rolloff / len(spec) > 0.92
```

合成语音的高频能量往往异常平坦。生产检测器应使用 AASIST，而不是这段代码；但其直觉是正确的。

### 第 2 步：使用 AudioSeal 嵌入并检测

```python
from audioseal import AudioSeal
import torch

generator = AudioSeal.load_generator("audioseal_wm_16bits")
detector = AudioSeal.load_detector("audioseal_detector_16bits")

audio = load_wav("generated.wav", sr=16000)[None, None, :]
payload = torch.tensor([[1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0]])
watermark = generator.get_watermark(audio, sample_rate=16000, message=payload)
watermarked = audio + watermark

result, decoded_payload = detector.detect_watermark(watermarked, sample_rate=16000)
# result: float in [0, 1] — probability of watermark presence
# decoded_payload: 16 bits; match against embedded payload
```

### 第 3 步：评估——EER

```python
def eer(real_scores, fake_scores):
    thresholds = sorted(set(real_scores + fake_scores))
    best = (1.0, 0.0)
    for t in thresholds:
        far = sum(1 for s in fake_scores if s >= t) / len(fake_scores)
        frr = sum(1 for s in real_scores if s < t) / len(real_scores)
        if abs(far - frr) < best[0]:
            best = (abs(far - frr), (far + frr) / 2)
    return best[1]
```

### 第 4 步：生产集成

```python
def safe_tts(text, voice, clone_reference=None):
    if clone_reference is not None:
        verify_consent(user_id, clone_reference)
    audio = tts_model.synthesize(text, voice)
    audio_with_wm = audioseal_embed(audio, payload=build_payload(user_id, model_id))
    manifest = c2pa_sign(audio_with_wm, user_id, timestamp=now())
    return audio_with_wm, manifest
```

每次生成都应附带：（1）水印；（2）签名清单；（3）符合保留策略的审计日志。

## 学以致用

| 用例 | 防御措施 |
|----------|---------|
| 交付 TTS/声音克隆 | 每次输出都嵌入 AudioSeal（不可妥协） |
| 生物识别语音解锁 | AASIST + ECAPA 集成，并加入活体挑战 |
| 呼叫中心欺诈检测 | 对 20% 的来电采样运行 AASIST |
| 播客真实性 | 上传时使用 C2PA 签名；若由 AI 生成，再加入 AudioSeal |
| 研究/训练检测器 | ASVspoof 5 的训练集、开发集、评估集 |

## 陷阱

- **添加水印后从不运行检测器。** 毫无意义。应把检测器加入 CI。
- **检测但不校准。** 在 ASVspoof LA 上训练的 AASIST 会过拟合，真实环境中的准确率会下降。必须在自己的领域上校准。
- **音高偏移缺口。** 激进的音高偏移可以移除大多数水印。需要检测器作为后备。
- **删除元数据后重新托管。** 重新编码即可轻易绕过 C2PA。始终同时使用加密签名与感知层（水印）防御。
- **把活体挑战当作检测。** 要求用户说出随机短语，可以防止重放攻击，却无法阻止实时克隆。

## 交付成果

保存为 `outputs/skill-spoof-defender.md`。为语音生成部署选择检测模型、水印、溯源清单和运维方案。

## 练习

1. **简单。** 运行 `code/main.py`。在合成音频上演示玩具检测器与玩具水印的嵌入/检测。
2. **中等。** 安装 `audioseal`，在一段 TTS 输出中嵌入 16 比特载荷，再解码。给音频加入噪声，并测量比特恢复准确率。
3. **困难。** 在 ASVspoof 2019 LA 上微调 RawNet2 或 AASIST，并测量 EER。再使用一组留出的 F5-TTS 生成片段进行测试，观察分布外检测如何退化。

## 关键术语

| 术语 | 人们通常怎么说 | 实际含义 |
|------|-----------------|-----------------------|
| ASVspoof | 基准 | 两年一届的挑战赛；2024 年版为 ASVspoof 5。 |
| CM（对抗措施） | 检测器 | 区分真实语音与合成/转换语音的分类器。 |
| SASV | 说话人验证 + CM | 集成生物识别与欺骗检测。 |
| AudioSeal | Meta 水印 | 可定位、16 比特载荷，速度比 WavMark 快 485 倍。 |
| 比特恢复准确率 | 水印存活程度 | 遭受攻击后成功恢复的载荷比特比例。 |
| C2PA | 溯源清单 | 关于创建过程与作者身份的加密元数据。 |
| AASIST | 检测器家族 | 基于图注意力的顶尖反欺骗模型。 |

## 延伸阅读

- [Todisco 等（2024），ASVspoof 5](https://dl.acm.org/doi/10.1016/j.csl.2025.101825)——当前基准。
- [Defossez 等（2024），AudioSeal](https://arxiv.org/abs/2401.17264)——默认水印方案。
- [San Roman 等（2025），WaveVerify](https://arxiv.org/abs/2507.21150)——用于抵抗时间变换攻击的混合专家检测器。
- [Jung 等（2022），AASIST](https://arxiv.org/abs/2110.01200)——顶尖检测骨干网络。
- [AudioMarkBench（2024）](https://proceedings.neurips.cc/paper_files/paper/2024/file/5d9b7775296a641a1913ab6b4425d5e8-Paper-Datasets_and_Benchmarks_Track.pdf)——稳健性评估。
- [C2PA 规范](https://c2pa.org/specifications/specifications/)——溯源清单格式。
