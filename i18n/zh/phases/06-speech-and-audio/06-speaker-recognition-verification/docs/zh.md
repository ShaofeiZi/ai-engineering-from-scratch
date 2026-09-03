# 说话人识别与验证

> ASR 问“他说了什么？”，说话人识别问“是谁说的？”二者的数学形式看起来相同——嵌入加余弦相似度——但每项生产决策都取决于一个 EER 数字。

**Type:** 构建
**Languages:** Python
**Prerequisites:** 阶段 6 · 02（频谱图与梅尔特征）、阶段 5 · 22（嵌入模型）
**Time:** 约 45 分钟

## 问题

用户说出一段口令。你要判断：此人是否就是其声称的身份（*验证*，1:1），还是注册库中的哪一个人（*识别*，1:N）？又或者两者都不是——这是一位未知说话人（*开放集*）？

2018 年以前：GMM-UBM + i-vector。EER 尚可，但容易受到通道变化（电话与笔记本电脑）和情绪影响。2018～2022 年：x-vector（使用角度间隔训练的 TDNN 骨干网络）。2022 年以后：ECAPA-TDNN 与 WavLM-large 嵌入。到 2026 年，这个领域由三个模型和一个指标主导。

这个指标就是 **EER**——等错误率。调整决策阈值，使错误接受率等于错误拒绝率，二者的交点就是 EER。每篇论文、每张排行榜和每次采购评估都会使用它。

## 概念

![注册与验证流水线：嵌入 + 余弦相似度 + EER](../../../../../../phases/06-speech-and-audio/06-speaker-recognition-verification/assets/speaker-verification.svg)

**流水线。** 注册：录制目标说话人 5～30 秒的语音，计算定长嵌入（ECAPA-TDNN 为 192 维，WavLM-large 为 256 维）。验证：获得测试话语的嵌入，计算余弦相似度，再与阈值比较。

**ECAPA-TDNN（2020，2026 年仍占主导）。** Emphasized Channel Attention, Propagation and Aggregation - Time-Delay Neural Network。它使用带压缩—激励的一维卷积块、多头注意力池化，再通过线性层输出 192 维向量。模型在 VoxCeleb 1+2（2700 位说话人、110 万段话语）上使用加性角度间隔损失（AAM-softmax）训练。

**WavLM-SV（2022+）。** 使用 AAM 损失微调预训练 WavLM-large 自监督学习骨干网络。质量更高，但速度更慢——体积超过 300 MB，而 ECAPA-TDNN 只有 15 MB。

**x-vector（基线）。** TDNN + 统计池化。经典方法，在 CPU/边缘端仍然实用。

**AAM-softmax。** 在角度空间中为正确类别增加间隔 `m` 的标准 softmax：`cos(θ + m)`。它会迫使不同类别在角度上彼此分离。典型值为 `m=0.2`，缩放系数 `s=30`。

### 评分

- **余弦相似度。** 计算注册嵌入与测试嵌入之间的余弦值，并根据阈值作出决策。
- **PLDA（概率 LDA）。** 把嵌入投影到一个潜在空间，使同一说话人与不同说话人的似然比可以闭式计算。在余弦相似度之上加入 PLDA，可把 EER 降低 10%～20%。它是 2020 年前的标准方案，如今只用于封闭集设置。
- **分数归一化。** `S-norm` 或 `AS-norm`：根据一组冒充者分数的均值与标准差，对每项分数进行归一化。这对于跨领域评估至关重要。

### 应当了解的数字（2026）

| 模型 | VoxCeleb1-O EER | 参数量 | 吞吐量（A100） |
|-------|-----------------|--------|-------------------|
| x-vector（经典） | 3.10% | 5 M | 400× 实时 |
| ECAPA-TDNN | 0.87% | 15 M | 200× 实时 |
| WavLM-SV large | 0.42% | 316 M | 20× 实时 |
| Pyannote 3.1 分割 + 嵌入 | 0.65% | 6 M | 100× 实时 |
| ReDimNet（2024） | 0.39% | 24 M | 100× 实时 |

### 说话人分离

在多人音频中判断“谁在何时说话”。流水线为：VAD → 分段 → 嵌入每个片段 → 聚类（凝聚式或谱聚类）→ 平滑边界。现代技术栈是 `pyannote.audio` 3.1，它把说话人分割、嵌入与聚类封装在一次调用之后。2026 年在 AMI 上的顶尖 DER 约为 15%（2022 年为 23%）。

```figure
sp-eer-crossover
```

## 动手构建

### 第 1 步：基于 MFCC 统计量的玩具嵌入

```python
def embed_mfcc_stats(signal, sr):
    frames = featurize_mfcc(signal, sr, n_mfcc=13)
    mean = [sum(f[i] for f in frames) / len(frames) for i in range(13)]
    std = [
        math.sqrt(sum((f[i] - mean[i]) ** 2 for f in frames) / len(frames))
        for i in range(13)
    ]
    return mean + std  # 26-d
```

它远远达不到顶尖水平，只用于教学。`code/main.py` 会把它作为合成说话人数据上的概念验证。

### 第 2 步：余弦相似度 + 阈值

```python
def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb) if na and nb else 0.0

def verify(enroll, test, threshold=0.75):
    return cosine(enroll, test) >= threshold
```

### 第 3 步：从相似度样本对计算 EER

```python
def eer(same_scores, diff_scores):
    thresholds = sorted(set(same_scores + diff_scores))
    best = (1.0, 1.0, 0.0)  # (fa, fr, threshold)
    for t in thresholds:
        fr = sum(1 for s in same_scores if s < t) / len(same_scores)
        fa = sum(1 for s in diff_scores if s >= t) / len(diff_scores)
        if abs(fa - fr) < abs(best[0] - best[1]):
            best = (fa, fr, t)
    return (best[0] + best[1]) / 2, best[2]
```

返回（EER，EER 对应阈值），两者都要报告。

### 第 4 步：使用 SpeechBrain 投入生产

```python
from speechbrain.pretrained import EncoderClassifier

clf = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")

# enroll: average the embeddings of 3-5 clean samples
enroll = torch.stack([clf.encode_batch(load(x)) for x in enrollment_clips]).mean(0)
# verify
score = clf.similarity(enroll, clf.encode_batch(load("test.wav"))).item()
verdict = score > 0.25   # ECAPA typical threshold; tune on your data
```

### 第 5 步：使用 pyannote 进行说话人分离

```python
from pyannote.audio import Pipeline

pipe = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1")
diarization = pipe("meeting.wav", num_speakers=None)
for turn, _, speaker in diarization.itertracks(yield_label=True):
    print(f"{turn.start:.1f}–{turn.end:.1f}  {speaker}")
```

## 学以致用

2026 年的技术栈：

| 场景 | 选择 |
|-----------|------|
| 封闭集 1:1 验证、边缘端 | ECAPA-TDNN + 余弦阈值 |
| 开放集验证、云端 | WavLM-SV + AS-norm |
| 说话人分离（会议、播客） | `pyannote/speaker-diarization-3.1` |
| 反欺骗（重放/深度伪造检测） | AASIST 或 RawNet2 |
| 微型嵌入式设备（KWS + 注册） | Titanet-Small（NeMo） |

## 陷阱

- **通道不匹配。** 在 VoxCeleb（网络视频）上训练的模型不等于电话音频模型。始终在目标通道上评估。
- **短话语。** 测试音频短于 3 秒时，EER 会急剧恶化。
- **注册音频带噪。** 一段嘈杂的注册音频会污染锚点。应使用至少 3 段干净样本并取平均。
- **对所有条件使用固定阈值。** 始终在目标领域的留出开发集上调节阈值。
- **对未归一化嵌入计算余弦相似度。** 应先执行 L2 归一化，否则模长会主导结果。

## 交付成果

保存为 `outputs/skill-speaker-verifier.md`。选择模型、注册规程、阈值调优方案和反欺诈保障措施。

## 练习

1. **简单。** 运行 `code/main.py`。它会构建合成“说话人”（不同音调特征），完成注册，并在包含 100 个样本对的试验列表上计算 EER。
2. **中等。** 在 30 段 VoxCeleb1 话语（5 位说话人 × 每人 6 段）上使用 SpeechBrain ECAPA，分别通过余弦相似度和 PLDA 计算 EER。
3. **困难。** 使用 `pyannote.audio` 构建完整的注册 → 说话人分离 → 验证流水线，在 AMI 开发集上评估 DER。

## 关键术语

| 术语 | 人们通常怎么说 | 实际含义 |
|------|-----------------|-----------------------|
| EER | 核心指标 | 错误接受率等于错误拒绝率时的阈值。 |
| 验证 | 1:1 | “这是 Alice 吗？” |
| 识别 | 1:N | “正在说话的是谁？” |
| 开放集 | 可能出现未知者 | 测试集可以包含尚未注册的说话人。 |
| 注册 | 登记身份 | 计算说话人的参考嵌入。 |
| AAM-softmax | 损失函数 | 带加性角度间隔的 softmax，迫使簇彼此分离。 |
| PLDA | 经典评分方法 | 概率 LDA；在嵌入上进行似然比评分。 |
| DER | 说话人分离指标 | 说话人分离错误率——漏检 + 误报 + 混淆。 |

## 延伸阅读

- [Snyder 等（2018），X-Vector：用于说话人识别的稳健深度神经网络嵌入](https://www.danielpovey.com/files/2018_icassp_xvectors.pdf)——经典深度嵌入论文。
- [Desplanques 等（2020），ECAPA-TDNN](https://arxiv.org/abs/2005.07143)——2020～2026 年占主导地位的架构。
- [Chen 等（2022），WavLM：用于全栈语音处理的大规模自监督预训练](https://arxiv.org/abs/2110.13900)——用于说话人验证与分离的自监督学习骨干网络。
- [Bredin 等（2023），pyannote.audio 3.1](https://github.com/pyannote/pyannote-audio)——生产级说话人分离 + 嵌入技术栈。
- [VoxCeleb 排行榜（更新至 2026）](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/)——各模型当前 EER 排名。
