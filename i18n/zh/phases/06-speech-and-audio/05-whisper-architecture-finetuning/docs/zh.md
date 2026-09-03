# Whisper——架构与微调

> Whisper 是一个采用 30 秒窗口的 Transformer 编码器—解码器，在 68 万小时的多语言弱监督音频—文本对上训练。一套架构完成多种任务，稳健覆盖 99 种语言，是 2026 年的参考 ASR。

**Type:** 构建
**Languages:** Python
**Prerequisites:** 阶段 6 · 04（ASR）、阶段 5 · 10（注意力）、阶段 7 · 05（完整 Transformer）
**Time:** 约 75 分钟

## 问题

OpenAI 于 2022 年 9 月发布的 Whisper，是第一个真正商品化的 ASR 模型：输入音频即可获得文本，覆盖 99 种语言，能够抵抗噪声，并可在笔记本电脑上运行。到 2024 年，OpenAI 已发布 Large-v3 和 Turbo 变体；到 2026 年，从播客转写、语音助手到 YouTube 字幕，Whisper 都是默认基线。

但你不能永远把 Whisper 当作黑盒流水线。领域漂移会让它失效——技术术语、说话人口音、专有名词、短音频和静音都可能带来问题。你需要知道：

1. 它的内部究竟是什么。
2. 如何正确向它提供分块、流式或长音频。
3. 何时需要微调，以及如何微调。

## 概念

![Whisper 编码器—解码器、任务、分块推理与微调](../../../../../../phases/06-speech-and-audio/05-whisper-architecture-finetuning/assets/whisper.svg)

**架构。** 标准 Transformer 编码器—解码器。

- 输入：30 秒对数梅尔频谱图，80 个梅尔分箱，10 毫秒步长 → 3000 帧。较短片段补零，较长片段分块。
- 编码器：卷积下采样（步幅 2）+ `N` 个 Transformer 块。Large-v3 使用 32 层、1280 维、20 个头。
- 解码器：`N` 个 Transformer 块，包含因果自注意力以及对编码器输出的交叉注意力，大小与编码器相同。
- 输出：覆盖 51865 词元词表的 BPE 词元。

Large-v3 有 15.5 亿参数。Turbo 把解码器从 32 层缩减至 4 层，以不到 1% 的 WER 损失换来 8 倍速度提升。

**提示格式。** Whisper 是一个通过解码器提示中的特殊词元控制任务的多任务模型：

```
<|startoftranscript|><|en|><|transcribe|><|notimestamps|> Hello world.<|endoftext|>
```

- `<|en|>`——语言标签；用于控制翻译与转写行为。
- `<|transcribe|>` 或 `<|translate|>`——选择逐字转写，或将任意语言输入翻译为英语输出。
- `<|notimestamps|>`——跳过词级时间戳，速度更快。

正是提示让一个模型能够完成多种任务。把 `<|en|>` 改成 `<|fr|>`，它就会转写法语。

**30 秒窗口。** 一切都以 30 秒为固定单位。较长音频需要分块，较短音频需要填充。窗口原生不支持流式处理——这就是 WhisperX、Whisper-Streaming 和 faster-whisper 存在的原因。

**对数梅尔归一化。** `(log_mel - mean) / std`，其中统计量来自 Whisper 自己的训练语料。你*必须*使用 Whisper 的预处理（`whisper.audio.log_mel_spectrogram`），而不是 `librosa.feature.melspectrogram`。

### 2026 年的变体

| 变体 | 参数量 | 延迟（A100） | WER（LibriSpeech-clean） |
|---------|--------|----------------|------------------------|
| Tiny | 39M | 1× 实时 | 5.4% |
| Base | 74M | 1× | 4.1% |
| Small | 244M | 1× | 3.0% |
| Medium | 769M | 1× | 2.7% |
| Large-v3 | 1.55B | 2× | 1.8% |
| Large-v3-turbo | 809M | 8× | 1.58% |
| Whisper-Streaming（2024） | 1.55B | 流式 | 2.0% |

### 微调

2026 年的标准流程：

1. 收集 10～100 小时带对齐转写的目标领域音频。
2. 使用 `transformers.Seq2SeqTrainer`，并配合 `generate_with_loss` 回调。
3. 参数高效方案：在注意力层的 `q_proj`、`k_proj`、`v_proj` 上应用 LoRA，可以用不到 0.3 的 WER 代价把 GPU 内存降至四分之一。
4. 如果数据少于 10 小时，就冻结编码器，只微调解码器。
5. 使用 Whisper 自己的分词器和提示格式，绝不要替换分词器。

社区结果显示：在 20 小时医学听写数据上微调 Medium，可以把医学词汇上的 WER 从 12% 降至 4.5%；在 4 小时冰岛语数据上微调 Turbo，可以把 WER 从 18% 降至 6%。

```figure
sp-asr-attention
```

## 动手构建

### 第 1 步：开箱运行 Whisper

```python
import whisper
model = whisper.load_model("large-v3-turbo")
result = model.transcribe(
    "clip.wav",
    language="en",
    task="transcribe",
    temperature=0.0,
    condition_on_previous_text=False,  # prevents runaway repetition
)
print(result["text"])
for seg in result["segments"]:
    print(f"[{seg['start']:.2f}–{seg['end']:.2f}] {seg['text']}")
```

有三个默认值应该始终显式覆盖：`temperature=0.0`（采样默认会采用 0.0 → 0.2 → 0.4……的回退链）、`condition_on_previous_text=False`（防止幻觉级联），以及 `no_speech_threshold=0.6`（静音检测）。

### 第 2 步：分块处理长音频

```python
# whisperx is the 2026 reference for long-form with word-level timestamps
import whisperx
model = whisperx.load_model("large-v3-turbo", device="cuda", compute_type="float16")
segments = model.transcribe("1hour.mp3", batch_size=16, chunk_size=30)
```

WhisperX 增加了三项能力：（1）Silero VAD 门控；（2）通过 wav2vec 2.0 实现词级对齐；（3）通过 `pyannote.audio` 实现说话人分离。它是 2026 年生产转写的主力工具。

### 第 3 步：使用 LoRA 微调

```python
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from peft import LoraConfig, get_peft_model

model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v3-turbo")
lora = LoraConfig(
    r=16, lora_alpha=32, target_modules=["q_proj", "v_proj"],
    lora_dropout=0.1, bias="none", task_type="SEQ_2_SEQ_LM",
)
model = get_peft_model(model, lora)
# model.print_trainable_parameters()  -> ~3M trainable / 809M total
```

然后使用标准 Trainer 循环。每 1000 步保存一次检查点，在留出集上使用 WER 评估。

### 第 4 步：检查每一层学到了什么

```python
# Grab cross-attention weights during decode to see what the decoder attends to.
with torch.inference_mode():
    out = model.generate(
        input_features=features,
        return_dict_in_generate=True,
        output_attentions=True,
    )
# out.cross_attentions: layer × head × step × src_len
```

用热力图可视化——你会看到解码器各步骤扫描编码器帧时形成对角对齐。这条对角线就是 Whisper 对单词时间戳的理解。

## 学以致用

2026 年的技术栈：

| 场景 | 选择 |
|-----------|------|
| 通用英语、离线 | 通过 `whisperx` 使用 Large-v3-turbo |
| 移动端/边缘端 | int8 量化的 Whisper-Tiny 或 Moonshine |
| 多语言长音频 | 通过 `whisperx` 使用 Large-v3 + 说话人分离 |
| 低资源语言 | 使用 LoRA 微调 Medium 或 Turbo |
| 流式（2 秒延迟） | Whisper-Streaming 或 Parakeet-TDT |
| 词级时间戳 | WhisperX（通过 wav2vec 2.0 强制对齐） |

`faster-whisper`（CTranslate2 后端）是 2026 年最快的 CPU + GPU 推理运行时——输出完全一致，速度是原始实现的 4 倍。

## 2026 年仍会进入生产的陷阱

- **静音上的幻觉文本。** Whisper 的训练数据包含字幕，因此会从静音中生成“Thanks for watching!”、“Subscribe!”或歌词。调用前始终使用 VAD 把关。
- **`condition_on_previous_text` 级联。** 一次幻觉会污染后续窗口。除非需要跨块流畅衔接，否则将其设为 `False`。
- **短片段填充。** 2 秒片段补到 30 秒后，尾部静音可能引发幻觉。使用 `pad=False` 或 VAD 门控。
- **错误的梅尔统计量。** 使用 librosa 的梅尔特征而不是 Whisper 自己的处理方式，会产生近乎随机的输出。应使用 `whisper.audio.log_mel_spectrogram`。

## 交付成果

保存为 `outputs/skill-whisper-tuner.md`。根据具体领域设计 Whisper 微调或推理流水线。

## 练习

1. **简单。** 运行 `code/main.py`。它会对 Whisper 风格提示进行分词，计算解码形状预算，并打印一段 10 分钟音频的分块计划。
2. **中等。** 安装 `faster-whisper`，转写一段 10 分钟播客，并与人工转写比较 WER。尝试 `language="auto"` 与强制 `language="en"`。
3. **困难。** 使用 Hugging Face `datasets` 选择一种 Whisper 表现较差的语言（例如乌尔都语），在 2 小时数据上用 LoRA 微调 Medium 两轮，并报告 WER 变化。

## 关键术语

| 术语 | 人们通常怎么说 | 实际含义 |
|------|-----------------|-----------------------|
| 30 秒窗口 | Whisper 的限制 | 硬性输入上限；更长音频必须分块。 |
| SOT | 转写开始 | `<\|startoftranscript\|>` 启动解码器提示。 |
| 时间戳词元 | 时间对齐 | 每 0.02 秒偏移都对应 5.1 万词表中的一个特殊词元。 |
| Turbo | 快速变体 | 4 层解码器，速度提高 8 倍，WER 退化不到 1%。 |
| WhisperX | 长音频封装 | VAD + Whisper + wav2vec 对齐 + 说话人分离。 |
| LoRA 微调 | 高效微调 | 为注意力增加低秩适配器，只训练约 0.3% 的参数。 |
| 幻觉 | 隐蔽的失败 | Whisper 从噪声或静音中生成流畅英语。 |

## 延伸阅读

- [Radford 等 / OpenAI（2022），Whisper 论文](https://arxiv.org/abs/2212.04356)——原始架构与训练方法。
- [OpenAI（2024），Whisper Large-v3-turbo 发布说明](https://github.com/openai/whisper/discussions/2363)——4 层解码器，8 倍加速。
- [Bain 等（2023），WhisperX](https://arxiv.org/abs/2303.00747)——长音频、词级对齐、说话人分离。
- [Systran——faster-whisper 代码库](https://github.com/SYSTRAN/faster-whisper)——由 CTranslate2 支持，速度提高 4 倍。
- [Hugging Face——Whisper 微调教程](https://huggingface.co/blog/fine-tune-whisper)——规范的 LoRA/全量微调教程。
