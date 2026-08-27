# 音频语言模型——Qwen2.5-Omni、Audio Flamingo、GPT-4o Audio

> 2026 年的音频语言模型能够综合推理语音、环境声音和音乐。Qwen2.5-Omni-7B 在 MMAU-Pro 上比肩 GPT-4o Audio，Audio Flamingo Next 在 LongAudioBench 上胜过 Gemini 2.5 Pro。开放与封闭模型之间的差距几乎已经消失——唯独多音频任务例外，所有模型都接近随机水平。

**Type:** 学习
**Languages:** Python
**Prerequisites:** 阶段 6 · 04（ASR）、阶段 12 · 03（视觉语言模型）、阶段 7 · 10（音频 Transformer）
**Time:** 约 45 分钟

## 问题

你有一段 5 秒音频：狗叫、有人喊“stop!”，随后归于寂静。可以提出的问题横跨多个维度：

- **转写。** “说了什么？”——属于 ASR 范畴。
- **语义推理。** “这个人有危险吗？”——需要综合理解狗叫、喊声与随后的静音。
- **音乐推理。** “哪些乐器在演奏旋律？”
- **长音频检索。** “在这场 90 分钟的讲座中，讲师在哪里解释了梯度下降？”

能够通过一个提示回答所有这些问题的模型，就是**音频语言模型**（LALM / ALM）。它不同于纯 ASR：LALM 会生成自由形式的自然语言答案，而不只是转写文本。

## 概念

![音频语言模型：音频编码器 + 投影器 + 大语言模型解码器](../assets/alm-architecture.svg)

### 三组件模板

2026 年的每个 LALM 都采用相同骨架：

1. **音频编码器。** Whisper 编码器、BEATs、CLAP、WavLM，或各模型自己的编码器。
2. **投影器。** 用线性层或 MLP 把音频编码器特征桥接到大语言模型的词元嵌入空间。
3. **大语言模型。** 基于 Llama / Qwen / Gemma 的解码器。接收交错的文本与音频词元，生成文本。

训练过程：

- **阶段 1。** 冻结编码器和大语言模型，只在 ASR/字幕数据上训练投影器。
- **阶段 2。** 在遵循指令的音频任务（问答、推理、音乐理解）上进行全量或 LoRA 微调。
- **阶段 3（可选）。** 语音输入/语音输出功能会增加语音解码器。Qwen2.5-Omni 和 AF3-Chat 都支持这种方式。

### 2026 年模型版图

| 模型 | 骨干网络 | 音频编码器 | 输出模态 | 获取方式 |
|-------|----------|---------------|-----------------|--------|
| Qwen2.5-Omni-7B | Qwen2.5-7B | 自定义 + Whisper | 文本 + 语音 | Apache-2.0 |
| Qwen3-Omni | Qwen3 | 自定义 | 文本 + 语音 | Apache-2.0 |
| Audio Flamingo 3 | Qwen2 | AF-CLAP | 文本 | NVIDIA 非商业许可 |
| Audio Flamingo Next | Qwen2 | AF-CLAP v2 | 文本 | NVIDIA 非商业许可 |
| SALMONN | Vicuna | Whisper + BEATs | 文本 | Apache-2.0 |
| LTU / LTU-AS | Llama | CAV-MAE | 文本 | Apache-2.0 |
| GAMA | Llama | AST + Q-Former | 文本 | Apache-2.0 |
| Gemini 2.5 Flash/Pro（封闭） | Gemini | 专有 | 文本 + 语音 | API |
| GPT-4o Audio（封闭） | GPT-4o | 专有 | 文本 + 语音 | API |

### 基准现实检验（2026）

**MMAU-Pro。** 包含 1800 个问答对，覆盖语音、声音、音乐和混合任务，也包括多音频子集。

| 模型 | 总体 | 语音 | 声音 | 音乐 | 多音频 |
|-------|---------|--------|-------|-------|-------------|
| Gemini 2.5 Pro | 约 60% | 73.4% | 51.9% | 64.9% | 约 22% |
| Gemini 2.5 Flash | 约 57% | 73.4% | 50.5% | 64.9% | 21.2% |
| GPT-4o Audio | 52.5% | — | — | — | 26.5% |
| Qwen2.5-Omni-7B | 52.2% | 57.4% | 47.6% | 61.5% | 约 20% |
| Audio Flamingo 3 | 约 54% | — | — | — | — |
| Audio Flamingo Next | LongAudioBench 顶尖水平 | — | — | — | — |

**多音频这一列对所有模型都很不利。** 四选一的随机准确率为 25%，大多数模型只在这一水平附近。LALM 仍不善于比较两段音频。

### LALM 在 2026 年适用的场景

- **呼叫中心录音的合规审计。** “客服是否提到了规定披露事项？”
- **无障碍功能。** 向听障用户描述声音事件，而不只是转写语音。
- **内容审核。** 综合检测暴力语言、威胁语气与背景环境。
- **播客/会议章节划分。** 生成语义摘要，而不只是划分说话轮次。
- **音乐目录分析。** “找出所有 B 段发生转调的曲目。”

### 它们还不适用的场景

- 精细音乐理论分析（低于和弦层级）。
- 长对话中带说话人归属的推理（超过 10 分钟后退化）。
- 多音频比较（22%～26% 只勉强达到随机水平）。
- 实时流式推理（大多数模型都采用离线批量推理）。

```figure
v4-alm-tokens
```

## 动手构建

### 第 1 步：查询 Qwen2.5-Omni

```python
from transformers import AutoModelForCausalLM, AutoProcessor

processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-Omni-7B")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-Omni-7B", torch_dtype="auto")

audio, sr = load_wav("clip.wav", sr=16000)
messages = [{
    "role": "user",
    "content": [
        {"type": "audio", "audio": audio},
        {"type": "text", "text": "What sounds do you hear, and what's happening?"},
    ],
}]
inputs = processor.apply_chat_template(messages, tokenize=True, return_tensors="pt")
output = model.generate(**inputs, max_new_tokens=200)
print(processor.decode(output[0], skip_special_tokens=True))
```

### 第 2 步：投影器模式

```python
import torch.nn as nn

class AudioProjector(nn.Module):
    def __init__(self, audio_dim=1280, llm_dim=4096):
        super().__init__()
        self.down = nn.Linear(audio_dim, llm_dim)
        self.act = nn.GELU()
        self.up = nn.Linear(llm_dim, llm_dim)

    def forward(self, audio_features):
        return self.up(self.act(self.down(audio_features)))
```

就是这样。投影器通常只有 1～3 个线性层。在 ASR 样本对（音频 → 转写）上训练它，就是阶段 1 的预训练任务。

### 第 3 步：评测 MMAU / LongAudioBench

```python
from datasets import load_dataset
mmau = load_dataset("MMAU/MMAU-Pro")

correct = 0
for item in mmau["test"]:
    answer = call_model(item["audio"], item["question"], item["choices"])
    if answer == item["correct_choice"]:
        correct += 1
print(f"Accuracy: {correct / len(mmau['test']):.3f}")
```

分别报告每个类别（语音/声音/音乐/多音频）的结果。汇总数字会掩盖模型究竟在哪里失败。

## 学以致用

| 任务 | 2026 年选择 |
|------|-----------|
| 自由形式音频问答（开放模型） | Qwen2.5-Omni-7B |
| 开放模型中的最佳长音频能力 | Audio Flamingo Next |
| 最佳封闭模型 | Gemini 2.5 Pro |
| 语音输入/语音输出智能体 | Qwen2.5-Omni 或 GPT-4o Audio |
| 音乐推理 | Audio Flamingo 3 或 2（音乐专用 AF-CLAP） |
| 呼叫中心审计 | 通过 API 使用 Gemini 2.5 Pro，并对政策文档执行 RAG |

## 陷阱

- **过度信任多音频能力。** 如果任务需要回答“哪段音频包含 X”，模型接近随机水平的表现是真实存在的。
- **长音频退化。** 超过 10 分钟后，大多数模型的说话人归属判断都会失效。应先进行说话人分离（第 6 课），再生成摘要。
- **静音上的幻觉。** 使用 Whisper 编码器的 LALM 会继承同样的问题。必须使用 VAD 把关。
- **挑选有利基准。** 供应商博客只会突出表现最好的类别。应亲自运行 MMAU-Pro 的多音频子集。

## 交付成果

保存为 `outputs/skill-alm-picker.md`。针对具体音频理解任务选择 LALM、基准子集和输出模态（文本或语音）。

## 练习

1. **简单。** 运行 `code/main.py`，观察玩具投影器模式如何把（音频嵌入、文本词元）路由为输出词元。
2. **中等。** 在 100 个 MMAU-Pro 语音项目上评测 Qwen2.5-Omni-7B，并与论文报告值比较。
3. **困难。** 构建最小音频字幕基线：BEATs 编码器 + 两层投影器 + 冻结的 Llama-3.2-1B。只在 AudioCaps 上微调投影器，并与 SALMONN 在 Clotho-AQA 上比较。

## 关键术语

| 术语 | 人们通常怎么说 | 实际含义 |
|------|-----------------|-----------------------|
| LALM | 音频版 ChatGPT | 音频编码器 + 投影器 + 大语言模型解码器。 |
| 投影器 | 适配器 | 把音频特征映射到大语言模型嵌入空间的小型 MLP。 |
| MMAU | 基准 | 覆盖语音、声音和音乐的 1 万个音频问答样本。 |
| MMAU-Pro | 更难的 MMAU | 1800 个强调多音频与推理的问题。 |
| LongAudioBench | 长音频评估 | 带语义查询的多分钟音频片段。 |
| 语音输入/语音输出 | 原生语音 | 模型直接接收语音并输出语音，不经过文本中转。 |

## 延伸阅读

- [Chu 等（2024），Qwen2-Audio](https://arxiv.org/abs/2407.10759)——参考架构。
- [阿里巴巴（2025），Qwen2.5-Omni](https://huggingface.co/Qwen/Qwen2.5-Omni-7B)——语音输入、语音输出。
- [NVIDIA（2025），Audio Flamingo 3](https://arxiv.org/abs/2507.08128)——开放长音频领先模型。
- [NVIDIA（2026），Audio Flamingo Next](https://arxiv.org/abs/2604.10905)——LongAudioBench 顶尖模型。
- [Tang 等（2023），SALMONN](https://arxiv.org/abs/2310.13289)——双编码器先驱。
- [MMAU-Pro 排行榜](https://mmaubenchmark.github.io/)——2026 年实时排名。
