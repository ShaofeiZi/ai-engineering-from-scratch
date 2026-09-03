# 流式语音到语音——Moshi、Hibiki 与全双工对话

> 2024～2026 年重新定义了语音 AI。Moshi 用单一模型以 200 毫秒延迟同时聆听与说话，Hibiki 则逐块完成语音到语音翻译。二者都放弃 ASR → 大语言模型 → TTS 流水线，改用基于 Mimi 编解码器词元的统一全双工架构。这是新的参考设计。

**Type:** 学习
**Languages:** Python
**Prerequisites:** 阶段 6 · 13（神经音频编解码器）、阶段 6 · 11（实时音频）、阶段 7 · 05（完整 Transformer）
**Time:** 约 75 分钟

## 问题

所有根据第 11 与第 12 课构建的语音智能体，都存在约 300～500 毫秒的基础延迟下限：VAD 触发、STT 处理、大语言模型推理、TTS 生成。每个阶段都有自己的最低延迟。你可以调优并并行处理，却仍会受到流水线形态的限制。

Moshi（Kyutai，2024～2026）提出了不同的问题：如果根本不存在流水线呢？如果由一个模型直接、持续地接收音频并输出音频，把文本仅作为中间的“内心独白”，而不是必经阶段呢？

答案就是**全双工语音到语音**。理论延迟为 160 毫秒（80 毫秒 Mimi 帧 + 80 毫秒声学延迟）；在单张 L4 GPU 上，实际延迟为 200 毫秒。这只有最佳级联语音智能体的一半。

## 概念

![Moshi 架构：两条并行 Mimi 流 + 内心独白文本](../../../../../../phases/06-speech-and-audio/15-streaming-speech-to-speech-moshi-hibiki/assets/moshi-hibiki.svg)

### Moshi 架构

**输入。** 两条 Mimi 编解码器流，均为 12.5 Hz × 8 个码本：

- 流 1：用户音频（由 Mimi 编码，持续到达）
- 流 2：Moshi 自己的音频（由 Moshi 生成）

**Transformer。** 一个拥有 70 亿参数的时间 Transformer 同时处理这两条流与一条文本“内心独白”流。在每个 80 毫秒步骤中，它会：

1. 接收最新的用户 Mimi 词元（8 个码本）。
2. 接收最近生成的 Moshi Mimi 词元（8 个码本）。
3. 生成下一个 Moshi 文本词元（内心独白）。
4. 通过小型深度 Transformer 生成下一组 Moshi Mimi 词元（8 个码本）。

三条流——用户音频、Moshi 音频、Moshi 文本——并行运行。Moshi 可以一边说话一边听用户，可以在用户打断时中断自己，也可以用“mhm”等附和声回应，而不打断主要话语。

**深度 Transformer。** 在同一帧中，8 个码本并非并行预测，因为码本之间存在依赖关系。一个小型两层“深度 Transformer”会在 80 毫秒内依次预测它们。这是自回归编解码器语言模型的标准分解方式，VALL-E 和 VibeVoice 也采用它。

### 内心独白文本为何有帮助

如果没有显式文本，模型就必须在声学流中隐式建模语言。Moshi 的洞见是：迫使模型在输出音频的同时输出文本词元。文本流本质上就是 Moshi 自己话语的转写。它可以提高语义连贯性，让语言模型头更容易替换，还能免费提供转写文本。

### Hibiki：流式语音到语音翻译

它采用同一种架构，但在翻译对上训练。源语言音频持续进入，目标语言音频持续输出。Hibiki-Zero（2026 年 2 月）不再需要词级对齐训练数据——它使用句子级数据，并通过 GRPO 强化学习优化延迟。

最初支持四个语言对；使用约 1000 小时数据即可适配一种新语言。

### 更广泛的 Kyutai 技术栈（2026）

- **Moshi**——全双工对话（最初面向法语，英语支持良好）
- **Hibiki / Hibiki-Zero**——同步语音翻译
- **Kyutai STT**——流式 ASR（500 毫秒或 2.5 秒前瞻）
- **Kyutai Pocket TTS**——可在 CPU 上运行的 1 亿参数 TTS（2026 年 1 月）
- **Unmute**——在公共服务器上组合上述组件的完整流水线

在 L40S GPU 上的吞吐量：以 3 倍实时速度同时处理 64 个会话。

### Sesame CSM——近亲架构

Sesame CSM（2025）采用类似思想——Llama-3 骨干网络配合 Mimi 编解码器头。但 CSM 是单向的（接收上下文 + 文本，生成语音），而不是全双工。它是市场上“声音存在感”最强的 TTS，但与 Moshi 的全双工能力并不完全相同。

### 2026 年性能数据

| 模型 | 延迟 | 用例 | 许可证 |
|-------|---------|----------|---------|
| Moshi | 200 毫秒（L4） | 英语/法语全双工对话 | CC-BY 4.0 |
| Hibiki | 12.5 Hz 帧率 | 法语 ↔ 英语流式翻译 | CC-BY 4.0 |
| Hibiki-Zero | 相同 | 5 个语言对，无须对齐数据 | CC-BY 4.0 |
| Sesame CSM-1B | 200 毫秒 TTFA | 上下文条件 TTS | Apache-2.0 |
| GPT-4o Realtime | 约 300 毫秒 | 封闭，OpenAI API | 商业许可 |
| Gemini 2.5 Live | 约 350 毫秒 | 封闭，Google API | 商业许可 |

```figure
sp-fullduplex
```

## 动手构建

### 第 1 步：接口

Moshi 提供一个 WebSocket 服务器，接收 80 毫秒的 Mimi 编码音频块，再返回 80 毫秒的 Mimi 编码音频块。两个方向持续运行。

```python
import asyncio
import websockets
from moshi.client_utils import encode_audio_mimi, decode_audio_mimi

async def moshi_chat():
    async with websockets.connect("ws://localhost:8998/api/chat") as ws:
        mic_task = asyncio.create_task(stream_mic_to(ws))
        spk_task = asyncio.create_task(stream_from_to_speaker(ws))
        await asyncio.gather(mic_task, spk_task)
```

### 第 2 步：全双工循环

```python
async def stream_mic_to(ws):
    async for chunk_80ms in mic_stream_at_12_5_hz():
        mimi_tokens = encode_audio_mimi(chunk_80ms)
        await ws.send(serialize(mimi_tokens))

async def stream_from_to_speaker(ws):
    async for msg in ws:
        mimi_tokens, text_token = deserialize(msg)
        audio = decode_audio_mimi(mimi_tokens)
        await play(audio)
```

两个方向同时运行。Python asyncio 或 Rust futures 是标准传输实现。

### 第 3 步：训练目标（概念）

对于每个 80 毫秒帧 `t`：

- 输入：`user_mimi[0..t]`、`moshi_mimi[0..t-1]`、`moshi_text[0..t-1]`
- 预测：先预测 `moshi_text[t]`，再预测 `moshi_mimi[t, codebook_0..7]`

文本先于音频预测（内心独白）；音频则在深度 Transformer 内部按码本顺序预测。

### 第 4 步：Moshi 擅长什么，又不擅长什么

Moshi 的优势：

- 在低成本硬件上实现低于 250 毫秒的端到端延迟。
- 自然地进行附和与打断。
- 不需要流水线胶水代码。

Moshi 的劣势：

- 工具调用（没有针对它训练；需要独立的大语言模型路径）。
- 长程推理（Moshi 是约 8B 的对话模型，不是 Claude/GPT-4）。
- 小众主题上的事实准确性。
- 大多数企业生产用例（2026 年仍使用流水线）。

## 学以致用

| 场景 | 选择 |
|-----------|------|
| 最低延迟的语音陪伴 | Moshi |
| 实时翻译通话 | Hibiki |
| 语音演示/研究 | Moshi、CSM |
| 带工具的企业智能体 | 使用流水线（第 12 课），而非 Moshi |
| 上下文中的自定义声音 TTS | Sesame CSM |
| 任意语言的语音到语音 | GPT-4o Realtime 或 Gemini 2.5 Live（商业） |

## 陷阱

- **工具调用能力有限。** Moshi 是对话模型，不是智能体框架。需要工具时，应与流水线组合。
- **特定声音条件。** Moshi 使用单一的已训练角色；克隆新声音需要单独训练。
- **语言覆盖。** 法语和英语表现出色，其他语言有限。Hibiki-Zero 能改善问题，但仍需要训练数据。
- **资源成本。** 一个完整 Moshi 会话会独占一个 GPU 槽位，不适合低成本多租户部署。

## 交付成果

保存为 `outputs/skill-duplex-pipeline.md`。为语音智能体工作负载选择流水线式或全双工架构，并说明理由。

## 练习

1. **简单。** 运行 `code/main.py`。它会以符号方式模拟双流 + 内心独白架构。
2. **中等。** 从 Hugging Face 拉取 Moshi，运行服务器并测试一轮对话。测量从用户结束说话到 Moshi 开始响应的实际延迟。
3. **困难。** 取出第 12 课中的流水线智能体，用 20 个相同测试话语比较它与 Moshi 的 P50 延迟，并说明流水线架构仍会在哪些场景中胜出。

## 关键术语

| 术语 | 人们通常怎么说 | 实际含义 |
|------|-----------------|-----------------------|
| 全双工 | 同时聆听与说话 | 同一个模型中同时激活两条音频流。 |
| 内心独白 | 模型的文本流 | Moshi 在音频输出之外同步生成文本词元。 |
| 深度 Transformer | 码本间预测器 | 在一个 80 毫秒帧内预测 8 个码本的小型 Transformer。 |
| Mimi | Kyutai 的编解码器 | 12.5 Hz × 8 个码本；语义 + 声学；支撑 Moshi。 |
| 流式 S2S | 实时音频 → 音频 | 逐块翻译/对话，无流水线阶段。 |
| 附和 | “嗯哼”等回应 | Moshi 可以发出简短回应，而不打断自己的主要话轮。 |

## 延伸阅读

- [Défossez 等（2024），Moshi——语音—文本基础模型](https://arxiv.org/html/2410.00037v2)——原始论文。
- [Kyutai Labs（2026），Hibiki-Zero](https://arxiv.org/abs/2602.12345)——无需对齐数据的同步翻译。
- [Sesame（2025），跨越声音的恐怖谷](https://www.sesame.com/research/crossing_the_uncanny_valley_of_voice)——CSM 规格。
- [Kyutai——Moshi 代码库](https://github.com/kyutai-labs/moshi)——安装与服务器。
- [OpenAI——Realtime API](https://platform.openai.com/docs/guides/realtime)——封闭商业同类方案。
- [Kyutai——延迟流建模](https://github.com/kyutai-labs/delayed-streams-modeling)——底层 STT/TTS 框架。
