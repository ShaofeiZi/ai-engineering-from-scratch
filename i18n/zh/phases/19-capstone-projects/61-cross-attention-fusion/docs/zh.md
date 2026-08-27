# 跨注意力融合

> projection layer 只能把一个图像向量和一个 caption 向量对齐。真正的视觉语言解码器需要让每个文本 token 都能 attend 到每个 patch token，这样模型才能把每个词落到具体区域上。cross-attention 就是这个 grounding 发生的地方。文本负责提问；视觉的 keys 和 values 负责回答。这一课会把 cross-attention block、causal text self-attention，以及保证两者都合法的 mask 形状一起搭起来。

**Type:** 构建
**Languages:** Python
**Prerequisites:** 第 19 阶段第 30–37 课（Track B 基础课）
**Time:** 约 90 分钟

## 学习目标

- 实现 multi-head cross-attention，其中 query stream 来自文本，key/value stream 来自视觉。
- 组合一个 decoder block：causal self-attention + cross-attention + feed-forward。
- 把 mask 形状做对：self-attention 用 causal mask，cross-attention 不使用 mask。
- 在 batched text tokens 和固定 image token pool 上跑通一次 forward pass。

## 问题

把 image tokens 和 text tokens 直接拼成一个序列，是一种融合方式，也就是 early fusion，Chameleon 和 Emu3 走的是这条路。另一条路是 cross-attention，也就是 late fusion，Flamingo 最先把它系统化，此后几乎所有 Flamingo 形状的解码器都在复用这条思路。在 late fusion 中，文本解码器只在 text-only tokens 上运行，并在每一层通过 cross-attention 伸手去读取图像流。

late fusion 有两个明显好处。第一，文本流保持干净，模型的 text-only 能力更容易保住。第二，图像流每张图只需要计算一次，之后每个 decode step 都可以反复复用，因此即使 caption 很长，生成开销也不高。代价则是：每个 block 都会多出一层 attention sub-layer。

## 概念

```mermaid
flowchart TB
  Image[image tokens B x Nv x D] --> Vis[frozen vision encoder]
  Vis --> Mem[memory tokens B x Nv x D]
  Text[text token ids] --> Emb[text embedding]
  Emb --> Self[masked self-attention]
  Self --> Cross[cross-attention queries=text keys/values=memory]
  Cross --> FFN[feed-forward]
  FFN --> Out[next-token logits]
  Mem --> Cross
```

```mermaid
flowchart LR
  Q[text Q B x H x Nt x d] --> Scores[Q K^T / sqrt d]
  K[image K B x H x Nv x d] --> Scores
  Scores --> Soft[softmax over Nv]
  V[image V B x H x Nv x d] --> Out
  Soft --> Out[output B x H x Nt x d]
```

### Mask 形状

decoder block 里的两种 attention 需要不同的 masks：

| 注意力类型 | 查询长度 | 键长度 | 掩码 | 原因 |
|-----------|--------------|------------|------|-----|
| Self-attention | `Nt`（文本） | `Nt`（文本） | 因果掩码：下三角矩阵 `(Nt, Nt)` | 自回归生成时，文本 token 不能查看未来位置 |
| Cross-attention | `Nt`（文本） | `Nv`（视觉） | 无掩码 | 每个文本位置都可以看到整张图像 |

本课还包含一个 shape-validation function，因此一旦把两种 mask 搞混，会直接抛出 `ValueError`，而不是悄悄得到一条已经损坏的 loss curve。

### 为什么 cross-attention 不需要 mask

在生成任何文本之前，整张图像已经是完全可见的。caption 的第 `t` 个 token 可以 attend 到图像里的任意一个 patch；图像 patches 本身不存在时间顺序。某些 Flamingo 变体在交织多个图像与多个文本片段时，会加上按样本变化的 masking pattern；但对于“一张图像 + 一条 caption”这一最基本情况，cross-attention 应该看到全部视觉内容。

### Key/value caching

图像的 keys 和 values 会在 decode 开始时计算一次，然后放进 cache。之后每来一个新的文本 token，都直接复用这个 cache，而不重新计算。这正是 captioning 在推理阶段足够快的原因：重的 ViT 只运行一次；cross-attention 在后续每一步都只重复利用现成的 keys 和 values。本课会显式暴露 cache，并测试 cache-hit 路径。

### Block 组合方式

一个 decoder block 的顺序是：pre-LN -> self-attention -> residual -> pre-LN -> cross-attention -> residual -> pre-LN -> feed-forward -> residual。总共三个 sub-layers，每一个都有自己的 LayerNorm。Flamingo 论文曾在 cross-attention 上加了一个 learned gate，让模型可以在训练稳定性代价下选择退出图像路径；本课采用的是最标准的 baseline，不加 gate。

```python
class DecoderBlock:
  def forward(self, text_tokens, image_tokens, text_mask, cross_mask):
      text_tokens = text_tokens + self.self_attn(self.ln1(text_tokens),
                                                 mask=text_mask)
      text_tokens = text_tokens + self.cross_attn(self.ln2(text_tokens),
                                                  image_tokens,
                                                  mask=cross_mask)
      text_tokens = text_tokens + self.ffn(self.ln3(text_tokens))
      return text_tokens
```

```figure
ch-crossattn-fan
```

## 动手实现

`code/main.py` 实现了：

- `CrossAttention(hidden, heads)`，一个带独立 `q` 和 `kv` projections 的 multi-head cross-attention。
- `CausalSelfAttention(hidden, heads)`，也就是标准 decoder 里的 masked self-attention。
- `DecoderBlock`，把三个 sub-layers 按 pre-LN residual 方式组合起来。
- `VisionLanguageDecoder`，一个四层 decoder，输入来自 mock vision encoder 输出和一个很小的文本 embedding table。
- `causal_mask(length)`，返回一个 `(length, length)` 的 lower-triangular boolean tensor。
- 一个 demo：给入 batch size 为 2、长度为 10 的文本序列，以及长度为 197 的 image memory，并打印输出形状、self-attention mask 形状，以及每个位置的 cross-attention 输出范数。

运行它：

```bash
python3 code/main.py
```

输出：decoder 会产生一个 `(2, 10, text_vocab)` 的 logits tensor。mask 形状是 `(10, 10)`。KV-cache 的复用检查会确认 cached 和 uncached 两条路径产出的 logits 完全一致。

## 实际使用

cross-attention 主要出现在两大生产模型家族中：

- **Flamingo 和 IDEFICS。** 每隔 K 个 language model blocks 插入一个 cross-attention sub-layer，并保持 LM 冻结。视觉语言 adapter 就是这块 cross-attention 再加上它的 gate。
- **BLIP-2。** Q-Former 会用一组固定的 32 个 query tokens 通过 cross-attention 去读取图像特征，然后再把这些 queries 投影到 LM embedding space。

本课这个 block 的形状，能够直接映射到这两类实现上。mask 纪律也是一样的：self 上用 causal，cross 上不用。

## 测试

`code/test_main.py` 覆盖：

- causal mask 是否真的是 lower-triangular，并且匹配预期的 boolean 形状
- cross-attention 输出形状是否始终等于 `(B, Nt, hidden)`，不受 key length 影响
- KV-cache 路径是否在浮点容差内与 uncached 路径一致
- text 和 image streams 之间发生 shape mismatch 时，是否会抛出清晰的 `ValueError`
- 一个完整 decoder forward pass 是否产出正确的 batch 和 sequence 形状

运行它们：

```bash
python3 -m unittest code/test_main.py
```

## 练习

1. 给 cross-attention residual 增加一个 learned tanh gate，也就是 Flamingo 的那套技巧，并验证从接近零的初始 gate 出发仍然可以收敛。gate 从 0 开始时，模型会先恢复 text-only 行为，再逐步把图像流混进来。

2. 实现 interleaved attention，使同一个 decoder 能同时消费多张图像和多个文本片段。构造按样本变化的 cross-attention mask，确保文本片段 2 不会 attend 到图像 1。

3. profile cross-attention 与 self-attention 在 `Nt=64, Nv=576`，也就是更高分辨率下 24x24 网格时的耗时。cross-attention 的复杂度是 `Nt * Nv`，在图像分辨率较高时会成为主导开销。

4. 给 cross-attention map 加一个 query-side dropout，并在 demo 上测量 caption diversity。随着 cross map 中 dropout 增强，caption 样本方差通常会上升。

5. 把 cross-attention layer 换成一个 Q-Former 风格的 attention block，让固定的 32-token query pool 在每层只读取一次图像特征。

## 关键术语

| 术语 | 含义 |
|------|---------------|
| Late fusion | 文本和视觉分别保留在独立的数据流中，由 cross-attention 在每个 block 连接两者 |
| Cross-attention | Q 来自一条数据流，K 和 V 来自另一条数据流 |
| Causal mask | 防止自回归生成时查看未来位置的下三角布尔掩码 |
| KV cache | 图像的键和值只存储一次，并在每个解码步骤中复用 |
| Memory tokens | 解码器反复读取的冻结图像 token |

## 延伸阅读

- Flamingo (2022) 介绍了带 gated cross-attention 的标准 late-fusion 设计。
- BLIP-2 (2023) 展示了 Q-Former，它本质上是披着 learned query pool 外衣的 cross-attention block。
- IDEFICS (2023) 是 Flamingo 配方的一份开源复现。
