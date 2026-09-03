---
name: skill-ctc-decoder
description: 从零编写贪心和束搜索 CTC 解码器，包括长度归一化
version: 1.0.0
phase: 4
lesson: 19
tags: [ocr, ctc, decoding, sequence-models]
---

# CTC 解码器

为 CTC 输出实现两种解码例程：贪心解码（速度快）和束搜索解码（在噪声输入上效果更好）。

## 适用场景

- 在自定义 CRNN 输出上运行 OCR 推理。
- 使用不同解码器对预训练 OCR 模型进行基准测试。
- 在不引入 ctcdecode 的情况下实现一个简单的束搜索。

## 输入

- `log_probs`：(T, N, C) 的 log-softmax，覆盖词表（按约定索引 0 为 blank）。
- `vocab`：长度为 C 的字符列表。
- `beam_width`（仅束搜索）：通常为 5-10。

## 贪心解码器

```python
def greedy_ctc_decode(log_probs, vocab, blank=0):
    preds = log_probs.argmax(dim=-1).transpose(0, 1).cpu().tolist()
    out = []
    for seq in preds:
        decoded = []
        prev = None
        for idx in seq:
            if idx != prev and idx != blank:
                decoded.append(vocab[idx])
            prev = idx
        out.append("".join(decoded))
    return out
```

## 束搜索解码器

```python
import heapq
import math

def beam_ctc_decode(log_probs, vocab, beam_width=5, blank=0):
    T, N, C = log_probs.shape
    lp = log_probs.cpu()
    results = []
    for n in range(N):
        beams = {("",): (0.0, -math.inf)}  # (prefix_tuple) -> (p_blank, p_nonblank)
        for t in range(T):
            logits_t = lp[t, n]
            new_beams = {}
            for prefix, (p_b, p_nb) in beams.items():
                for c in range(C):
                    p = logits_t[c].item()
                    if c == blank:
                        nb = p_b + p
                        nnb = p_nb + p
                        upd = new_beams.get(prefix, (-math.inf, -math.inf))
                        new_beams[prefix] = (
                            _logsumexp(upd[0], _logsumexp(nb, nnb)),
                            upd[1],
                        )
                    else:
                        last = prefix[-1] if prefix else ""
                        char = vocab[c]
                        if char == last:
                            # Case 1: stay on same prefix (collapse from p_nb)
                            upd = new_beams.get(prefix, (-math.inf, -math.inf))
                            new_beams[prefix] = (upd[0], _logsumexp(upd[1], p_nb + p))
                            # Case 2: extend prefix via blank-separated repeat ("a_a" -> "aa")
                            new_prefix = prefix + (char,)
                            upd = new_beams.get(new_prefix, (-math.inf, -math.inf))
                            new_beams[new_prefix] = (upd[0], _logsumexp(upd[1], p_b + p))
                        else:
                            new_prefix = prefix + (char,)
                            upd = new_beams.get(new_prefix, (-math.inf, -math.inf))
                            nb = _logsumexp(p_b, p_nb) + p
                            new_beams[new_prefix] = (upd[0], _logsumexp(upd[1], nb))
            beams = dict(heapq.nlargest(
                beam_width,
                new_beams.items(),
                key=lambda kv: _logsumexp(kv[1][0], kv[1][1]),
            ))
        best = max(beams.items(), key=lambda kv: _logsumexp(kv[1][0], kv[1][1]))[0]
        results.append("".join(best))
    return results


def _logsumexp(a, b):
    if a == -math.inf: return b
    if b == -math.inf: return a
    m = max(a, b)
    return m + math.log(math.exp(a - m) + math.exp(b - m))
```

## 规则

- 在 PyTorch 的 `nn.CTCLoss` 中，按约定 CTC 的 blank 索引为 0。
- 束搜索能在低置信度输入上提升准确率；在干净输入上提升不到 1% CER。
- 不要将束宽降到 5 以下；低于该值时准确率与延迟的权衡会趋于平缓。
- 在紧凑的延迟预算内运行束搜索时，可退化为贪心解码；在大多数生产 OCR 数据上，质量损失很小。
- 对于大词表（CJK 3000+ 字符），应改用 `ctcdecode`（C++），而不要使用上面的纯 Python 版本；Python 束搜索很快会成为瓶颈。
