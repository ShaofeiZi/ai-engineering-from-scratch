"""ASR 基础：贪心 CTC 解码、束搜索 CTC 解码和词错误率。

仅使用标准库。构建一个手写的微型 CTC 示例并计算 WER。
运行：python3 code/main.py
"""

import math
import random
from collections import Counter


BLANK = 0
VOCAB = "_abcdefghijklmnopqrstuvwxyz "  # 索引 0 表示 blank


def ctc_greedy(frame_probs):
    preds = [max(range(len(p)), key=lambda i: p[i]) for p in frame_probs]
    out = []
    prev = -1
    for p in preds:
        if p != prev and p != BLANK:
            out.append(p)
        prev = p
    return "".join(VOCAB[i] for i in out)


def ctc_beam(frame_probs, beam_width=8):
    beams = [((), 0.0)]
    for p in frame_probs:
        log_p = [math.log(max(pi, 1e-10)) for pi in p]
        new_beams = {}
        for seq, lp in beams:
            for t, lpt in enumerate(log_p):
                if t == BLANK:
                    new_seq = seq
                else:
                    if seq and seq[-1] == t:
                        new_seq = seq
                    else:
                        new_seq = seq + (t,)
                if new_seq in new_beams:
                    new_beams[new_seq] = math.log(math.exp(new_beams[new_seq]) + math.exp(lp + lpt))
                else:
                    new_beams[new_seq] = lp + lpt
        beams = sorted(new_beams.items(), key=lambda x: -x[1])[:beam_width]
    best = beams[0][0]
    return "".join(VOCAB[i] for i in best)


def wer(ref, hyp):
    r = ref.split()
    h = hyp.split()
    nr = len(r)
    if nr == 0:
        return 0.0 if not h else 1.0
    dp = [[0] * (len(h) + 1) for _ in range(nr + 1)]
    for i in range(nr + 1):
        dp[i][0] = i
    for j in range(len(h) + 1):
        dp[0][j] = j
    for i in range(1, nr + 1):
        for j in range(1, len(h) + 1):
            cost = 0 if r[i - 1] == h[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)
    return dp[nr][len(h)] / nr


def one_hot_like(char, noise=0.02, vocab_size=len(VOCAB)):
    base = [noise] * vocab_size
    idx = VOCAB.index(char)
    base[idx] = 1.0 - noise * (vocab_size - 1)
    return base


def build_frame_probs(target, duration_per_char=3, blank_runs=1):
    random.seed(0)
    frames = []
    for c in target:
        for _ in range(duration_per_char):
            frames.append(one_hot_like(c))
        for _ in range(blank_runs):
            frames.append(one_hot_like("_"))
    return frames


def corrupt(probs, n_swaps=3, swap_strength=0.4):
    random.seed(1)
    out = [list(p) for p in probs]
    for _ in range(n_swaps):
        i = random.randrange(len(out))
        j1 = random.randrange(len(out[i]))
        j2 = random.randrange(len(out[i]))
        swap = swap_strength
        out[i][j1] -= swap
        out[i][j2] += swap
    return out


def main():
    target = "hello world"
    print("=== 步骤 1：为目标文本构建逐帧 CTC 输出 ===")
    print(f"  目标：{target!r}")
    probs = build_frame_probs(target, duration_per_char=3, blank_runs=1)
    print(f"  帧数：{len(probs)}  词表大小：{len(VOCAB)}  （索引 0 = blank）")

    print()
    print("=== 步骤 2：贪心解码（折叠重复项，丢弃 blank）===")
    greedy = ctc_greedy(probs)
    print(f"  贪心解码：{greedy!r}")

    print()
    print("=== 步骤 3：束搜索解码（宽度 8，简化版）===")
    beam = ctc_beam(probs, beam_width=8)
    print(f"  束搜索解码：{beam!r}")
    print(f"  注意：该束搜索没有 blank 间隔状态，会合并连续重复项；")
    print(f"  正确的前缀树束搜索（如 ctcdecode）会跟踪 P_blank / P_nonblank，")
    print(f"  从而保留 'hello' 中两个 l 这样的重复字母。")

    print()
    print("=== 步骤 4：扰动 logits；束搜索应优于贪心解码 ===")
    corrupted = corrupt(probs, n_swaps=6, swap_strength=0.6)
    g2 = ctc_greedy(corrupted)
    b2 = ctc_beam(corrupted, beam_width=16)
    print(f"  贪心解码：{g2!r}")
    print(f"  束搜索：  {b2!r}")

    print()
    print("=== 步骤 5：WER ===")
    ref = "hello world this is a test"
    hyps = {
        "完全匹配":     "hello world this is a test",
        "一次替换":     "hello world this is the test",
        "一次删除":     "hello world this a test",
        "一次插入":     "hello world this is a big test",
        "无关内容":     "bye everyone nothing here",
    }
    for label, hyp in hyps.items():
        print(f"  {label:<14} WER = {wer(ref, hyp):.3f}  假设={hyp!r}")

    print()
    print("=== 步骤 6：LibriSpeech test-clean 上的最佳模型（2026）===")
    table = [
        ("Parakeet-TDT-1.1B", 1.40, "1.1B"),
        ("Canary-1B Flash",   1.48, "1B"),
        ("Whisper-L-v3-turbo", 1.58, "809M"),
        ("Seamless M4T v2",    1.70, "2.3B"),
        ("wav2vec 2.0 Large",  1.92, "317M"),
    ]
    print("  | 模型                  | WER  | 参数量 |")
    for name, w, p in table:
        print(f"  | {name:<21} | {w:.2f} | {p:<6} |")


if __name__ == "__main__":
    main()
