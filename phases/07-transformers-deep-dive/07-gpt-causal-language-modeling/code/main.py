"""GPT 风格因果语言建模——因果掩码、损失移位和采样。

仅使用标准库。通过带随机权重的微型“GPT”，在包含 20 个 token 的词表上
演示因果掩码、下一 token 预测和四种采样策略。
"""

import math
import random


def softmax(logits, temperature=1.0):
    if temperature != 1.0:
        logits = [x / temperature for x in logits]
    m = max(logits)
    exps = [math.exp(x - m) for x in logits]
    s = sum(exps)
    return [e / s for e in exps]


def causal_mask(n):
    return [[0.0 if j <= i else float("-inf") for j in range(n)] for i in range(n)]


def attention_scores_with_mask(raw_scores, mask):
    return [[s + m for s, m in zip(row, mrow)] for row, mrow in zip(raw_scores, mask)]


def apply_softmax_row(row):
    finite = [x for x in row if x != float("-inf")]
    if not finite:
        return [0.0] * len(row)
    m = max(finite)
    exps = [math.exp(x - m) if x != float("-inf") else 0.0 for x in row]
    s = sum(exps)
    return [e / s if s > 0 else 0.0 for e in exps]


def cross_entropy_shifted(logits_per_pos, target_ids):
    """下一 token 的交叉熵：logit_i 与 target_{i+1} 对比。"""
    total = 0.0
    count = 0
    for i in range(len(target_ids) - 1):
        probs = softmax(logits_per_pos[i])
        p = probs[target_ids[i + 1]]
        total += -math.log(max(p, 1e-12))
        count += 1
    return total / count


def sample_greedy(probs):
    return max(range(len(probs)), key=lambda i: probs[i])


def sample_temperature(logits, t, rng):
    probs = softmax(logits, temperature=t)
    return sample_from_distribution(probs, rng)


def sample_from_distribution(probs, rng):
    r = rng.random()
    cum = 0.0
    for i, p in enumerate(probs):
        cum += p
        if r <= cum:
            return i
    return len(probs) - 1


def sample_top_k(logits, k, rng, temperature=1.0):
    indexed = sorted(enumerate(logits), key=lambda x: -x[1])
    keep = indexed[:k]
    keep_ids = [i for i, _ in keep]
    keep_logits = [v for _, v in keep]
    probs = softmax(keep_logits, temperature=temperature)
    chosen = sample_from_distribution(probs, rng)
    return keep_ids[chosen]


def sample_top_p(logits, p, rng, temperature=1.0):
    probs = softmax(logits, temperature=temperature)
    indexed = sorted(enumerate(probs), key=lambda x: -x[1])
    cum = 0.0
    cutoff = len(indexed)
    for i, (_, pi) in enumerate(indexed):
        cum += pi
        if cum >= p:
            cutoff = i + 1
            break
    kept = indexed[:cutoff]
    total = sum(pi for _, pi in kept)
    renorm = [(idx, pi / total) for idx, pi in kept]
    r = rng.random()
    cum = 0.0
    for idx, pi in renorm:
        cum += pi
        if r <= cum:
            return idx
    return renorm[-1][0]


def sample_min_p(logits, min_p, rng, temperature=1.0):
    probs = softmax(logits, temperature=temperature)
    max_p = max(probs)
    threshold = min_p * max_p
    kept = [(i, pi) for i, pi in enumerate(probs) if pi >= threshold]
    total = sum(pi for _, pi in kept)
    renorm = [(i, pi / total) for i, pi in kept]
    r = rng.random()
    cum = 0.0
    for i, pi in renorm:
        cum += pi
        if r <= cum:
            return i
    return renorm[-1][0]


def demo_causal_mask():
    print("=== 因果注意力矩阵（softmax 后）===")
    n = 6
    rng = random.Random(42)
    raw = [[rng.gauss(0, 1) for _ in range(n)] for _ in range(n)]
    mask = causal_mask(n)
    masked = attention_scores_with_mask(raw, mask)
    attn = [apply_softmax_row(row) for row in masked]
    for i, row in enumerate(attn):
        print("  " + "  ".join(f"{v:.3f}" for v in row))
    print("  （每一行都是位置 0..i 上的有效概率分布）")
    print()


def demo_sampling():
    print("=== 在模拟下一 token 分布上使用采样策略 ===")
    vocab = ["the", "cat", "dog", "sat", "ran", "jumped", "on", "mat", "floor", "."]
    logits = [3.2, 1.1, 2.8, 0.4, 0.9, 1.5, -0.2, 2.1, 0.7, 0.1]
    probs = softmax(logits)
    print("token       logit   概率")
    for w, l, p in zip(vocab, logits, probs):
        print(f"  {w:<8}  {l:+.2f}   {p:.3f}")
    print()

    rng = random.Random(0)
    print("贪心：          " + vocab[sample_greedy(probs)])
    print("temp=0.7:       " + vocab[sample_temperature(logits, 0.7, rng)])
    print("temp=2.0:       " + vocab[sample_temperature(logits, 2.0, rng)])
    print("top-k=3:        " + vocab[sample_top_k(logits, 3, rng)])
    print("top-p=0.9:      " + vocab[sample_top_p(logits, 0.9, rng)])
    print("min-p=0.1:      " + vocab[sample_min_p(logits, 0.1, rng)])
    print()


def demo_ce_loss():
    print("=== 下一 token 的交叉熵损失 ===")
    vocab_size = 10
    seq = [3, 1, 7, 0, 4, 9]
    rng = random.Random(7)
    logits = [[rng.gauss(0, 1) for _ in range(vocab_size)] for _ in seq]
    # 略微提高正确下一 token 的分数，以模拟“略经训练”的模型
    for i in range(len(seq) - 1):
        logits[i][seq[i + 1]] += 2.0
    loss_trained = cross_entropy_shifted(logits, seq)
    # 无偏随机值
    logits_rand = [[rng.gauss(0, 1) for _ in range(vocab_size)] for _ in seq]
    loss_rand = cross_entropy_shifted(logits_rand, seq)
    print(f"偏置 logits 的损失（近似已训练）：{loss_trained:.3f}")
    print(f"随机 logits 的损失：              {loss_rand:.3f}")
    print(f"随机基线损失（ln V = ln {vocab_size}）：{math.log(vocab_size):.3f}")
    print()


def main():
    demo_causal_mask()
    demo_sampling()
    demo_ce_loss()
    print("要点：掩码只需一行代码，其余部分仍是同一个 transformer。")


if __name__ == "__main__":
    main()
