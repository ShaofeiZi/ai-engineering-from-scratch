import math
import random


def simulate_copy_accuracy(seq_len, context_dim=8, epochs=200, n_train=300, seed=0):
    rng = random.Random(seed)
    vocab = list("abcdefghij")
    vocab_size = len(vocab)

    embed = [[rng.gauss(0, 0.3) for _ in range(context_dim)] for _ in range(vocab_size)]
    context = [0.0] * context_dim

    def encode(sequence):
        c = [0.0] * context_dim
        decay = 0.85
        for token in sequence:
            idx = vocab.index(token)
            for d in range(context_dim):
                c[d] = c[d] * decay + embed[idx][d]
        return c

    def decode_score(context, target):
        total = 0.0
        recovery = 1.0
        for token in target:
            idx = vocab.index(token)
            score = sum(context[d] * embed[idx][d] for d in range(context_dim))
            normed = math.tanh(score) * recovery
            total += max(0.0, normed)
            recovery *= 0.9
        return total / max(1, len(target))

    hits = 0
    trials = 100
    for _ in range(trials):
        seq = [rng.choice(vocab) for _ in range(seq_len)]
        c = encode(seq)
        target_score = decode_score(c, seq)

        noise_score = decode_score(c, [rng.choice(vocab) for _ in range(seq_len)])
        if target_score > noise_score:
            hits += 1
    return hits / trials


def main():
    print("编码器-解码器瓶颈的简化模拟")
    print("上下文向量具有固定大小：8 个浮点数")
    print("编码器状态每步按 0.85 的比率衰减（模拟遗忘）")
    print()
    print(f"{'序列长度':>8}  {'准确率':>10}")
    for length in [5, 10, 20, 40, 80]:
        acc = simulate_copy_accuracy(length)
        print(f"{length:>8}  {acc:>9.0%}")
    print()
    print("真正的 LSTM 衰减更平缓，但仍会遇到相同上限。")
    print("注意力机制（课程 10）消除了固定大小限制。")


if __name__ == "__main__":
    main()
