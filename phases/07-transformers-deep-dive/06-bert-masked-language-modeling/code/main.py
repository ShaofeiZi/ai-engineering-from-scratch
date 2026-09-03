"""BERT 风格掩码语言建模——解密掩码规则。

仅使用标准库。演示 80/10/10 规则、全词遮蔽，以及对大批 token
进行分布合理性检查。
"""

import random
from collections import Counter


MASK_ID = 0  # 在这个玩具词表中为 [MASK] 保留 ID 0
CLS_ID = 1
SEP_ID = 2
SPECIAL_IDS = {MASK_ID, CLS_ID, SEP_ID}
IGNORE_INDEX = -100


def create_mlm_batch(tokens, vocab_size, mask_prob=0.15, rng=None):
    """应用 BERT 掩码。

    返回 (input_ids, labels)。若位置被选中用于预测，则 labels[i] 为原始
    token，否则为 IGNORE_INDEX。
    """
    if rng is None:
        rng = random.Random()
    input_ids = list(tokens)
    labels = [IGNORE_INDEX] * len(tokens)
    for i, t in enumerate(tokens):
        if t in SPECIAL_IDS:
            continue
        if rng.random() >= mask_prob:
            continue
        labels[i] = t
        r = rng.random()
        if r < 0.8:
            input_ids[i] = MASK_ID
        elif r < 0.9:
            rand_id = t
            while rand_id in SPECIAL_IDS or rand_id == t:
                rand_id = rng.randrange(vocab_size)
            input_ids[i] = rand_id
    return input_ids, labels


def whole_word_mlm(tokens, word_spans, vocab_size, mask_prob=0.15, rng=None):
    """遮蔽完整单词：若选中一个跨度内的任意子词，则遮蔽全部子词。

    word_spans：由 (start, end) 组成的列表，表示 token 序列中的左闭右开区间。
    """
    if rng is None:
        rng = random.Random()
    input_ids = list(tokens)
    labels = [IGNORE_INDEX] * len(tokens)
    for start, end in word_spans:
        if any(tokens[i] in SPECIAL_IDS for i in range(start, end)):
            continue
        if rng.random() >= mask_prob:
            continue
        r = rng.random()
        if r < 0.8:
            for i in range(start, end):
                labels[i] = tokens[i]
                input_ids[i] = MASK_ID
        elif r < 0.9:
            for i in range(start, end):
                labels[i] = tokens[i]
                rand_id = tokens[i]
                while rand_id in SPECIAL_IDS or rand_id == tokens[i]:
                    rand_id = rng.randrange(vocab_size)
                input_ids[i] = rand_id
        else:
            for i in range(start, end):
                labels[i] = tokens[i]
    return input_ids, labels


def distribution_check(n_tokens, vocab_size, mask_prob=0.15, seed=42):
    rng = random.Random(seed)
    tokens = [rng.randrange(3, vocab_size) for _ in range(n_tokens)]
    input_ids, labels = create_mlm_batch(tokens, vocab_size, mask_prob, rng)

    selected = sum(1 for l in labels if l != IGNORE_INDEX)
    masked = sum(1 for t, l in zip(input_ids, labels) if l != IGNORE_INDEX and t == MASK_ID)
    randomized = sum(1 for t, l in zip(input_ids, labels) if l != IGNORE_INDEX and t != MASK_ID and t != l)
    unchanged = sum(1 for t, l in zip(input_ids, labels) if l != IGNORE_INDEX and t == l)

    return {
        "tokens": n_tokens,
        "selected": selected,
        "selected_pct": 100 * selected / n_tokens,
        "masked_of_selected_pct": 100 * masked / selected if selected else 0.0,
        "random_of_selected_pct": 100 * randomized / selected if selected else 0.0,
        "unchanged_of_selected_pct": 100 * unchanged / selected if selected else 0.0,
    }


def toy_predict(masked_inputs, vocab):
    """模拟 MLM 头：返回词表上的均匀分布。
    真正的 BERT 会将每个位置的编码器输出投影到词表空间。
    """
    V = len(vocab)
    return [[1.0 / V for _ in range(V)] for _ in masked_inputs]


def main():
    vocab_words = [
        "[MASK]", "[CLS]", "[SEP]",
        "the", "quick", "brown", "fox", "jumps", "over",
        "lazy", "dog", "a", "stitch", "in", "time",
        "saves", "nine", "sat", "on", "mat",
    ]
    vocab_size = len(vocab_words)
    id_of = {w: i for i, w in enumerate(vocab_words)}

    sentence = ["[CLS]", "the", "quick", "brown", "fox", "jumps", "over", "the", "lazy", "dog", "[SEP]"]
    tokens = [id_of[w] for w in sentence]
    rng = random.Random(42)
    inp, labels = create_mlm_batch(tokens, vocab_size, mask_prob=0.5, rng=rng)

    print("=== MLM 掩码演示（prob=0.5，便于观察）===")
    print(f"{'索引':>4}  {'单词':>9}  {'输入 ID':>9}  {'输入词':>11}  {'标签':>6}")
    for i, (t_in, t_orig, lab) in enumerate(zip(inp, tokens, labels)):
        print(f"{i:>4}  {vocab_words[t_orig]:>9}  {t_in:>9}  {vocab_words[t_in]:>11}  {lab:>6}")

    print()
    print("=== 10 万个随机 token 上的 80/10/10 分布 ===")
    stats = distribution_check(n_tokens=100_000, vocab_size=vocab_size, mask_prob=0.15)
    print(f"已选中：               {stats['selected_pct']:.2f}%   （目标 15.0%）")
    print(f"  -> 替换为 [MASK]：   {stats['masked_of_selected_pct']:.2f}%   （目标 80.0%）")
    print(f"  -> 替换为随机 token：{stats['random_of_selected_pct']:.2f}%   （目标 10.0%）")
    print(f"  -> 保持不变：         {stats['unchanged_of_selected_pct']:.2f}%   （目标 10.0%）")

    print()
    print("=== 全词掩码演示 ===")
    # 演示时将 "quick brown" 和 "lazy dog" 分别视为含两个子词的单词
    tokens2 = [id_of[w] for w in sentence]
    spans = [(0, 1), (1, 2), (2, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 10), (10, 11)]
    rng2 = random.Random(7)
    inp2, labels2 = whole_word_mlm(tokens2, spans, vocab_size, mask_prob=0.5, rng=rng2)
    print("跨度：      " + " ".join(f"[{s}:{e}]" for s, e in spans))
    print("输入单词：  " + " ".join(vocab_words[t] for t in inp2))
    print("标签掩码：  " + " ".join(("P" if l != IGNORE_INDEX else ".") for l in labels2))
    print("P = 该位置有标签，. = 忽略")


if __name__ == "__main__":
    main()
