import math
from collections import Counter


def tokenize(text):
    return text.lower().replace(".", " .").replace(",", " ,").split()


def ngrams(tokens, n):
    return [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]


def ngram_precision(hyp_tokens, ref_tokens, n):
    hyp_counts = Counter(ngrams(hyp_tokens, n))
    ref_counts = Counter(ngrams(ref_tokens, n))
    clipped = 0
    total = 0
    for ngram, count in hyp_counts.items():
        clipped += min(count, ref_counts.get(ngram, 0))
        total += count
    if total == 0:
        return 0.0
    return clipped / total


def brevity_penalty(hyp_len, ref_len):
    if hyp_len >= ref_len:
        return 1.0
    if hyp_len == 0:
        return 0.0
    return math.exp(1 - ref_len / hyp_len)


def simple_bleu(hypothesis, reference, max_n=4):
    hyp = tokenize(hypothesis)
    ref = tokenize(reference)
    precisions = [ngram_precision(hyp, ref, n) for n in range(1, max_n + 1)]
    if any(p == 0 for p in precisions):
        return 0.0
    log_mean = sum(math.log(p) for p in precisions) / max_n
    bp = brevity_penalty(len(hyp), len(ref))
    return 100 * bp * math.exp(log_mean)


def simple_bleu_note():
    return (
        "上面的 simple_bleu 没有平滑处理：只要一个 n-gram 的精确率为零，"
        "分数就会降至 0.0。这会过度惩罚较短的候选译文；生产环境会通过 "
        "sacrebleu 使用 epsilon / NIST / add-k 平滑。"
    )


def chrf(hypothesis, reference, n=6, beta=2):
    def char_ngrams(text, k):
        return [text[i:i + k] for i in range(len(text) - k + 1)]

    hyp = hypothesis.lower()
    ref = reference.lower()
    precisions = []
    recalls = []
    for k in range(1, n + 1):
        hyp_c = Counter(char_ngrams(hyp, k))
        ref_c = Counter(char_ngrams(ref, k))
        match = sum((hyp_c & ref_c).values())
        if sum(hyp_c.values()) == 0 or sum(ref_c.values()) == 0:
            continue
        precisions.append(match / sum(hyp_c.values()))
        recalls.append(match / sum(ref_c.values()))
    if not precisions:
        return 0.0
    p = sum(precisions) / len(precisions)
    r = sum(recalls) / len(recalls)
    if p + r == 0:
        return 0.0
    beta2 = beta * beta
    return 100 * (1 + beta2) * p * r / (beta2 * p + r)


def main():
    cases = [
        ("Les chats courent.", "Les chats courent."),
        ("Les chats sont en train de courir.", "Les chats courent."),
        ("Les chiens mangent.", "Les chats courent."),
        ("Les", "Les chats courent."),
    ]
    print(f"{'候选译文':40s}  {'参考译文':25s}  {'BLEU':>6}  {'chrF':>6}")
    for hyp, ref in cases:
        b = simple_bleu(hyp, ref)
        c = chrf(hyp, ref)
        print(f"{hyp:40s}  {ref:25s}  {b:6.1f}  {c:6.1f}")
    print()
    print(simple_bleu_note())
    print("BLEU 相差不到 1 分属于噪声。chrF 能捕捉 BLEU 遗漏的部分形态匹配。")
    print("实际工作请使用 sacrebleu（pip install sacrebleu），而不是这个教学版本。")


if __name__ == "__main__":
    main()
