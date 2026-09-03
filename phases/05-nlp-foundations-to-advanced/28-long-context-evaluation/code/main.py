import random
import re


FILLER_WORDS = (
    "the quick brown fox jumps over the lazy dog and then runs across the field "
    "where birds sing and flowers bloom while the river flows gently toward the sea "
    "and clouds drift slowly across a sky painted in hues of orange and pink "
    "as evening approaches and the day begins to fade into quiet darkness "
).split()


def make_filler(n_tokens, seed):
    rng = random.Random(seed)
    return " ".join(rng.choice(FILLER_WORDS) for _ in range(n_tokens))


def insert_needle(filler, needle, depth_ratio):
    words = filler.split()
    pos = int(len(words) * depth_ratio)
    return " ".join(words[:pos] + [needle] + words[pos:])


def mock_retrieval_model(context, question, effective_capacity):
    needle_pattern = re.compile(r"(?:the magic word is|the secret code is|the pass phrase is|x1\s*=|x2\s*=|x3\s*=) [A-Z0-9a-z_]+", re.IGNORECASE)
    matches = list(needle_pattern.finditer(context))
    if not matches:
        return "no answer"
    total_len = len(context.split())
    for m in matches:
        before = len(context[:m.start()].split())
        if before <= effective_capacity:
            return m.group(0).split()[-1].strip(",.")
    return "no answer"


def score_single_needle(context, expected, effective_capacity):
    question = "What is the magic word?"
    answer = mock_retrieval_model(context, question, effective_capacity)
    return 1 if expected.lower() in answer.lower() else 0


def score_multi_needle(context, expected_list, effective_capacity):
    total_len = len(context.split())
    needle_pattern = re.compile(r"the magic word is ([A-Za-z0-9_]+)", re.IGNORECASE)
    found = []
    for m in needle_pattern.finditer(context):
        before = len(context[:m.start()].split())
        if before <= effective_capacity:
            found.append(m.group(1).lower())
    hits = sum(1 for e in expected_list if e.lower() in found)
    return hits / len(expected_list)


def run_niah_grid(lengths, depths, seed=0):
    needle = "the magic word is pineapple"
    expected = "pineapple"
    capacity = 20000
    corner = "depth\\len"
    print(f"  {corner:<12}  " + "  ".join(f"{n:>6}" for n in lengths))
    for d in depths:
        row = []
        for n in lengths:
            filler = make_filler(n, seed=seed + n)
            haystack = insert_needle(filler, needle, d)
            row.append(score_single_needle(haystack, expected, capacity))
        tag = "  PASS" if all(row) else f"{sum(row)}/{len(row)}"
        print(f"  depth={d:<5}    " + "  ".join(f"{x:>6}" for x in row) + "    " + tag)


def run_multi_needle(length, n_needles=3, seed=42):
    needles = ["the magic word is pineapple",
               "the magic word is compass",
               "the magic word is whisper"][:n_needles]
    expected = ["pineapple", "compass", "whisper"][:n_needles]
    filler = make_filler(length, seed=seed)
    depths = [0.2, 0.5, 0.8][:n_needles]
    words = filler.split()
    for d, n in sorted(zip(depths, needles), reverse=True):
        pos = int(len(words) * d)
        words = words[:pos] + [n] + words[pos:]
    haystack = " ".join(words)
    return score_multi_needle(haystack, expected, effective_capacity=length)


def main():
    lengths = [500, 2000, 8000, 20000, 40000]
    depths = [0.1, 0.3, 0.5, 0.7, 0.9]

    print("=== 简化 NIAH 网格（模拟模型的有效容量 = 20k）===")
    print("标记：1 = 在上下文中找到针，0 = 未找到针")
    print()
    run_niah_grid(lengths, depths)

    print()
    print("=== 多针测试：长度=10000，n=3 ===")
    score = run_multi_needle(10000, n_needles=3)
    print(f"  找到 {score * 3:.0f} / 3 根针")

    print()
    print("注意事项：")
    print("  模拟模型采用硬性有效容量截断；真正的 LLM 会逐渐退化。")
    print("  真正的 NIAH：扫描 5 个深度 × 6 个长度，为每个模型生成热力图。")
    print("  始终搭配一个多跳/聚合任务（RULER）——单针测试容易饱和。")


if __name__ == "__main__":
    main()
