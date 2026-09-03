"""CLIP / SigLIP 对比损失玩具示例 — 纯标准库 Python。

在手工构造的相似度矩阵上实现 InfoNCE（softmax）和 sigmoid 逐对损失。
还用合成图像和文本嵌入走查一个零样本分类小演示。

无需 numpy。无需 torch。重点是看清损失数学和 argmax 模式。
"""

from __future__ import annotations

import math
import random


def normalize(v: list[float]) -> list[float]:
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v]


def cosine(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def similarity_matrix(images: list[list[float]],
                      texts: list[list[float]],
                      tau: float) -> list[list[float]]:
    I = [normalize(v) for v in images]
    T = [normalize(v) for v in texts]
    N = len(I)
    S = [[0.0] * N for _ in range(N)]
    for i in range(N):
        for j in range(N):
            S[i][j] = cosine(I[i], T[j]) / tau
    return S


def log_sum_exp(row: list[float]) -> float:
    m = max(row)
    return m + math.log(sum(math.exp(x - m) for x in row))


def infonce_loss(S: list[list[float]]) -> float:
    """对行和列做对称 InfoNCE。"""
    N = len(S)
    loss_i2t = 0.0
    for i in range(N):
        loss_i2t += -S[i][i] + log_sum_exp(S[i])
    loss_t2i = 0.0
    for j in range(N):
        col = [S[i][j] for i in range(N)]
        loss_t2i += -S[j][j] + log_sum_exp(col)
    return (loss_i2t + loss_t2i) / (2 * N)


def sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def sigmoid_loss(S: list[list[float]], bias: float = 0.0) -> float:
    """SigLIP 风格的逐样本对 BCE；正样本在对角线上。"""
    N = len(S)
    total = 0.0
    count = 0
    for i in range(N):
        for j in range(N):
            logit = S[i][j] + bias
            y = 1.0 if i == j else 0.0
            p = sigmoid(logit)
            eps = 1e-9
            term = y * math.log(p + eps) + (1 - y) * math.log(1 - p + eps)
            total += -term
            count += 1
    return total / count


def zero_shot_classify(image: list[float],
                       class_texts: dict[str, list[float]]) -> list[tuple[str, float]]:
    """在类提示上做余弦相似度 argmax。"""
    img = normalize(image)
    scores = []
    for name, vec in class_texts.items():
        scores.append((name, cosine(img, normalize(vec))))
    scores.sort(key=lambda p: p[1], reverse=True)
    return scores


def make_fake_embedding(seed: int, dim: int = 64) -> list[float]:
    rng = random.Random(seed)
    return [rng.gauss(0, 1) for _ in range(dim)]


def demo_infonce() -> None:
    print("\n演示 1：4 对对齐样本上的 InfoNCE")
    print("-" * 60)
    images = [make_fake_embedding(i) for i in range(4)]
    texts = [[x + 0.05 * make_fake_embedding(i + 100)[k] for k, x in enumerate(v)]
             for i, v in enumerate(images)]

    for tau in (0.07, 0.1, 1.0):
        S = similarity_matrix(images, texts, tau=tau)
        loss = infonce_loss(S)
        slip = sigmoid_loss(S)
        print(f"  tau={tau:4.2f}  InfoNCE={loss:.4f}  SigLIP={slip:.4f}")


def demo_shuffled() -> None:
    print("\n演示 2：错位样本会发生什么")
    print("-" * 60)
    images = [make_fake_embedding(i) for i in range(6)]
    texts = [make_fake_embedding(i + 500) for i in range(6)]
    S = similarity_matrix(images, texts, tau=0.07)
    loss = infonce_loss(S)
    slip = sigmoid_loss(S)
    print(f"  错位：InfoNCE={loss:.4f}  SigLIP={slip:.4f}")
    aligned_imgs = [make_fake_embedding(i) for i in range(6)]
    aligned_txt = [[x + 0.02 for x in v] for v in aligned_imgs]
    S2 = similarity_matrix(aligned_imgs, aligned_txt, tau=0.07)
    print(f"  对齐      ：InfoNCE={infonce_loss(S2):.4f}  "
          f"SigLIP={sigmoid_loss(S2):.4f}")
    print("  对齐损失 < 错位损失，验证了梯度信号。")


def demo_zero_shot() -> None:
    print("\n演示 3：零样本分类")
    print("-" * 60)
    classes = {
        "cat": make_fake_embedding(42),
        "dog": make_fake_embedding(43),
        "bird": make_fake_embedding(44),
        "car": make_fake_embedding(45),
    }
    query_image = [c + 0.3 * make_fake_embedding(999)[i]
                   for i, c in enumerate(classes["dog"])]

    ranked = zero_shot_classify(query_image, classes)
    print("  查询图像（接近“狗”原型）：")
    display_names = {"cat": "猫", "dog": "狗", "bird": "鸟", "car": "汽车"}
    for name, score in ranked:
        print(f"    {display_names[name]:6s}：{score:+.4f}")
    print(f"  第一名：{display_names[ranked[0][0]]}")


def demo_prompt_ensemble() -> None:
    print("\n演示 4：提示模板集成")
    print("-" * 60)
    templates = [
        "一张{class}的照片",
        "一幅{class}的图片",
        "一张展示{class}的图像",
    ]
    class_name = "金毛寻回犬"
    ensemble_vec = [0.0] * 64
    count = 0
    for t in templates:
        prompt = t.format(**{"class": class_name})
        seed = sum(ord(c) for c in prompt)
        emb = make_fake_embedding(seed)
        for k in range(64):
            ensemble_vec[k] += emb[k]
        count += 1
    ensemble_vec = [x / count for x in ensemble_vec]
    print(f"  为 {count} 个 '{class_name}' 提示做集成")
    print(f"  前 6 维：{[round(x, 3) for x in ensemble_vec[:6]]}")
    print("  单模板：噪声更大；集成：在真实基准上可提高 1-3 分。")


def main() -> None:
    print("=" * 60)
    print("CLIP / SIGLIP 对比训练（第 12 阶段，第 02 课）")
    print("=" * 60)
    demo_infonce()
    demo_shuffled()
    demo_zero_shot()
    demo_prompt_ensemble()
    print("\n" + "=" * 60)
    print("要点")
    print("-" * 60)
    print("  · InfoNCE 同时惩罚行与列（对称）")
    print("  · tau 越小，softmax 越尖锐，困难负样本压力越大")
    print("  · Sigmoid 损失将样本对解耦，分布式运行无需 all-gather")
    print("  · 零样本分类是在类别提示上取 argmax cos(image, prompt)")


if __name__ == "__main__":
    main()
