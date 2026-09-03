"""WMDP 风格评测框架——仅使用 Python 标准库。

模拟模型是一个三领域专家，具有各领域的准确率向量。
模拟跨生物、网络安全和化学领域的 WMDP 风格选择题评测。
展示 RMU 风格遗忘的权衡：抑制特定领域能力，同时衡量通用能力代价。

用法：python3 code/main.py
"""

from __future__ import annotations

import random


random.seed(47)


DOMAINS = {
    "biosecurity":   {"n_questions": 200, "accuracy": 0.72},
    "cybersecurity": {"n_questions": 200, "accuracy": 0.80},
    "chemistry":     {"n_questions": 200, "accuracy": 0.64},
    "mmlu_general":  {"n_questions": 200, "accuracy": 0.78},
}


def evaluate(model_accuracy: dict) -> dict:
    """运行玩具 WMDP 风格基准测试，返回各领域分数。"""
    results = {}
    for domain, cfg in DOMAINS.items():
        correct = 0
        for _ in range(cfg["n_questions"]):
            acc = model_accuracy.get(domain, cfg["accuracy"])
            if random.random() < acc:
                correct += 1
        results[domain] = correct / cfg["n_questions"]
    return results


def apply_rmu_style_unlearning(model_accuracy: dict,
                               targets: list[str],
                               strength: float = 0.9,
                               collateral: float = 0.03) -> dict:
    """遗忘干预：按 `strength` 降低目标领域准确率，并使其他领域承受
    `collateral` 的准确率损失（通用能力损失）。"""
    new = dict(model_accuracy)
    for d in targets:
        new[d] = max(0.25, new[d] * (1 - strength))
    for d in new:
        if d not in targets:
            new[d] = max(0.0, new[d] - collateral)
    return new


def baseline_model() -> dict:
    return {d: cfg["accuracy"] for d, cfg in DOMAINS.items()}


def report(title: str, r: dict) -> None:
    print(f"\n{title}")
    for d, score in r.items():
        print(f"  {d:18s} : {score:.3f}")


def main() -> None:
    print("=" * 70)
    print("WMDP 风格评测框架（阶段 18，第 17 课）")
    print("=" * 70)

    base = baseline_model()
    report("基线模型的各领域准确率", base)
    baseline_results = evaluate(base)
    report("测得分数（遗忘前）", baseline_results)

    # 遗忘生物 + 化学领域。
    post = apply_rmu_style_unlearning(base, targets=["biosecurity", "chemistry"],
                                       strength=0.85, collateral=0.04)
    post_results = evaluate(post)
    report("测得分数（遗忘后：生物 + 化学）", post_results)

    print("\nuplift 风格计算（新手基线 ~= 0.25 随机水平）：")
    novice = 0.25
    for d in ("biosecurity", "cybersecurity", "chemistry"):
        pre = baseline_results[d]
        pst = post_results[d]
        uplift_pre = pre / novice
        uplift_post = pst / novice
        print(f"  {d:18s}  遗忘前={uplift_pre:.2f}x 新手  遗忘后={uplift_post:.2f}x 新手")

    print("\n" + "=" * 70)
    print("要点：WMDP 无需诱导有害输出，即可给出各领域的能力数值。RMU 风格遗忘")
    print("会降低目标领域分数，同时造成约 3%–4% 的通用能力附带损失。2025 年")
    print("该领域的叙事是“轻度提升”->“临近门槛”->“不足以排除 ASL-3”，")
    print("每次转变都由不同研究支持。")
    print("=" * 70)


if __name__ == "__main__":
    main()
