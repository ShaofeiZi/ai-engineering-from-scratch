"""STaR 循环模拟器——使用 Python 标准库。

这是一个玩具算术任务。“模型”会采用三种策略生成推理过程：
  1. 可靠推理（始终正确）
  2. 惰性捷径（在分布内问题上的答案正确率为 40%，
     在分布外问题上则接近零）
  3. 随机猜测

STaR 的自举轮次只保留答案正确的推理过程。若没有防护，
捷径推理会因在分布内看似正确而得到强化。

模拟器还会运行 V-STaR 风格的推理选择器：采样 N 条推理过程，
选取验证器评分最高的一条。验证器本身也使用同一数据训练，
因此在 OOD 场景中，它可能把确信但错误的推理排在坦诚表达
不确定性的推理之前。
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field


@dataclass
class Trace:
    strategy: str  # "sound"、"shortcut" 或 "random"
    answer_correct: bool
    rationale_sound: bool


@dataclass
class Model:
    prob_sound: float
    prob_shortcut: float
    # 隐含的 prob_random = 1 - sound - shortcut

    def sample(self, on_ood: bool) -> Trace:
        r = random.random()
        if r < self.prob_sound:
            return Trace("sound", True, True)
        elif r < self.prob_sound + self.prob_shortcut:
            ok = random.random() < (0.05 if on_ood else 0.40)
            return Trace("shortcut", ok, False)
        else:
            ok = random.random() < 0.10
            return Trace("random", ok, False)


def evaluate(model: Model, n: int, on_ood: bool) -> tuple[float, float]:
    """返回（答案准确率，可靠推理占比）。"""
    correct = 0
    sound = 0
    for _ in range(n):
        t = model.sample(on_ood)
        if t.answer_correct:
            correct += 1
        if t.rationale_sound:
            sound += 1
    return correct / n, sound / n


def star_round(model: Model, n_samples: int = 1000) -> Model:
    """运行一轮 STaR：保留答案正确的轨迹并重新训练。"""
    kept = []
    for _ in range(n_samples):
        t = model.sample(on_ood=False)
        if t.answer_correct:
            kept.append(t)

    if not kept:
        return model

    sound_kept = sum(1 for k in kept if k.strategy == "sound")
    shortcut_kept = sum(1 for k in kept if k.strategy == "shortcut")
    random_kept = sum(1 for k in kept if k.strategy == "random")
    total = len(kept)

    # 按被强化的轨迹更新比例，并混入旧先验以避免坍缩。
    alpha = 0.6
    new_sound = alpha * (sound_kept / total) + (1 - alpha) * model.prob_sound
    new_short = alpha * (shortcut_kept / total) + (1 - alpha) * model.prob_shortcut

    # 重新归一化。
    s = new_sound + new_short
    if s > 1.0:
        new_sound /= s
        new_short /= s
    return Model(new_sound, new_short)


def run_star(rounds: int, initial: Model) -> list[Model]:
    models = [initial]
    m = initial
    for _ in range(rounds):
        m = star_round(m)
        models.append(m)
    return models


def vstar_infer(model: Model, samples_per_problem: int, n_problems: int,
                on_ood: bool) -> float:
    """V-STaR 风格的 best-of-N：选出最可信的轨迹。这里把验证器
    建模为置信度评分，而评分本身会因可靠推理或捷径推理而产生偏差
    （可靠推理的排序可靠性为 0.9，捷径推理为 0.55）。

    注意：这是一个理想化验证器——它能读取真实的 ``rationale_sound``
    标记，因此代表训练良好的验证器所能达到的上限。真实验证器必须从
    轨迹本身推断推理是否可靠，所以实际收益会更小。
    """
    correct = 0
    for _ in range(n_problems):
        traces = [model.sample(on_ood) for _ in range(samples_per_problem)]
        # 验证器尝试选出正确轨迹，但并不完美。
        best = None
        best_score = -1.0
        for t in traces:
            score = 0.9 if t.rationale_sound else (0.55 if t.answer_correct else 0.3)
            score += random.random() * 0.1
            if score > best_score:
                best_score = score
                best = t
        if best and best.answer_correct:
            correct += 1
    return correct / n_problems


def report_round(label: str, models: list[Model]) -> None:
    print(f"\n{label}")
    print("-" * 70)
    print(f"  {'轮次':>5}  {'p(可靠)':>10}  {'p(捷径)':>12}  "
          f"{'ID 准确率':>8}  {'OOD 准确率':>8}  {'可靠占比':>10}")
    for i, m in enumerate(models):
        id_acc, id_sound = evaluate(m, 500, on_ood=False)
        ood_acc, _ = evaluate(m, 500, on_ood=True)
        print(f"  {i:>5}  {m.prob_sound:>10.3f}  {m.prob_shortcut:>12.3f}  "
              f"{id_acc:>8.1%}  {ood_acc:>8.1%}  {id_sound:>10.1%}")


def vstar_report(model: Model) -> None:
    print("\nV-STaR best-of-N 推理")
    print("-" * 70)
    for n in (1, 4, 16):
        for ood in (False, True):
            acc = vstar_infer(model, n, 500, ood)
            tag = "OOD" if ood else "ID"
            print(f"  n={n:>3}  {tag:<3}  准确率 {acc:.1%}")


def main() -> None:
    random.seed(42)
    print("=" * 70)
    print("STaR、V-STaR、QUIET-STaR（阶段 15，第 2 课）")
    print("=" * 70)

    print("\n场景 A：无捷径的基础模型（纯净推理先验）")
    models = run_star(5, Model(prob_sound=0.20, prob_shortcut=0.0))
    report_round("STaR 自举轮次（纯净）", models)

    print("\n场景 B：倾向使用捷径的基础模型（分布内命中率 0.4）")
    models = run_star(5, Model(prob_sound=0.20, prob_shortcut=0.40))
    report_round("STaR 自举轮次（含捷径）", models)

    vstar_report(models[-1])

    print()
    print("=" * 70)
    print("要点：STaR 会强化任何能够得到答案的方式")
    print("-" * 70)
    print("  场景 A 的 ID 与 OOD 表现都得到提升。")
    print("  场景 B 的 ID 表现提升，但 OOD 表现崩溃——捷径因在训练数据中")
    print("  看似正确而得到强化。")
    print("  V-STaR 的验证器有助于推理阶段，但无法消除训练时习得的偏差。")


if __name__ == "__main__":
    main()
