"""AI Scientist v2 循环模拟器 — 纯标准库 Python。

将研究循环建模为状态机，每个阶段可配置失败概率，
种子数据来自 Beel 等人 (2025) 对 AI Scientist 真实行为的
研究发现。运行多次试验并报告结果分布，包括关键的
"打磨过的论文伴随有缺陷实验"类别。
"""

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass


DEFAULT_SEED = 42


@dataclass
class LoopConfig:
    # 想法在并不新颖时被误标为新颖的概率。
    novelty_mislabel: float = 0.25
    # 实验因代码错误而失败的概率（Beel 等人约 0.42）。
    experiment_failure: float = 0.42
    # 实验失败后通过重试可恢复的比例。
    retry_recovery: float = 0.55
    # 视觉-语言图形批评在底层实验已损坏时
    # 仍产出干净视觉效果的概率。
    polish_masks_weakness: float = 0.70
    # 自动写作阶段在给定（可能有缺陷的）实验数据时
    # 产出一篇连贯论文的概率。
    writeup_success: float = 0.85
    # 内部审稿人接受概率（弱审稿人）。
    internal_review_accept: float = 0.50


@dataclass
class Outcome:
    submitted: bool
    has_novelty_flaw: bool
    has_experiment_flaw: bool
    polished_but_flawed: bool
    polished_ok: bool
    abandoned_stage: str


def run_one(cfg: LoopConfig) -> Outcome:
    # 在本玩具模型中，想法生成总是成功。
    has_novelty_flaw = random.random() < cfg.novelty_mislabel

    # 实验执行：失败 + 重试恢复。
    failed = random.random() < cfg.experiment_failure
    if failed:
        recovered = random.random() < cfg.retry_recovery
        if not recovered:
            return Outcome(
                submitted=False,
                has_novelty_flaw=has_novelty_flaw,
                has_experiment_flaw=True,
                polished_but_flawed=False,
                polished_ok=False,
                abandoned_stage="experiment",
            )
        # 建模选择：重试恢复的实验仍带有残余缺陷
        #（静默错误的数值、形状不匹配仅打补丁而未重新
        # 验证等）。该残余缺陷正是打磨阶段后续可掩盖的
        # 内容，也是"打磨过但有缺陷"类别的主要驱动因素。
        has_experiment_flaw = True
    else:
        has_experiment_flaw = False

    # 视觉-语言图形打磨。
    polished_hides_weakness = (
        has_experiment_flaw and random.random() < cfg.polish_masks_weakness
    )

    # 写作阶段。
    if random.random() > cfg.writeup_success:
        return Outcome(
            submitted=False,
            has_novelty_flaw=has_novelty_flaw,
            has_experiment_flaw=has_experiment_flaw,
            polished_but_flawed=False,
            polished_ok=False,
            abandoned_stage="writeup",
        )

    # 内部审稿。
    if random.random() > cfg.internal_review_accept:
        return Outcome(
            submitted=False,
            has_novelty_flaw=has_novelty_flaw,
            has_experiment_flaw=has_experiment_flaw,
            polished_but_flawed=False,
            polished_ok=False,
            abandoned_stage="internal_review",
        )

    polished_ok = not has_experiment_flaw and not has_novelty_flaw
    # 任何带有缺陷的已提交论文都计为 polished_but_flawed：
    # 弱内部审稿人无论打磨阶段是否将其隐藏都会放行。
    # 这使得两个桶对已提交论文做到穷尽
    #（polished_ok + polished_but_flawed == len(submitted)）。
    polished_but_flawed = has_experiment_flaw or has_novelty_flaw
    return Outcome(
        submitted=True,
        has_novelty_flaw=has_novelty_flaw,
        has_experiment_flaw=has_experiment_flaw,
        polished_but_flawed=polished_but_flawed,
        polished_ok=polished_ok,
        abandoned_stage="",
    )


def report(n: int, cfg: LoopConfig) -> None:
    outs = [run_one(cfg) for _ in range(n)]

    submitted = [o for o in outs if o.submitted]
    abandoned = [o for o in outs if not o.submitted]
    polished_ok = [o for o in submitted if o.polished_ok]
    polished_but_flawed = [o for o in submitted if o.polished_but_flawed]

    print("  配置")
    print(f"    新颖性误标率：          {cfg.novelty_mislabel:.2f}")
    print(f"    实验失败率：            {cfg.experiment_failure:.2f}")
    print(f"    重试恢复比例：          {cfg.retry_recovery:.2f}")
    print(f"    打磨掩盖缺陷的概率：    {cfg.polish_masks_weakness:.2f}")
    print(f"    写作成功率：            {cfg.writeup_success:.2f}")
    print(f"    内部审稿接受率：        {cfg.internal_review_accept:.2f}")

    print()
    print(f"  试验数                    : {n}")
    print(f"  提交数                    : {len(submitted)} ({len(submitted) / n:.1%})")
    print(f"  放弃数                    : {len(abandoned)} ({len(abandoned) / n:.1%})")
    by_stage = {}
    for o in abandoned:
        by_stage[o.abandoned_stage] = by_stage.get(o.abandoned_stage, 0) + 1
    for stage, count in sorted(by_stage.items()):
        print(f"    阶段 {stage:<18}: {count}")

    print()
    print("  提交质量分布")
    print(f"    干净（新颖且有效）        : {len(polished_ok)} "
          f"(占试验 {len(polished_ok) / n:.1%}，"
          f"占提交 {len(polished_ok) / max(1, len(submitted)):.1%})")
    print(f"    打磨过但有缺陷            : {len(polished_but_flawed)} "
          f"(占试验 {len(polished_but_flawed) / n:.1%}，"
          f"占提交 {len(polished_but_flawed) / max(1, len(submitted)):.1%})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-failure", type=float, default=None,
                        help="覆盖基线运行中的 LoopConfig.experiment_failure")
    parser.add_argument("--novelty-mislabel", type=float, default=None,
                        help="覆盖基线运行中的 LoopConfig.novelty_mislabel")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help="随机数种子（默认：%(default)s）")
    args = parser.parse_args()

    random.seed(args.seed)
    print("=" * 70)
    print("AI SCIENTIST V2 循环模拟器（阶段 15，第 5 课）")
    print("=" * 70)

    overrides = {}
    if args.experiment_failure is not None:
        overrides["experiment_failure"] = args.experiment_failure
    if args.novelty_mislabel is not None:
        overrides["novelty_mislabel"] = args.novelty_mislabel
    baseline_cfg = LoopConfig(**overrides)

    label = "基线（Beel 式参数）" if not overrides else "基线（已覆盖）"
    print(f"\n{label}")
    print("-" * 70)
    report(1000, baseline_cfg)

    print("\n乐观场景（更严格的参数）")
    print("-" * 70)
    report(1000, LoopConfig(
        novelty_mislabel=0.10,
        experiment_failure=0.20,
        retry_recovery=0.80,
        polish_masks_weakness=0.40,
        writeup_success=0.92,
        internal_review_accept=0.60,
    ))

    print()
    print("=" * 70)
    print("头条：提交量超过可靠研究量")
    print("-" * 70)
    print("  即便在乐观场景下，仍有不可忽视比例的已提交论文")
    print("  带有打磨阶段帮助隐藏的缺陷。这正是")
    print("  '展示质量差距'的操作含义——也是")
    print("  循环与任何投稿场所之间需要设置人工审稿关卡的原因。")


if __name__ == "__main__":
    main()
