"""分层 cost-governor 模拟器 — 纯标准库 Python。

模拟一个 agent 在 30 轮后陷入轮询循环的过程。对比
三种配置：

  1. 无上限：无限制花费
  2. 仅月度上限：最终会捕获，但在此之前已花费大量
  3. 分层堆叠：per-request + 迭代 + 速率限制 + 月度上限

指标：执行轮数、总 token 数、总花费美元、触发的限制器。
"""

from __future__ import annotations

from dataclasses import dataclass, field


# ---------- 模拟运行配置 ----------

NORMAL_TURN_TOKENS = 2_500
LOOP_TURN_TOKENS = 8_000
LOOP_STARTS_AT = 30

# $/token（input+output 混合），用于 Sonnet-class 模型，mid-2026 费率
DOLLARS_PER_KTOK = 0.003


def turn_cost(turn: int) -> int:
    return LOOP_TURN_TOKENS if turn >= LOOP_STARTS_AT else NORMAL_TURN_TOKENS


# ---------- 治理器 ----------

@dataclass
class Governor:
    max_tokens_per_request: int = 10_000
    max_turns: int = 200
    max_budget_usd: float = 50.0
    velocity_usd_per_min: float = 5.0       # 在超过此滚动速率时截断
    velocity_window_min: float = 10.0
    monthly_cap_usd: float = 500.0

    enable_request_cap: bool = True
    enable_iter_cap: bool = True
    enable_velocity: bool = True
    enable_session_cap: bool = True
    enable_monthly_cap: bool = True

    # 模拟器的 per-minute 轮次速率（每轮秒数）
    seconds_per_turn: float = 30.0


@dataclass
class Run:
    turns: int = 0
    tokens: int = 0
    dollars: float = 0.0
    history: list[tuple[float, float]] = field(default_factory=list)  # （分钟，dollars-at-that-minute）
    stopped_by: str = ""


EPSILON_MIN = 1e-9


def velocity_exceeded(run: Run, gov: Governor, now_min: float) -> bool:
    if not run.history:
        return False
    cutoff = now_min - gov.velocity_window_min
    window = [(t, d) for (t, d) in run.history if t >= cutoff]
    if not window:
        return False
    start_min, start_dollars = window[0]
    window_dollars = run.dollars - start_dollars
    # 使用窗口内的实际经过时间，而非标称
    # 窗口宽度。在 warm-up 期间（now_min < velocity_window_min），这会
    # 阻止速率变为 under-reported.
    elapsed = max(now_min - start_min, EPSILON_MIN)
    rate = window_dollars / elapsed
    return rate > gov.velocity_usd_per_min


def simulate(gov: Governor, label: str) -> Run:
    run = Run()
    now_min = 0.0

    for turn in range(1, 10_001):
        tok = turn_cost(turn)
        if gov.enable_request_cap and tok > gov.max_tokens_per_request:
            tok = gov.max_tokens_per_request
        run.turns = turn
        run.tokens += tok
        run.dollars += (tok / 1000.0) * DOLLARS_PER_KTOK
        now_min += gov.seconds_per_turn / 60.0
        run.history.append((now_min, run.dollars))

        if gov.enable_iter_cap and turn >= gov.max_turns:
            run.stopped_by = "max_turns"
            break
        if gov.enable_session_cap and run.dollars >= gov.max_budget_usd:
            run.stopped_by = "max_budget_usd"
            break
        if gov.enable_velocity and velocity_exceeded(run, gov, now_min):
            run.stopped_by = "velocity_limit"
            break
        if gov.enable_monthly_cap and run.dollars >= gov.monthly_cap_usd:
            run.stopped_by = "monthly_cap"
            break

    if not run.stopped_by:
        run.stopped_by = "模拟轮次已耗尽"

    print(f"  {label:<24}  轮次={run.turns:>5}  token={run.tokens:>8,}  "
          f"费用=${run.dollars:>7.2f}  停止原因={run.stopped_by}")
    return run


def main() -> None:
    print("=" * 85)
    print("LAYERED COST GOVERNORS（第 15 阶段，第 13 课）")
    print("=" * 85)
    print()
    print("Agent 在第 30 轮进入轮询循环。")
    print("-" * 85)

    # 1. 无上限
    g = Governor(
        enable_request_cap=False,
        enable_iter_cap=False,
        enable_velocity=False,
        enable_session_cap=False,
        enable_monthly_cap=False,
    )
    # 上限设得极大以使模拟终止；此行为"无限制"情形。
    g.max_turns = 10_000
    g.enable_iter_cap = True
    simulate(g, "无上限（模拟 1 万轮）")

    # 2. 仅月度上限
    g = Governor(
        enable_request_cap=False,
        enable_iter_cap=False,
        enable_velocity=False,
        enable_session_cap=False,
        enable_monthly_cap=True,
    )
    simulate(g, "仅月度上限")

    # 3. 分层堆叠
    g = Governor()
    simulate(g, "分层堆叠")

    print()
    print("=" * 85)
    print("要点：上限必须分层，因为故障模式随时间尺度而异")
    print("-" * 85)
    print("  月度上限触发较晚：预算已经消耗近半。")
    print("  速率限制（$5/分钟滚动）可在几分钟内捕获循环。")
    print("  迭代上限防止单次运行超过 N 轮。")
    print("  Per-request 上限防止单次补全变为无限制。")
    print("  会话美元上限（max_budget_usd）为成本系紧安全带。")
    print("  每层覆盖不同的故障（循环、泄漏、激增、释放）。")


if __name__ == "__main__":
    main()
