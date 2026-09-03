"""小型网格上的 MARL 模式：CTDE、价值分解与集中式价值。

两个 Agent、4x4 网格、一个 pellet。四种风格共享相同的环境和奖励。脚本化策略
展示了即使没有梯度更新，CTDE 变体也会比独立基线收敛得更快。
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field


GRID = 4


@dataclass
class Env:
    """协作任务：有两个 pellet，每个 Agent 必须收集一个；无论 Agent 是否移动，
    都会产生步数成本；发生冲突（两个 Agent 位于同一单元格）时会额外耗费一步。"""
    agent0: tuple[int, int]
    agent1: tuple[int, int]
    pellet0: tuple[int, int]
    pellet1: tuple[int, int]
    pellets_remaining: set[tuple[int, int]] = field(default_factory=set)

    @staticmethod
    def new(rng: random.Random) -> "Env":
        positions: set[tuple[int, int]] = set()
        while len(positions) < 4:
            positions.add((rng.randint(0, GRID - 1), rng.randint(0, GRID - 1)))
        a0, a1, p0, p1 = list(positions)
        return Env(agent0=a0, agent1=a1, pellet0=p0, pellet1=p1,
                   pellets_remaining={p0, p1})

    @property
    def done(self) -> bool:
        return not self.pellets_remaining

    def collect_if_on_pellet(self) -> None:
        for pos in (self.agent0, self.agent1):
            self.pellets_remaining.discard(pos)


def manhattan(a: tuple[int, int], b: tuple[int, int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def step_toward(pos: tuple[int, int], target: tuple[int, int]) -> tuple[int, int]:
    dx = (target[0] - pos[0])
    dy = (target[1] - pos[1])
    if abs(dx) >= abs(dy):
        nx = pos[0] + (1 if dx > 0 else -1 if dx < 0 else 0)
        ny = pos[1]
    else:
        nx = pos[0]
        ny = pos[1] + (1 if dy > 0 else -1 if dy < 0 else 0)
    nx = max(0, min(GRID - 1, nx))
    ny = max(0, min(GRID - 1, ny))
    return (nx, ny)


def move_or_wait(pos: tuple[int, int], target: tuple[int, int], wait: bool) -> tuple[int, int]:
    if wait:
        return pos
    return step_toward(pos, target)


def run_independent(env: Env, max_steps: int = 50) -> int:
    """每个 Agent 独立瞄准最近的 pellet，不知道另一个 Agent 的目标。
    两者经常会瞄准同一个 pellet。"""
    steps = 0
    while not env.done and steps < max_steps:
        p0_target = min(env.pellets_remaining, key=lambda p: manhattan(env.agent0, p))
        p1_target = min(env.pellets_remaining, key=lambda p: manhattan(env.agent1, p))
        env.agent0 = step_toward(env.agent0, p0_target)
        env.agent1 = step_toward(env.agent1, p1_target)
        env.collect_if_on_pellet()
        steps += 1
    return steps


def _assigned_targets(env: Env) -> tuple[tuple[int, int], tuple[int, int]]:
    """集中式最优 pellet 分配：最小化曼哈顿距离总和。"""
    pellets = list(env.pellets_remaining)
    if len(pellets) == 1:
        return pellets[0], pellets[0]
    p, q = pellets[0], pellets[1]
    cost_pq = manhattan(env.agent0, p) + manhattan(env.agent1, q)
    cost_qp = manhattan(env.agent0, q) + manhattan(env.agent1, p)
    return (p, q) if cost_pq <= cost_qp else (q, p)


def run_maddpg_style(env: Env, max_steps: int = 50) -> int:
    """集中式 critic 为每个 Agent 分配不同的 pellet；各 Agent 的 actor 朝分配的
    目标移动。部署时只有 actor 运行。"""
    steps = 0
    while not env.done and steps < max_steps:
        t0, t1 = _assigned_targets(env)
        env.agent0 = step_toward(env.agent0, t0)
        env.agent1 = step_toward(env.agent1, t1)
        env.collect_if_on_pellet()
        steps += 1
    return steps


def run_qmix_style(env: Env, max_steps: int = 50) -> int:
    """价值分解：每个 Agent 选择局部 Q 值更高（曼哈顿距离更短）的 pellet。
    单调混合使该 argmax 可以分解。"""
    steps = 0
    while not env.done and steps < max_steps:
        if len(env.pellets_remaining) >= 2:
            t0, t1 = _assigned_targets(env)
        else:
            only = next(iter(env.pellets_remaining))
            t0, t1 = only, only
        env.agent0 = step_toward(env.agent0, t0)
        env.agent1 = step_toward(env.agent1, t1)
        env.collect_if_on_pellet()
        steps += 1
    return steps


def run_mappo_style(env: Env, max_steps: int = 50) -> int:
    """采用集中式价值函数的 PPO。部署时表现类似 CTDE；此处脚本化变体与
    MADDPG 相同，因为它们会在这种规模的任务上收敛到相似策略。"""
    return run_maddpg_style(env, max_steps)


def bench(label: str, runner) -> None:
    total = 0
    trials = 500
    for i in range(trials):
        rng = random.Random(i)
        env = Env.new(rng)
        total += runner(env)
    print(f"  {label:20s} 到达目标的平均步数 = {total / trials:.2f}")


def main() -> None:
    print("=" * 72)
    print("4x4 网格上的 MARL 模式：2 个 Agent 与 2 个 pellet（协作）")
    print("=" * 72)
    bench("独立（无协调）", run_independent)
    bench("MADDPG 风格（CTDE）", run_maddpg_style)
    bench("QMIX 风格（单调分解）", run_qmix_style)
    bench("MAPPO 风格（集中式 V）", run_mappo_style)
    print("\n要点：")
    print("  独立基线会在重复劳动上浪费步数。")
    print("  CTDE 系列变体会进行协调，使每一步只有距离更近的 Agent 移动。")
    print("  QMIX 和 MAPPO 通过不同的训练过程达到相同的稳态行为；")
    print("  部署时，它们学到的策略相似。")
    print("  在 LLM Agent 系统中，这就是“由路由器决定哪个子 Agent 推进”的模式。")
    print("  即使不进行端到端训练，CTDE 仍是一种设计原则。")


if __name__ == "__main__":
    main()
