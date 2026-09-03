"""熔断开关 + 断路器 + 金丝雀模拟器 — 纯标准库 Python。

三个检测器：
  1. 熔断开关：代理外部的布尔值；每轮检查一次
  2. 断路器：连续 N 次相同工具调用时断开
  3. 金丝雀令牌：虚假凭证，被读取即触发告警

轨迹演练了每个检测器。第 10 课和第 13 课的分类器与预算层
相互正交；本课关注的是这些触发式防线。
"""

from __future__ import annotations

from dataclasses import dataclass, field


# ---------- 外部状态（代理不可写入） ----------

EXTERNAL_KILL_SWITCH = {"enabled": False}


# 已在工作区中植入金丝雀令牌。
CANARY_FILES = {
    "~/.env.canary": "AWS_SECRET_ACCESS_KEY=canary-xxxxxxxxxxxxxxxx",
    "~/notes/fake-credentials.txt": "token = 'canary-sk-ABC-NEVER-USE'",
}


# ---------- 断路器 ----------

@dataclass
class CircuitBreaker:
    threshold: int = 5
    state: str = "closed"   # closed | open | half_open
    recent: list[str] = field(default_factory=list)

    def record(self, action_key: str) -> bool:
        """如果动作可以继续执行则返回 True，如果断路器处于 open 状态则返回 False。"""
        if self.state == "open":
            return False
        self.recent.append(action_key)
        if len(self.recent) >= self.threshold:
            tail = self.recent[-self.threshold:]
            if all(a == tail[0] for a in tail):
                self.state = "open"
                return False
        return True


# ---------- 金丝雀检测器 ----------

@dataclass
class Canary:
    triggered: list[tuple[int, str]] = field(default_factory=list)

    def check_read(self, turn: int, path: str) -> bool:
        if path in CANARY_FILES:
            self.triggered.append((turn, path))
            return True
        return False


# ---------- 运行轨迹 ----------

@dataclass
class Action:
    kind: str    # "tool" | "read"
    payload: str


def run_trajectory(traj: list[Action], kill_switch: dict) -> None:
    breaker = CircuitBreaker(threshold=5)
    canary = Canary()
    kill_fired = False
    breaker_fired = False

    for i, a in enumerate(traj, 1):
        # 检测器 1：熔断开关
        if kill_switch["enabled"]:
            print(f"  {i:>2}。[KILL SWITCH 已触发] 拒绝动作 {a.kind}:{a.payload}")
            kill_fired = True
            break

        # 检测器 2：断路器
        allowed = breaker.record(f"{a.kind}:{a.payload}")
        if not allowed:
            print(f"  {i:>2}. [断路器已打开] {a.kind}:{a.payload}  "
                  f"原因=连续 5 次相同调用")
            breaker_fired = True
            break

        # 检测器 3：金丝雀
        if a.kind == "read":
            hit = canary.check_read(i, a.payload)
            if hit:
                print(f"  {i:>2}. [金丝雀已触发] 读取 {a.payload!r}  "
                      f"-> 已发出警报")
                continue

        print(f"  {i:>2}. 正常  {a.kind}:{a.payload}")

    print(f"  摘要：熔断开关触发={kill_fired}  断路器触发={breaker_fired}  "
          f"金丝雀命中={len(canary.triggered)}")


def main() -> None:
    print("=" * 80)
    print("触发式防线：熔断开关、断路器与金丝雀（第 15 阶段，第 14 课）")
    print("=" * 80)

    traj = [
        Action("tool", "read:src/app.py"),
        Action("tool", "edit:src/app.py"),
        Action("tool", "read:logs/app.log"),   # 开始 identical-read 连发
        Action("tool", "read:logs/app.log"),
        Action("tool", "read:logs/app.log"),
        Action("tool", "read:logs/app.log"),
        Action("tool", "read:logs/app.log"),   # 第 5 次相同调用 -> 断路器触发
        Action("read", "~/notes/checklist.md"),
        Action("read", "~/.env.canary"),       # 金丝雀命中
    ]

    print("\n熔断开关关闭")
    print("-" * 80)
    run_trajectory(traj, EXTERNAL_KILL_SWITCH)

    print("\n熔断开关开启（操作员从外部翻转了它）")
    print("-" * 80)
    EXTERNAL_KILL_SWITCH["enabled"] = True
    run_trajectory(traj, EXTERNAL_KILL_SWITCH)
    EXTERNAL_KILL_SWITCH["enabled"] = False

    print()
    print("=" * 80)
    print("要点：三个检测器，三种不同的故障类别")
    print("-" * 80)
    print("  熔断开关在操作员动作下停止整个代理。")
    print("  断路器暂停特定模式，而非整个代理。")
    print("  金丝雀令牌无需检测内容即可识别意图。")
    print("  以上均无法捕获语义组合攻击（见第 10 课）。")
    print("  硬性宪法约束完善了防御体系（第 17 课）。")


if __name__ == "__main__":
    main()
