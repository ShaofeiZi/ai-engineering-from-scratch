"""失败模式：MAST 分类器、熔断器与重试风暴模拟器。

仅使用 stdlib。模拟器展示在没有熔断器时，10% 的下游错误率如何经由重试放大为
10 倍负载；熔断器则会限制这种放大。
"""
from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from enum import Enum


# ---------- MAST 分类器 ----------

MAST_CATEGORIES = {
    "spec": "规格问题（占失败的 41.77%）",
    "coord": "协调失败（占失败的 36.94%）",
    "verify": "验证缺口（占失败的 21.30%）",
}

GROUPTHINK = {
    "monoculture": "monoculture 崩溃（相同基础模型 → 相关错误）",
    "conformity": "从众偏差（Agent 与声音最大的同伴保持一致）",
    "tom": "心智理论不足（无法对彼此建模）",
    "mixed_motive": "混合动机漂移（折中方案不能满足任何一方）",
    "cascade": "级联可靠性故障（重试风暴）",
}


def categorize_incident(symptoms: dict) -> tuple[str, str]:
    if symptoms.get("role_conflict") or symptoms.get("task_ambiguity"):
        return "spec", MAST_CATEGORIES["spec"]
    if symptoms.get("state_drift") or symptoms.get("message_lost") or symptoms.get("sync_error"):
        return "coord", MAST_CATEGORIES["coord"]
    if symptoms.get("no_verifier") or symptoms.get("hallucination_propagation"):
        return "verify", MAST_CATEGORIES["verify"]
    return "unknown", "没有匹配的 MAST 类别"


def detect_groupthink(symptoms: dict) -> list[tuple[str, str]]:
    hits = []
    if symptoms.get("correlated_errors"):
        hits.append(("monoculture", GROUPTHINK["monoculture"]))
    if symptoms.get("agreement_rate_spike"):
        hits.append(("conformity", GROUPTHINK["conformity"]))
    if symptoms.get("coordination_drop_long_horizon"):
        hits.append(("tom", GROUPTHINK["tom"]))
    if symptoms.get("compromise_outputs"):
        hits.append(("mixed_motive", GROUPTHINK["mixed_motive"]))
    if symptoms.get("retry_amplification"):
        hits.append(("cascade", GROUPTHINK["cascade"]))
    return hits


# ---------- 断路器 ----------

class BreakerState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass
class CircuitBreaker:
    failure_threshold: float = 0.5
    window_size: int = 20
    open_cooldown_s: float = 0.5
    state: BreakerState = BreakerState.CLOSED
    outcomes: list[bool] = field(default_factory=list)
    opened_at: float = 0.0

    def _error_rate(self) -> float:
        if not self.outcomes:
            return 0.0
        recent = self.outcomes[-self.window_size:]
        return 1.0 - (sum(recent) / len(recent))

    def allow(self) -> bool:
        if self.state == BreakerState.OPEN:
            if time.monotonic() - self.opened_at >= self.open_cooldown_s:
                self.state = BreakerState.HALF_OPEN
            else:
                return False
        return True

    def record(self, success: bool) -> None:
        self.outcomes.append(success)
        if self.state == BreakerState.HALF_OPEN:
            if success:
                self.state = BreakerState.CLOSED
            else:
                self.state = BreakerState.OPEN
                self.opened_at = time.monotonic()
        elif self.state == BreakerState.CLOSED:
            if self._error_rate() > self.failure_threshold and len(self.outcomes) >= self.window_size:
                self.state = BreakerState.OPEN
                self.opened_at = time.monotonic()


# ---------- 重试风暴模拟器 ----------

@dataclass
class DownstreamService:
    base_failure_rate: float = 0.1
    load: int = 0

    def handle(self, rng: random.Random) -> bool:
        # 负载增加 -> 失败率上升（退化模型）
        effective_rate = self.base_failure_rate + (self.load * 0.02)
        effective_rate = min(effective_rate, 0.99)
        return rng.random() > effective_rate


def simulate_retry_storm(requests: int, use_breaker: bool, seed: int = 0) -> tuple[int, int, int]:
    rng = random.Random(seed)
    service = DownstreamService()
    breaker = CircuitBreaker()
    total_calls = 0
    successes = 0
    short_circuits = 0

    for _ in range(requests):
        attempts_for_req = 0
        while attempts_for_req < 4:
            if use_breaker and not breaker.allow():
                short_circuits += 1
                break
            service.load = min(total_calls // 10, 50)
            total_calls += 1
            ok = service.handle(rng)
            if use_breaker:
                breaker.record(ok)
            if ok:
                successes += 1
                break
            attempts_for_req += 1
    return total_calls, successes, short_circuits


def demo_incident_categorization() -> None:
    print("=" * 72)
    print("事件分类 — 将症状映射到 MAST + Groupthink 类别")
    print("=" * 72)
    incidents = [
        {"role_conflict": True, "name": "两个 Agent 都在进行审查"},
        {"state_drift": True, "name": "Agent A 认为已完成，Agent B 仍在运行"},
        {"no_verifier": True, "hallucination_propagation": True, "name": "幻觉事实跨 Agent 传播"},
        {"correlated_errors": True, "agreement_rate_spike": True, "name": "3 个 Agent 给出相同的错误答案"},
        {"retry_amplification": True, "name": "支付重试级联到库存系统"},
    ]
    for inc in incidents:
        name = inc.pop("name")
        cat, desc = categorize_incident(inc)
        gt = detect_groupthink(inc)
        print(f"\n  事件：{name}")
        print(f"    MAST：       {cat} — {desc}")
        if gt:
            for code, d in gt:
                print(f"    群体思维：{code} — {d}")


def demo_retry_storm() -> None:
    print("\n" + "=" * 72)
    print("重试风暴 — 向基线失败率为 10% 的服务发送 200 个请求")
    print("=" * 72)
    total_no_cb, succ_no_cb, _ = simulate_retry_storm(200, use_breaker=False, seed=0)
    total_cb, succ_cb, sc = simulate_retry_storm(200, use_breaker=True, seed=0)
    print(f"  无熔断器：总调用数={total_no_cb:4d}  成功={succ_no_cb:4d}")
    print(f"  有熔断器：总调用数={total_cb:4d}  成功={succ_cb:4d}  已短路={sc}")
    print("  失败率一旦超过阈值，熔断器就会短路调用，从而停止放大。")
    print("  成功次数可能会下降，但下游服务得以存活。")


def main() -> None:
    demo_incident_categorization()
    demo_retry_storm()
    print("\n要点：")
    print("  MAST 类别为失败命名；分类是修复的第一步。")
    print("  熔断器是规范的级联缓解措施，与分布式系统相同。")
    print("  STRATUS 风格的检测 / 诊断 / 验证可将缓解成功率提升 1.5 倍。")
    print("  慢性失败代理指标（一致率、重试率）能在明显错误出现前发现漂移。")


if __name__ == "__main__":
    main()
