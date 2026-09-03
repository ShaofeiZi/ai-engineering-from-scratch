"""结果评估器：改进检验、配对 t 检验、对数归一化、判定结论。

概念参考：
- ./docs/en.md（本课）
- Phase 19 Track A 第 20-29 课（agent harness 原语）

仅依赖标准库与 numpy（读取结果列表）。t 检验的数学计算为纯标准库实现。
运行：python3 code/main.py
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np


HIGHER = "higher_is_better"
LOWER = "lower_is_better"
LOG = "log"
LINEAR = "linear"


@dataclass
class ExperimentResultLike:
    """评估器所依赖的 ExperimentResult 字段的子集。

    第 52 课的 runner 输出相同的字段。第 53 课的测试会直接构造轻量实例，
    以便对评估器进行独立测试。
    """
    spec_id: str
    terminal: str
    metrics: dict


@dataclass
class MetricSpec:
    name: str
    direction: str = HIGHER
    scale: str = LINEAR

    def validate(self) -> None:
        if self.direction not in (HIGHER, LOWER):
            raise ValueError(f"未知的 direction {self.direction!r}")
        if self.scale not in (LINEAR, LOG):
            raise ValueError(f"未知的 scale {self.scale!r}")


@dataclass
class Verdict:
    hypothesis_id: int
    metric: str
    direction: str
    scale: str
    candidate_mean: float
    baseline_mean: float
    improvement: float
    p_value: float | None
    significance_threshold: float
    improvement_threshold: float
    verdict: str
    rationale: str

    def to_dict(self) -> dict:
        return {
            "hypothesis_id": self.hypothesis_id,
            "metric": self.metric,
            "direction": self.direction,
            "scale": self.scale,
            "candidate_mean": round(self.candidate_mean, 6),
            "baseline_mean": round(self.baseline_mean, 6),
            "improvement": round(self.improvement, 6),
            "p_value": None if self.p_value is None else round(self.p_value, 6),
            "significance_threshold": self.significance_threshold,
            "improvement_threshold": self.improvement_threshold,
            "verdict": self.verdict,
            "rationale": self.rationale,
        }


class PairingError(ValueError):
    """当候选与基线结果无法按 seed 配对时抛出。"""


def _lentz_betacf(a: float, b: float, x: float, max_iter: int = 200, eps: float = 1e-12) -> float:
    """通过 Lentz 算法计算正则化不完全 beta 函数的连分数。"""
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < eps:
        d = eps
    d = 1.0 / d
    h = d
    for m in range(1, max_iter + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < eps:
            d = eps
        c = 1.0 + aa / c
        if abs(c) < eps:
            c = eps
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < eps:
            d = eps
        c = 1.0 + aa / c
        if abs(c) < eps:
            c = eps
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            return h
    return h


def regularised_incomplete_beta(a: float, b: float, x: float) -> float:
    """通过标准 Lentz 连分数计算 I_x(a, b)，关于 x 对称。"""
    if x < 0.0 or x > 1.0:
        raise ValueError("x 不在 [0, 1] 范围内")
    if x == 0.0:
        return 0.0
    if x == 1.0:
        return 1.0
    log_bt = (
        math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
        + a * math.log(x) + b * math.log(1.0 - x)
    )
    bt = math.exp(log_bt)
    if x < (a + 1.0) / (a + b + 2.0):
        return bt * _lentz_betacf(a, b, x) / a
    return 1.0 - bt * _lentz_betacf(b, a, 1.0 - x) / b


def two_sided_t_p_value(t_stat: float, df: int) -> float:
    """自由度为 df 的 t 统计量的双侧 p 值。"""
    if df <= 0:
        return 1.0
    x = df / (df + t_stat * t_stat)
    p = regularised_incomplete_beta(df / 2.0, 0.5, x)
    return max(0.0, min(1.0, p))


def paired_t_test(candidate: list[float], baseline: list[float]) -> tuple[float, float | None, int]:
    """返回 (mean_diff, p_value, n)。当 n < 2 时 p_value 为 None。"""
    if len(candidate) != len(baseline):
        raise PairingError("候选与基线长度不一致")
    n = len(candidate)
    if n < 2:
        if n == 0:
            return 0.0, None, 0
        return float(candidate[0] - baseline[0]), None, n
    diffs = np.array(candidate) - np.array(baseline)
    mean_diff = float(diffs.mean())
    var_diff = float(diffs.var(ddof=1))
    if var_diff == 0.0:
        return mean_diff, (0.0 if mean_diff != 0.0 else 1.0), n
    t_stat = mean_diff / math.sqrt(var_diff / n)
    p = two_sided_t_p_value(t_stat, n - 1)
    return mean_diff, p, n


def _pair_by_seed(
    candidates: list[ExperimentResultLike],
    baselines: list[ExperimentResultLike],
    metric: str,
) -> tuple[list[float], list[float]]:
    cand_map: dict[int, float] = {}
    for r in candidates:
        seed = r.metrics.get("seed")
        if seed is None:
            raise PairingError(f"候选 {r.spec_id} 缺少 seed")
        seed_i = int(seed)
        if seed_i in cand_map:
            raise PairingError(f"候选 seed {seed_i} 重复（spec {r.spec_id}）")
        cand_map[seed_i] = float(r.metrics[metric])
    base_map: dict[int, float] = {}
    for r in baselines:
        seed = r.metrics.get("seed")
        if seed is None:
            raise PairingError(f"基线 {r.spec_id} 缺少 seed")
        seed_i = int(seed)
        if seed_i in base_map:
            raise PairingError(f"基线 seed {seed_i} 重复（spec {r.spec_id}）")
        base_map[seed_i] = float(r.metrics[metric])
    shared = sorted(set(cand_map.keys()) & set(base_map.keys()))
    if not shared:
        raise PairingError("候选与基线之间没有共享的 seed")
    return [cand_map[s] for s in shared], [base_map[s] for s in shared]


def _improvement(candidate_mean: float, baseline_mean: float, direction: str) -> float:
    denom = abs(baseline_mean) if baseline_mean != 0.0 else 1.0
    raw = (candidate_mean - baseline_mean) / denom
    if direction == LOWER:
        return -raw
    return raw


def _log_transform(values: list[float], scale: str) -> list[float]:
    if scale != LOG:
        return list(values)
    out = []
    for v in values:
        if v <= 0.0:
            raise ValueError(f"对数尺度指标必须为正，得到 {v}")
        out.append(math.log(v))
    return out


@dataclass
class EvaluatorConfig:
    improvement_threshold: float = 0.02
    significance_threshold: float = 0.05


class Evaluator:
    """对（候选, 基线）结果列表的纯函数；返回一个 Verdict。"""

    def __init__(self, config: EvaluatorConfig | None = None) -> None:
        self._cfg = config or EvaluatorConfig()

    def evaluate(
        self,
        hypothesis_id: int,
        metric_spec: MetricSpec,
        candidates: list[ExperimentResultLike],
        baselines: list[ExperimentResultLike],
    ) -> Verdict:
        metric_spec.validate()
        bad = [r for r in [*candidates, *baselines] if r.terminal != "ok"]
        if bad:
            return self._failed_verdict(hypothesis_id, metric_spec, bad)
        cand_vals, base_vals = _pair_by_seed(candidates, baselines, metric_spec.name)
        cand_t = _log_transform(cand_vals, metric_spec.scale)
        base_t = _log_transform(base_vals, metric_spec.scale)
        mean_diff, p_value, n = paired_t_test(cand_t, base_t)
        cand_mean = float(np.mean(cand_vals))
        base_mean = float(np.mean(base_vals))
        cand_mean_t = float(np.mean(cand_t))
        base_mean_t = float(np.mean(base_t))
        improvement = _improvement(cand_mean_t, base_mean_t, metric_spec.direction)
        verdict, rationale = self._verdict(improvement, p_value, n)
        return Verdict(
            hypothesis_id=hypothesis_id,
            metric=metric_spec.name,
            direction=metric_spec.direction,
            scale=metric_spec.scale,
            candidate_mean=cand_mean,
            baseline_mean=base_mean,
            improvement=improvement,
            p_value=p_value,
            significance_threshold=self._cfg.significance_threshold,
            improvement_threshold=self._cfg.improvement_threshold,
            verdict=verdict,
            rationale=rationale,
        )

    def _failed_verdict(
        self, hypothesis_id: int, metric_spec: MetricSpec, bad: list[ExperimentResultLike]
    ) -> Verdict:
        terminals = sorted({r.terminal for r in bad})
        rationale = f"运行失败，terminal 为 {terminals}"
        return Verdict(
            hypothesis_id=hypothesis_id,
            metric=metric_spec.name,
            direction=metric_spec.direction,
            scale=metric_spec.scale,
            candidate_mean=0.0,
            baseline_mean=0.0,
            improvement=0.0,
            p_value=None,
            significance_threshold=self._cfg.significance_threshold,
            improvement_threshold=self._cfg.improvement_threshold,
            verdict="failed",
            rationale=rationale,
        )

    def _verdict(self, improvement: float, p_value: float | None, n: int) -> tuple[str, str]:
        if abs(improvement) < self._cfg.improvement_threshold:
            return "noise", (
                f"改进 {improvement:.4f} 低于阈值 "
                f"{self._cfg.improvement_threshold}"
            )
        if p_value is None:
            return "noise", f"仅有 {n} 个配对样本；无法进行显著性检验"
        if p_value > self._cfg.significance_threshold:
            return "noise", (
                f"p 值 {p_value:.4f} 超过显著性阈值 "
                f"{self._cfg.significance_threshold}"
            )
        if improvement > 0:
            return "improved", (
                f"改进 {improvement:.4f} 在 p={p_value:.4f} 下显著"
            )
        return "regressed", (
            f"回归 {improvement:.4f} 在 p={p_value:.4f} 下显著"
        )


def _demo() -> None:
    def result(seed: int, perplexity: float) -> ExperimentResultLike:
        return ExperimentResultLike(
            spec_id=f"demo_{seed}",
            terminal="ok",
            metrics={"seed": seed, "perplexity": perplexity, "final_loss": math.log(perplexity)},
        )

    candidates = [result(s, 28.0 - 0.5 * s + (s % 3) * 0.2) for s in range(8)]
    baselines = [result(s, 32.0 - 0.4 * s + (s % 3) * 0.2) for s in range(8)]
    metric_spec = MetricSpec(name="perplexity", direction=LOWER, scale=LOG)
    evaluator = Evaluator()
    verdict = evaluator.evaluate(1, metric_spec, candidates, baselines)
    print(json.dumps(verdict.to_dict(), indent=2))


if __name__ == "__main__":
    _demo()
