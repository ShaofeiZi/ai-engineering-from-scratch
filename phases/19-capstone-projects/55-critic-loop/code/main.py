"""针对论文草稿的多轮评审循环，包含五个固定评分维度。

概念参考：
- ./docs/en.md（本课）
- 第 19 阶段第 54 课（论文写作器；提供草稿结构）
- 第 19 阶段第 50-53 课（更早的自动研究阶段）

仅使用标准库。运行：python3 code/main.py
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Callable, Protocol


DIMENSIONS: tuple[str, ...] = (
    "clarity",
    "novelty",
    "evidence",
    "methodology",
    "related_work",
)


@dataclass
class MiniSection:
    """评审循环所用的最小章节结构。对应第 54 课的 Section。"""
    id: str
    title: str
    body: str = ""
    figure_refs: list[str] = field(default_factory=list)
    cites: list[str] = field(default_factory=list)


@dataclass
class MiniPaper:
    """评审循环所用的最小论文结构。对应第 54 课的 Paper。"""
    title: str
    abstract: str
    sections: list[MiniSection] = field(default_factory=list)
    originality_tag: str = "low"
    citation_count_target: int = 4
    figure_count_target: int = 2


@dataclass
class Suggestion:
    dimension: str
    target_section_id: str | None
    edit: str

    def to_dict(self) -> dict:
        return {
            "dimension": self.dimension,
            "target_section_id": self.target_section_id,
            "edit": self.edit,
        }


@dataclass
class Critique:
    round: int
    scores: dict[str, float]
    suggestions: list[Suggestion]
    reason: str

    def mean(self) -> float:
        if not self.scores:
            return 0.0
        return sum(self.scores.values()) / len(self.scores)

    def to_dict(self) -> dict:
        return {
            "round": self.round,
            "scores": dict(self.scores),
            "mean": self.mean(),
            "suggestions": [s.to_dict() for s in self.suggestions],
            "reason": self.reason,
        }


class Critic(Protocol):
    def __call__(self, paper: MiniPaper, round_: int) -> Critique: ...


class Reviser(Protocol):
    def __call__(self, paper: MiniPaper, suggestions: list[Suggestion]) -> MiniPaper: ...


@dataclass
class LoopTrace:
    round: int
    scores: dict[str, float]
    mean: float
    suggestions_applied: int
    verdict: str

    def to_dict(self) -> dict:
        return {
            "round": self.round,
            "scores": dict(self.scores),
            "mean": self.mean,
            "suggestions_applied": self.suggestions_applied,
            "verdict": self.verdict,
        }


@dataclass
class LoopResult:
    status: str
    reason: str
    rounds_used: int
    final_scores: dict[str, float]
    final_mean: float
    paper: MiniPaper
    trace: list[LoopTrace]

    def to_dict(self) -> dict:
        return {
            "status": self.status,
            "reason": self.reason,
            "rounds_used": self.rounds_used,
            "final_scores": dict(self.final_scores),
            "final_mean": self.final_mean,
            "trace": [t.to_dict() for t in self.trace],
        }


class CriticLoop:
    """驱动 评审 -> 修订 -> 收敛检查 的循环，直到触发停止条件。"""

    def __init__(
        self,
        critic: Critic,
        reviser: Reviser,
        max_rounds: int = 5,
        target_score: float = 8.0,
        plateau_epsilon: float = 0.1,
        plateau_window: int = 2,
    ) -> None:
        if max_rounds < 1:
            raise ValueError("max_rounds 必须 >= 1")
        if plateau_window < 1:
            raise ValueError("plateau_window 必须 >= 1")
        self.critic = critic
        self.reviser = reviser
        self.max_rounds = max_rounds
        self.target_score = target_score
        self.plateau_epsilon = plateau_epsilon
        self.plateau_window = plateau_window

    def _target_met(self, critique: Critique) -> bool:
        return all(critique.scores.get(d, 0.0) >= self.target_score for d in DIMENSIONS)

    def _plateau(self, trace: list[LoopTrace]) -> bool:
        if len(trace) < self.plateau_window + 1:
            return False
        recent = trace[-(self.plateau_window + 1):]
        for i in range(1, len(recent)):
            if recent[i].mean - recent[i - 1].mean > self.plateau_epsilon:
                return False
        return True

    def run(self, paper: MiniPaper) -> LoopResult:
        trace: list[LoopTrace] = []
        critique: Critique | None = None

        for round_ in range(1, self.max_rounds + 1):
            critique = self.critic(paper, round_)
            applied = len(critique.suggestions)

            if self._target_met(critique):
                trace.append(LoopTrace(
                    round=round_, scores=dict(critique.scores),
                    mean=critique.mean(), suggestions_applied=0,
                    verdict="target",
                ))
                return LoopResult(
                    status="converged", reason="target",
                    rounds_used=round_, final_scores=dict(critique.scores),
                    final_mean=critique.mean(), paper=paper, trace=trace,
                )

            interim = LoopTrace(
                round=round_, scores=dict(critique.scores),
                mean=critique.mean(), suggestions_applied=applied,
                verdict="continue",
            )
            trace.append(interim)

            if self._plateau(trace):
                interim.verdict = "plateau"
                return LoopResult(
                    status="converged", reason="plateau",
                    rounds_used=round_, final_scores=dict(critique.scores),
                    final_mean=critique.mean(), paper=paper, trace=trace,
                )

            paper = self.reviser(paper, critique.suggestions)

        final_scores = dict(critique.scores) if critique is not None else {d: 0.0 for d in DIMENSIONS}
        final_mean = critique.mean() if critique is not None else 0.0
        if trace:
            trace[-1].verdict = "budget"
        return LoopResult(
            status="stopped", reason="budget",
            rounds_used=self.max_rounds, final_scores=final_scores,
            final_mean=final_mean, paper=paper, trace=trace,
        )


def deterministic_score(paper: MiniPaper) -> dict[str, float]:
    """对论文在五个维度上进行确定性评分，每项 0..10 分。"""
    body_lens = [len(s.body) for s in paper.sections]
    avg_body = (sum(body_lens) / len(body_lens)) if body_lens else 0.0
    section_titles = {s.title.lower() for s in paper.sections}

    clarity = min(10.0, 3.0 + avg_body / 50.0)

    if paper.originality_tag == "high":
        novelty = 9.0
    elif paper.originality_tag == "medium":
        novelty = 6.0
    else:
        novelty = 3.0

    fig_refs = sum(1 for s in paper.sections for _ in s.figure_refs)
    cites = sum(len(s.cites) for s in paper.sections)
    evidence = min(10.0, 2.0 + 2.0 * fig_refs + 1.5 * cites)

    has_method = any(
        s.title.lower().startswith("method") and s.body for s in paper.sections
    )
    methodology = 9.0 if has_method else 4.0

    has_related = (
        "related work" in section_titles
        and any(
            s.title.lower() == "related work" and s.body
            for s in paper.sections
        )
    )
    related_work = 9.0 if has_related else 4.0

    return {
        "clarity": round(clarity, 2),
        "novelty": round(novelty, 2),
        "evidence": round(evidence, 2),
        "methodology": round(methodology, 2),
        "related_work": round(related_work, 2),
    }


def deterministic_critic(paper: MiniPaper, round_: int) -> Critique:
    """为论文评分，并为每个未达目标的维度生成一条建议。"""
    scores = deterministic_score(paper)
    suggestions: list[Suggestion] = []

    def first_section_id(default: str = "intro") -> str:
        return paper.sections[0].id if paper.sections else default

    if scores["clarity"] < 8.0:
        target = paper.sections[-1].id if paper.sections else None
        suggestions.append(Suggestion(
            dimension="clarity",
            target_section_id=target,
            edit="expand-body",
        ))
    if scores["novelty"] < 8.0:
        suggestions.append(Suggestion(
            dimension="novelty",
            target_section_id=None,
            edit="bump-originality",
        ))
    if scores["evidence"] < 8.0:
        suggestions.append(Suggestion(
            dimension="evidence",
            target_section_id=first_section_id(),
            edit="add-figure-and-cite",
        ))
    if scores["methodology"] < 8.0:
        suggestions.append(Suggestion(
            dimension="methodology",
            target_section_id=None,
            edit="add-method-section",
        ))
    if scores["related_work"] < 8.0:
        suggestions.append(Suggestion(
            dimension="related_work",
            target_section_id=None,
            edit="add-related-work-section",
        ))

    reason = "fully-met" if not suggestions else f"{len(suggestions)} below target"
    return Critique(round=round_, scores=scores, suggestions=suggestions, reason=reason)


def deterministic_reviser(paper: MiniPaper, suggestions: list[Suggestion]) -> MiniPaper:
    """确定性应用每条建议的编辑，返回原地修改后的同一 paper 对象。"""
    fig_counter = 0
    cite_counter = 0
    for s in paper.sections:
        fig_counter += len(s.figure_refs)
        cite_counter += len(s.cites)

    for sug in suggestions:
        if sug.edit == "expand-body":
            for sec in paper.sections:
                if sec.id == sug.target_section_id:
                    sec.body = (sec.body + " " + ("x" * 80)).strip()
                    break
        elif sug.edit == "bump-originality":
            if paper.originality_tag == "low":
                paper.originality_tag = "medium"
            elif paper.originality_tag == "medium":
                paper.originality_tag = "high"
        elif sug.edit == "add-figure-and-cite":
            target_id = sug.target_section_id or (paper.sections[0].id if paper.sections else None)
            for sec in paper.sections:
                if sec.id == target_id:
                    fig_counter += 1
                    cite_counter += 1
                    sec.figure_refs.append(f"f{fig_counter}")
                    sec.cites.append(f"c{cite_counter}")
                    break
        elif sug.edit == "add-method-section":
            if not any(s.title.lower().startswith("method") for s in paper.sections):
                paper.sections.append(MiniSection(
                    id="method", title="Method",
                    body="A description of the method follows. " + ("x" * 200),
                ))
            else:
                for sec in paper.sections:
                    if sec.title.lower().startswith("method") and not sec.body:
                        sec.body = "A description of the method follows. " + ("x" * 200)
                        break
        elif sug.edit == "add-related-work-section":
            if not any(s.title.lower() == "related work" for s in paper.sections):
                paper.sections.append(MiniSection(
                    id="related-work", title="Related Work",
                    body="We survey adjacent work. " + ("x" * 200),
                ))
            else:
                for sec in paper.sections:
                    if sec.title.lower() == "related work" and not sec.body:
                        sec.body = "We survey adjacent work. " + ("x" * 200)
                        break
    return paper


def make_deterministic_critic_pair() -> tuple[Critic, Reviser]:
    return deterministic_critic, deterministic_reviser


def demo() -> dict:
    paper = MiniPaper(
        title="Auto-Research Loop",
        abstract="abstract",
        sections=[
            MiniSection(id="intro", title="Introduction", body="short intro"),
        ],
        originality_tag="low",
    )
    critic, reviser = make_deterministic_critic_pair()
    loop = CriticLoop(critic=critic, reviser=reviser, max_rounds=6, target_score=8.0)
    result = loop.run(paper)
    return result.to_dict()


if __name__ == "__main__":
    print(json.dumps(demo(), indent=2))
