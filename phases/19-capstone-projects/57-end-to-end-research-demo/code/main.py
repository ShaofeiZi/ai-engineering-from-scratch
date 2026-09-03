"""端到端自动科研演示：种子 -> 调度器 -> 评审循环 -> 论文撰写器。

概念参考：
- ./docs/en.md（本课）
- 第 19 阶段第 54 课（论文撰写器）
- 第 19 阶段第 55 课（评审循环）
- 第 19 阶段第 56 课（迭代调度器）
- 第 19 阶段第 50–53 课（更早的自动科研阶段；这里的种子/运行器桩代码代为承担其职责）

仅依赖标准库与 numpy。运行：python3 code/main.py
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
from dataclasses import dataclass, field
from typing import Awaitable, Callable


HERE = os.path.dirname(os.path.abspath(__file__))
LESSON_ROOT = os.path.dirname(os.path.dirname(HERE))


def _add(path: str) -> None:
    if path not in sys.path:
        sys.path.insert(0, path)


_add(os.path.join(LESSON_ROOT, "54-paper-writer", "code"))
_add(os.path.join(LESSON_ROOT, "55-critic-loop", "code"))
_add(os.path.join(LESSON_ROOT, "56-iteration-scheduler", "code"))

import importlib
import importlib.util


def _load_module(name: str, file_path: str):
    spec = importlib.util.spec_from_file_location(name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法从 {file_path} 加载 {name}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


paper_writer_mod = _load_module(
    "_t_d2_paper_writer",
    os.path.join(LESSON_ROOT, "54-paper-writer", "code", "main.py"),
)
critic_loop_mod = _load_module(
    "_t_d2_critic_loop",
    os.path.join(LESSON_ROOT, "55-critic-loop", "code", "main.py"),
)
scheduler_mod = _load_module(
    "_t_d2_scheduler",
    os.path.join(LESSON_ROOT, "56-iteration-scheduler", "code", "main.py"),
)


Paper = paper_writer_mod.Paper
Section = paper_writer_mod.Section
Figure = paper_writer_mod.Figure
BibEntry = paper_writer_mod.BibEntry
PaperWriter = paper_writer_mod.PaperWriter
PaperValidationError = paper_writer_mod.PaperValidationError
MockProseGenerator = paper_writer_mod.MockProseGenerator

MiniPaper = critic_loop_mod.MiniPaper
MiniSection = critic_loop_mod.MiniSection
CriticLoop = critic_loop_mod.CriticLoop
deterministic_critic = critic_loop_mod.deterministic_critic
deterministic_reviser = critic_loop_mod.deterministic_reviser

Hypothesis = scheduler_mod.Hypothesis
Result = scheduler_mod.Result
IterationScheduler = scheduler_mod.IterationScheduler
SchedulerReport = scheduler_mod.SchedulerReport
make_deterministic_runner = scheduler_mod.make_deterministic_runner


class NoTriggerError(Exception):
    """没有分支越过论文阈值；演示无法选出最佳结果。"""


class BestResultError(Exception):
    """选择器收到了空的触发列表。"""


@dataclass
class DemoReport:
    scheduler_report: dict
    best_branch: str
    best_reward: float
    critic_result: dict
    paper_manifest: dict
    stop_reason: str

    def to_dict(self) -> dict:
        return {
            "scheduler_report": self.scheduler_report,
            "best_branch": self.best_branch,
            "best_reward": self.best_reward,
            "critic_result": self.critic_result,
            "paper_manifest": self.paper_manifest,
            "stop_reason": self.stop_reason,
        }


def make_seed_hypotheses() -> list[Hypothesis]:
    """三条种子假设，每条科研分支一条。作为第 50–53 课的桩代码替身。"""
    return [
        Hypothesis(id="h-alpha-1", branch="alpha", payload={"q": "method-x"}),
        Hypothesis(id="h-beta-1", branch="beta", payload={"q": "method-y"}),
        Hypothesis(id="h-gamma-1", branch="gamma", payload={"q": "method-z"}),
    ]


def pick_best_branch(scheduler_report: SchedulerReport) -> tuple[str, float]:
    """在被触发的分支中挑选平均奖励最高的一条。

    平局时按分支 id 字母序打破（确定性）。
    """
    if not scheduler_report.paper_triggers:
        raise NoTriggerError("没有分支越过论文阈值")
    by_branch = {b.branch: b for b in scheduler_report.branches}
    triggered = [by_branch[name] for name in scheduler_report.paper_triggers if name in by_branch]
    if not triggered:
        raise BestResultError("查找后触发列表为空")
    triggered.sort(key=lambda b: (-b.mean, b.branch))
    best = triggered[0]
    return best.branch, best.mean


def _originality_for_reward(reward: float) -> str:
    if reward >= 0.8:
        return "high"
    if reward >= 0.6:
        return "medium"
    return "low"


def build_mini_paper(branch: str, reward: float) -> MiniPaper:
    return MiniPaper(
        title=f"Auto-Research Findings on Branch {branch}",
        abstract=f"We summarise the best yielding branch {branch} from the auto-research loop.",
        sections=[
            MiniSection(id="intro", title="Introduction", body="initial observations"),
            MiniSection(id="results", title="Results", body=""),
        ],
        originality_tag=_originality_for_reward(reward),
    )


def mini_to_full_paper(mini: MiniPaper, branch: str) -> Paper:
    """将收敛后的 MiniPaper 提升为撰写器使用的完整 Paper。

    为每个使用到的引用键添加一个图和一个参考文献条目。每个章节的
    引用列表都被保留；参考文献由其并集构建。
    """
    cites: list[str] = []
    for sec in mini.sections:
        for c in sec.cites:
            if c not in cites:
                cites.append(c)

    bib = [
        BibEntry(
            key=key, entry_type="article",
            fields={"title": f"Source {key}", "author": "Synthesised", "year": "2026"},
        )
        for key in cites
    ]
    if not bib:
        bib = [BibEntry(
            key=f"{branch}-baseline", entry_type="article",
            fields={"title": "Baseline", "author": "Synthesised", "year": "2026"},
        )]
        for sec in mini.sections:
            if sec.id == "intro":
                sec.cites.append(f"{branch}-baseline")
                break

    fig = Figure(
        id=f"{branch}-results",
        path=f"figs/{branch}.pdf",
        caption=f"Reward trajectory on branch {branch}",
    )

    sections = []
    for s in mini.sections:
        figure_refs = [fig.id] if s.id == "results" else []
        sections.append(Section(
            id=s.id, title=s.title, body=s.body,
            cites=list(s.cites), figure_refs=figure_refs,
        ))
    return Paper(
        title=mini.title,
        authors=["Auto-Research Demo"],
        abstract=mini.abstract,
        sections=sections,
        figures=[fig],
        bibliography=bib,
    )


async def _run_demo_async(out_dir: str, seed: int = 11) -> DemoReport:
    seed_list = make_seed_hypotheses()
    if not seed_list:
        raise BestResultError("种子列表为空")

    runner = make_deterministic_runner(
        base_rewards={"alpha": 0.82, "beta": 0.55, "gamma": 0.15},
        noise=0.04,
        delay_ms=2.0,
        seed=seed,
    )
    sched = IterationScheduler(
        runner=runner, slots=3, max_experiments=6,
        paper_threshold=0.7, prune_floor=0.2, prune_after_runs=3,
        expander=scheduler_mod.deterministic_expander,
    )
    sched_report = await sched.run(seed_list)

    branch, reward = pick_best_branch(sched_report)

    mini = build_mini_paper(branch, reward)
    loop = CriticLoop(
        critic=deterministic_critic,
        reviser=deterministic_reviser,
        max_rounds=6,
        target_score=8.0,
    )
    critic_result = loop.run(mini)

    full_paper = mini_to_full_paper(critic_result.paper, branch)
    prose = MockProseGenerator(outlines={
        "intro": f"motivation for branch {branch}",
        "results": f"summary of reward trajectory on branch {branch}",
        "method": "method description",
        "related-work": "related work survey",
    })
    writer = PaperWriter(prose=prose)
    manifest = writer.write(full_paper, out_dir)

    return DemoReport(
        scheduler_report=sched_report.to_dict(),
        best_branch=branch,
        best_reward=reward,
        critic_result=critic_result.to_dict(),
        paper_manifest=manifest,
        stop_reason=sched_report.stop_reason,
    )


def run_demo(out_dir: str | None = None, seed: int = 11) -> DemoReport:
    if out_dir is None:
        out_dir = tempfile.mkdtemp(prefix="auto-research-demo-")
    return asyncio.run(_run_demo_async(out_dir, seed=seed))


def demo() -> dict:
    return run_demo().to_dict()


if __name__ == "__main__":
    rep = demo()
    print(json.dumps({
        "stop_reason": rep["stop_reason"],
        "best_branch": rep["best_branch"],
        "best_reward": rep["best_reward"],
        "critic_status": rep["critic_result"]["status"],
        "critic_rounds": rep["critic_result"]["rounds_used"],
        "paper_sections": [s["id"] for s in rep["paper_manifest"]["sections"]],
        "paper_figures": [f["id"] for f in rep["paper_manifest"]["figures"]],
        "experiments_run": rep["scheduler_report"]["experiments_run"],
    }, indent=2))
