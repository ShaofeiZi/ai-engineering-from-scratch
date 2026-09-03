"""带分解漂移演示的分层多 Agent 系统。

三级层次结构：top manager -> sub-manager -> worker。
分别运行正常路径和扰动路径；在扰动路径中，top manager 错误标记了一个分支。
观察错误如何层层传播。
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class LeafOutput:
    worker: str
    question: str
    answer: str


@dataclass
class SubSummary:
    sub_manager: str
    leaves: list[LeafOutput]
    summary: str


@dataclass
class TopSynthesis:
    top_manager: str
    branches: list[SubSummary]
    synthesis: str


class Worker:
    def __init__(self, name: str, canned: dict[str, str]) -> None:
        self.name = name
        self.canned = canned

    def run(self, question: str) -> LeafOutput:
        key = self._match_key(question)
        ans = self.canned.get(key, f"[没有针对“{question}”的预设答案]")
        return LeafOutput(worker=self.name, question=question, answer=ans)

    def _match_key(self, q: str) -> str:
        ql = q.lower()
        for k in self.canned:
            if k in ql:
                return k
        return "default"


class SubManager:
    def __init__(self, name: str, workers: list[Worker], split: dict[str, str]) -> None:
        self.name = name
        self.workers = workers
        self.split = split

    def run(self, task: str) -> SubSummary:
        leaves = []
        for w in self.workers:
            sub_q = self.split.get(w.name, task)
            leaves.append(w.run(sub_q))
        summary = f"[{self.name}] aggregated: " + " | ".join(l.answer for l in leaves)
        return SubSummary(sub_manager=self.name, leaves=leaves, summary=summary)


class TopManager:
    def __init__(self, name: str, subs: dict[str, SubManager]) -> None:
        self.name = name
        self.subs = subs

    def run(self, task: str, branch_labels: list[str]) -> TopSynthesis:
        summaries: list[SubSummary] = []
        for label in branch_labels:
            if label not in self.subs:
                summaries.append(
                    SubSummary(
                        sub_manager=f"MISSING[{label}]",
                        leaves=[],
                        summary=f"[top] 尝试委派给“{label}”，但该 sub-manager 不存在",
                    )
                )
                continue
            summaries.append(self.subs[label].run(f"{task} -- branch: {label}"))
        synth = "top 综合结果：" + " || ".join(s.summary for s in summaries)
        return TopSynthesis(top_manager=self.name, branches=summaries, synthesis=synth)


def build_hierarchy() -> TopManager:
    fe = Worker("fe", {"frontend": "React 组件审计完成；发现 2 个问题。"})
    be = Worker("be", {"backend": "API 端点审计完成；发现 1 个问题。"})
    eng = SubManager(
        "eng-manager",
        [fe, be],
        {"fe": "对该功能进行前端审查", "be": "对该功能进行后端审查"},
    )
    lw = Worker("lawyer", {"legal": "合同条款 A 和 B 不合规。"})
    legal = SubManager("legal-manager", [lw], {"lawyer": "对该功能进行法律审查"})
    fw = Worker(
        "finance",
        {"finance": "预计成本为每月 $42k；超出预算 12%。"},
    )
    finance = SubManager("finance-manager", [fw], {"finance": "对该功能进行财务审查"})
    return TopManager("vp-eng", {"engineering": eng, "legal": legal, "finance": finance})


def render(label: str, synth: TopSynthesis) -> None:
    print(f"\n=== {label} ===")
    for branch in synth.branches:
        print(f"  [子管理者] {branch.sub_manager}")
        for leaf in branch.leaves:
            print(f"    [leaf] {leaf.worker:8s} 收到问题：{leaf.question}")
            print(f"           回答：{leaf.answer}")
        print(f"    [总结] {branch.summary}")
    print(f"  [顶层] {synth.synthesis}")


def main() -> None:
    print("带分解漂移的分层多 Agent 演示")
    print("-" * 60)

    top = build_hierarchy()
    task = "将高级套餐功能发布到生产环境。"

    happy = top.run(task, branch_labels=["engineering", "legal"])
    render("正常路径（分支正确）", happy)

    perturbed = top.run(task, branch_labels=["engineering", "finance"])
    render("扰动路径（top manager 将“legal”错标为“finance”）", perturbed)

    print("\n用户要求进行法律和工程审查。")
    print("正常路径：legal 和 engineering 都如实回答。")
    print("扰动路径：finance 尽职回答，但法律问题无人作答。")
    print("错误出现在 top 综合结果中，距离人类能发现它的位置隔了一层。")


if __name__ == "__main__":
    main()
