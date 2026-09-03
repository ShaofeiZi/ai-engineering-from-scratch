#本课程序旨在先明确可观察的结果，再选择输出方案。
#阅读：phases/14-agent-engineering/47-outcomes-before-output/docs/en.md
#参考文献：Nuseibeh 与 Easterbrook，《需求工程：路线图》，2000 年。
#参考文献：Dardenne、van Lamsweerde 与 Fickas，《Goal-Directed 需求获取》，1993 年。
#运行此文件以生成 outputs/outcome-frame.json.
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class OutcomeFrame:
    user: str
    situation: str
    current_behavior: str
    desired_outcome: str
    constraints: list[str]
    non_goals: list[str]
    proposed_output: str = ""


def validate(frame: OutcomeFrame) -> list[str]:
    issues: list[str] = []
    for name in ("user", "situation", "current_behavior", "desired_outcome"):
        if not getattr(frame, name).strip():
            issues.append(f"{name} is empty")
    if not frame.constraints:
        issues.append("constraints are empty")
    if not frame.non_goals:
        issues.append("non-goals are empty")
    if frame.proposed_output and frame.proposed_output.lower() in frame.desired_outcome.lower():
        issues.append("desired outcome names the proposed output")
    return issues


def decision(frame: OutcomeFrame) -> dict:
    issues = validate(frame)
    return {
        "status": "ready-to-discover" if not issues else "needs-framing",
        "issues": issues,
        "frame": asdict(frame),
        "next_question": f"What evidence would show that the desired outcome was achieved for the {frame.user}?",
    }


def example() -> OutcomeFrame:
    return OutcomeFrame(
        user="on-call engineer",
        situation="a production alert arrives during an incident",
        current_behavior="searches three dashboards before finding the affected service",
        desired_outcome="identifies the failing service and safe next action within two minutes",
        constraints=["read-only during diagnosis", "must preserve the audit trail"],
        non_goals=["automatic remediation", "replacing the incident commander"],
        proposed_output="incident assistant",
    )


def main() -> None:
    output = Path(__file__).resolve().parents[1] / "outputs" / "outcome-frame.json"
    output.write_text(json.dumps(decision(example()), indent=2) + "\n", encoding="utf-8")
    print(output.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
