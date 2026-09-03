"""带有覆盖率下限、--strict 模式和签名覆盖的确定性验证关卡。

将任务的 scope_report、rule_report、反馈日志以及可选的
coverage_report 合并为一个 verification_report.json.。没有 LLM 评判；LLM
评判位于评审方（Phase 14 · 39）。覆盖要求在 overrides.jsonl 中有一条签名记录，
包含原因、用户和 HEAD 提交。

运行：python3 code/main.py
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import math
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

HERE = Path(__file__).parent
OVERRIDES_PATH = HERE / "overrides.jsonl"
COVERAGE_FLOOR_DEFAULT = 0.80
COVERAGE_REGRESSION_DELTA = 0.01

# 用于签署覆盖条目的审计密钥。在生产环境中应从密钥管理
# 系统中读取。采用失败即关闭策略：仅在显式设置 VERIFY_DEMO_MODE=1
# 时才回退到演示密钥，并明确告警，以确保其不会意外落入 CI 中。
_OVERRIDE_SECRET_ENV = "VERIFY_OVERRIDE_SECRET"
_DEMO_MODE_ENV = "VERIFY_DEMO_MODE"


def _load_override_secret() -> str:
    secret = os.environ.get(_OVERRIDE_SECRET_ENV)
    if secret:
        return secret
    if os.environ.get(_DEMO_MODE_ENV) == "1":
        print(
            f"警告：未设置 {_OVERRIDE_SECRET_ENV}，且 {_DEMO_MODE_ENV}=1；"
            "正在使用不安全的 demo secret。请勿在此模式下记录真实覆盖项。",
            file=sys.stderr,
        )
        return "demo-override-secret-do-not-ship"
    raise RuntimeError(
        f"拒绝启动：未设置 {_OVERRIDE_SECRET_ENV}。"
        f"请设置该环境变量，或传入 {_DEMO_MODE_ENV}=1 以仅运行课程演示。"
    )


@dataclass
class Finding:
    code: str
    severity: str
    detail: str


@dataclass
class Artifacts:
    task_id: str
    acceptance_commands: list[str]
    feedback: list[dict[str, object]]
    scope_report: dict[str, object]
    rule_report: list[dict[str, object]]
    coverage_report: dict[str, float] | None = None  # {"current": 0.84, "previous": 0.85}
    head_commit: str = ""


@dataclass
class VerdictReport:
    task_id: str
    passed: bool
    strict: bool
    findings: list[Finding] = field(default_factory=list)
    coverage: dict[str, float] | None = None
    head_commit: str = ""


def _acceptance_findings(art: Artifacts) -> list[Finding]:
    findings: list[Finding] = []
    commands_run = [str(rec.get("command")) for rec in art.feedback]
    accept_set = set(art.acceptance_commands)
    for cmd in art.acceptance_commands:
        if cmd not in commands_run:
            findings.append(Finding("acceptance.missing", "block", f"never ran: {cmd}"))
    for rec in art.feedback:
        cmd_str = str(rec.get("command"))
        if rec.get("exit_code") is None:
            findings.append(Finding("feedback.null_exit", "block", f"missing exit for {cmd_str}"))
        elif rec.get("exit_code") != 0 and cmd_str in accept_set:
            findings.append(
                Finding("acceptance.failed", "block", f"acceptance exit {rec.get('exit_code')} on {cmd_str}")
            )
    return findings


def _scope_findings(art: Artifacts) -> list[Finding]:
    findings: list[Finding] = []
    if art.scope_report.get("forbidden_writes"):
        findings.append(Finding("scope.forbidden", "block",
                                f"forbidden writes: {art.scope_report['forbidden_writes']}"))
    if art.scope_report.get("off_scope_writes"):
        findings.append(Finding("scope.off_scope", "warn",
                                f"off-scope writes: {art.scope_report['off_scope_writes']}"))
    return findings


def _rule_findings(art: Artifacts) -> list[Finding]:
    return [Finding("rule.failed", "block", f"rule failed: {row.get('slug')}")
            for row in art.rule_report if not row.get("passed")]


def _coverage_findings(art: Artifacts, floor: float) -> list[Finding]:
    """Anthropic 混合准则：将可验证奖励（测试 + 覆盖率）与评分细则评判相结合。

    未达下限即为阻塞。相对于上次合并的覆盖率回退超过
    COVERAGE_REGRESSION_DELTA 即为阻塞；较小的下降仅为警告。
    """
    findings: list[Finding] = []
    if not art.coverage_report:
        findings.append(Finding("coverage.missing", "warn",
                                "no coverage_report.json; cannot enforce floor"))
        return findings
    current = float(art.coverage_report.get("current", 0.0))
    previous = float(art.coverage_report.get("previous", current))
    if current < floor:
        findings.append(Finding("coverage.below_floor", "block",
                                f"coverage {current:.2%} below floor {floor:.0%}"))
    delta = previous - current
    if delta > COVERAGE_REGRESSION_DELTA and not math.isclose(
        delta, COVERAGE_REGRESSION_DELTA, rel_tol=1e-9
    ):
        findings.append(Finding("coverage.regression", "block",
                                f"coverage dropped {delta:.2%} (prev {previous:.2%} -> {current:.2%})"))
    elif delta > 0 and not math.isclose(delta, 0.0, abs_tol=1e-12):
        findings.append(Finding("coverage.minor_regression", "warn",
                                f"coverage dropped {delta:.2%}"))
    return findings


def verify(
    art: Artifacts,
    strict: bool = False,
    coverage_floor: float = COVERAGE_FLOOR_DEFAULT,
) -> VerdictReport:
    findings = (
        _acceptance_findings(art)
        + _scope_findings(art)
        + _rule_findings(art)
        + _coverage_findings(art, coverage_floor)
    )
    if strict:
        # --strict 会将所有警告提升为阻塞。Opt-in 仅限发布分支。
        findings = [Finding(f.code, "block" if f.severity == "warn" else f.severity, f.detail)
                    for f in findings]
    blocking = [f for f in findings if f.severity == "block"]
    return VerdictReport(
        task_id=art.task_id,
        passed=not blocking,
        strict=strict,
        findings=findings,
        coverage=art.coverage_report,
        head_commit=art.head_commit,
    )


def _sign(payload: dict[str, object]) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hmac.new(_load_override_secret().encode(), canonical, hashlib.sha256).hexdigest()[:32]


def record_override(
    task_id: str, finding_code: str, reason: str, user_id: str, head_commit: str
) -> dict[str, object]:
    """追加一条签名覆盖条目。若五个字段未全部填写则拒绝执行。"""
    if not all([task_id, finding_code, reason, user_id, head_commit]):
        raise ValueError("override requires task_id, finding_code, reason, user_id, head_commit")
    payload = {
        "task_id": task_id,
        "finding_code": finding_code,
        "reason": reason,
        "user_id": user_id,
        "head_commit": head_commit,
        "ts": time.time(),
    }
    payload["signature"] = _sign({k: v for k, v in payload.items() if k != "signature"})
    with OVERRIDES_PATH.open("a") as fh:
        fh.write(json.dumps(payload) + "\n")
    return payload


def verify_signature(entry: dict[str, object]) -> bool:
    expected = entry.get("signature")
    payload = {k: v for k, v in entry.items() if k != "signature"}
    return hmac.compare_digest(_sign(payload), str(expected))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--strict", action="store_true", help="promote every warn to block")
    ap.add_argument("--floor", type=float, default=COVERAGE_FLOOR_DEFAULT)
    args = ap.parse_args()

    accept = ["pytest -x test_app.py::test_signup_rejects_short_password"]
    cases = [
        Artifacts(
            task_id="T-001",
            acceptance_commands=accept,
            feedback=[{"command": accept[0], "exit_code": 0}],
            scope_report={"forbidden_writes": [], "off_scope_writes": []},
            rule_report=[{"slug": "done/tests-pass", "passed": True}],
            coverage_report={"current": 0.84, "previous": 0.85},
            head_commit="a1b2c3d",
        ),
        Artifacts(
            task_id="T-002",
            acceptance_commands=accept,
            feedback=[{"command": accept[0], "exit_code": 0}],
            scope_report={"forbidden_writes": ["scripts/release.sh"], "off_scope_writes": ["README.md"]},
            rule_report=[{"slug": "forbidden/no-release-script-edits", "passed": False}],
            coverage_report={"current": 0.62, "previous": 0.80},
            head_commit="b2c3d4e",
        ),
        Artifacts(
            task_id="T-003",
            acceptance_commands=accept,
            feedback=[],
            scope_report={"forbidden_writes": [], "off_scope_writes": []},
            rule_report=[{"slug": "done/tests-pass", "passed": False}],
            head_commit="c3d4e5f",
        ),
    ]

    for art in cases:
        report = verify(art, strict=args.strict, coverage_floor=args.floor)
        path = HERE / f"verification_report_{art.task_id}.json"
        path.write_text(json.dumps(
            {"task_id": report.task_id, "passed": report.passed, "strict": report.strict,
             "head_commit": report.head_commit, "coverage": report.coverage,
             "findings": [asdict(f) for f in report.findings]},
            indent=2) + "\n")
        flag = " (strict)" if report.strict else ""
        print(f"任务 {report.task_id}{flag}: 通过={report.passed} 发现={len(report.findings)}")
        for f in report.findings:
            print(f"  [{f.severity}] {f.code}: {f.detail}")
        print()

    # 演示对 off-scope 警告的签名覆盖，该警告由 T-002 实际触发。
    try:
        entry = record_override(
            task_id="T-002",
            finding_code="scope.off_scope",
            reason="reviewer approved README update for the new signup contract",
            user_id="rohitg00",
            head_commit="b2c3d4e",
        )
        print(f"覆盖已记录: 签名={entry['signature']} 已验证={verify_signature(entry)}")
    except RuntimeError as exc:
        print(f"覆盖演示已跳过: {exc}")


if __name__ == "__main__":
    main()
