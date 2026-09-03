"""用结构化捕获、密钥脱敏、轮转和命令谱系来封装 subprocess.run。

每条 shell 命令都经过 run_with_feedback.。记录包含 argv、已脱敏的
stdout/stderr 尾部、退出码、耗时、started_at、代理备注以及一个
command_id/parent_command_id 对，使重试可追溯到其来源。JSONL 文件在 1 MB 时轮转，以保持加载器内存有界。

运行：python3 code/main.py
"""

from __future__ import annotations

import json
import re
import shlex
import subprocess
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path

HERE = Path(__file__).parent
RECORD = HERE / "feedback_record.jsonl"

HEAD_LINES = 5
TAIL_LINES = 30
ROTATE_BYTES = 1 * 1024 * 1024  # 1 MB
MAX_ROTATIONS = 5

# 密钥模式。每季度根据生产运行时观测到的泄漏形状进行审计。
REDACTION_PATTERNS = [
    (re.compile(r"(?i)bearer\s+[A-Za-z0-9._\-]+"), "Bearer [REDACTED]"),
    (re.compile(r"(?i)\b(password|passwd|secret|api[_-]?key|access[_-]?key|token)\s*[:=]\s*\S+"),
     r"\1=[REDACTED]"),
    (re.compile(r"\bAKIA[0-9A-Z]{16}\b"), "AKIA[REDACTED]"),
    (re.compile(r"\bxox[baprs]-[A-Za-z0-9\-]+"), "xox-[REDACTED]"),
    (re.compile(r"-----BEGIN [A-Z ]+ PRIVATE KEY-----[\s\S]*?-----END [A-Z ]+ PRIVATE KEY-----"),
     "[REDACTED PRIVATE KEY]"),
]


@dataclass
class FeedbackRecord:
    command_id: str
    parent_command_id: str | None
    command: list[str]
    stdout_tail: str
    stderr_tail: str
    exit_code: int | None
    duration_ms: int
    started_at: float
    agent_note: str
    error: str | None = None
    truncations: dict[str, int] = field(default_factory=dict)
    redactions: dict[str, int] = field(default_factory=dict)


def redact(text: str) -> tuple[str, int]:
    """在 JSONL 追加之前剥离密钥。读取时才脱敏容易引发安全问题。"""
    if not text:
        return text, 0
    hits = 0
    out = text
    for pattern, replacement in REDACTION_PATTERNS:
        out, n = pattern.subn(replacement, out)
        hits += n
    return out, hits


def deterministic_tail(text: str, head: int = HEAD_LINES, tail: int = TAIL_LINES) -> tuple[str, int]:
    lines = text.splitlines()
    if len(lines) <= head + tail:
        return text, 0
    cut = len(lines) - head - tail
    return "\n".join(lines[:head] + [f"...truncated {cut} lines..."] + lines[-tail:]), cut


def _process_capture(text: str) -> tuple[str, int, int]:
    """先截断，再脱敏。返回（文本、截断行数、脱敏命中数）。"""
    tailed, cut = deterministic_tail(text)
    redacted, hits = redact(tailed)
    return redacted, cut, hits


def maybe_rotate() -> None:
    """将活跃文件上限设为 ROTATE_BYTES；按 .1 到 .MAX 轮转，丢弃最旧的。"""
    if not RECORD.exists() or RECORD.stat().st_size < ROTATE_BYTES:
        return
    for idx in range(MAX_ROTATIONS, 0, -1):
        src = RECORD.with_suffix(RECORD.suffix + (f".{idx - 1}" if idx > 1 else ""))
        if src == RECORD:
            src = RECORD
        dst = RECORD.with_suffix(RECORD.suffix + f".{idx}")
        if src.exists():
            if idx == MAX_ROTATIONS and dst.exists():
                dst.unlink()
            try:
                src.rename(dst)
            except FileNotFoundError:
                pass


def run_with_feedback(
    command: list[str],
    agent_note: str = "",
    timeout_s: float = 30.0,
    parent_command_id: str | None = None,
) -> FeedbackRecord:
    started = time.time()
    command_id = uuid.uuid4().hex[:12]
    base_kwargs = dict(
        command_id=command_id,
        parent_command_id=parent_command_id,
        command=command,
        started_at=started,
        agent_note=agent_note,
    )
    try:
        completed = subprocess.run(command, capture_output=True, text=True, timeout=timeout_s)
        out, cut_out, red_out = _process_capture(completed.stdout)
        err, cut_err, red_err = _process_capture(completed.stderr)
        record = FeedbackRecord(
            stdout_tail=out, stderr_tail=err,
            exit_code=completed.returncode,
            duration_ms=int((time.time() - started) * 1000),
            truncations={"stdout": cut_out, "stderr": cut_err},
            redactions={"stdout": red_out, "stderr": red_err},
            **base_kwargs,
        )
    except subprocess.TimeoutExpired as exc:
        partial_out = exc.stdout.decode(errors="replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        partial_err = exc.stderr.decode(errors="replace") if isinstance(exc.stderr, bytes) else (exc.stderr or "")
        out, cut_out, red_out = _process_capture(partial_out)
        err, cut_err, red_err = _process_capture(partial_err)
        record = FeedbackRecord(
            stdout_tail=out, stderr_tail=err,
            exit_code=None,
            duration_ms=int((time.time() - started) * 1000),
            error=f"timeout after {timeout_s}s",
            truncations={"stdout": cut_out, "stderr": cut_err},
            redactions={"stdout": red_out, "stderr": red_err},
            **base_kwargs,
        )
    except FileNotFoundError as exc:
        record = FeedbackRecord(
            stdout_tail="", stderr_tail="",
            exit_code=None,
            duration_ms=int((time.time() - started) * 1000),
            error=str(exc),
            **base_kwargs,
        )

    maybe_rotate()
    with RECORD.open("a") as fh:
        fh.write(json.dumps(asdict(record)) + "\n")
    return record


def loop_can_advance(record: FeedbackRecord) -> bool:
    """当退出码缺失时拒绝推进循环。"""
    return record.exit_code is not None


def load_all() -> list[FeedbackRecord]:
    """读取活跃文件和轮转文件，使父命令谱系在轮转后仍然保留。"""
    def _rotation_key(p: Path) -> int:
        suffix = p.name[len(RECORD.name):]
        if not suffix:
            return 0  # 活跃文件
        try:
            return int(suffix.lstrip("."))
        except ValueError:
            return 99
    paths = sorted(HERE.glob(RECORD.name + "*"), key=_rotation_key, reverse=True)
    by_id: dict[str, FeedbackRecord] = {}
    for path in paths:
        try:
            text = path.read_text()
        except FileNotFoundError:
            continue
        for line in text.splitlines():
            if not line.strip():
                continue
            try:
                record = FeedbackRecord(**json.loads(line))
            except (json.JSONDecodeError, TypeError):
                continue
            by_id[record.command_id] = record  # 活跃文件优先（最后加载的为准）
    return list(by_id.values())


def retry_chain(command_id: str) -> list[FeedbackRecord]:
    """沿 parent_command_id 指针遍历以重建重试链。"""
    records = {r.command_id: r for r in load_all()}
    chain: list[FeedbackRecord] = []
    cursor: str | None = command_id
    while cursor and cursor in records:
        chain.append(records[cursor])
        cursor = records[cursor].parent_command_id
    return list(reversed(chain))


def main() -> None:
    for path in HERE.glob("feedback_record.jsonl*"):
        path.unlink()

    ok = run_with_feedback(["python3", "-c", "print('hello')"], agent_note="expect hello")
    leak = run_with_feedback(
        ["python3", "-c",
         "print('Authorization: Bearer ya29.AbCdEf'); print('password=hunter2'); print('AKIAIOSFODNN7EXAMPLE')"],
        agent_note="expect redaction"
    )
    fail = run_with_feedback(["python3", "-c", "import sys; sys.exit(2)"], agent_note="first attempt; will retry")
    retry = run_with_feedback(
        ["python3", "-c", "print('recovered'); import sys; sys.exit(0)"],
        agent_note="retry after non-zero",
        parent_command_id=fail.command_id,
    )
    missing = run_with_feedback([shlex.split("does-not-exist")[0]], agent_note="probe missing binary")

    for label, rec in (("ok", ok), ("leak", leak), ("fail", fail), ("retry", retry), ("missing", missing)):
        print(f"{label}: cid={rec.command_id} parent={rec.parent_command_id or '-'} exit={rec.exit_code} "
              f"duration_ms={rec.duration_ms} redactions={rec.redactions or '-'}")
        if rec.error:
            print(f"  错误：{rec.error}")
        if rec.stdout_tail and "REDACTED" in rec.stdout_tail:
            print(f"  脱敏后的 stdout：{rec.stdout_tail!r}")

    chain = retry_chain(retry.command_id)
    print(f"\n{retry.command_id} 的重试链：{[r.command_id for r in chain]}（从最旧到最新）")
    print(f"{len(load_all())} 条记录已持久化到 {RECORD.name}")


if __name__ == "__main__":
    main()
