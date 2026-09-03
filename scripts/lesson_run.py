#!/usr/bin/env python3
"""对每节课程的 Python 代码执行冒烟检查。

默认使用 `py_compile` 对 `phases/**/[0-9][0-9]-*/code/` 下的每个 `.py` 文件
执行字节编译，但不会执行代码，因为执行可能需要 API key 和课程未固定的重量级
ML 依赖。仅检查语法足以发现贡献者最常引入的回归（错误缩进、损坏的 f-string、
意外编辑）。

使用 `--execute` 可显式启用真实执行，每个文件超时为 10 秒。若课程入口文件
以 `# requires: pkg1, pkg2` 注释列出标准库之外的 import，则以
"needs <deps>" 原因跳过，避免重量级课程（torch、anthropic 等）拖垮运行。

Usage:
    python3 scripts/lesson_run.py                      # syntax check, full curriculum
    python3 scripts/lesson_run.py --phase 14           # one phase only
    python3 scripts/lesson_run.py --strict             # exit 1 on any failure
    python3 scripts/lesson_run.py --json               # JSON report on stdout
    python3 scripts/lesson_run.py --execute            # actually run each lesson

退出码：
    0 — 检查通过，或非严格模式下仅报告失败
    1 — 使用 `--strict` 且至少一节课程失败

仅使用标准库。需要 Python 3.10+ 语法（PEP 604 union）。
"""

from __future__ import annotations

import argparse
import json
import py_compile
import re
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parent.parent
PHASES_DIR = ROOT / "phases"

PHASE_DIR_RE = re.compile(r"^([0-9]{2})-[a-z0-9][a-z0-9-]*$")
LESSON_DIR_RE = re.compile(r"^([0-9]{2})-[a-z0-9][a-z0-9-]*$")
REQUIRES_RE = re.compile(r"^\s*#\s*requires:\s*(.+?)\s*$")

EXECUTE_TIMEOUT_SEC = 10


@dataclass
class LessonResult:
    lesson: str
    files: list[str] = field(default_factory=list)
    status: str = "passed"  # passed | failed | skipped（通过、失败、跳过）
    reason: str = ""

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def iter_lesson_dirs(phase_filter: int | None) -> Iterable[Path]:
    if not PHASES_DIR.is_dir():
        return
    for phase in sorted(PHASES_DIR.iterdir()):
        if not phase.is_dir():
            continue
        m = PHASE_DIR_RE.match(phase.name)
        if not m:
            continue
        if phase_filter is not None and int(m.group(1)) != phase_filter:
            continue
        for lesson in sorted(phase.iterdir()):
            if lesson.is_dir() and LESSON_DIR_RE.match(lesson.name):
                yield lesson


def list_python_files(code_dir: Path) -> list[Path]:
    if not code_dir.is_dir():
        return []
    return sorted(p for p in code_dir.rglob("*.py") if p.is_file())


def pick_entry_file(py_files: list[Path]) -> Path | None:
    for path in py_files:
        if path.name.startswith("main."):
            return path
    return py_files[0] if py_files else None


def read_requires(path: Path) -> list[str]:
    try:
        text = path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        return []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if not stripped.startswith("#"):
            break
        match = REQUIRES_RE.match(line)
        if match:
            deps = [d.strip() for d in match.group(1).split(",")]
            return [d for d in deps if d]
    return []


def syntax_check(py_files: list[Path]) -> tuple[bool, str]:
    for path in py_files:
        try:
            py_compile.compile(str(path), doraise=True)
        except py_compile.PyCompileError as exc:
            return False, f"{path.relative_to(ROOT).as_posix()}: {exc.msg.strip()}"
    return True, ""


def execute_lesson(entry: Path) -> tuple[bool, str]:
    try:
        proc = subprocess.run(
            [sys.executable, str(entry)],
            cwd=str(entry.parent),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            timeout=EXECUTE_TIMEOUT_SEC,
            check=False,
            text=True,
        )
    except subprocess.TimeoutExpired:
        return False, f"timeout after {EXECUTE_TIMEOUT_SEC}s"
    except OSError as exc:
        return False, f"failed to launch interpreter: {exc}"
    if proc.returncode == 0:
        return True, ""
    stderr = (proc.stderr or "").strip()
    last_line = stderr.splitlines()[-1] if stderr else f"exit {proc.returncode}"
    return False, f"exit {proc.returncode}: {last_line}"


def check_lesson(lesson: Path, execute: bool) -> LessonResult:
    rel = lesson.relative_to(ROOT).as_posix()
    code_dir = lesson / "code"
    py_files = list_python_files(code_dir)
    result = LessonResult(
        lesson=rel,
        files=[p.relative_to(ROOT).as_posix() for p in py_files],
    )
    if not py_files:
        result.status = "skipped"
        result.reason = "no python files"
        return result

    ok, msg = syntax_check(py_files)
    if not ok:
        result.status = "failed"
        result.reason = msg
        return result

    if execute:
        entry = pick_entry_file(py_files)
        if entry is None:
            result.status = "skipped"
            result.reason = "no entry file"
            return result
        deps = read_requires(entry)
        if deps:
            result.status = "skipped"
            result.reason = f"needs {', '.join(deps)}"
            return result
        ok, msg = execute_lesson(entry)
        if not ok:
            result.status = "failed"
            result.reason = msg

    return result


def render_report(results: list[LessonResult], execute: bool) -> str:
    passed = [r for r in results if r.status == "passed"]
    failed = [r for r in results if r.status == "failed"]
    skipped = [r for r in results if r.status == "skipped"]
    mode = "execute" if execute else "syntax"
    total_files = sum(len(r.files) for r in results)
    lines = [
        f"lesson_run.py ({mode}) — {len(results)} lesson(s), "
        f"{total_files} python file(s): "
        f"passed={len(passed)} failed={len(failed)} skipped={len(skipped)}",
    ]
    if failed:
        lines.append("")
        lines.append("Failures:")
        for r in failed:
            lines.append(f"  [FAIL] {r.lesson}: {r.reason}")
    if skipped and execute:
        lines.append("")
        lines.append("Skipped:")
        for r in skipped:
            lines.append(f"  [SKIP] {r.lesson}: {r.reason}")
    return "\n".join(lines)


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase", type=int, default=None, help="仅检查一个 phase 编号"
    )
    parser.add_argument(
        "--json", action="store_true", help="向 stdout 输出 JSON 报告"
    )
    parser.add_argument(
        "--strict", action="store_true", help="任一课程失败时以状态码 1 退出"
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help=f"运行每课的入口文件，超时为 {EXECUTE_TIMEOUT_SEC} 秒",
    )
    args = parser.parse_args(argv)

    results = [check_lesson(lesson, args.execute) for lesson in iter_lesson_dirs(args.phase)]
    failed = [r for r in results if r.status == "failed"]

    if args.json:
        payload = {
            "mode": "execute" if args.execute else "syntax",
            "checked": len(results),
            "passed": [r.to_dict() for r in results if r.status == "passed"],
            "failed": [r.to_dict() for r in results if r.status == "failed"],
            "skipped": [r.to_dict() for r in results if r.status == "skipped"],
        }
        json.dump(payload, sys.stdout, indent=2)
        sys.stdout.write("\n")
    else:
        sys.stdout.write(render_report(results, args.execute) + "\n")

    return 1 if (args.strict and failed) else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
