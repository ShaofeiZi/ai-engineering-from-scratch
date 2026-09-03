#!/usr/bin/env python3
"""验证 README.md 中的硬编码数量与 catalog.json 汇总值一致。

需要 Python 3.10+，仅使用标准库。

catalog.json 是文件系统事实源（CI 中由 scripts/build_catalog.py 重建；本地
缺失时临时构建）。README 中散布着硬编码数量（"428 lessons"、
"373 skills, 99 prompts, ..."），课程增减时容易漂移。本脚本将每个硬编码
数量绑定到 catalog.json 的 `totals` 字段，不一致时失败。

Usage:
    python3 scripts/check_readme_counts.py            # exit 1 on any drift
    python3 scripts/check_readme_counts.py --json     # machine-readable report
    python3 scripts/check_readme_counts.py --fix      # rewrite README to match catalog

`--fix` 需要显式启用。CI 不带 `--fix` 运行，发现不一致时构建失败，并在
workflow 日志中暴露漂移。

匹配模式会刻意锚定 README 上下文（徽章 URL、alt 属性、特定正文），因此不会
改动目录表中的 `<code>22 lessons</code>` 等分阶段数量。每个模式声明对应的
catalog 字段和简短说明；不匹配项会连同行号及上下文一起报告。
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CATALOG_PATH = ROOT / "catalog.json"
README_PATH = ROOT / "README.md"
VOLUMES_PATH = ROOT / "book" / "volumes.json"
BOOK_README_PATH = ROOT / "book" / "README.md"
INDEX_HTML_PATH = ROOT / "site" / "index.html"


@dataclass(frozen=True)
class CountPattern:
    """README 中绑定到 catalog 汇总字段的单个硬编码数量。"""

    regex: re.Pattern[str]
    field: str  # totals.<field> 字段
    description: str


PATTERNS: tuple[CountPattern, ...] = (
    CountPattern(
        regex=re.compile(r"lessons-(\d+)-3553ff"),
        field="lessons",
        description="lesson-count badge URL",
    ),
    CountPattern(
        regex=re.compile(r'alt="(\d+) lessons"'),
        field="lessons",
        description="lesson-count badge alt text",
    ),
    CountPattern(
        regex=re.compile(r"^> (\d+) lessons\. \d+ phases\.", re.MULTILINE),
        field="lessons",
        description="hero blockquote lesson count",
    ),
    CountPattern(
        regex=re.compile(r"^> \d+ lessons\. (\d+) phases\.", re.MULTILINE),
        field="phases",
        description="hero blockquote phase count",
    ),
    CountPattern(
        regex=re.compile(r"This curriculum is the spine\. (\d+) phases,"),
        field="phases",
        description="'spine' prose phase count",
    ),
    CountPattern(
        regex=re.compile(r"This curriculum is the spine\. \d+ phases, (\d+) lessons,"),
        field="lessons",
        description="'spine' prose lesson count",
    ),
    CountPattern(
        regex=re.compile(r"phases-(\d+)-3553ff"),
        field="phases",
        description="phase-count badge URL",
    ),
    CountPattern(
        regex=re.compile(r'alt="(\d+) phases"'),
        field="phases",
        description="phase-count badge alt text",
    ),
    CountPattern(
        regex=re.compile(r"portfolio of (\d+) artifacts"),
        field="lessons",
        description="'portfolio of N artifacts' (one artifact per lesson)",
    ),
    CountPattern(
        regex=re.compile(r"The repo ships (\d+) skills"),
        field="skills",
        description="toolkit section skill count",
    ),
    CountPattern(
        regex=re.compile(r"The repo ships \d+ skills and (\d+) prompts"),
        field="prompts",
        description="toolkit section prompt count",
    ),
    CountPattern(
        regex=re.compile(r"MIT-licensed, (\d+) lessons\."),
        field="lessons",
        description="sponsor section lesson count",
    ),
)


@dataclass
class Mismatch:
    pattern: CountPattern
    found: int
    expected: int
    line: int
    snippet: str


def load_totals() -> dict[str, int]:
    if CATALOG_PATH.exists():
        with CATALOG_PATH.open(encoding="utf-8") as fh:
            catalog = json.load(fh)
    else:
        sys.path.insert(0, str(ROOT / "scripts"))
        from build_catalog import build_catalog

        catalog = build_catalog()
    totals = catalog.get("totals")
    if not isinstance(totals, dict):
        raise SystemExit("catalog.json 缺少 'totals' block")
    return totals


def line_for(text: str, offset: int) -> int:
    return text.count("\n", 0, offset) + 1


def snippet_for(text: str, offset: int, end: int) -> str:
    line_start = text.rfind("\n", 0, offset) + 1
    line_end = text.find("\n", end)
    if line_end == -1:
        line_end = len(text)
    return text[line_start:line_end].strip()


def find_mismatches(readme_text: str, totals: dict[str, int]) -> list[Mismatch]:
    mismatches: list[Mismatch] = []
    for pattern in PATTERNS:
        expected = totals.get(pattern.field)
        if expected is None:
            raise SystemExit(f"catalog.json totals 缺少字段：{pattern.field}")
        matched_any = False
        for match in pattern.regex.finditer(readme_text):
            matched_any = True
            found = int(match.group(1))
            if found != expected:
                mismatches.append(
                    Mismatch(
                        pattern=pattern,
                        found=found,
                        expected=expected,
                        line=line_for(readme_text, match.start()),
                        snippet=snippet_for(readme_text, match.start(), match.end()),
                    )
                )
        if not matched_any:
            raise SystemExit(
                f"pattern 完全未匹配 README：{pattern.description} "
                f"({pattern.regex.pattern!r})。README 结构已变化；"
                f"请更新 scripts/check_readme_counts.py。"
            )
    return mismatches


def apply_fixes(readme_text: str, totals: dict[str, int]) -> str:
    for pattern in PATTERNS:
        expected = totals[pattern.field]

        def replace(match: re.Match[str], expected: int = expected) -> str:
            whole = match.group(0)
            old = match.group(1)
            start = match.start(1) - match.start()
            return whole[:start] + str(expected) + whole[start + len(old):]

        readme_text = pattern.regex.sub(replace, readme_text)
    return readme_text


def expand_phase_display(display: str) -> list[str]:
    """将阶段范围显示（如 "00-02"、"00–02"、"03, 04, 06"）解析为两位 ID。"""
    display = display.strip()
    parts = [p.strip() for p in display.split(",")]
    out: list[str] = []
    for part in parts:
        m = re.fullmatch(r"(\d{2})\s*[-–]\s*(\d{2})", part)
        if m:
            out.extend(f"{i:02d}" for i in range(int(m.group(1)), int(m.group(2)) + 1))
        elif re.fullmatch(r"\d{2}", part):
            out.append(part)
        else:
            return []
    return out


def check_book_volumes() -> list[str]:
    """确保三份分卷表展示副本都与 book/volumes.json 一致。

    volumes.json 驱动构建（产物文件名、标题和阶段范围）；README 表格、
    book/README.md 表格和首页 books 数组各自保存一份手工渲染的副本。重命名或
    移动阶段范围时若漏改任一副本，发布下载链接会静默变成 404，因此任何差异都在此失败。
    """
    volumes = json.loads(VOLUMES_PATH.read_text(encoding="utf-8"))["volumes"]
    readme = README_PATH.read_text(encoding="utf-8")
    book_readme = BOOK_README_PATH.read_text(encoding="utf-8")
    index_html = INDEX_HTML_PATH.read_text(encoding="utf-8")
    errors: list[str] = []

    for vol in volumes:
        n, slug = vol["number"], vol["slug"]
        title, subtitle = vol["title"], vol["subtitle"]
        shorts = [p.split("-")[0] for p in vol["phases"]]
        where = f"volume {n} ({slug})"

        m = re.search(
            rf"^\| {n} \| (.+?) \| (.+?) \| (.+) \|$", readme, re.MULTILINE
        )
        if not m or "aiefs-vol" not in m.group(3):
            errors.append(f"README.md book table: no row for {where}")
        else:
            if m.group(1) != f"{title} · {subtitle}":
                errors.append(
                    f"README.md book table {where}: name cell {m.group(1)!r} != "
                    f"{title!r} · {subtitle!r}"
                )
            if expand_phase_display(m.group(2)) != shorts:
                errors.append(
                    f"README.md book table {where}: phases {m.group(2)!r} != {shorts}"
                )
            for ext in ("epub", "pdf"):
                if f"aiefs-vol{n}-{slug}.{ext}" not in m.group(3):
                    errors.append(
                        f"README.md book table {where}: missing aiefs-vol{n}-{slug}.{ext} link"
                    )

        m = re.search(rf"^\| {n} \| (.+?) \| (.+?) \|$", book_readme, re.MULTILINE)
        if not m:
            errors.append(f"book/README.md table: no row for {where}")
        else:
            if m.group(1) != title:
                errors.append(
                    f"book/README.md table {where}: title {m.group(1)!r} != {title!r}"
                )
            if expand_phase_display(m.group(2)) != shorts:
                errors.append(
                    f"book/README.md table {where}: phases {m.group(2)!r} != {shorts}"
                )

        m = re.search(
            rf"\{{ n: {n}, slug: '([^']*)', title: '([^']*)', "
            rf"subtitle: '([^']*)', phases: '([^']*)' \}}",
            index_html,
        )
        if not m:
            errors.append(f"site/index.html books array: no entry for {where}")
        else:
            if m.group(1) != slug:
                errors.append(
                    f"site/index.html books array {where}: slug {m.group(1)!r} != {slug!r}"
                )
            if m.group(2) != title or m.group(3) != subtitle:
                errors.append(
                    f"site/index.html books array {where}: title/subtitle drift "
                    f"({m.group(2)!r}, {m.group(3)!r})"
                )
            if expand_phase_display(m.group(4)) != shorts:
                errors.append(
                    f"site/index.html books array {where}: phases {m.group(4)!r} != {shorts}"
                )

    return errors


def render_text_report(mismatches: list[Mismatch]) -> str:
    if not mismatches:
        return "README.md 计数与 catalog.json totals 一致。\n"
    out = [f"检测到 README.md 漂移：{len(mismatches)} 处不匹配。\n"]
    for m in mismatches:
        out.append(
            f"  README.md:{m.line}  {m.pattern.description}\n"
            f"    预期 totals.{m.pattern.field} = {m.expected}，实际为 {m.found}\n"
            f"    >>> {m.snippet}\n"
        )
    out.append(
        "\n请运行 `python3 scripts/check_readme_counts.py --fix` 更新 README.md。\n"
    )
    return "".join(out)


def render_json_report(mismatches: list[Mismatch], totals: dict[str, int]) -> str:
    payload = {
        "ok": not mismatches,
        "totals": totals,
        "mismatches": [
            {
                "line": m.line,
                "field": m.pattern.field,
                "description": m.pattern.description,
                "expected": m.expected,
                "found": m.found,
                "snippet": m.snippet,
            }
            for m in mismatches
        ],
    }
    return json.dumps(payload, indent=2) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--json", action="store_true", help="向 stdout 输出 JSON 报告")
    parser.add_argument(
        "--fix",
        action="store_true",
        help="重写 README.md，使硬编码数量与 catalog.json 一致",
    )
    args = parser.parse_args(argv)

    totals = load_totals()
    readme_text = README_PATH.read_text(encoding="utf-8")

    if args.fix:
        initial_mismatches = find_mismatches(readme_text, totals)
        if not initial_mismatches:
            if args.json:
                sys.stdout.write(render_json_report([], totals))
            else:
                sys.stdout.write("README.md 已与 catalog.json totals 一致。\n")
            return 0
        new_text = apply_fixes(readme_text, totals)
        README_PATH.write_text(new_text, encoding="utf-8")
        remaining = find_mismatches(new_text, totals)
        if args.json:
            sys.stdout.write(render_json_report(remaining, totals))
        else:
            sys.stdout.write("已更新 README.md，使其与 catalog.json totals 一致。\n")
            if remaining:
                sys.stdout.write(render_text_report(remaining))
        return 1 if remaining else 0

    mismatches = find_mismatches(readme_text, totals)
    volume_errors = check_book_volumes()
    if args.json:
        report = json.loads(render_json_report(mismatches, totals))
        report["ok"] = report["ok"] and not volume_errors
        report["volume_errors"] = volume_errors
        sys.stdout.write(json.dumps(report, indent=2) + "\n")
    else:
        sys.stdout.write(render_text_report(mismatches))
        if volume_errors:
            sys.stdout.write(
                f"检测到图书分卷漂移：{len(volume_errors)} 个错误。\n"
            )
            for err in volume_errors:
                sys.stdout.write(f"  {err}\n")
            sys.stdout.write(
                "book/volumes.json 是事实源；请同步上面的副本。\n"
            )
        else:
            sys.stdout.write("Book volume tables match book/volumes.json.\n")
    return 1 if (mismatches or volume_errors) else 0


if __name__ == "__main__":
    sys.exit(main())
