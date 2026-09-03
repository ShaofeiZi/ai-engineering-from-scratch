"""足以解析宪法文件的极简 YAML 子集解析器。

支持：
- 两空格缩进的嵌套块映射
- 块序列（采用 '- key: value' 形式）
- 字符串标量（普通、单引号、双引号）
- 整数标量
- 在 ':' 右侧的内联列表值（用于简单原子）
- 行内 '#' 之后的注释

不支持：锚点、别名、标签、流式风格、多文档、多行折叠/字面块。
宪法格式在设计时即有意避开这些特性。

若已安装 PyYAML，则通过 load_yaml 优先使用它；此回退实现存在的目的，
是让本课程在任何标准的 Python 安装环境下都能运行。
"""

from __future__ import annotations

import re
from typing import Any


def load_yaml(text: str) -> Any:
    try:
        import yaml  # type: ignore[import-not-found]
        return yaml.safe_load(text)
    except ModuleNotFoundError:
        return _parse(text)


def _strip_comment(line: str) -> str:
    in_single = False
    in_double = False
    for i, ch in enumerate(line):
        if ch == "'" and not in_double:
            in_single = not in_single
        elif ch == '"' and not in_single:
            in_double = not in_double
        elif ch == "#" and not in_single and not in_double:
            return line[:i]
    return line


def _indent_of(line: str) -> int:
    return len(line) - len(line.lstrip(" "))


def _coerce(value: str) -> Any:
    s = value.strip()
    if s == "":
        return None
    if s == "{}":
        return {}
    if s == "[]":
        return []
    if s in ("true", "True"):
        return True
    if s in ("false", "False"):
        return False
    if s in ("null", "None", "~"):
        return None
    if (s.startswith("'") and s.endswith("'")) or (s.startswith('"') and s.endswith('"')):
        inner = s[1:-1]
        if s[0] == '"':
            inner = inner.encode("utf-8").decode("unicode_escape")
        return inner
    try:
        return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        pass
    return s


def _parse(text: str) -> Any:
    raw_lines = []
    for raw in text.splitlines():
        stripped = _strip_comment(raw).rstrip()
        if stripped.strip() == "":
            continue
        raw_lines.append(stripped)
    if not raw_lines:
        return None
    return _parse_block(raw_lines, 0, _indent_of(raw_lines[0]))[0]


def _parse_block(lines: list[str], start: int, indent: int) -> tuple[Any, int]:
    if start >= len(lines):
        return None, start
    first = lines[start]
    cur_indent = _indent_of(first)
    if cur_indent < indent:
        return None, start
    stripped = first.lstrip()
    if stripped.startswith("- "):
        return _parse_sequence(lines, start, cur_indent)
    return _parse_mapping(lines, start, cur_indent)


def _parse_mapping(lines: list[str], start: int, indent: int) -> tuple[dict[str, Any], int]:
    out: dict[str, Any] = {}
    i = start
    while i < len(lines):
        line = lines[i]
        cur_indent = _indent_of(line)
        if cur_indent < indent:
            break
        if cur_indent > indent:
            i += 1
            continue
        stripped = line.lstrip()
        if stripped.startswith("- "):
            break
        m = re.match(r"^([A-Za-z_][\w-]*)\s*:\s*(.*)$", stripped)
        if not m:
            raise ValueError(f"格式错误的行: {line!r}")
        key = m.group(1)
        rest = m.group(2)
        if rest.strip() == "":
            if i + 1 < len(lines) and _indent_of(lines[i + 1]) > indent:
                value, i = _parse_block(lines, i + 1, _indent_of(lines[i + 1]))
                out[key] = value
                continue
            out[key] = None
            i += 1
        else:
            out[key] = _coerce(rest)
            i += 1
    return out, i


def _parse_sequence(lines: list[str], start: int, indent: int) -> tuple[list[Any], int]:
    out: list[Any] = []
    i = start
    while i < len(lines):
        line = lines[i]
        cur_indent = _indent_of(line)
        if cur_indent < indent:
            break
        stripped = line.lstrip()
        if not stripped.startswith("- "):
            break
        if cur_indent > indent:
            i += 1
            continue
        rest = stripped[2:]
        m = re.match(r"^([A-Za-z_][\w-]*)\s*:\s*(.*)$", rest)
        if m:
            child_indent = indent + 2
            synthetic = " " * child_indent + rest
            j = i + 1
            extra_lines = []
            while j < len(lines) and _indent_of(lines[j]) > indent and not lines[j].lstrip().startswith("- "):
                extra_lines.append(lines[j])
                j += 1
            block = [synthetic] + extra_lines
            value, _ = _parse_mapping(block, 0, child_indent)
            out.append(value)
            i = j
        else:
            out.append(_coerce(rest))
            i += 1
    return out, i
