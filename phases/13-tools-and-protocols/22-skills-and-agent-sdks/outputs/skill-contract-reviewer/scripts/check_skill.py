#!/usr/bin/env python3
"""Read-only、stdlib-only 验证器，用于 SKILL.md. 的核心身份"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


NAME_PATTERN = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
CORE_FIELDS = {
    "name",
    "description",
    "license",
    "compatibility",
    "metadata",
    "allowed-tools",
}


def validate(directory: Path) -> dict[str, object]:
    errors: list[dict[str, str]] = []
    path = directory / "SKILL.md"
    fields: dict[str, object] = {}
    body = ""
    if not path.is_file() or path.is_symlink():
        errors.append({"code": "skill-file", "message": "需要常规的 SKILL.md 文件"})
    else:
        lines = path.read_text(encoding="utf-8").splitlines()
        if not lines or lines[0] != "---" or "---" not in lines[1:]:
            errors.append({"code": "frontmatter", "message": "需要精确的 frontmatter 分隔符"})
        else:
            end = lines.index("---", 1)
            index = 1
            while index < end:
                line = lines[index]
                if not line.strip() or line.lstrip().startswith("#"):
                    index += 1
                    continue
                if line[:1].isspace() or ":" not in line:
                    errors.append(
                        {
                            "code": "frontmatter-syntax",
                            "message": f"第 {index + 1} 行的顶层语法格式错误",
                        }
                    )
                    index += 1
                    continue
                key, value = line.split(":", 1)
                key, value = key.strip(), value.strip()
                if not re.fullmatch(r"[A-Za-z][A-Za-z0-9-]*", key):
                    errors.append(
                        {
                            "code": "frontmatter-syntax",
                            "message": f"字段名 {key!r} 无效",
                        }
                    )
                    index += 1
                    continue
                if key in fields:
                    errors.append({"code": "duplicate", "message": f"第 {index + 1} 行重复定义了 {key}"})
                if value in {">", "|"}:
                    block: list[str] = []
                    index += 1
                    while index < end and (not lines[index] or lines[index][:1].isspace()):
                        block.append(lines[index].lstrip())
                        index += 1
                    fields[key] = (" " if value == ">" else "\n").join(block).strip()
                    continue
                if key == "metadata" and not value:
                    nested: dict[str, str] = {}
                    index += 1
                    while index < end and (
                        not lines[index] or lines[index][:1].isspace()
                    ):
                        nested_line = lines[index].strip()
                        if nested_line:
                            if ":" not in nested_line:
                                errors.append(
                                    {
                                        "code": "metadata-shape",
                                        "message": f"第 {index + 1} 行的 metadata 格式错误",
                                    }
                                )
                            else:
                                nested_key, nested_value = nested_line.split(":", 1)
                                nested_key = nested_key.strip()
                                if nested_key in nested:
                                    errors.append(
                                        {
                                            "code": "duplicate",
                                            "message": f"metadata 字段 {nested_key!r} 重复",
                                        }
                                    )
                                nested[nested_key] = nested_value.strip().strip("\"'")
                        index += 1
                    fields[key] = nested
                    continue
                fields[key] = value.strip("\"'")
                index += 1
            body = "\n".join(lines[end + 1 :]).strip()

    name_value = fields.get("name", "")
    description_value = fields.get("description", "")
    name = name_value if isinstance(name_value, str) else ""
    description = description_value if isinstance(description_value, str) else ""
    if not name:
        errors.append({"code": "name-required", "message": "name 为必填项"})
    elif len(name) > 64 or not NAME_PATTERN.fullmatch(name):
        errors.append({"code": "name-format", "message": "name 必须使用 kebab-case，且最多包含 64 个字符"})
    elif name != directory.name:
        errors.append({"code": "directory-mismatch", "message": "name 必须与目录名一致"})
    if not description:
        errors.append({"code": "description-required", "message": "description 为必填项"})
    elif len(description) > 1024:
        errors.append({"code": "description-length", "message": "description 超过 1024 个字符"})
    if "compatibility" in fields:
        compatibility = fields["compatibility"]
        if not isinstance(compatibility, str) or not 1 <= len(compatibility) <= 500:
            errors.append(
                {
                    "code": "compatibility-length",
                    "message": "compatibility 必须包含 1 到 500 个字符",
                }
            )
    if "metadata" in fields and not isinstance(fields["metadata"], dict):
        errors.append(
            {
                "code": "metadata-shape",
                "message": "metadata 必须将字符串键映射到字符串值",
            }
        )
    if "allowed-tools" in fields:
        allowed_tools = fields["allowed-tools"]
        if not isinstance(allowed_tools, str) or not allowed_tools.strip():
            errors.append(
                {
                    "code": "allowed-tools-shape",
                    "message": "allowed-tools 必须是以空格分隔的非空字符串",
                }
            )
    for unknown in sorted(set(fields) - CORE_FIELDS):
        errors.append(
            {
                "code": "unsupported-field",
                "message": f"{unknown!r} 不属于可移植核心字段",
            }
        )
    if not body:
        errors.append({"code": "body-required", "message": "指令正文为必填项"})
    return {
        "path": str(directory),
        "valid": not errors,
        "name": name or None,
        "description": description or None,
        "errors": errors,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, help="包含 SKILL.md 的技能包目录")
    args = parser.parse_args()
    result = validate(args.directory.resolve())
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
