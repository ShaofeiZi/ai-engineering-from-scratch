"""供 scripts/ 工具共用的辅助函数。

当前提供：
- parse_frontmatter：解析 Markdown 中 `--- ... ---` 块的最小 YAML 子集解析器。
- validate_repository_directory/file：检查课程产物是否位于仓库内。
- validate_skill_bundle：安全、确定性地发现 skill bundle 文件。

无外部依赖。需要 Python 3.10+（类型注解使用 PEP 604 union）。
"""

from __future__ import annotations

import os
from pathlib import Path


class BundleValidationError(ValueError):
    """skill bundle 不安全或格式错误时抛出。"""


class ArtifactPathError(ValueError):
    """课程产物路径不安全或逃逸出仓库时抛出。"""


def _resolve_within_repository(
    target: Path,
    repository_root: Path,
    label: str,
    error_type: type[ValueError],
) -> Path:
    try:
        resolved_target = target.resolve(strict=True)
        resolved_root = repository_root.resolve(strict=True)
    except (OSError, RuntimeError) as error:
        raise error_type(f"could not resolve {label}: {target}") from error
    if not resolved_target.is_relative_to(resolved_root):
        raise error_type(f"{label} escapes the repository: {target}")
    return resolved_target


def validate_repository_directory(
    directory: Path, repository_root: Path, label: str
) -> Path:
    """解析普通目录，并要求其始终位于仓库内。"""

    if directory.is_symlink():
        _resolve_within_repository(
            directory, repository_root, label, ArtifactPathError
        )
        raise ArtifactPathError(f"{label} must be a regular directory: {directory}")
    if not directory.is_dir():
        raise ArtifactPathError(f"{label} must be a regular directory: {directory}")
    return _resolve_within_repository(
        directory, repository_root, label, ArtifactPathError
    )


def validate_repository_file(file_path: Path, repository_root: Path, label: str) -> Path:
    """解析仓库内不属于符号链接的普通文件。"""

    if file_path.is_symlink() or not file_path.is_file():
        raise ArtifactPathError(f"{label} must be a regular file: {file_path}")
    return _resolve_within_repository(
        file_path, repository_root, label, ArtifactPathError
    )


def validate_skill_bundle(bundle_root: Path, repository_root: Path) -> list[str]:
    """按字符串顺序返回已验证 bundle 的相对 POSIX 文件路径。"""

    if bundle_root.is_symlink() or not bundle_root.is_dir():
        raise BundleValidationError(
            f"skill bundle must be a regular directory: {bundle_root}"
        )
    _resolve_within_repository(
        bundle_root, repository_root, "skill bundle", BundleValidationError
    )

    skill_path = bundle_root / "SKILL.md"
    if skill_path.is_symlink() or not skill_path.is_file():
        raise BundleValidationError(
            f"skill bundle entrypoint must be a regular file: {skill_path}"
        )

    bundle_files: list[str] = []
    for current, dirs, files in os.walk(bundle_root, followlinks=False):
        dirs.sort()
        files.sort()
        current_path = Path(current)
        for name in dirs:
            entry = current_path / name
            if entry.is_symlink() or not entry.is_dir():
                raise BundleValidationError(
                    f"skill bundle contains an unsafe directory entry: {entry}"
                )
        for name in files:
            entry = current_path / name
            if entry.is_symlink() or not entry.is_file():
                raise BundleValidationError(
                    f"skill bundle contains an unsafe file entry: {entry}"
                )
            bundle_files.append(entry.relative_to(bundle_root).as_posix())
    return sorted(bundle_files)


def parse_frontmatter(text: str) -> dict[str, object] | None:
    """解析 Markdown 字符串开头的 YAML 子集 frontmatter 块。

    返回解析后的键值映射；没有 frontmatter 或缺少结束 `---` 时返回 None。

    支持裸字符串、单引号/双引号字符串、列表，以及以 `#` 开头的行内注释。
    """
    if not text.startswith("---\n"):
        return None
    # 结束分隔符：文件内部的 "\n---\n"，或文件结尾的 "\n---"。
    end = text.find("\n---\n", 4)
    if end == -1 and text.endswith("\n---"):
        end = len(text) - 4
    if end == -1:
        return None
    block = text[4:end].strip("\n")
    result: dict[str, object] = {}
    for raw in block.splitlines():
        # 锚定第 0 列：跳过注释和缩进行。
        if not raw or raw.startswith("#") or raw[0] in (" ", "\t"):
            continue
        if ":" not in raw:
            continue
        key, _, value = raw.partition(":")
        key = key.strip()
        if not key:
            continue
        value = value.strip()
        if value.startswith("[") and value.endswith("]"):
            inner = value[1:-1].strip()
            result[key] = (
                [item.strip().strip("'\"") for item in inner.split(",") if item.strip()]
                if inner
                else []
            )
        elif (value.startswith('"') and value.endswith('"')) or (
            value.startswith("'") and value.endswith("'")
        ):
            result[key] = value[1:-1]
        else:
            result[key] = value
    return result
