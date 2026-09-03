#!/usr/bin/env bash
set -euo pipefail

# 将 agent 工作台 pack 安装到当前仓库。
# 用法：bin/install.sh [--force]

FORCE="${1:-}"
TARGET="$(pwd)"
PACK_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

required=("AGENTS.md" "VERSION" "docs" "schemas" "scripts")
for path in "${required[@]}"; do
    if [[ ! -e "$PACK_ROOT/$path" ]]; then
        echo "缺少 pack 源文件：$PACK_ROOT/$path" >&2
        exit 1
    fi
done

if [[ -e "$TARGET/AGENTS.md" && "$FORCE" != "--force" ]]; then
    echo "AGENTS.md 已存在。请传入 --force 以覆盖。" >&2
    exit 1
fi

cp "$PACK_ROOT/AGENTS.md" "$TARGET/AGENTS.md"
mkdir -p "$TARGET/docs" "$TARGET/schemas" "$TARGET/scripts"
cp -r "$PACK_ROOT/docs/." "$TARGET/docs/"
cp -r "$PACK_ROOT/schemas/." "$TARGET/schemas/"
cp -r "$PACK_ROOT/scripts/." "$TARGET/scripts/"
cat "$PACK_ROOT/VERSION" > "$TARGET/.workbench-version"

echo "pack 已安装，版本为 $(cat "$PACK_ROOT/VERSION")"
echo "下一步：编辑 task_board.json，设置验收命令，运行 scripts/init_agent.py"
