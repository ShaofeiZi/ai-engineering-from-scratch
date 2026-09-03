#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  cat <<'USAGE' >&2
用法：scripts/scaffold-lesson.sh <phase-dir> <lesson-slug> [title]

示例：
  scripts/scaffold-lesson.sh 05-nlp-foundations-to-advanced 03-tokenizers
  scripts/scaffold-lesson.sh 05-nlp-foundations-to-advanced 03-tokenizers "Tokenizers from Scratch"

创建 phases/<phase-dir>/<lesson-slug>/，其中包含 code/、notebook/、docs/、outputs/，
并根据 LESSON_TEMPLATE.md 预填充 docs/en.md 骨架。
USAGE
  exit 2
fi

PHASE="$1"
LESSON="$2"
TITLE="${3:-}"

REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null)"
if [[ -z "$REPO_ROOT" ]]; then
  echo "错误：请在 ai-engineering-from-scratch Git 仓库内运行此脚本" >&2
  exit 1
fi

PHASE_DIR="$REPO_ROOT/phases/$PHASE"
LESSON_DIR="$PHASE_DIR/$LESSON"

if [[ ! -d "$PHASE_DIR" ]]; then
  echo "错误：找不到 phase 目录：phases/$PHASE" >&2
  echo "      运行 ls phases/ 查看有效 phase" >&2
  exit 1
fi

if [[ -e "$LESSON_DIR" ]]; then
  echo "错误：课程已存在：phases/$PHASE/$LESSON" >&2
  exit 1
fi

if [[ ! "$LESSON" =~ ^[0-9]{2}-[a-z0-9-]+$ ]]; then
  echo "错误：课程 slug 必须匹配 NN-kebab-case（例如 03-tokenizers）" >&2
  exit 1
fi

mkdir -p "$LESSON_DIR/code" "$LESSON_DIR/notebook" "$LESSON_DIR/docs" "$LESSON_DIR/outputs"

PRETTY_TITLE="$TITLE"
if [[ -z "$PRETTY_TITLE" ]]; then
  PRETTY_TITLE="$(echo "${LESSON#[0-9][0-9]-}" | tr '-' ' ' | awk '{for (i=1; i<=NF; i++) $i=toupper(substr($i,1,1)) substr($i,2);}1')"
fi

PHASE_NUM="${PHASE%%-*}"
LESSON_NUM="${LESSON%%-*}"

cat >"$LESSON_DIR/docs/en.md" <<EOF
# $PRETTY_TITLE

> [One-line motto. The core idea that sticks.]

**Type:** Build
**Languages:** Python
**Prerequisites:** [prior lessons]
**Time:** ~75 minutes

## The Problem

[2-3 paragraphs. What can't a learner do without this? Make it concrete.]

## The Concept

[Intuition first. Diagrams, tables, mental models. No code yet.]

## Build It

### Step 1: [name]

[explanation]

\`\`\`python
# Write code here
\`\`\`

### Step 2: [name]

[explanation]

\`\`\`python
# Write code here
\`\`\`

## Use It

[How a real framework solves the same thing. Compare your version.]

## Ship It

[The reusable artifact this lesson produces. Save in outputs/.]

## Exercises

1. [Easy — reinforce core concept]
2. [Medium — apply to a different problem]
3. [Hard — extend or combine with prior lessons]

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|----------------------|
|      |                |                      |

## Further Reading

- []() — []
EOF

cat >"$LESSON_DIR/code/main.py" <<'EOF'
def main():
    raise NotImplementedError("Implement this lesson")


if __name__ == "__main__":
    main()
EOF

touch "$LESSON_DIR/notebook/.gitkeep"
touch "$LESSON_DIR/outputs/.gitkeep"

echo "created phases/$PHASE/$LESSON/"
echo ""
echo "next:"
echo "  1. edit phases/$PHASE/$LESSON/docs/en.md"
echo "  2. write phases/$PHASE/$LESSON/code/main.py"
echo "  3. add a markdown-link row to ROADMAP.md under Phase $PHASE_NUM:"
echo "     | $LESSON_NUM | [$PRETTY_TITLE](phases/$PHASE/$LESSON) | ✅ | ~75 min |"
echo "  4. atomic commit: git add phases/$PHASE/$LESSON ROADMAP.md && git commit -m \"feat(phase-$PHASE_NUM/$LESSON_NUM): $PRETTY_TITLE\""
