#!/usr/bin/env python3
"""分散测验正确答案的位置，避免总在同一选项。

生成的测验中有 61.5% 的正确答案位于选项 B（索引 1），使测验容易被猜中。
本脚本使用由内容作为种子的确定性排列重写每题选项顺序，并更新 `correct` 索引
以跟随移动后的答案。该过程具备幂等性：排列前会把选项规范化为有序基准，
因此重复运行会产生逐字节一致的输出。

若题目选项按位置互相引用（如 "all of the above"、"both A and B"），
则保留原顺序，因为重排会破坏语义。

Usage:
    python3 scripts/debias_quizzes.py            # rewrite in place
    python3 scripts/debias_quizzes.py --check     # report distribution, no writes
"""
import argparse
import collections
import glob
import hashlib
import json
import random
import re
import sys

QUIZ_GLOB = "phases/*/*/quiz.json"

ANCHOR = re.compile(
    r"\b(all|none|both|neither)\s+of\s+(the|these)\b"
    r"|\b(both|neither|either)\s+[A-D]\b"
    r"|\b[A-D]\s+and\s+[A-D]\b"
    r"|\b(above|below|following)\b",
    re.IGNORECASE,
)


def seed_for(path, question_text):
    h = hashlib.sha256(f"{path}\x00{question_text}".encode("utf-8")).hexdigest()
    return int(h[:16], 16)


def has_positional_anchor(options):
    return any(ANCHOR.search(str(o)) for o in options)


def debias_question(path, q):
    """题目选项顺序发生变化时返回 True。"""
    options = q.get("options")
    correct = q.get("correct")
    if not isinstance(options, list) or not isinstance(correct, int):
        return False
    if not (0 <= correct < len(options)) or len(options) < 2:
        return False
    if has_positional_anchor(options):
        return False
    if len(set(map(str, options))) != len(options):
        return False  # 重复选项会让身份追踪产生歧义

    correct_val = options[correct]
    base = sorted(options, key=str)
    perm = list(range(len(base)))
    random.Random(seed_for(path, q.get("question", ""))).shuffle(perm)
    new_options = [base[i] for i in perm]
    new_correct = new_options.index(correct_val)

    if new_options == options and new_correct == correct:
        return False
    assert sorted(map(str, new_options)) == sorted(map(str, options))
    q["options"] = new_options
    q["correct"] = new_correct
    return True


def serialize(data, inline_options):
    """以两空格缩进美化 JSON，并匹配文件原有的选项排版。

    有些测验文件把每题的 `options` 数组放在一行，另一些则展开显示。保留原始形式，
    可让 diff 仅包含重新排序的值，而不是整文件的空白变化。
    """
    text = json.dumps(data, ensure_ascii=False, indent=2)
    if not inline_options:
        return text + "\n"
    out = []
    lines = text.split("\n")
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.rstrip()
        if stripped.endswith('"options": ['):
            indent = line[: len(line) - len(line.lstrip())]
            block = []
            i += 1
            while not lines[i].strip().startswith("]"):
                block.append(lines[i])
                i += 1
            trailing = "," if lines[i].strip().endswith(",") else ""
            items = json.loads("[" + "\n".join(block) + "]")
            out.append(f'{indent}"options": {json.dumps(items, ensure_ascii=False)}{trailing}')
        else:
            out.append(line)
        i += 1
    return "\n".join(out) + "\n"


def iter_questions(data):
    if isinstance(data, dict):
        qs = data.get("questions")
    elif isinstance(data, list):
        qs = data
    else:
        qs = None
    return qs if isinstance(qs, list) else []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="仅报告，不写入")
    args = ap.parse_args()

    pos = collections.Counter()
    total = 0
    changed_files = 0
    changed_qs = 0
    skipped_anchor = 0

    for path in sorted(glob.glob(QUIZ_GLOB)):
        with open(path, encoding="utf-8") as fh:
            raw = fh.read()
        data = json.loads(raw)
        inline_options = '"options": [\n' not in raw
        questions = iter_questions(data)
        if not questions:
            continue
        file_changed = False
        for q in questions:
            if not isinstance(q, dict):
                continue
            opts, c = q.get("options"), q.get("correct")
            if isinstance(opts, list) and isinstance(c, int) and 0 <= c < len(opts):
                if has_positional_anchor(opts):
                    skipped_anchor += 1
                elif debias_question(path, q):
                    file_changed = True
                    changed_qs += 1
                pos[q.get("correct")] += 1
                total += 1
        if file_changed and not args.check:
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(serialize(data, inline_options))
        if file_changed:
            changed_files += 1

    verb = "将重写" if args.check else "已重写"
    print(f"题目：{total}  受影响文件：{changed_files}  {verb}：{changed_qs}")
    print(f"保持不变的位置锚定题目：{skipped_anchor}")
    print("正确选项位置分布：")
    for k in sorted(pos):
        label = chr(65 + k) if isinstance(k, int) else "?"
        print(f"  {label}: {pos[k]:4d}  {100 * pos[k] / max(total, 1):5.1f}%")

    if args.check and changed_qs:
        print(
            f"\n失败：{changed_qs} 道 quiz 题目尚未消除位置偏差。"
            "请运行：python3 scripts/debias_quizzes.py",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
