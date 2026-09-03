#!/usr/bin/env python3
"""根据规范英文 README 构建各语言 README。

README 主要由结构组成：横幅、徽章、523 行课程表和 HTML 块。只翻译正文与
标题，其余字节保持不变，避免译文破坏布局、课程表或链接。生成器只替换原文件
副本中的已翻译行区间，因此没有译文的语言往返处理后仍与 README 逐字节一致
（每次运行都会断言这一点）。

译文由人工编写（保证落地页质量），存放在 scripts/readme_translations.py，
并以精确英文块作为键。没有译文的块回退到英文；简体中文除外，其翻译表必须
精确覆盖当前全部可翻译块。

    python3 scripts/build_readme_i18n.py --dump     # list translatable blocks
    python3 scripts/build_readme_i18n.py            # write i18n/<lang>/README.md
    python3 scripts/build_readme_i18n.py --check     # fail if any output is stale

输出写入 i18n/<lang>/README.md 并提交到 main（课程译文则位于 translations
分支）。英文始终是规范源。
"""
import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
README = ROOT / "README.md"
OUT_ROOT = ROOT / "i18n"

sys.path.insert(0, str(Path(__file__).resolve().parent))

FENCE = re.compile(r"^\s*```")
HEADING = re.compile(r"^(#{1,6})\s+(.*)$")
BLOCKQUOTE = re.compile(r"^>\s?(.*)$")
STRUCTURAL = re.compile(r"^\s*(<|\||!\[|<!--|\[!\[|-{3,}\s*$|\d+\.\s|[-*]\s)")
COMPLETE_LANGUAGES = ("zh",)
ZH_CATALOG_DIR = ROOT / "i18n" / "zh" / "catalog"
ZH_STRUCTURAL_PATH = ROOT / "i18n" / "zh" / "readme-structural.json"
LESSON_ROW_RE = re.compile(
    r"^(\|\s*\d+\s*\|\s*)\[([^]]+)\]\((phases/[^)]+)\)"
    r"(\s*\|\s*)([^|]+?)(\s*\|\s*)([^|]+?)(\s*\|)$"
)
PHASE_ZERO_RE = re.compile(r"^### Phase 0: .*?(`\d+ lessons`)$")
PHASE_SUMMARY_RE = re.compile(
    r"^<summary><b>Phase (\d+) — .*?</b> &nbsp;<code>(\d+) lessons</code>"
    r"&nbsp; <em>.*?</em></summary>$"
)
CATALOG_COLUMN_TRANSLATIONS = {
    "Build": "实践",
    "Learn": "学习",
    "Reference": "参考",
    "A. Agent harness": "A. 智能体框架",
    "B. NLP LLM": "B. NLP 大语言模型",
    "C. Train end-to-end": "C. 端到端训练",
    "D. Auto research": "D. 自动化研究",
    "E. Multimodal VLM": "E. 多模态 VLM",
    "F. Advanced RAG": "F. 进阶 RAG",
    "G. Eval framework": "G. 评估框架",
    "H. Distributed train": "H. 分布式训练",
    "I. Safety harness": "I. 安全框架",
}
CATALOG_HEADER_TRANSLATIONS = {
    "| # | Lesson | Type | Lang |": "| # | 课程 | 类型 | 语言 |",
    "| # | Project | Combines | Lang |": "| # | 项目 | 综合运用 | 语言 |",
}
HTML_ALT_RE = re.compile(r'\balt="([^"]*)"')
HTML_TAG_RE = re.compile(r"<[^>]+>")


def is_prose(line):
    s = line.strip()
    if not s or STRUCTURAL.match(line) or s.startswith("```"):
        return False
    if re.fullmatch(r"\[[^\]]+\]\([^)]+\)", s):  # 单独占一行的链接或徽章
        return False
    return bool(re.search(r"[A-Za-z]{3,}", s))


def block_key(lines):
    return re.sub(r"\s+", " ", " ".join(lines)).strip()


def spans(text):
    """以包含原始行索引的 dict 返回可翻译 span。

    kind='heading' 覆盖一行；kind='prose' 覆盖最长连续正文行（或 blockquote 正文行）。
    其他内容保持不变。
    """
    lines = text.split("\n")
    i, in_code, out = 0, False, []
    while i < len(lines):
        line = lines[i]
        if FENCE.match(line):
            in_code = not in_code
            i += 1
            continue
        if in_code:
            i += 1
            continue
        m = HEADING.match(line)
        if m:
            out.append({"kind": "heading", "start": i, "end": i + 1,
                        "prefix": m.group(1) + " ", "key": block_key([m.group(2)])})
            i += 1
            continue
        bq = BLOCKQUOTE.match(line)
        if bq and is_prose(bq.group(1)):
            start, block = i, []
            while i < len(lines):
                inner = BLOCKQUOTE.match(lines[i])
                if not (inner and is_prose(inner.group(1))):
                    break
                block.append(inner.group(1))
                i += 1
            out.append({"kind": "prose", "start": start, "end": i,
                        "prefix": "> ", "key": block_key(block)})
            continue
        if is_prose(line):
            start, block = i, []
            while i < len(lines) and is_prose(lines[i]):
                block.append(lines[i])
                i += 1
            out.append({"kind": "prose", "start": start, "end": i,
                        "prefix": "", "key": block_key(block)})
            continue
        i += 1
    return out


# 相对仓库根目录的链接/图片目标，不包括绝对 URL、锚点和已经向上引用的路径。
# 两个捕获组：起始部分与目标。
_HTML_LINK = re.compile(r'((?:href|src)=")(?!https?://|/|#|mailto:|data:|\.\.?/)([^"]+)')
_MD_LINK = re.compile(r'(\]\()(?!https?://|/|#|mailto:|data:|\.\.?/)([^)]+)')


def localize_links(md):
    """为仓库根目录相对链接添加 ../../ 前缀，使 i18n/<lang>/ 下两层深的
    README 仍能正确解析图片、课程表和语言栏。

    fenced code block 保持不变，因此 ``tools[call.name](**kw)`` 等形似 Markdown
    链接的代码绝不会被重写。"""
    out, in_code = [], False
    for line in md.split("\n"):
        if FENCE.match(line):
            in_code = not in_code
            out.append(line)
            continue
        if in_code:
            out.append(line)
            continue
        line = _HTML_LINK.sub(lambda m: m.group(1) + "../../" + m.group(2), line)
        line = _MD_LINK.sub(lambda m: m.group(1) + "../../" + m.group(2), line)
        out.append(line)
    return "\n".join(out)


def render(text, lang, translations):
    table = translations.get(lang, {})
    lines = text.split("\n")
    # 自后向前替换，使前面的索引保持有效。
    for sp in sorted(spans(text), key=lambda s: s["start"], reverse=True):
        t = table.get(sp["key"])
        if not t:
            continue
        replacement = [sp["prefix"] + ln for ln in t.split("\n")]
        lines[sp["start"]:sp["end"]] = replacement
    in_code = False
    for index, line in enumerate(lines):
        if FENCE.match(line):
            in_code = not in_code
            continue
        if not in_code:
            lines[index] = table.get(line, line)
    return "\n".join(lines)


def load_zh_assets(catalog_dir=ZH_CATALOG_DIR, structural_path=ZH_STRUCTURAL_PATH):
    """加载完整、可审阅的简体中文 README 清单。"""
    phases, lessons = {}, {}
    for path in sorted(catalog_dir.glob("phase-*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        for key, value in payload.get("phases", {}).items():
            if key in phases:
                raise ValueError(f"重复的 zh 阶段翻译 {key!r}")
            phases[key] = value
        for key, value in payload.get("lessons", {}).items():
            if key in lessons:
                raise ValueError(f"重复的 zh 课程翻译 {key!r}")
            lessons[key] = value
    structural = json.loads(structural_path.read_text(encoding="utf-8"))
    return phases, lessons, structural


def _phase_by_number(phases, number):
    prefix = f"{int(number):02d}-"
    matches = [(key, value) for key, value in phases.items() if key.startswith(prefix)]
    if len(matches) != 1:
        raise ValueError(f"{prefix!r} 应有一个 zh 阶段条目，实际找到 {len(matches)} 个")
    return matches[0]


def render_zh_catalog_line(line, phases, lessons):
    """翻译一行 README 课程内容，不改变其中链接。"""
    if line in CATALOG_HEADER_TRANSLATIONS:
        return CATALOG_HEADER_TRANSLATIONS[line]
    row = LESSON_ROW_RE.match(line)
    if row:
        prefix, _, href, sep1, column, sep2, language, suffix = row.groups()
        key = href if href.endswith("/") else href + "/"
        title = lessons.get(key)
        if title is None:
            return line
        column = CATALOG_COLUMN_TRANSLATIONS.get(column.strip(), column.strip())
        return f"{prefix}[{title}]({href}){sep1}{column}{sep2}{language.strip()}{suffix}"

    phase_zero = PHASE_ZERO_RE.match(line)
    if phase_zero:
        _, phase = _phase_by_number(phases, 0)
        lesson_count = phase_zero.group(1).replace(" lessons", " 节课")
        return f"### 第 0 阶段：{phase['title']} {lesson_count}"

    summary = PHASE_SUMMARY_RE.match(line)
    if summary:
        number, lesson_count = summary.groups()
        _, phase = _phase_by_number(phases, number)
        return (
            f"<summary><b>第 {int(number)} 阶段 — {phase['title']}</b> "
            f"&nbsp;<code>{lesson_count} 节课</code>&nbsp; "
            f"<em>{phase['description']}</em></summary>"
        )
    return line


def render_zh_catalog(text, phases, lessons):
    """在通用正文翻译前渲染由 catalog 管理的 README 文案。"""
    lines = text.split("\n")
    rendered = []
    index = 0
    while index < len(lines):
        line = lines[index]
        if PHASE_ZERO_RE.match(line):
            rendered.append(render_zh_catalog_line(line, phases, lessons))
            index += 1
            description = []
            while index < len(lines):
                blockquote = BLOCKQUOTE.match(lines[index])
                if not (blockquote and is_prose(blockquote.group(1))):
                    break
                description.append(blockquote.group(1))
                index += 1
            if description:
                _, phase = _phase_by_number(phases, 0)
                rendered.append(f"> {phase['description']}")
            continue
        rendered.append(render_zh_catalog_line(line, phases, lessons))
        index += 1
    return "\n".join(rendered)


def render_complete_zh(text, translations, phases, lessons, structural):
    """渲染正文及所有已注册的可见 README 结构行。"""
    body = render_zh_catalog(text, phases, lessons)
    body = render(body, "zh", translations)
    line_translations = structural.get("lineTranslations", {})
    rendered = []
    in_code = False
    for line in body.split("\n"):
        if FENCE.match(line):
            in_code = not in_code
        elif not in_code:
            line = line_translations.get(line, line)
        rendered.append(line)
    return "\n".join(rendered)


def phase_zero_catalog_span_keys(text):
    """返回中文文案来自 phase catalog 的正文 key。"""
    keys = set()
    lines = text.split("\n")
    for index, line in enumerate(lines):
        match = PHASE_ZERO_RE.match(line)
        if not match:
            continue
        heading = HEADING.match(line)
        keys.add(block_key([heading.group(2)]))
        description = []
        index += 1
        while index < len(lines):
            blockquote = BLOCKQUOTE.match(lines[index])
            if not (blockquote and is_prose(blockquote.group(1))):
                break
            description.append(blockquote.group(1))
            index += 1
        if description:
            keys.add(block_key(description))
    return keys


def zh_structural_candidates(text):
    """返回普通正文/catalog span 之外的可见英文源行。"""
    covered = {
        index
        for span in spans(text)
        for index in range(span["start"], span["end"])
    }
    candidates = set()
    in_fence = False
    for index, line in enumerate(text.split("\n")):
        if FENCE.match(line):
            in_fence = not in_fence
            continue
        if in_fence or index in covered:
            continue
        stripped = line.strip()
        if (
            not stripped
            or stripped.startswith("<!--")
            or re.fullmatch(r"[|:\- ]+", stripped)
            or LESSON_ROW_RE.match(line)
            or line in CATALOG_HEADER_TRANSLATIONS
            or PHASE_ZERO_RE.match(line)
            or PHASE_SUMMARY_RE.match(line)
        ):
            continue
        visible = HTML_ALT_RE.sub(lambda match: " " + match.group(1) + " ", line)
        visible = HTML_TAG_RE.sub(" ", visible)
        if re.search(r"[A-Za-z]{3,}", visible):
            candidates.add(line)
    return candidates


def validate_zh_assets(text, phases, lessons, structural):
    """返回 zh README 清单 key 缺失、多余或过期的错误。"""
    errors = []
    expected_lessons = set()
    for line in text.splitlines():
        match = LESSON_ROW_RE.match(line)
        if match:
            href = match.group(3)
            expected_lessons.add(href if href.endswith("/") else href + "/")
    expected_phases = {path.parents[2].name for path in ROOT.glob("phases/*/*/docs/en.md")}
    missing_lessons = sorted(expected_lessons - set(lessons))
    extra_lessons = sorted(set(lessons) - expected_lessons)
    missing_phases = sorted(expected_phases - set(phases))
    extra_phases = sorted(set(phases) - expected_phases)
    if missing_lessons:
        errors.append(f"缺少 {len(missing_lessons)} 个 zh 课程标题：{missing_lessons[:5]}")
    if extra_lessons:
        errors.append(f"存在 {len(extra_lessons)} 个过期 zh 课程标题：{extra_lessons[:5]}")
    if missing_phases:
        errors.append(f"缺少 {len(missing_phases)} 个 zh 阶段条目：{missing_phases}")
    if extra_phases:
        errors.append(f"存在 {len(extra_phases)} 个过期 zh 阶段条目：{extra_phases}")
    source_lines = set(text.splitlines())
    line_translations = structural.get("lineTranslations", {})
    allowed = set(structural.get("allowedEnglishLines", []))
    candidates = zh_structural_candidates(text)
    classified = set(line_translations) | allowed
    missing_lines = sorted(candidates - classified)
    stale_lines = sorted(classified - source_lines)
    overlap = sorted(set(line_translations) & allowed)
    empty_lines = sorted(key for key, value in line_translations.items() if not value.strip())
    if missing_lines:
        errors.append(
            f"缺少 {len(missing_lines)} 个 zh 可见行决策："
            f"{missing_lines[:5]}"
        )
    if stale_lines:
        errors.append(f"存在 {len(stale_lines)} 个过期 zh 结构行 key：{stale_lines[:5]}")
    if overlap:
        errors.append(f"conflicting {len(overlap)} zh structural line key(s): {overlap[:5]}")
    if empty_lines:
        errors.append(f"empty {len(empty_lines)} zh structural translation(s): {empty_lines[:5]}")
    return errors


def translation_coverage_issues(text, lang, translations):
    """返回完整语言中缺失和过期的精确 span key。

    空译文值按缺失处理，因为 ``render`` 会有意把它们视为回退到规范英文文本的请求。
    当前源代码的精确行可作为有效补充 key，因为 ``render`` 也支持 fenced code block
    之外的逐行翻译。
    """
    required = {sp["key"] for sp in spans(text)}
    if lang == "zh":
        required -= phase_zero_catalog_span_keys(text)
    table = translations.get(lang, {})
    missing = sorted(key for key in required if not table.get(key))
    supplemental_lines = zh_structural_candidates(text) if lang == "zh" else set()
    stale = sorted(set(table) - required - supplemental_lines)
    return missing, stale


def validate_complete_translations(text, translations):
    """完整语言发生漂移时返回人类可读错误。"""
    failures = []
    for lang in COMPLETE_LANGUAGES:
        missing, stale = translation_coverage_issues(text, lang, translations)
        if not missing and not stale:
            continue
        lines = [f"{lang!r} 的 README 翻译覆盖错误："]
        if missing:
            lines.append(f"  缺少 {len(missing)} 个当前 span key：")
            lines.extend(f"    - {key!r}" for key in missing)
        if stale:
            lines.append(f"  存在 {len(stale)} 个过期翻译 key：")
            lines.extend(f"    - {key!r}" for key in stale)
        failures.append("\n".join(lines))
    if not failures:
        return ""
    return (
        "\n".join(failures)
        + "\nTRANSLATIONS['zh'] 必须精确覆盖当前 README span key；"
          "拒绝回退到英文。"
    )


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", action="store_true")
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args(argv)
    text = README.read_text(encoding="utf-8")

    assert render(text, "en", {}) == text, "生成器未做到结构无损"

    if args.dump:
        keys = [sp["key"] for sp in spans(text)]
        for k in keys:
            print(f"- {k}")
        print(f"\n{len(keys)} 个可翻译 block；往返一致性正常", file=sys.stderr)
        return 0

    from readme_translations import TRANSLATIONS, README_NOTE

    coverage_error = validate_complete_translations(text, TRANSLATIONS)
    if coverage_error:
        print(coverage_error, file=sys.stderr)
        return 1

    try:
        zh_phases, zh_lessons, zh_structural = load_zh_assets()
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"简体中文 README 资源错误：{exc}", file=sys.stderr)
        return 1
    zh_asset_errors = validate_zh_assets(
        text, zh_phases, zh_lessons, zh_structural
    )
    if zh_asset_errors:
        print(
            "简体中文 README 覆盖错误：\n  - "
            + "\n  - ".join(zh_asset_errors),
            file=sys.stderr,
        )
        return 1

    stale = []
    for lang in TRANSLATIONS:
        note = README_NOTE.get(lang, "")
        if lang == "zh":
            body = render_complete_zh(
                text, TRANSLATIONS, zh_phases, zh_lessons, zh_structural
            )
        else:
            body = render(text, lang, TRANSLATIONS)
        body = localize_links(body)
        content = f"{note}\n{body}" if note else body
        dst = OUT_ROOT / lang / "README.md"
        if args.check:
            if not dst.is_file() or dst.read_text(encoding="utf-8") != content:
                stale.append(lang)
        else:
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_text(content, encoding="utf-8")
            print(f"已写入 {dst.relative_to(ROOT)}")
    if args.check and stale:
        print(f"README 译文已过期：{stale}；请运行 build_readme_i18n.py", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
