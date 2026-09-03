#!/usr/bin/env python3
"""将课程组装为分卷图书，并使用 pandoc 渲染。

Usage:
    python3 scripts/build_book.py                 # assemble + epub for all volumes
    python3 scripts/build_book.py --volume language
    python3 scripts/build_book.py --pdf           # also render PDF (xelatex)
    python3 scripts/build_book.py --assemble-only # markdown only, no pandoc

本书特意作为仓库与网站的配套内容，而非替代品。交互图、测验和可运行代码
仍保留在线版本；每章末尾都提供前往这些内容的链接。
"""

import argparse
import functools
import hashlib
import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_catalog import LESSON_DIR_RE, read_h1, slug_to_title  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
PHASES = ROOT / "phases"
BUILD = ROOT / "book" / "_build"
DIST = ROOT / "dist" / "book"

CONFIG = json.loads((ROOT / "book" / "volumes.json").read_text(encoding="utf-8"))
SITE = CONFIG["site"].rstrip("/")
REPO = CONFIG["repo"].rstrip("/")

FENCE = re.compile(r"^ {0,3}```(?P<info>.*)$")
ASSET_IMG = re.compile(r"\]\(\.\./assets/")
HEADING2 = re.compile(r"^## ")
CANONICAL_H2_KIND_BY_TITLE = {
    "Ship It": "artifact",
    "Shipped Artifact": "artifact",
    "Exercises": "practice",
    "Practice Lab": "practice"
}
BOOK_SECTION_TITLE_ALIASES = {
    "ar": {
        "artifact": frozenset({"أرسله", "الأثاث المُرسل"}),
        "practice": frozenset({"التمارين", "مختبر التدريب"}),
    },
    "es": {
        "artifact": frozenset({"Envío", "Artículo enviado"}),
        "practice": frozenset({"Los ejercicios", "Laboratorio de práctica"}),
    },
    "fr": {
        "artifact": frozenset({"La faire partir", "Artéfact expédié"}),
        "practice": frozenset({"Exercices", "Laboratoire de pratique"}),
    },
    "hi": {
        "artifact": frozenset({"इसे भेजें", "शिप की गई कलाकृतियाँ"}),
        "practice": frozenset({"व्यायाम", "अभ्यास प्रयोगशाला"}),
    },
    "tr": {
        "artifact": frozenset({"Gönder", "Nakliye edilen Sanatlı"}),
        "practice": frozenset({"Egzersizler", "Pratik Laboratuvar"}),
    },
    "pt": {
        "artifact": frozenset({"Envia-o", "Artefato enviado"}),
        "practice": frozenset({"Exercícios", "Laboratório de prática"}),
    },
    "vi": {
        "artifact": frozenset({"Chuyển nó", "Hiện vật đã vận chuyển"}),
        "practice": frozenset({"Các bài tập", "Phòng thực hành"}),
    },
    "zh": {
        "artifact": frozenset({
            "交付成果",
            "交付物",
            "交付它",
            "交付上线",
            "交付产物",
            "产出",
            "放进系统里",
        }),
        "practice": frozenset({
            "练习",
            "动手练习",
            "实践实验",
        }),
    },
}

MERMAID_OK = shutil.which("mmdc") is not None
ROMAN = ["I", "II", "III", "IV", "V", "VI", "VII", "VIII"]


def lesson_dirs(phase):
    base = PHASES / phase
    if not base.is_dir():
        return []
    return [
        d
        for d in sorted(base.iterdir())
        if d.is_dir() and LESSON_DIR_RE.match(d.name) and (d / "docs" / "en.md").is_file()
    ]


def phase_title(phase):
    return read_h1(PHASES / phase / "README.md") or slug_to_title(phase.split("-", 1)[-1])


def urls_for(phase, lesson):
    rel = f"phases/{phase}/{lesson}"
    return {
        "web": f"{SITE}/lesson?path={rel}",
        "code": f"{REPO}/tree/main/{rel}/code",
        "repo": f"{REPO}/tree/main/{rel}",
    }


def fenced_div(cls, *lines):
    return ["", "::: {." + cls + "}", *lines, ":::", ""]


def continue_box(u, has_quiz):
    lines = [
        "**Continue online.** The living edition of this chapter has more than the page can hold:",
        "",
        f"- Animated, interactive figures and the web text: <{u['web']}>",
        f"- Runnable code for every step: <{u['code']}>",
    ]
    if has_quiz:
        lines.append(f"- The chapter quiz, graded in the browser: <{u['web']}>")
    lines += [
        "",
        "The repository moves faster than any printing. When the book and the repo disagree, trust the repo.",
    ]
    return fenced_div("continue-online", *lines)


def fence_end(src, i):
    """返回关闭 src[i] 所开 fence 的行索引；未关闭时返回 len(src)。"""
    j = i + 1
    while j < len(src) and src[j].strip() != "```":
        j += 1
    return j


BOOK_LANG = "en"  # 由 --lang 设置；存在译文时选择翻译后的来源


def _lesson_source(phase, lesson, source_root=ROOT, book_lang=None):
    book_lang = BOOK_LANG if book_lang is None else book_lang
    en = source_root / "phases" / phase / lesson / "docs" / "en.md"
    if book_lang != "en":
        tr = source_root / "i18n" / book_lang / "phases" / phase / lesson / "docs" / f"{book_lang}.md"
        if tr.is_file():
            return tr
    return en


def translation_coverage(vol, book_lang=None):
    """返回一个分卷及所选语言的 ``(localized, total)``。"""
    book_lang = BOOK_LANG if book_lang is None else book_lang
    localized = total = 0
    for phase in vol["phases"]:
        for lesson_dir in lesson_dirs(phase):
            total += 1
            canonical = lesson_dir / "docs" / "en.md"
            if _lesson_source(phase, lesson_dir.name, ROOT, book_lang) != canonical:
                localized += 1
    return localized, total


def require_translation_coverage(vol, book_lang=None):
    """拒绝语言标记错误的版本，并报告逐课程 fallback。"""
    book_lang = BOOK_LANG if book_lang is None else book_lang
    if book_lang == "en":
        return
    localized, total = translation_coverage(vol, book_lang)
    if total and localized == 0:
        raise SystemExit(
            f"分卷 {vol['slug']}：未找到 {book_lang} 课程译文；"
            "请先恢复配置的翻译分支再构建"
        )
    if localized < total:
        raise SystemExit(
            f"分卷 {vol['slug']}：{book_lang} 翻译覆盖不完整 "
            f"({localized}/{total})；请先恢复配置的翻译分支再构建"
        )


def require_translation_provenance(volumes, book_lang=None):
    """写入任何图书输出前，审计所有选中的已翻译 phase。"""
    book_lang = BOOK_LANG if book_lang is None else book_lang
    if book_lang == "en":
        return

    # 避免让英文构建路径依赖翻译流水线。audit_translations 是规范的 cache/provenance
    # 实现，因此图书构建器有意委托给它，而不是自行解析 cache 记录。
    import audit_translations as translation_audit

    source = translation_audit.LocalTranslationSource(ROOT)
    phases = dict.fromkeys(
        phase for volume in volumes for phase in volume["phases"]
    )
    for phase in phases:
        try:
            result = translation_audit.audit_translations(
                ROOT, book_lang, source, phase
            )
        except (translation_audit.TranslationSourceError, ValueError) as exc:
            raise SystemExit(
                f"{book_lang} 阶段 {phase} 的翻译预检失败：{exc}"
            ) from exc
        if result.issues:
            raise SystemExit(
                f"{book_lang} 阶段 {phase} 的翻译预检失败：\n"
                f"{translation_audit.render_report(result)}"
            )


def _canonical_h2_kinds(phase, lesson, source_root=ROOT):
    """将每个二级标题映射到与语言无关的图书角色。

    译文课程保留规范标题顺序，但会自然地本地化 "Ship It"、"Exercises" 等标题。
    图书转换应依据匹配的英文标题驱动，无需每份译文都保留这两个英文标签。
    """
    source = source_root / "phases" / phase / lesson / "docs" / "en.md"
    return [
        CANONICAL_H2_KIND_BY_TITLE.get(title)
        for title in _h2_titles(source.read_text(encoding="utf-8").splitlines())
    ]


def _localized_h2_kind(title, book_lang):
    """在不要求英文标题的情况下解析特殊图书角色。"""
    canonical_kind = CANONICAL_H2_KIND_BY_TITLE.get(title)
    if canonical_kind is not None:
        return canonical_kind
    for kind, aliases in BOOK_SECTION_TITLE_ALIASES.get(book_lang, {}).items():
        if title in aliases:
            return kind
    return None


def _validate_h2_sections(canonical_lines, localized_lines, source, book_lang):
    """本地化章节无法安全对齐时按失败处理。

    翻译审计把标题顺序和数量作为契约。标题可以翻译，但特殊图书角色使用显式的逐语言
    alias 契约，因此等量删除/插入或重排不会悄悄把规范角色分配给错误的 H2。
    未知语言默认失败，除非这些特殊标题保留其规范英文标题。
    """
    canonical_titles = _h2_titles(canonical_lines)
    localized_titles = _h2_titles(localized_lines)
    if len(localized_titles) != len(canonical_titles):
        raise ValueError(
            f"{source} 的 H2 结构不匹配：预期 "
            f"{len(canonical_titles)} 个 H2 标题，实际为 {len(localized_titles)} 个"
        )

    canonical_roles = tuple(
        (index, title, kind)
        for index, title in enumerate(canonical_titles, start=1)
        if (kind := CANONICAL_H2_KIND_BY_TITLE.get(title)) is not None
    )
    expected = tuple((index, kind) for index, _, kind in canonical_roles)
    localized_roles = tuple(
        (
            index,
            localized_titles[index - 1],
            _localized_h2_kind(localized_titles[index - 1], book_lang),
        )
        for index, _, _ in canonical_roles
    )
    found = tuple((index, kind) for index, _, kind in localized_roles)
    if found != expected:
        raise ValueError(
            f"{source} 的 H2 章节不匹配：预期规范特殊章节 "
            f"{canonical_roles!r}，实际本地化特殊章节为 "
            f"{localized_roles!r}"
        )
    return [CANONICAL_H2_KIND_BY_TITLE.get(title) for title in canonical_titles]


def _h2_titles(lines):
    """返回 fenced code block 之外的二级标题。"""
    titles = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if FENCE.match(line):
            i = fence_end(lines, i) + 1
            continue
        i += 1
        if not HEADING2.match(line):
            continue
        titles.append(line[3:].strip())
    return titles


def transform_lesson(phase, lesson_dir, source_root=ROOT, book_lang=None):
    lesson = lesson_dir.name
    u = urls_for(phase, lesson)
    has_quiz = (lesson_dir / "quiz.json").is_file()
    source = _lesson_source(phase, lesson, source_root, book_lang)
    src = source.read_text(encoding="utf-8").splitlines()
    canonical_source = (
        source_root / "phases" / phase / lesson / "docs" / "en.md"
    )
    canonical_src = canonical_source.read_text(encoding="utf-8").splitlines()
    canonical_h2_kinds = _validate_h2_sections(
        canonical_src, src, source, book_lang or BOOK_LANG
    )

    out = []
    balanced = True
    h2_index = 0
    i = 0
    while i < len(src):
        line = src[i]

        if FENCE.match(line):
            end = fence_end(src, i)
            if end >= len(src):
                balanced = False
            info = FENCE.match(line).group("info").strip()
            block = src[i + 1 : end]
            if info == "figure":
                fig_id = block[0].strip() if block else "figure"
                out += fenced_div(
                    "interactive-figure",
                    f"**Interactive figure: `{fig_id}`.** This one moves. Watch it animate and drag its controls in the web edition: <{u['web']}>",
                )
            elif info == "mermaid":
                rendered = render_mermaid(block)
                if rendered:
                    out += ["", f"![diagram]({rendered})", ""]
                else:
                    out += fenced_div(
                        "interactive-figure",
                        f"**Diagram.** Rendered live in the web edition: <{u['web']}>",
                    )
            else:
                out += src[i : end + 1]
            i = end + 1
            continue

        section_kind = None
        if HEADING2.match(line):
            if h2_index < len(canonical_h2_kinds):
                section_kind = canonical_h2_kinds[h2_index]
            h2_index += 1

        if section_kind == "artifact":
            out += fenced_div(
                "continue-online",
                f"**This chapter ships an artifact.** The course version of this lesson produces a reusable prompt or agent skill. It lives in the repository, ready to install: <{u['repo']}>",
            )
            i += 1
            while i < len(src):
                if FENCE.match(src[i]):
                    end = fence_end(src, i)
                    if end >= len(src):
                        balanced = False
                    i = end + 1
                    continue
                if HEADING2.match(src[i]):
                    break
                i += 1
            continue

        if section_kind == "practice":
            out.append(line)
            out.append("")
            out.append(f"Starter code and the lesson's working implementation: <{u['code']}>")
            i += 1
            continue

        out.append(ASSET_IMG.sub(f"](phases/{phase}/{lesson}/assets/", line))
        i += 1

    if h2_index != len(canonical_h2_kinds):
        raise ValueError(
            f"{source} 的 H2 结构不匹配：预期转换 "
            f"{len(canonical_h2_kinds)} 个 H2 标题，实际为 {h2_index} 个"
        )

    if not balanced:
        raise ValueError(f"{lesson_dir / 'docs' / 'en.md'} 中的代码 fence 未闭合")

    out += continue_box(u, has_quiz)
    return out


def _transform_lesson_fixture(fixture):
    """使用隔离的课程源执行生产图书转换。"""
    canonical = fixture.get("canonical")
    localized = fixture.get("localized")
    if not isinstance(canonical, str) or not isinstance(localized, str):
        raise ValueError("fixture 的 canonical 和 localized 字段必须是字符串")

    phase = "99-book-transform-fixture"
    lesson = "01-localized-sections"
    book_lang = fixture.get("lang", "zh")
    if not isinstance(book_lang, str):
        raise ValueError("fixture 的 lang 必须是字符串")
    with tempfile.TemporaryDirectory(prefix="build-book-fixture-") as temp_dir:
        source_root = Path(temp_dir)
        lesson_dir = source_root / "phases" / phase / lesson
        docs_dir = lesson_dir / "docs"
        docs_dir.mkdir(parents=True)
        (docs_dir / "en.md").write_text(canonical, encoding="utf-8")

        localized_doc = (
            source_root
            / "i18n"
            / book_lang
            / "phases"
            / phase
            / lesson
            / "docs"
            / f"{book_lang}.md"
        )
        localized_doc.parent.mkdir(parents=True)
        localized_doc.write_text(localized, encoding="utf-8")

        return {
            "canonicalH2Kinds": _canonical_h2_kinds(phase, lesson, source_root),
            "transformed": "\n".join(
                transform_lesson(
                    phase,
                    lesson_dir,
                    source_root=source_root,
                    book_lang=book_lang,
                )
            ),
        }


@functools.lru_cache(maxsize=None)
def font_families():
    if not shutil.which("fc-list"):
        return frozenset()
    r = subprocess.run(["fc-list", ":", "family"], capture_output=True, text=True)
    return frozenset(
        fam.strip() for fam_line in r.stdout.splitlines() for fam in fam_line.split(",")
    )


def pick_font(candidates):
    families = font_families()
    for c in candidates:
        if c in families:
            return c
    return None


def render_mermaid(block):
    if not MERMAID_OK:
        return None
    assets = BUILD / "diagrams"
    assets.mkdir(parents=True, exist_ok=True)
    stem = hashlib.sha1("\n".join(block).encode()).hexdigest()[:16]
    svg = assets / f"{stem}.svg"
    if svg.is_file():
        return str(svg.relative_to(ROOT))
    mmd = assets / f"{stem}.mmd"
    mmd.write_text("\n".join(block), encoding="utf-8")
    try:
        subprocess.run(
            ["mmdc", "-i", str(mmd), "-o", str(svg), "-b", "transparent", "--quiet"],
            check=True, capture_output=True, timeout=60,
        )
        return str(svg.relative_to(ROOT))
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or b"").decode(errors="replace").strip()[:300]
        print(f"警告：{mmd.name} 的 Mermaid 渲染失败：{detail}", file=sys.stderr)
        return None
    except subprocess.TimeoutExpired:
        print(f"警告：{mmd.name} 的 Mermaid 渲染超时", file=sys.stderr)
        return None


@functools.lru_cache(maxsize=None)
def git_date():
    return subprocess.run(
        ["git", "log", "-1", "--format=%cs"], capture_output=True, text=True, cwd=ROOT
    ).stdout.strip()


@functools.lru_cache(maxsize=None)
def git_edition():
    return subprocess.run(
        ["git", "log", "-1", "--format=%cd", "--date=format:%Y.%m"],
        capture_output=True, text=True, cwd=ROOT,
    ).stdout.strip() or "0000.00"


@functools.lru_cache(maxsize=None)
def titlepage_template(book_lang=None):
    book_lang = BOOK_LANG if book_lang is None else book_lang
    if book_lang == "en":
        return r"""\begin{titlepage}
\thispagestyle{empty}
\vspace*{0.1in}
\noindent{\ttfamily\bfseries\color{blueprint}\small VOLUME\ @VOLNUM3@\ \ —\ \ REFERENCE\ MANUAL}\hfill{\ttfamily\color{inkmute}\small \textcopyright\ 2026\ \ \textperiodcentered\ \ OPEN\ SOURCE\ \ \textperiodcentered\ \ MIT\ LICENSE}\\[6pt]
\noindent{\color{blueprint}\rule{\textwidth}{0.8pt}}\\[64pt]
\noindent{\ttfamily\bfseries\color{blueprint}\fontsize{38}{44}\selectfont AI\ ENGINEERING}\\[6pt]
\noindent{\ttfamily\bfseries\color{ink}\fontsize{38}{44}\selectfont FROM\ SCRATCH.}\\[40pt]
\noindent{\Large\color{ink}\itshape @TITLE@: @SUBTITLE@}\\[14pt]
\noindent{\ttfamily\color{inksoft}\small VOLUME\ @ROMAN@\ /\ OF\ @TOTALVOL@\ \ \textperiodcentered\ \ @CHAPTERS@\ CHAPTERS\ \ \textperiodcentered\ \ PHASES\ @PHASES@}
\vfill
\noindent{\ttfamily\footnotesize\color{inkmute}EDITION\ @EDITION@\ \ \textperiodcentered\ \ A\ SNAPSHOT\ OF\ A\ LIVING\ CURRICULUM\ \ \textperiodcentered\ \ THE\ LINKS\ BELOW\ ALWAYS\ POINT\ TO\ THE\ LATEST\ BUILD}\\[6pt]
\noindent{\color{blueprint}\rule{\textwidth}{0.8pt}}\\[10pt]
\noindent{\ttfamily\large\href{https://aiengineeringfromscratch.com}{\color{blueprint}aiengineeringfromscratch.com}}\hfill{\ttfamily\footnotesize\color{inkmute}\href{https://github.com/rohitg00/ai-engineering-from-scratch}{github.com/rohitg00/ai-engineering-from-scratch}}
\end{titlepage}
"""
    return (ROOT / "book" / "titlepage.tex").read_text(encoding="utf-8")


def clean_phase_title(raw):
    return re.sub(r"^Phase\s+\d+\s*[:—-]\s*", "", raw).strip()


def series_map(vol):
    rows = []
    for v in CONFIG["volumes"]:
        marker = "**" if v["slug"] == vol["slug"] else ""
        phases = ", ".join(p.split("-")[0] for p in v["phases"])
        rows.append(f"| {marker}{v['number']}{marker} | {marker}{v['title']}{marker} — {v['subtitle']} | {phases} |")
    return "\n".join([
        "| Vol | Title | Course phases |",
        "|-----|-------|---------------|",
    ] + rows)


def how_to_use(vol):
    return f"""# About This Volume {{.unnumbered}}

This is Volume {vol['number']} of *{CONFIG['series']}*, a six-volume compilation of the open course of the same name. Each volume stands alone; cross-references cite course phase numbers, which map to volumes like this:

{series_map(vol)}

The chapters in this volume come from course phases {', '.join(p.split('-')[0] for p in vol['phases'])}. Chapter prerequisites name phases, not volumes; use the table above to translate.

# How to Use This Book {{.unnumbered}}

This volume is one loop of a larger machine, and it works best when you run the whole loop:

1. **Read the chapter here.** The prose, the derivations, and the code walkthroughs are complete on the page.
2. **Run the code from the repository.** Every chapter has a `code/` directory with a working implementation you can run and break: <{REPO}>
3. **Open the web edition for what paper cannot do.** Animated figures you can watch and drag, and a quiz per chapter that grades itself: <{SITE}>

The repository is the living edition. Lessons are updated as the field moves; the book is a snapshot with a version number. When they disagree, the repo is right.

## Learning with an AI {{.unnumbered}}

This course is built to be read by agents as well as people. The machine-readable index of every lesson lives at <{SITE}/llms.txt>. If you learn with an AI assistant, paste this and go:

> I am working through *{CONFIG["series"]}, Volume {vol["number"]}: {vol["title"]}*. Fetch {SITE}/llms.txt, find the lesson I name, and act as my tutor: quiz me on its Key Terms, review my solutions to its Exercises, and walk me through its code from the repository.
"""


def assemble(vol):
    require_translation_coverage(vol)
    BUILD.mkdir(parents=True, exist_ok=True)
    parts = [how_to_use(vol)]
    chapters = 0
    for part_idx, phase in enumerate(vol["phases"]):
        title = clean_phase_title(phase_title(phase))
        parts.append(
            f"\n# Part {ROMAN[part_idx]} — {title} {{.unnumbered .part}}\n\n"
            f"*Course phase {phase.split('-')[0]}. Live edition with animated figures and quizzes: <{SITE}/catalog.html>*\n"
        )
        for lesson_dir in lesson_dirs(phase):
            parts.append("\n".join(transform_lesson(phase, lesson_dir)))
            chapters += 1
    text = "\n\n".join(parts)
    md = BUILD / f"{vol['slug']}.md"
    md.write_text(text, encoding="utf-8")
    return md, chapters, len(text.split())


def metadata(vol, book_lang="en"):
    meta = BUILD / f"{vol['slug']}-meta.yaml"
    meta.write_text(
        "---\n"
        f"title: \"{CONFIG['series']}\"\n"
        f"subtitle: \"Volume {vol['number']} — {vol['title']}: {vol['subtitle']}\"\n"
        f"author: \"{CONFIG['author']}\"\n"
        f"lang: {book_lang}\n"
        "toc-title: Contents\n"
        "---\n",
        encoding="utf-8",
    )
    return meta


def render(vol, md, chapters, pdf=False):
    DIST.mkdir(parents=True, exist_ok=True)
    meta = metadata(vol, BOOK_LANG)
    suffix = "" if BOOK_LANG == "en" else f"-{BOOK_LANG}"
    epub = DIST / f"aiefs-vol{vol['number']}-{vol['slug']}{suffix}.epub"
    cmd = [
        "pandoc", str(meta), str(md),
        "-o", str(epub),
        "--from", "markdown+fenced_divs",
        "--toc", "--toc-depth=1",
        "--top-level-division=chapter",
        "--css", str(ROOT / "book" / "epub.css"),
        "--resource-path", str(ROOT),
        "--metadata", f"date={git_date()}",
    ]
    subprocess.run(cmd, check=True, cwd=ROOT)
    results = [epub]
    if pdf and BOOK_LANG in ("ar", "fa", "ur", "he"):
        # 从右到左文字需要 bidi 引擎及 xelatex 主题未提供的阿拉伯语/希伯来语字体；
        # 上方 EPUB 原生处理 RTL，因此跳过 PDF，避免产出错误的从左到右版本。
        print(f"提示：跳过 {vol['slug']} 的 {BOOK_LANG} PDF（PDF 尚未接入 RTL）；已生成 EPUB", file=sys.stderr)
        pdf = False
    if pdf:
        titlepage = BUILD / f"{vol['slug']}-titlepage.tex"
        titlepage.write_text(
            titlepage_template(BOOK_LANG)
            .replace("@VOLNUM3@", f"{vol['number']:03d}")
            .replace("@EDITION@", git_edition())
            .replace("@ROMAN@", ROMAN[vol["number"] - 1])
            .replace("@TOTALVOL@", ROMAN[len(CONFIG["volumes"]) - 1])
            .replace("@CHAPTERS@", str(chapters))
            .replace("@PHASES@", "\\ \\textperiodcentered\\ ".join(p.split("-")[0] for p in vol["phases"]))
            .replace("@TITLE@", vol["title"])
            .replace("@SUBTITLE@", vol["subtitle"]),
            encoding="utf-8",
        )
        pdf_out = DIST / f"aiefs-vol{vol['number']}-{vol['slug']}{suffix}.pdf"
        cmd_pdf = [
            "pandoc", str(md),
            "-o", str(pdf_out),
            "--from", "markdown+fenced_divs",
            "--toc", "--toc-depth=1",
            "--top-level-division=chapter",
            "--pdf-engine=xelatex",
            "--columns=40",
            "--resource-path", str(ROOT),
            "--include-in-header", str(ROOT / "book" / "theme.tex"),
            "--include-before-body", str(titlepage),
            "-M", f"title-meta={CONFIG['series']} Volume {vol['number']}: {vol['title']}",
            "-M", "author-meta=aiengineeringfromscratch.com",
            "-M", f"lang={BOOK_LANG}",
            "-V", "toc-title=Contents",
            "-V", "documentclass=book",
            "-V", "classoption=oneside,openany",
            "-V", "geometry=margin=1in",
            "-V", "fontsize=10pt",
        ]
        serif = pick_font(["DejaVu Serif", "STIX Two Text", "Georgia"])
        mono = pick_font(["DejaVu Sans Mono", "Menlo", "Consolas"])
        if serif:
            cmd_pdf += ["-V", f"mainfont={serif}"]
        if mono:
            cmd_pdf += ["-V", f"monofont={mono}"]
        # CJK 文字需要匹配字体；DejaVu 已覆盖其他语言所需的
        # Latin/Cyrillic/Greek/Devanagari 字符。
        cjk_candidates = {
            "zh": ["Noto Sans CJK SC", "Noto Serif CJK SC", "Source Han Serif SC"],
            "zh-TW": ["Noto Sans CJK TC", "Noto Serif CJK TC", "Source Han Serif TC"],
            "ja": ["Noto Sans CJK JP", "Noto Serif CJK JP", "Source Han Serif JP"],
            "ko": ["Noto Sans CJK KR", "Noto Serif CJK KR", "Source Han Serif KR"],
        }
        if BOOK_LANG in cjk_candidates:
            cjk = pick_font(cjk_candidates[BOOK_LANG])
            if cjk:
                cmd_pdf += ["-V", f"CJKmainfont={cjk}"]
        try:
            subprocess.run(cmd_pdf, check=True, cwd=ROOT)
            results.append(pdf_out)
        except subprocess.CalledProcessError:
            print(f"警告：{vol['slug']} 的 PDF 渲染失败（非致命错误）", file=sys.stderr)
    return results


def check_phases():
    claimed = set()
    for vol in CONFIG["volumes"]:
        for phase in vol["phases"]:
            claimed.add(phase)
            if not (PHASES / phase).is_dir() or not lesson_dirs(phase):
                sys.exit(f"分卷 {vol['slug']}：阶段 {phase} 缺失或没有课程")
    for d in sorted(PHASES.iterdir()):
        if d.is_dir() and d.name not in claimed:
            print(f"警告：阶段目录 {d.name} 不属于任何分卷", file=sys.stderr)


def main():
    global BOOK_LANG
    ap = argparse.ArgumentParser()
    ap.add_argument("--volume", help="按 slug 构建一个分卷")
    ap.add_argument("--pdf", action="store_true", help="同时通过 xelatex 渲染 PDF")
    ap.add_argument("--assemble-only", action="store_true", help="跳过 pandoc")
    ap.add_argument(
        "--lang",
        default="en",
        help="从 i18n/<lang>/ 构建经过完整审计的版本",
    )
    ap.add_argument("--test-transform-fixture", action="store_true",
                    help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.test_transform_fixture:
        json.dump(_transform_lesson_fixture(json.load(sys.stdin)), sys.stdout, ensure_ascii=False)
        sys.stdout.write("\n")
        return

    BOOK_LANG = args.lang

    check_phases()

    vols = CONFIG["volumes"]
    if args.volume:
        vols = [v for v in vols if v["slug"] == args.volume]
        if not vols:
            sys.exit(f"未知分卷：{args.volume}")

    require_translation_provenance(vols)

    for vol in vols:
        md, chapters, words = assemble(vol)
        print(f"分卷 {vol['number']} {vol['slug']}：{chapters} 章，{words:,} 词 -> {md}")
        if not args.assemble_only:
            for artifact in render(vol, md, chapters, pdf=args.pdf):
                size = artifact.stat().st_size // 1024
                print(f"  {artifact} ({size} KB)")


if __name__ == "__main__":
    main()
