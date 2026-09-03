"""文档 AI 栈示例——LayoutLMv3 风格输入、Donut schema 与 token 预算。

标准库实现。为示例页面生成三流 LayoutLM 输入（text、bbox、patch-ids），生成 Donut 风格 JSON schema，并比较 OCR 流水线、Donut、Nougat 与原生 VLM 的总输入 token 数。
"""

from __future__ import annotations

import json
from dataclasses import dataclass


@dataclass
class Token:
    text: str
    bbox: tuple[int, int, int, int]


def mock_page() -> list[Token]:
    """一张合成的发票页面。"""
    return [
        Token("发票",         (100, 50,  300, 80)),
        Token("ACME Co.",     (100, 100, 250, 130)),
        Token("商品",         (100, 200, 200, 230)),
        Token("小部件 A",     (100, 240, 250, 270)),
        Token("价格",         (400, 200, 500, 230)),
        Token("$120.00",      (400, 240, 500, 270)),
        Token("合计",         (400, 400, 500, 430)),
        Token("$1,245.00",    (400, 440, 550, 470)),
    ]


def layoutlm_input(tokens: list[Token], patch_grid: tuple[int, int] = (16, 16)) -> dict:
    """生成三流输入：text、bbox、patch-ids。"""
    text_ids = [hash(t.text) % 10000 for t in tokens]
    bbox_stream = [t.bbox for t in tokens]
    n_patches = patch_grid[0] * patch_grid[1]
    patch_ids = list(range(n_patches))
    return {"text_ids": text_ids, "bbox_stream": bbox_stream,
            "patch_ids": patch_ids}


def donut_schema(task: str = "invoice") -> dict:
    schemas = {
        "invoice": {
            "vendor": "<string>",
            "invoice_number": "<string>",
            "line_items": [
                {"description": "<string>", "quantity": "<int>", "price": "<float>"}
            ],
            "total": "<float>",
            "currency": "<string>",
        },
        "form": {
            "form_id": "<string>",
            "fields": [
                {"name": "<string>", "value": "<string>", "confidence": "<float>"}
            ],
        },
    }
    return schemas.get(task, {})


def token_budget() -> None:
    print("\n每页输入 TOKEN 预算（A4 分辨率 300 DPI，约 2500x3500 px）")
    print("-" * 60)
    rows = [
        ("OCR 流水线 + LayoutLMv3", 512, "文本 + bbox + 小图像"),
        ("Donut（无 OCR）",          4096, "Swin 编码器，约 4k 个 patch"),
        ("Nougat（论文页面）",       4096, "896x896，4 瓦片 AnyRes"),
        ("VLM AnyRes 4 瓦片（LLaVA）", 2916, "336 瓦片 + 缩略图"),
        ("原生 VLM 2048（Qwen2.5-VL）", 8192, "原生分辨率"),
        ("原生 VLM 2576（Claude 4.7）", 12000, "前沿方案，准确率最佳"),
    ]
    print(f"  {'技术栈':<28}{'token 数':<10}  注释")
    for name, toks, note in rows:
        print(f"  {name:<28}{toks:<10}  {note}")


def demo_pipeline_output() -> None:
    print("\nLAYOUTLMv3 风格输入（发票页面）")
    print("-" * 60)
    tokens = mock_page()
    data = layoutlm_input(tokens)
    print(f"  text_ids[0:4]    : {data['text_ids'][:4]}...")
    print(f"  bbox_stream[0:2] : {data['bbox_stream'][:2]}")
    print(f"  patch_ids 数量  : {len(data['patch_ids'])}")

    print("\nDONUT SCHEMA（发票）")
    print("-" * 60)
    schema = donut_schema("invoice")
    print(json.dumps(schema, indent=2))


def eras_table() -> None:
    print("\n文档 AI 的三个时代")
    print("-" * 60)
    rows = [
        ("时代 1：OCR 流水线",    "Tesseract, TrOCR, LayoutLMv3", "确定性"),
        ("时代 2：无 OCR",        "Donut, Nougat, DocLLM",         "泛化能力较弱"),
        ("时代 3：原生 VLM",      "Qwen2.5-VL, PaliGemma 2, Claude 4.7", "2026 年前沿方案"),
    ]
    for era, examples, trait in rows:
        print(f"  {era:<20}{examples:<36}{trait}")


def main() -> None:
    print("=" * 60)
    print("文档与图表理解（第 12 阶段，第 22 课）")
    print("=" * 60)

    demo_pipeline_output()
    token_budget()
    eras_table()

    print("\n配方选择器")
    print("-" * 60)
    print("  每天 1000 万张发票：OCR 流水线 + LayoutLMv3，成本低")
    print("  科学论文    : Nougat 处理数学公式，VLM 处理图表")
    print("  混合 + 手写  ：原生 VLM（PaliGemma 2 或 Qwen2.5-VL）")
    print("  受监管场景   ：OCR + VLM 交叉检查，可审计")


if __name__ == "__main__":
    main()
