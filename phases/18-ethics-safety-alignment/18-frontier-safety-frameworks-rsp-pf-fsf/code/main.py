"""前沿安全框架对比——仅使用 Python 标准库。

并排比较 Anthropic RSP v3.0、OpenAI PF v2 和 DeepMind FSF v3.0 的四个维度：
等级结构、CBRN 阈值、AI R&D 阈值和竞争者调整条款。

仅供参考，不进行模拟。主要来源在文内引用。

用法：python3 code/main.py
"""

from __future__ import annotations


LABS = [
    {
        "name": "Anthropic RSP v3.0（2026 年 2 月）",
        "tier_structure": "ASL-1 .. ASL-5+；类比生物安全等级",
        "cbrn_threshold": "ASL-3（2025 年 5 月启用）",
        "ai_rd_threshold": "AI R&D-2 + AI R&D-4（v3.0 中拆分）",
        "adjustment_clause": "有；允许因同行发布而降低要求",
        "safety_case": "跨越 AI R&D-4 时必须提交",
    },
    {
        "name": "OpenAI PF v2（2025 年 4 月 15 日）",
        "tier_structure": "各项跟踪能力分为 Low / Medium / High / Critical",
        "cbrn_threshold": "生物领域为 High",
        "ai_rd_threshold": "AI R&D 为 High；Critical 定义待定",
        "adjustment_clause": "有；领导层可以降低要求",
        "safety_case": "分别提交 Capabilities 和 Safeguards 报告",
    },
    {
        "name": "DeepMind FSF v3.0（2025 年 9 月）",
        "tier_structure": "各领域使用 CCL：生物/网络安全/ML R&D/操纵",
        "cbrn_threshold": "Bioweapon Uplift CCL",
        "ai_rd_threshold": "ML R&D Acceleration CCL（v2.0 提高安全等级）",
        "adjustment_clause": "有；2025 年新增",
        "safety_case": "按 CCL 提交；v2.0 新增 Deceptive Alignment 章节",
    },
]


def print_row(header: str, key: str) -> None:
    print(f"\n{header}")
    for lab in LABS:
        name = lab["name"]
        val = lab[key]
        print(f"  {name:32s} : {val}")


def main() -> None:
    print("=" * 78)
    print("前沿安全框架（阶段 18，第 18 课）")
    print("=" * 78)

    print_row("等级结构", "tier_structure")
    print_row("CBRN 阈值", "cbrn_threshold")
    print_row("AI R&D 阈值", "ai_rd_threshold")
    print_row("竞争者调整条款", "adjustment_clause")
    print_row("安全论证要求", "safety_case")

    print("\n" + "=" * 78)
    print("要点：三家实验室在结构上趋同——前沿能力分为三个等级，已定义 CBRN")
    print("阈值，AI R&D 阈值正在形成，竞争者调整条款普遍存在。目前没有统一的")
    print("行业术语，安全论证正成为趋同的交付物。UK AISI、US CAISI 和 EU AI")
    print("Office 构成相应的外部机构。")
    print("=" * 78)


if __name__ == "__main__":
    main()
