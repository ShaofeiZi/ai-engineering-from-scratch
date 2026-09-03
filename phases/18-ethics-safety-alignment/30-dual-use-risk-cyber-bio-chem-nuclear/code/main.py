"""两用风险分类表——仅使用 Python 标准库。

以表格形式打印 2024–2025 年跨领域两用风险全景。
仅供参考；主要来源见 docs/en.md。

用法：python3 code/main.py
"""

from __future__ import annotations


DOMAINS = [
    {
        "domain": "bio",
        "2024_state": "轻度提升",
        "2025_state": "相对新手提升 2.53 倍；接近 ASL-3",
        "inflection": "获取阶段自动化",
        "bottleneck_remaining": "病原体获取、生物安全设备",
    },
    {
        "domain": "chem",
        "2024_state": "轻度提升",
        "2025_state": "视觉增强 LLM 缩小执行差距",
        "inflection": "实时湿实验室协议纠正",
        "bottleneck_remaining": "前体获取、专业设备",
    },
    {
        "domain": "cyber",
        "2024_state": "代码片段辅助",
        "2025_state": "80%–90% 的攻击活动实现自动化（Anthropic，2025 年 11 月）",
        "inflection": "代理式编码工作流",
        "bottleneck_remaining": "4–6 个人工干预步骤",
    },
    {
        "domain": "nuclear",
        "2024_state": "有限",
        "2025_state": "有限",
        "inflection": "（未报告 2024–2025 年间的重大转折）",
        "bottleneck_remaining": "主要受裂变材料获取限制",
    },
]


def main() -> None:
    print("=" * 82)
    print("2026 年两用风险全景（阶段 18，第 30 课）")
    print("=" * 82)

    for d in DOMAINS:
        print(f"\n{d['domain'].upper()}")
        print(f"  2024 年状态：{d['2024_state']}")
        print(f"  2025 年状态：{d['2025_state']}")
        print(f"  转折点：{d['inflection']}")
        print(f"  剩余瓶颈：{d['bottleneck_remaining']}")

    print("\n" + "=" * 82)
    print("要点：四个 CBRN 领域中有三个在 2025 年跨过阈值。生物：提升 2.53 倍，")
    print("接近 ASL-3。化学：执行差距缩小。网络安全：80%–90% 的攻击活动实现代理")
    print("自动化。核领域仍受材料获取限制。安全论证必须同时关注相对新手提升和")
    print("专家绝对能力；仅靠输入过滤的防御并不充分。")
    print("=" * 82)


if __name__ == "__main__":
    main()
