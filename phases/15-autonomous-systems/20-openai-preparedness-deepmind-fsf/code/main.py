"""跨政策决策表差异比较——使用 Python 标准库。

读取三张小表格，分别记录了 OpenAI PF v2、Anthropic RSP v3.0
和 DeepMind FSF v3 如何对一小批能力进行分类。输出一份
side-by-side 对比。这些表格是对三份原始文档的教学化提炼；
实际政策判断需查阅原文档。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Policy:
    name: str
    # 能力 ->（分类，触发动作）
    table: dict[str, tuple[str, str]]


# 示例性提炼；实际决策请参阅原始文档。
OPENAI_PF_V2 = Policy(
    name="OpenAI Preparedness v2（2025 年 4 月）",
    table={
        "long_range_autonomy": ("研究型", "已观察；可能需要缓解措施"),
        "sandbagging": ("研究型", "已观察；可能需要缓解措施"),
        "autonomous_replication": ("研究型", "已观察；可能需要缓解措施"),
        "undermining_safeguards": ("研究型", "已观察；可能需要缓解措施"),
        "rnd_automation": ("已追踪", "能力与防护报告；SAG 审查"),
        "cyber_uplift": ("已追踪", "能力与防护报告；SAG 审查"),
        "bio_uplift": ("已追踪", "能力与防护报告；SAG 审查"),
    },
)

ANTHROPIC_RSP_V3 = Policy(
    name="Anthropic RSP v3.0（2026 年 2 月）",
    table={
        "long_range_autonomy": ("已命名风险", "达到阈值时需提交肯定情形"),
        "sandbagging": ("通过评测环境差距命名",
                        "在测量方法中处理"),
        "autonomous_replication": ("未明确命名",
                                   "纳入 AI R&D-4"),
        "undermining_safeguards": ("硬编码禁令",
                                   "拒绝训练或部署"),
        "rnd_automation": ("AI R&D-4 阈值",
                           "需要肯定情形"),
        "cyber_uplift": ("ASL-3 触发项",
                         "安全与部署缓解措施"),
        "bio_uplift": ("ASL-3 触发项",
                       "安全与部署缓解措施"),
    },
)

DEEPMIND_FSF_V3 = Policy(
    name="DeepMind FSF v3（2025 年 9 月 + 2026 年 4 月）",
    table={
        "long_range_autonomy": ("归入 ML R&D / 网络安全领域",
                                "CCL + 已追踪能力等级"),
        "sandbagging": ("欺骗性对齐监控",
                        "自动化工具性推理监控器"),
        "autonomous_replication": ("归入 ML R&D 领域",
                                   "CCL 阈值"),
        "undermining_safeguards": ("欺骗性对齐监控",
                                   "自动监控 + 红队测试"),
        "rnd_automation": ("ML R&D 自主性等级 1",
                           "2026 年 4 月新增已追踪能力等级"),
        "cyber_uplift": ("网络安全 CCL",
                         "安全与部署缓解措施"),
        "bio_uplift": ("生物安全 CCL",
                       "安全与部署缓解措施"),
    },
)


POLICIES = [OPENAI_PF_V2, ANTHROPIC_RSP_V3, DEEPMIND_FSF_V3]


def diff(capability: str) -> None:
    print(f"\n能力：{capability}")
    print("-" * 80)
    for p in POLICIES:
        entry = p.table.get(capability, ("表中未列出", "—"))
        print(f"  {p.name}")
        print(f"    分类：      {entry[0]}")
        print(f"    处置：      {entry[1]}")


def main() -> None:
    print("=" * 80)
    print("跨政策差异比较（第 15 阶段，第 20 课）")
    print("=" * 80)

    for cap in ("long_range_autonomy", "sandbagging", "autonomous_replication",
                "undermining_safeguards", "rnd_automation"):
        diff(cap)

    print()
    print("=" * 80)
    print("要点：同一能力，三种不同分类")
    print("-" * 80)
    print("  Long-range 自主性：")
    print("   - OpenAI：研究型（未触发）")
    print("   - Anthropic：已命名风险（肯定情形）")
    print("   - DeepMind：domain-folded（CCL + 被追踪能力等级）")
    print()
    print("  破坏安全防护：")
    print("   - OpenAI：研究型（未触发）")
    print("   - Anthropic：硬编码禁令（拒绝）")
    print("   - DeepMind：欺骗性对齐监控")
    print()
    print("  将三者结合起来阅读，才是实际所需的核心技能。")


if __name__ == "__main__":
    main()
