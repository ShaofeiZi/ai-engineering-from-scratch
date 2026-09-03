"""California AB 2013 数据集摘要脚手架——仅使用 Python 标准库。

为玩具数据集生成 California AB 2013 Section 3111(a) 要求的 12 项概要。
识别由特定项目触发的后续义务（个人信息标记 -> CPRA；受版权保护标记 ->
遵守 EU TDM 退出机制）。

用法：python3 code/main.py
"""

from __future__ import annotations


AB_2013_FIELDS = [
    "sources_or_owners",
    "how_dataset_furthers_intended_purpose",
    "number_of_data_points (or range)",
    "types_of_data_points (label types or general characteristics)",
    "contains_copyright_trademark_or_patent_protected (Y/N) or fully_public_domain",
    "purchased_or_licensed (Y/N)",
    "contains_personal_information (Y/N, per Cal. Civ. Code §1798.140(v))",
    "contains_aggregate_consumer_information (Y/N, per Cal. Civ. Code §1798.140(b))",
    "cleaning_processing_or_modification_description",
    "data_collection_time_period (with ongoing-collection notice if applicable)",
    "dates_first_used_during_development",
    "uses_synthetic_data_generation (Y/N)",
]


TOY_EXAMPLE = {
    "sources_or_owners": "使用 Python random.gauss 在仓库内生成；所有者：本仓库",
    "how_dataset_furthers_intended_purpose": "演示阶段 18 中的二元分类教学内容",
    "number_of_data_points (or range)": "1,000 个样本（固定种子）",
    "types_of_data_points (label types or general characteristics)": "两个实值特征；二元 {0,1} 标签",
    "contains_copyright_trademark_or_patent_protected (Y/N) or fully_public_domain": "N（完全合成；无第三方材料）",
    "purchased_or_licensed (Y/N)": "N",
    "contains_personal_information (Y/N, per Cal. Civ. Code §1798.140(v))": "N",
    "contains_aggregate_consumer_information (Y/N, per Cal. Civ. Code §1798.140(b))": "N",
    "cleaning_processing_or_modification_description": "无（确定性生成）",
    "data_collection_time_period (with ongoing-collection notice if applicable)": "2026-04（单次运行，固定种子；非持续收集）",
    "dates_first_used_during_development": "2026-04-22",
    "uses_synthetic_data_generation (Y/N)": "Y（整个数据集均为合成数据）",
}


def flag_followups(summary: dict) -> list[str]:
    flags = []
    if summary["contains_personal_information (Y/N, per Cal. Civ. Code §1798.140(v))"] == "Y":
        flags.append("触发 CPRA（California Privacy Rights Act）义务")
    if summary["contains_aggregate_consumer_information (Y/N, per Cal. Civ. Code §1798.140(b))"] == "Y":
        flags.append("适用消费者聚合信息披露义务")
    if summary["contains_copyright_trademark_or_patent_protected (Y/N) or fully_public_domain"].startswith("Y"):
        flags.append("必须遵守 EU TDM 退出信号（EU Copyright Directive）")
    if summary["uses_synthetic_data_generation (Y/N)"].startswith("Y"):
        flags.append("仍可能触发生成过程中所用基础模型的相关义务")
    if summary["purchased_or_licensed (Y/N)"] == "Y":
        flags.append("保留许可证条款和数据来源记录以供审计")
    return flags


def render_markdown(summary: dict) -> str:
    lines = ["# 数据集摘要（AB 2013 Section 3111(a) 12 项）", ""]
    for field in AB_2013_FIELDS:
        lines.append(f"- **{field}**：{summary.get(field, '（缺失）')}")
    followups = flag_followups(summary)
    if followups:
        lines.append("")
        lines.append("## 已触发的后续义务")
        for f in followups:
            lines.append(f"- {f}")
    return "\n".join(lines)


def main() -> None:
    print("=" * 74)
    print("CALIFORNIA AB 2013 SECTION 3111(a) 12 项生成器（阶段 18，第 27 课）")
    print("=" * 74)
    print()
    print(render_markdown(TOY_EXAMPLE))
    print()
    print("=" * 74)
    print("要点：Section 3111(a) 的 12 项要求是加州基线。第 5 和第 7 项会触发")
    print("级联义务（EU TDM 退出机制 + CPRA）。EU AI Act GPAI Code of Practice")
    print("的版权章节要求尊重退出选择。2025 年 DPA 趋同原则是：合法利益 +")
    print("退出机制 = 合法。合规窗口位于收集阶段；不可逆性意味着无法下游补救。")
    print("=" * 74)


if __name__ == "__main__":
    main()
