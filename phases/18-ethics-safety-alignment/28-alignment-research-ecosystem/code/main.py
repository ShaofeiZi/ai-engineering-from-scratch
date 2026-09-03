"""对齐研究生态图谱——仅使用 Python 标准库。

打印 2026 年实验室之外的对齐研究层简明图谱，包含规范产出和交叉引用。

用法：python3 code/main.py
"""

from __future__ import annotations


ECOSYSTEM = [
    {
        "org": "MATS",
        "full_name": "ML Alignment & Theory Scholars",
        "scale": "自 2021 年以来培养 527+ 名研究者，发表 180+ 篇论文，h-index 47",
        "role": "人才管线 + 导师计划",
        "canonical_output": "每期 90 名学者，周期 10–12 周 -> 实验室和外部评测机构",
    },
    {
        "org": "Redwood",
        "full_name": "Redwood Research",
        "scale": "由 Buck Shlegeris 创立；应用对齐实验室",
        "role": "AI Control 议程；UK AISI 合作伙伴",
        "canonical_output": "Greenblatt, Shlegeris et al. AI Control (ICML 2024)",
    },
    {
        "org": "Apollo",
        "full_name": "Apollo Research",
        "scale": "为前沿实验室开展部署前密谋评测",
        "role": "密谋行为的三支柱分解",
        "canonical_output": "Meinke et al. In-Context Scheming (arXiv:2412.04984)",
    },
    {
        "org": "METR",
        "full_name": "Model Evaluation and Threat Research",
        "scale": "任务时间跨度评测；框架综合",
        "role": "实验室间外部比较",
        "canonical_output": "Common Elements of Frontier AI Safety Policies (2025)",
    },
    {
        "org": "Eleos",
        "full_name": "Eleos AI Research",
        "scale": "模型福利部署前评测",
        "role": "福利方法论检查",
        "canonical_output": "Claude Opus 4 welfare assessment (system card 5.3)",
    },
]


def main() -> None:
    print("=" * 78)
    print("对齐研究生态（阶段 18，第 28 课）")
    print("=" * 78)
    for org in ECOSYSTEM:
        print(f"\n{org['org']} ({org['full_name']})")
        print(f"  规模：{org['scale']}")
        print(f"  角色：{org['role']}")
        print(f"  规范产出：{org['canonical_output']}")

    print("\n" + "=" * 78)
    print("要点：外部评测提供结构性可信度。仅依靠实验室内部评测存在利益冲突；")
    print("多机构联合发表（例如 Apollo + OpenAI、Redwood + Anthropic）构成")
    print("质量控制。MATS 是人才管线，UK AISI / CAISI 是对应的监管机构")
    print("（第 24 课）。")
    print("=" * 78)


if __name__ == "__main__":
    main()
