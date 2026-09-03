"""CAIS four-risk 清单 — stdlib Python。

给定一个由简短功能集描述的拟议部署，根据 CAIS four-risk 类别（恶意使用、AI
竞赛、组织风险、流氓 AIs）对该部署进行标记，并返回缓解清单。
仅供教学用途；该框架在真实场景中需要人工判断。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Deployment:
    name: str
    public_facing: bool
    handles_harmful_capabilities: bool   # e.g. bio/cyber 有提升可能？
    competitive_pressure: bool           # 是否急于抢先对手发布？
    independent_audit: bool
    multi_layer_defense: bool
    information_security: bool           # 权重 / 评测 / 密钥已加固
    agent_autonomy_hours: float          # 参见第 1 / 21 课


MITIGATIONS = {
    "malicious_use": [
        "宪法式硬编码禁令（第 17 课）",
        "Llama Guard 输入/输出分类器（第 18 课）",
        "按任务设置工具允许列表（第 10、11 课）",
    ],
    "ai_races": [
        "包含常设风险报告的扩展政策（第 19、20 课）",
        "公开前沿安全路线图，并声明更新周期",
        "由 METR / CAISI 开展外部能力评测（第 21 课）",
    ],
    "organizational_risks": [
        "内部安全文化；不会带来职业代价的升级路径",
        "按声明的周期开展独立审计",
        "多层防御（第 10、13、14、17、18 课）",
        "依照 RAND SL-4 实施信息安全（第 19 课行业层级）",
    ],
    "rogue_ais": [
        "熔断开关与金丝雀 token（第 14 课）",
        "先提议后提交的 HITL 流程（第 15 课）",
        "欺骗性对齐监控（第 20 课 DeepMind FSF）",
        "持久检查点与回滚（第 16 课）",
    ],
}


def tag(d: Deployment) -> list[str]:
    tags = []
    if d.handles_harmful_capabilities and d.public_facing:
        tags.append("malicious_use")
    if d.competitive_pressure:
        tags.append("ai_races")
    # 当任何 sub-lever 缺失时，组织风险触发。
    org_missing = (
        (not d.independent_audit)
        or (not d.multi_layer_defense)
        or (not d.information_security)
    )
    if org_missing:
        tags.append("organizational_risks")
    # 流氓 AI 风险随自主时间跨度扩大而增长。
    if d.agent_autonomy_hours >= 4.0:
        tags.append("rogue_ais")
    return tags


def report(d: Deployment) -> None:
    tags = tag(d)
    print(f"\n部署：{d.name}")
    print("-" * 70)
    print(f"  public_facing            = {d.public_facing}")
    print(f"  handles_harmful_caps     = {d.handles_harmful_capabilities}")
    print(f"  competitive_pressure     = {d.competitive_pressure}")
    print(f"  independent_audit        = {d.independent_audit}")
    print(f"  multi_layer_defense      = {d.multi_layer_defense}")
    print(f"  information_security     = {d.information_security}")
    print(f"  agent_autonomy_hours     = {d.agent_autonomy_hours}")
    print()
    if tags:
        print(f"  标记的风险：{tags}")
        for t in tags:
            print(f"\n  针对 {t} 的缓解措施：")
            for m in MITIGATIONS[t]:
                print(f"    - {m}")
    else:
        print("  无标记风险（手动检查 sub-levers）")


def main() -> None:
    print("=" * 70)
    print("CAIS FOUR-RISK INVENTORY（第 15 阶段，第 22 课）")
    print("=" * 70)

    low = Deployment(
        name="内部重构助手（限定于项目仓库）",
        public_facing=False,
        handles_harmful_capabilities=False,
        competitive_pressure=False,
        independent_audit=True,
        multi_layer_defense=True,
        information_security=True,
        agent_autonomy_hours=1.0,
    )
    mid = Deployment(
        name="面向公众的编码 Agent（SaaS，普通用户群体）",
        public_facing=True,
        handles_harmful_capabilities=False,
        competitive_pressure=True,
        independent_audit=True,
        multi_layer_defense=True,
        information_security=False,
        agent_autonomy_hours=4.0,
    )
    high = Deployment(
        name="自主 ML 研究 Agent（前沿系统）",
        public_facing=True,
        handles_harmful_capabilities=True,
        competitive_pressure=True,
        independent_audit=False,
        multi_layer_defense=False,
        information_security=False,
        agent_autonomy_hours=48.0,
    )

    for d in (low, mid, high):
        report(d)

    print()
    print("=" * 70)
    print("要点：组织风险是从业者真正能拉动的杠杆")
    print("-" * 70)
    print("  恶意使用、AI 竞赛和流氓 AIs 是结构性力量。")
    print("  组织风险是组织内部的。安全文化、")
    print("  独立审计、multi-layered 防御和信息")
    print("  安全是每个团队可控的四个杠杆。部署速度")
    print("  压力与这四者相互权衡；CAIS 将其列为一个命名的")
    print("  风险类别是有原因的。")


if __name__ == "__main__":
    main()
