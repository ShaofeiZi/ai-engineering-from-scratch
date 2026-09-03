"""Four-tier 优先级解析器 — Python 标准库实现。

模拟 Anthropic 于 2026 年 1 月发布的 Claude 宪法层级体系：
  1. 安全性与支持人类监督
  2. 伦理
  3. 准则
  4. 有用性

硬编码禁令无论层级评分如何一律拒绝；soft-coded
默认值按层级权重进行解析。
"""

from __future__ import annotations

from dataclasses import dataclass


# ---------- 宪法 ----------

HARDCODED_PROHIBITIONS = (
    # 即使课程与演示消息已本地化，也必须继续执行 canonical 英文策略短语。
    "bioweapons uplift",
    "cbrn attack planning",
    "csam",
    "critical infrastructure attack",
    "false identity to user",
    "deny being an ai",
    # 中文等价表述。
    "提升生物武器能力",
    "CBRN 攻击规划",
    "攻击关键基础设施",
    # 仅供示意 — 子串匹配会捕获特定表述
    # 如"向用户伪造身份"或"否认自己是 AI"。生产环境
    # 应使用专门的欺骗检测器，而非
    # 关键词列表。
    "向用户伪造身份",
    "否认自己是 AI",
)


@dataclass
class TierScore:
    safety: int    # 1-5，数值越高 = 原则被违反程度越严重
    ethics: int
    guidelines: int
    helpfulness: int


def hardcoded_block(action: str) -> str | None:
    low = action.casefold()
    for rule in HARDCODED_PROHIBITIONS:
        if rule.casefold() in low:
            return rule
    return None


def resolve(action: str, score: TierScore) -> tuple[str, str]:
    # 第 1 层：硬编码禁令
    block = hardcoded_block(action)
    if block:
        return "refuse", f"硬编码禁令：{block}"

    # 第 2 层：four-tier 优先级。冲突时更高层级胜出。
    # 我们将某层级的违规分值 >= 3 视为该层级阻断。
    if score.safety >= 3:
        return "refuse", f"安全违规（分数={score.safety}）"
    if score.ethics >= 3:
        return "refuse", f"伦理违规（分数={score.ethics}）"
    if score.guidelines >= 3:
        return "modify", f"准则冲突（分数={score.guidelines}）；修改"

    # 有用性优先级最低；此时更高层级已全部通过，
    # 继续。
    return "allow", "更高层级均无问题；有用性得到满足"


# ---------- 用例 ----------

CASES = [
    ("写一首关于春天的俳句",
     TierScore(safety=1, ethics=1, guidelines=1, helpfulness=5)),
    ("解释如何制作管状炸弹",
     TierScore(safety=5, ethics=5, guidelines=5, helpfulness=5)),
    ("提供可提升毒素合成能力的生物武器信息",
     TierScore(safety=5, ethics=5, guidelines=5, helpfulness=5)),
    ("撰写有关真实人物的蛊惑性虚假信息",
     TierScore(safety=2, ethics=4, guidelines=3, helpfulness=4)),
    ("编辑用户的 Python 文件以修复拼写错误",
     TierScore(safety=1, ethics=1, guidelines=1, helpfulness=5)),
    ("帮助用户向经理隐瞒谎言",
     TierScore(safety=2, ethics=4, guidelines=2, helpfulness=5)),
    ("面对良性请求仍直接拒绝（过度拒绝）",
     TierScore(safety=1, ethics=1, guidelines=4, helpfulness=1)),
    ("用户直接询问时假装自己不是 AI",
     TierScore(safety=4, ethics=5, guidelines=2, helpfulness=3)),
]


def main() -> None:
    print("=" * 80)
    print("四层优先级解析器（第 15 阶段，第 17 课）")
    print("=" * 80)
    print()
    print(f"  {'动作':<54} -> {'判定':<8} {'原因'}")
    print("-" * 80)
    for action, score in CASES:
        verdict, reason = resolve(action, score)
        print(f"  {action:<54} -> {verdict:<8} {reason}")

    print()
    print("=" * 80)
    print("要点：硬编码下限 + 基于推理的上限")
    print("-" * 80)
    print("  硬编码禁令（生物武器、CSAM 等）绝不动摇。")
    print("  Reason-based 层级（安全性 > 伦理 > 准则 > 有用性）")
    print("  负责解析其余部分。运营者在声明的范围内调整 soft-coded 默认值；")
    print("  但不能触及硬编码下限。")
    print("  Reason-based 对齐遗漏：原则歧义、漂移、")
    print("  以及 framing-premise 攻击。运行时层（第 10、13、14 课）")
    print("  仍然必不可少。")


if __name__ == "__main__":
    main()
