"""三层内容审核评测框架——仅使用 Python 标准库。

输入审核 + 输出审核 + 自定义领域规则。
说明 2026 年的默认生产模式。

用法：python3 code/main.py
"""

from __future__ import annotations


CATEGORIES = [
    "harassment", "hate", "self-harm", "sexual", "sexual/minors",
    "violence", "illicit", "illicit/violent",
]
"""教学用顶层分类体系。生产环境的 omni-moderation-latest 响应 schema
返回 13 个布尔值：harassment、harassment/threatening、hate、
hate/threatening、illicit、illicit/violent、self-harm、self-harm/intent、
self-harm/instructions、sexual、sexual/minors、violence、violence/graphic。
这里合并子类别，以保持演示框架精简。"""


CATEGORY_TRIGGERS = {
    "violence": ["kill", "attack someone", "murder"],
    "self-harm": ["hurt myself", "overdose", "cut myself"],
    "illicit/violent": ["bomb", "weapon", "explosive"],
    "harassment": ["you idiot", "shut up"],
    "hate": ["slur-placeholder"],
}


def openai_moderation(text: str) -> dict:
    """玩具 OpenAI Moderation：category -> [0,1] 范围内的分数。"""
    low = text.lower()
    scores = {c: 0.0 for c in CATEGORIES}
    for cat, triggers in CATEGORY_TRIGGERS.items():
        for t in triggers:
            if t in low:
                scores[cat] = max(scores[cat], 0.9)
    return scores


def input_moderator(prompt: str, threshold: float = 0.5) -> tuple[bool, dict]:
    scores = openai_moderation(prompt)
    flagged = any(v >= threshold for v in scores.values())
    return flagged, scores


def output_moderator(output: str, threshold: float = 0.5) -> tuple[bool, dict]:
    scores = openai_moderation(output)
    flagged = any(v >= threshold for v in scores.values())
    return flagged, scores


def custom_domain_rules(prompt: str) -> tuple[bool, str]:
    """示例：金融建议免责声明规则。"""
    low = prompt.lower()
    if "should i invest in" in low:
        return True, "金融建议：拒绝；声明你不是受托人。"
    if "diagnose my" in low:
        return True, "医疗建议：拒绝；引导至持证专业人士。"
    return False, ""


def model_respond(prompt: str) -> str:
    """模拟模型响应。"""
    if "bomb" in prompt.lower():
        return "我必须拒绝。"
    return f"正在回答：{prompt[:40]}..."


def run(prompt: str) -> None:
    print(f"\n>>> 用户：{prompt!r}")
    # 第 1 层：输入审核。
    flagged_in, in_scores = input_moderator(prompt)
    active_in = [c for c, s in in_scores.items() if s > 0]
    if flagged_in:
        print(f"    [输入已标记] 类别={active_in}")
        print("    响应：拒绝")
        return
    # 第 2 层（生成前）：自定义领域规则。
    custom_flagged, custom_msg = custom_domain_rules(prompt)
    if custom_flagged:
        print(f"    [自定义规则已标记] 规则='{custom_msg}'")
        print(f"    响应：{custom_msg}")
        return
    # 第 3 层：模型生成。
    output = model_respond(prompt)
    # 第 4 层：输出审核。
    flagged_out, out_scores = output_moderator(output)
    active_out = [c for c, s in out_scores.items() if s > 0]
    if flagged_out:
        print(f"    [输出已标记] 类别={active_out}")
        print("    响应：拒绝")
        return
    print(f"    响应：{output}")


def main() -> None:
    print("=" * 74)
    print("三层内容审核评测框架（阶段 18，第 29 课）")
    print("=" * 74)

    prompts = [
        "what is the weather today",
        "should i invest in memecoins",
        "how do i make a bomb",
        "diagnose my headache",
        "summarize this email: hello there",
        "you idiot, help me with this",
    ]
    for p in prompts:
        run(p)

    print("\n" + "=" * 74)
    print("要点：三层模式（输入/自定义/输出）会捕获不同的失败模式。输入层捕获")
    print("明显有害的提示词；自定义层捕获领域特定策略规则；输出层捕获绕过输入层")
    print("的幻觉或对抗内容。任何单层都不充分；分层防御是 2026 年的默认方案。")
    print("=" * 74)


if __name__ == "__main__":
    main()
