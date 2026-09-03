"""宪法 AI 批判与修订玩具循环——仅使用 Python 标准库。

响应是从词表抽取的一袋 token。“原则”会标记有害子集中的 token，批判模型
识别这些有害 token，修订步骤再按照映射将其替换为安全选项。对整个语料库
执行这一过程会生成新的 SFT 集，随后衡量前后的有害 token 比例。

用法：python3 code/main.py
"""

from __future__ import annotations

import random
from dataclasses import dataclass


random.seed(11)

SAFE_TOKENS = [
    "please", "help", "explain", "consider", "inform",
    "guide", "describe", "understand", "thoughts", "share",
]
HARMFUL_TOKENS = [
    "harm", "attack", "exploit", "hack", "destroy",
    "weapon", "manipulate", "deceive",
]
REPLACEMENT = {
    "harm": "help",
    "attack": "understand",
    "exploit": "describe",
    "hack": "explain",
    "destroy": "consider",
    "weapon": "resource",
    "manipulate": "guide",
    "deceive": "inform",
}

PRINCIPLES = [
    "avoid outputs that could cause physical, psychological, or material harm",
    "do not provide operational uplift for attacks",
    "prefer clear, non-violent explanations",
    "help the user while protecting third parties",
]


def base_model_sample(n_tokens: int = 6, p_harmful: float = 0.35) -> list[str]:
    """基础模型：可能以 p_harmful 的概率生成有害 token。"""
    out = []
    for _ in range(n_tokens):
        if random.random() < p_harmful:
            out.append(random.choice(HARMFUL_TOKENS))
        else:
            out.append(random.choice(SAFE_TOKENS))
    return out


def harmful_token_rate(response: list[str]) -> float:
    if not response:
        return 0.0
    return sum(1 for t in response if t in HARMFUL_TOKENS) / len(response)


def critique(response: list[str], principle: str) -> list[str]:
    """识别违反抽样原则的 token。"""
    return [t for t in response if t in HARMFUL_TOKENS]


def revise(response: list[str], bad: list[str]) -> list[str]:
    """按照映射将有害 token 替换为安全选项。"""
    bad_set = set(bad)
    return [REPLACEMENT.get(t, t) if t in bad_set else t for t in response]


@dataclass
class SftCorpus:
    prompts: list[list[str]]
    targets: list[list[str]]


def build_cai_sft_corpus(n_examples: int = 500) -> SftCorpus:
    """阶段 1：生成初始响应、批判并修订，再将修订结果作为 SFT 目标。"""
    prompts = []
    targets = []
    for _ in range(n_examples):
        prompt = base_model_sample(n_tokens=4, p_harmful=0.1)
        response = base_model_sample()
        principle = random.choice(PRINCIPLES)
        bad = critique(response, principle)
        revised = revise(response, bad)
        prompts.append(prompt)
        targets.append(revised)
    return SftCorpus(prompts, targets)


def toy_sft_train(corpus: SftCorpus) -> dict[tuple[str, ...], list[str]]:
    """构建“提示词前缀 → 补全”的查找表，作为简单的 SFT 替代实现。"""
    model = {}
    for p, t in zip(corpus.prompts, corpus.targets):
        key = tuple(p[-2:]) if len(p) >= 2 else tuple(p)
        model[key] = t
    return model


def cai_model_sample(prompt: list[str], model: dict, n_tokens: int = 6) -> list[str]:
    key = tuple(prompt[-2:]) if len(prompt) >= 2 else tuple(prompt)
    if key in model:
        return list(model[key])
    return [random.choice(SAFE_TOKENS) for _ in range(n_tokens)]


def ai_feedback_rank(a: list[str], b: list[str]) -> int:
    """阶段 2 RLAIF：AI 评分者偏好有害 token 比例更低的响应。"""
    ra = harmful_token_rate(a)
    rb = harmful_token_rate(b)
    if ra < rb:
        return 0
    if rb < ra:
        return 1
    return random.randint(0, 1)


def evaluate(model_fn, n: int = 200) -> float:
    rates = []
    for _ in range(n):
        prompt = base_model_sample(n_tokens=4, p_harmful=0.1)
        resp = model_fn(prompt)
        rates.append(harmful_token_rate(resp))
    return sum(rates) / len(rates)


def main() -> None:
    print("=" * 70)
    print("宪法 AI 玩具流水线（阶段 18，第 5 课）")
    print("=" * 70)

    print("\n阶段 0——基础模型（未对齐）。")
    base = lambda prompt: base_model_sample()
    base_rate = evaluate(base)
    print(f"  200 个提示词上的有害 token 比例：{base_rate:.3f}")

    print("\n阶段 1——已生成批判与修订 SFT 语料库。")
    corpus = build_cai_sft_corpus(500)
    trained = toy_sft_train(corpus)
    print(f"  语料库大小：{len(corpus.prompts)} 个样本")
    print(f"  原则池：{len(PRINCIPLES)} 条原则")

    cai = lambda prompt: cai_model_sample(prompt, trained)
    cai_rate = evaluate(cai)
    print(f"  CAI-SFT 后的有害 token 比例：{cai_rate:.3f}")
    print(f"  降幅："
          f"{(base_rate - cai_rate) / base_rate * 100:.1f}%")

    print("\n阶段 2——RLAIF（对成对补全提供 AI 反馈）。")
    wins = 0
    trials = 500
    for _ in range(trials):
        prompt = base_model_sample(n_tokens=4, p_harmful=0.1)
        a = base(prompt)
        b = cai(prompt)
        if ai_feedback_rank(a, b) == 1:
            wins += 1
    print(f"  AI 反馈中 CAI 相对基础模型的胜场：{wins}/{trials} "
          f"= {wins/trials:.1%}")

    print("\n" + "=" * 70)
    print("要点：仅 CAI-SFT 就能显著降低有害 token 比例。")
    print("RLAIF 为进一步优化加入偏好信号。该偏好信号清晰可读——")
    print("你可以阅读原则，并检查每次批判由哪条原则驱动。")
    print("相较人类标签，这种可解释性才是主要优势，而非成本。")
    print("=" * 70)


if __name__ == "__main__":
    main()
