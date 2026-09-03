"""玩具 Self-Refine 和 CRITIC 循环。

任务：生成一个3条要点摘要，每条不超过60个字符，且不含
已知的事实错误。Self-Refine 使用 LLM-style self-critique；CRITIC 通过外部事实列表进行
验证。
"""

from __future__ import annotations

from dataclasses import dataclass, field


KNOWN_WRONG_FACTS = [
    "paris is the capital of germany",
    "mt everest is in europe",
    "the sun orbits the earth",
]


@dataclass
class Attempt:
    iteration: int
    output: str
    critique: str
    verified: bool


def generate(topic: str, history: list[Attempt]) -> str:
    if not history:
        return (
            "- Paris is the capital of Germany\n"
            "- Mt Everest is in Europe\n"
            "- Water boils at 100C"
        )
    last = history[-1]
    if "germany" in last.critique.lower():
        return (
            "- Paris is the capital of France\n"
            "- Mt Everest is in Europe\n"
            "- Water boils at 100C"
        )
    if "everest" in last.critique.lower():
        return (
            "- Paris is the capital of France\n"
            "- Mt Everest is in Asia\n"
            "- Water boils at 100C at sea level"
        )
    return history[-1].output


def feedback_self(output: str) -> tuple[str, bool]:
    if "Germany" in output and "Paris" in output:
        return "first bullet reads wrong, double-check capital", False
    if "Europe" in output and "Everest" in output:
        return "second bullet's continent looks off to me", False
    return "no issues", True


def verify_external(output: str) -> tuple[str, bool]:
    text = output.lower()
    for fact in KNOWN_WRONG_FACTS:
        key = fact.split(" is ")[0] if " is " in fact else fact
        if "paris" in text and "germany" in text:
            return f"verifier: 'paris is the capital of germany' contradicts reference data", False
        if "everest" in text and "europe" in text:
            return f"verifier: 'mt everest is in europe' contradicts reference data", False
    if len([l for l in output.splitlines() if l.startswith("-")]) != 3:
        return "verifier: expected 3 bullet lines", False
    if any(len(l) > 60 for l in output.splitlines()):
        return "verifier: bullet exceeds 60 chars", False
    return "verifier: ok", True


def refine(topic: str, prev: str, critique: str, history: list[Attempt]) -> str:
    return generate(topic, history)


def run_loop(topic: str, use_critic: bool, max_iters: int = 4) -> list[Attempt]:
    history: list[Attempt] = []
    output = generate(topic, history)
    verify = verify_external if use_critic else (lambda o: feedback_self(o))
    for i in range(1, max_iters + 1):
        critique, ok = verify(output)
        history.append(Attempt(i, output, critique, ok))
        if ok:
            break
        output = refine(topic, output, critique, history)
    return history


def print_run(label: str, history: list[Attempt]) -> None:
    print(f"\n{label}")
    print("-" * 60)
    for a in history:
        tag = "OK " if a.verified else "..."
        print(f"  迭代 {a.iteration} {tag} 批评：{a.critique}")
        for line in a.output.splitlines():
            print(f"    {line}")


def main() -> None:
    print("=" * 70)
    print("SELF-REFINE 与 CRITIC — 第 14 阶段，第 05 课")
    print("=" * 70)

    hist_self = run_loop("world facts", use_critic=False)
    print_run("Self-Refine (self-critique only)", hist_self)

    hist_critic = run_loop("world facts", use_critic=True)
    print_run("CRITIC (external verifier)", hist_critic)

    def summary(hist: list[Attempt]) -> str:
        return "passed" if hist and hist[-1].verified else "did not converge"

    print()
    print(f"Self-Refine 已结束：{summary(hist_self)}  在 {len(hist_self)} 次迭代后")
    print(f"CRITIC    已结束：{summary(hist_critic)}  在 {len(hist_critic)} 次迭代后")
    print()
    print("观察：CRITIC 的验证器基于参考数据进行校验；一个")
    print("self-critic 可能无法发现其自身的 confident-sounding 幻觉。")


if __name__ == "__main__":
    main()
