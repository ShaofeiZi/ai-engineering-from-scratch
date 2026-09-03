"""Phase 13 第21课 - LLM 路由网关，标准库。

OpenAI-compatible 请求进入；优先级回退链选择后端；成本
追踪器累计支出 per-request. PII 脱敏运行 pre-dispatch.

后端提供者均为桩实现。将某个切换为 "outage" 即可观察回退行为。

运行：python code/main.py
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Callable


# 每100万token的成本（输入、输出）；演示用模拟费率
PRICES = {
    "openai/gpt-4o":           (5.0, 15.0),
    "openai/gpt-4o-mini":      (0.15, 0.60),
    "anthropic/claude-sonnet": (3.0, 15.0),
    "anthropic/claude-haiku":  (0.80, 4.0),
    "google/gemini-pro":       (1.25, 5.0),
}

OUTAGE: set[str] = set()


def provider_call(model: str, messages: list[dict]) -> dict:
    if model in OUTAGE:
        raise RuntimeError(f"{model} 返回模拟的 5xx 错误")
    time.sleep(0.01)
    last = messages[-1]["content"]
    out_toks = max(20, len(last) // 3)
    return {
        "id": f"resp_{model.replace('/', '_')}",
        "model": model,
        "choices": [{"message": {"role": "assistant",
                                 "content": f"[{model}] 回显：{last[:60]}"}}],
        "usage": {"prompt_tokens": len(last) // 4, "completion_tokens": out_toks},
    }


# 别名 -> 回退链
ROUTES = {
    "smart": ["openai/gpt-4o", "anthropic/claude-sonnet", "google/gemini-pro"],
    "fast":  ["openai/gpt-4o-mini", "anthropic/claude-haiku"],
}


PII_PATTERNS = [
    re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),  # SSN
    re.compile(r"\b\d{16}\b"),               # 信用卡
]


def redact_pii(text: str) -> tuple[str, bool]:
    redacted = False
    for pat in PII_PATTERNS:
        if pat.search(text):
            text = pat.sub("[REDACTED]", text)
            redacted = True
    return text, redacted


@dataclass
class Invocation:
    alias: str
    chosen_model: str = ""
    attempts: list[str] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    redacted: bool = False
    response: dict | None = None
    error: str | None = None


def route(alias: str, messages: list[dict]) -> Invocation:
    inv = Invocation(alias=alias)
    # 对输入进行PII脱敏
    new_msgs = []
    for m in messages:
        txt, r = redact_pii(m["content"])
        if r:
            inv.redacted = True
        new_msgs.append({"role": m["role"], "content": txt})
    chain = ROUTES.get(alias, [alias])
    for model in chain:
        inv.attempts.append(model)
        try:
            resp = provider_call(model, new_msgs)
            inv.chosen_model = model
            inv.response = resp
            u = resp["usage"]
            inv.input_tokens = u["prompt_tokens"]
            inv.output_tokens = u["completion_tokens"]
            in_rate, out_rate = PRICES.get(model, (0, 0))
            inv.cost_usd = (u["prompt_tokens"] * in_rate +
                            u["completion_tokens"] * out_rate) / 1_000_000
            return inv
        except RuntimeError as e:
            continue
    inv.error = "所有提供者均失败"
    return inv


def demo() -> None:
    print("=" * 72)
    print("第 13 阶段第 21 课 - LLM 路由网关")
    print("=" * 72)

    print("\n--- 场景1：智能路由，主后端可用 ---")
    inv = route("smart", [{"role": "user", "content": "解释 MCP"}])
    print(f"  选择    : {inv.chosen_model}")
    print(f"  尝试    : {inv.attempts}")
    print(f"  token数 : 输入={inv.input_tokens} 输出={inv.output_tokens}")
    print(f"  成本    : ${inv.cost_usd:.6f}")
    print(f"  回复    : {inv.response['choices'][0]['message']['content']}")

    print("\n--- 场景2：openai/gpt-4o 故障 -> 回退到 Claude ---")
    OUTAGE.add("openai/gpt-4o")
    inv = route("smart", [{"role": "user", "content": "相同的请求"}])
    print(f"  选择    : {inv.chosen_model}")
    print(f"  尝试    : {inv.attempts}")
    print(f"  成本    : ${inv.cost_usd:.6f}")
    OUTAGE.clear()

    print("\n--- 场景3：在分发前对输入中的 PII 脱敏 ---")
    inv = route("fast", [{"role": "user",
                           "content": "请通过 SSN 123-45-6789 联系我"}])
    print(f"  脱敏后  : {inv.redacted}")
    print(f"  回复    : {inv.response['choices'][0]['message']['content']}")

    print("\n--- 场景4：所有提供者均不可用 ---")
    OUTAGE.update(ROUTES["fast"])
    inv = route("fast", [{"role": "user", "content": "help"}])
    print(f"  尝试    : {inv.attempts}")
    print(f"  错误    : {inv.error}")


if __name__ == "__main__":
    demo()
