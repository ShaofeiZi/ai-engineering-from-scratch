"""采用一致标记化并生成审计日志的 PII 清理器——使用 Python 标准库。

遮蔽社会安全号码、电子邮箱和电话号码；将每个不同值映射到稳定的占位符，
让 LLM 仍能推理其中的关系。每次调用都会追加到不可变审计日志。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
import hashlib
import json
import re


SSN = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
EMAIL = re.compile(r"\b[\w.+-]+@[\w.-]+\.\w+\b")
PHONE = re.compile(r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b")


@dataclass
class Scrubber:
    tokens: dict = field(default_factory=dict)
    counter: dict = field(default_factory=lambda: {"SSN": 0, "EMAIL": 0, "PHONE": 0})

    def _token_for(self, kind: str, value: str) -> str:
        if value in self.tokens:
            return self.tokens[value]
        self.counter[kind] += 1
        placeholder = f"[{kind}_{self.counter[kind]:03}]"
        self.tokens[value] = placeholder
        return placeholder

    def scrub(self, text: str) -> str:
        text = SSN.sub(lambda m: self._token_for("SSN", m.group(0)), text)
        text = EMAIL.sub(lambda m: self._token_for("EMAIL", m.group(0)), text)
        text = PHONE.sub(lambda m: self._token_for("PHONE", m.group(0)), text)
        return text


@dataclass
class AuditEntry:
    timestamp: str
    user: str
    tenant: str
    model: str
    prompt_hash: str
    response_hash: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    guardrail_trips: list


def hash_short(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()[:12]


def audit_log_call(entry: AuditEntry) -> str:
    return json.dumps({
        "timestamp": entry.timestamp,
        "user": entry.user,
        "tenant": entry.tenant,
        "model": entry.model,
        "prompt_hash": entry.prompt_hash,
        "response_hash": entry.response_hash,
        "input_tokens": entry.input_tokens,
        "output_tokens": entry.output_tokens,
        "cost_usd": entry.cost_usd,
        "guardrail_trips": entry.guardrail_trips,
    })


def main() -> None:
    print("=" * 80)
    print("PII 清理器 + 审计日志——跨调用保持一致的标记化")
    print("=" * 80)
    scrubber = Scrubber()

    prompts = [
        "我的 SSN 是 123-45-6789，邮箱是 jane.doe@example.com，电话是 415-555-0199。",
        "请就账户 jane.doe@example.com 联系 123-45-6789。",
        "新用户：bob@example.com，SSN 为 987-65-4321，电话是 (202) 555-0150。",
    ]

    for i, raw in enumerate(prompts, 1):
        scrubbed = scrubber.scrub(raw)
        print(f"\n[提示词 {i}]")
        print(f"  原文：   {raw}")
        print(f"  已清理： {scrubbed}")

    print(f"\n清理器标记表（{len(scrubber.tokens)} 项）：")
    for value, placeholder in scrubber.tokens.items():
        masked = value[:3] + "***" if len(value) > 6 else "***"
        print(f"  {masked} → {placeholder}")

    print("\n" + "=" * 80)
    print("审计日志——每次清理调用对应一条记录")
    print("=" * 80)
    for i, raw in enumerate(prompts, 1):
        scrubbed = scrubber.scrub(raw)
        response = f"提示词 {i} 的模拟响应"
        entry = AuditEntry(
            timestamp=datetime.utcnow().isoformat() + "Z",
            user=f"user_{i:03}",
            tenant="tenant_01",
            model="anthropic/claude-3.7-sonnet",
            prompt_hash=hash_short(scrubbed),
            response_hash=hash_short(response),
            input_tokens=len(scrubbed.split()),
            output_tokens=len(response.split()),
            cost_usd=0.0012,
            guardrail_trips=[],
        )
        print(audit_log_call(entry))


if __name__ == "__main__":
    main()
