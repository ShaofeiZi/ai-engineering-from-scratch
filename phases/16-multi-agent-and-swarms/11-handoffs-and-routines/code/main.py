"""Handoff 驱动的编排：OpenAI Swarm 的微型实现。

两个原语：
  - Agent(name, instructions, functions)
  - handoff = 返回 Agent 的函数

运行循环会检测返回值是否为 Agent，并切换当前活动的 Agent。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional, Union


@dataclass
class Agent:
    name: str
    instructions: str
    functions: list[Callable] = field(default_factory=list)


@dataclass
class Msg:
    role: str
    content: str
    sender: Optional[str] = None


def triage_agent_factory() -> Agent:
    def transfer_to_refunds() -> "Agent":
        return refund_agent

    def transfer_to_sales() -> "Agent":
        return sales_agent

    def transfer_to_support() -> "Agent":
        return support_agent

    return Agent(
        name="triage",
        instructions="将用户路由到退款、销售或技术支持。",
        functions=[transfer_to_refunds, transfer_to_sales, transfer_to_support],
    )


def refund_agent_factory() -> Agent:
    def process_refund(order_id: str) -> str:
        return f"订单 {order_id} 的退款已处理。"

    return Agent(
        name="refund",
        instructions="处理退款请求。",
        functions=[process_refund],
    )


def sales_agent_factory() -> Agent:
    def quote_product(product: str) -> str:
        return f"{product} 的报价：每月 $99。"

    return Agent(
        name="sales",
        instructions="处理销售咨询。",
        functions=[quote_product],
    )


def support_agent_factory() -> Agent:
    def open_ticket(issue: str) -> str:
        return f"已为以下问题创建工单：{issue}"

    return Agent(
        name="support",
        instructions="处理技术支持。",
        functions=[open_ticket],
    )


triage_agent = triage_agent_factory()
refund_agent = refund_agent_factory()
sales_agent = sales_agent_factory()
support_agent = support_agent_factory()


def scripted_router(current: Agent, user_msg: str) -> Union[str, Agent]:
    """代替真实 LLM：它读取用户消息和当前 Agent 的 system prompt，随后输出文本
    或调用工具（工具可能返回另一个 Agent）。在真实 Swarm 中，这是一次 LLM 工具调用。"""
    text = user_msg.lower()
    if current.name == "triage":
        if "refund" in text or "money back" in text or "退款" in text:
            return next(f for f in current.functions if f.__name__ == "transfer_to_refunds")()
        if "buy" in text or "price" in text or "购买" in text or "价格" in text:
            return next(f for f in current.functions if f.__name__ == "transfer_to_sales")()
        if "broken" in text or "bug" in text or "故障" in text:
            return next(f for f in current.functions if f.__name__ == "transfer_to_support")()
        return "请告诉我您需要哪方面的帮助？"
    if current.name == "refund":
        order = "42"
        for word in user_msg.split():
            if word.isdigit():
                order = word
                break
        return next(f for f in current.functions if f.__name__ == "process_refund")(order)
    if current.name == "sales":
        product = "企业套餐"
        return next(f for f in current.functions if f.__name__ == "quote_product")(product)
    if current.name == "support":
        return next(f for f in current.functions if f.__name__ == "open_ticket")(user_msg)
    return "[无响应]"


def run_swarm(start_agent: Agent, user_messages: list[str]) -> list[Msg]:
    history: list[Msg] = []
    active = start_agent
    for user in user_messages:
        history.append(Msg(role="user", content=user))
        out = scripted_router(active, user)
        if isinstance(out, Agent):
            history.append(
                Msg(role="assistant", content=f"（handoff 至 {out.name}）", sender=active.name)
            )
            active = out
            out = scripted_router(active, user)
        history.append(Msg(role="assistant", content=str(out), sender=active.name))
    return history


def render(history: list[Msg]) -> None:
    for m in history:
        tag = m.sender if m.sender else m.role
        print(f"  [{tag:>8s}]: {m.content}")


def main() -> None:
    print("Handoff 驱动的编排 — OpenAI Swarm 结构")
    print("-" * 54)

    scenarios = [
        ("退款流程", ["我需要为订单 77 办理退款"]),
        ("销售流程", ["我想购买企业套餐，价格是多少？"]),
        ("支持流程", ["我的看板出现故障"]),
        ("意图不明确", ["你好"]),
    ]
    for label, msgs in scenarios:
        print(f"\n=== {label} ===")
        history = run_swarm(triage_agent, msgs)
        render(history)

    print("\n关键洞察：每次 handoff 都是一次返回 Agent 的工具调用。")
    print("框架唯一的工作就是检测 Agent 类型的返回值，并切换活动 Agent。")
    print("无需状态机，无需 DSL；Agent prompt 本身就是路由逻辑。")


if __name__ == "__main__":
    main()
