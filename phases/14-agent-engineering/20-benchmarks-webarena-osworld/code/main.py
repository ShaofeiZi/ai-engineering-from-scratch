"""简易网页智能体测试框架，带有基于执行的评估和轨迹效率分析。

模拟一个最小化购物应用；3 个任务配有黄金轨迹；一个脚本化代理
尝试每个任务；我们记录每个 OSWorld-Human. 的成功 + steps-over-gold
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


class ShoppingApp:
    def __init__(self) -> None:
        self.items = {
            "sku-001": {"name": "headphones", "price": 199},
            "sku-002": {"name": "keyboard", "price": 129},
            "sku-003": {"name": "mouse", "price": 59},
        }
        self.cart: dict[str, int] = {}
        self.orders: list[dict[str, Any]] = []

    def list_items(self) -> list[dict[str, Any]]:
        return [{"sku": sku, **meta} for sku, meta in self.items.items()]

    def add_to_cart(self, sku: str, qty: int = 1) -> str:
        if sku not in self.items:
            return "error: unknown sku"
        self.cart[sku] = self.cart.get(sku, 0) + qty
        return f"added {qty} x {sku}"

    def remove_from_cart(self, sku: str) -> str:
        if sku not in self.cart:
            return "error: not in cart"
        del self.cart[sku]
        return f"removed {sku}"

    def checkout(self) -> str:
        if not self.cart:
            return "error: empty cart"
        total = sum(self.items[sku]["price"] * qty
                    for sku, qty in self.cart.items())
        oid = f"ord-{len(self.orders) + 1:03d}"
        self.orders.append({"oid": oid, "items": dict(self.cart), "total": total})
        self.cart = {}
        return oid


@dataclass
class Task:
    tid: str
    description: str
    agent: Callable[[ShoppingApp], list[str]]
    gold_steps: int
    success: Callable[[ShoppingApp], bool]


def _agent_task_1(app: ShoppingApp) -> list[str]:
    trace: list[str] = []
    trace.append(f"list_items -> {len(app.list_items())} items")
    trace.append(f"add_to_cart sku-001 -> {app.add_to_cart('sku-001')}")
    trace.append(f"checkout -> {app.checkout()}")
    return trace


def _agent_task_2(app: ShoppingApp) -> list[str]:
    trace: list[str] = []
    trace.append(f"list_items")
    app.list_items()
    trace.append(f"add_to_cart sku-002 -> {app.add_to_cart('sku-002')}")
    trace.append(f"add_to_cart sku-003 -> {app.add_to_cart('sku-003')}")
    trace.append(f"checkout -> {app.checkout()}")
    return trace


def _agent_task_3(app: ShoppingApp) -> list[str]:
    trace: list[str] = []
    trace.append(f"list_items")
    app.list_items()
    trace.append(f"add_to_cart sku-001 -> {app.add_to_cart('sku-001')}")
    trace.append(f"add_to_cart sku-002 -> {app.add_to_cart('sku-002')}")
    trace.append("revised_choice: remove keyboard")
    trace.append(f"remove_from_cart sku-002 -> {app.remove_from_cart('sku-002')}")
    trace.append(f"add_to_cart sku-003 -> {app.add_to_cart('sku-003')}")
    trace.append(f"checkout -> {app.checkout()}")
    return trace


def main() -> None:
    print("=" * 70)
    print("WEBARENA/OSWORLD 风格测试框架 — 第 14 阶段，第 20 课")
    print("=" * 70)

    tasks = [
        Task(
            tid="buy_headphones",
            description="buy the headphones",
            agent=_agent_task_1,
            gold_steps=3,
            success=lambda app: any(
                o["items"].get("sku-001") == 1 for o in app.orders
            ),
        ),
        Task(
            tid="buy_bundle",
            description="buy keyboard + mouse as a bundle",
            agent=_agent_task_2,
            gold_steps=4,
            success=lambda app: any(
                o["items"].get("sku-002") == 1 and o["items"].get("sku-003") == 1
                for o in app.orders
            ),
        ),
        Task(
            tid="revised_order",
            description="swap keyboard for mouse mid-order",
            agent=_agent_task_3,
            gold_steps=5,
            success=lambda app: any(
                o["items"].get("sku-001") == 1 and
                o["items"].get("sku-003") == 1 and
                "sku-002" not in o["items"]
                for o in app.orders
            ),
        ),
    ]

    total_success = 0
    total_steps = 0
    total_gold = 0
    for task in tasks:
        app = ShoppingApp()
        trace = task.agent(app)
        ok = task.success(app)
        steps = len(trace)
        efficiency = steps / task.gold_steps
        print(f"\n[{task.tid}] {task.description}")
        print(f"  成功: {ok}")
        print(f"  步骤数： {steps}  （黄金轨迹 {task.gold_steps}，"
              f"efficiency {efficiency:.2f}x)")
        for line in trace:
            print(f"    - {line}")
        if ok:
            total_success += 1
        total_steps += steps
        total_gold += task.gold_steps

    print(f"\n汇总")
    print(f"  成功率:     {total_success}/{len(tasks)}")
    print(f"  步骤效率:  {total_steps / total_gold:.2f}x 相对黄金轨迹")
    print()
    print("WebArena：基于执行，使用 gym API，由状态检查决定成功与否。")
    print("OSWorld-Human：黄金轨迹揭示了 1.4-2.7x 的步骤低效。")


if __name__ == "__main__":
    main()
