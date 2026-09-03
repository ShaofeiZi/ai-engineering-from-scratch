"""多模态智能体毕业项目 — 动作模式 + 智能体循环 + 10 任务基准测试。

仅使用标准库。包含一个确定性页面跳转的模拟浏览器，一个从固定策略表发射动作的 VLM，
以及一个跟踪 10 个合成预订网站任务进度的外层循环。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field


ACTION_SCHEMA = {
    "click": ["x", "y", "element_desc"],
    "type":  ["text", "x", "y"],
    "scroll": ["direction", "amount"],
    "drag": ["x0", "y0", "x1", "y1"],
    "select": ["option_index"],
    "hover": ["x", "y"],
    "navigate": ["url"],
    "wait": ["ms"],
    "screenshot_region": ["x0", "y0", "x1", "y1"],
    "done": ["success", "explanation"],
}


@dataclass
class BrowserState:
    url: str = "https://mock-booking/"
    page: str = "home"
    filled: dict = field(default_factory=dict)


@dataclass
class Task:
    goal: str
    plan: list[dict]
    expected_page: str


def mock_tasks() -> list[Task]:
    return [
        Task(goal="预订 4 月 15 日从纽约到东京的航班",
             plan=[
                 {"action": "click", "x": 120, "y": 200, "element_desc": "搜索"},
                 {"action": "type",  "text": "Tokyo",  "x": 300, "y": 240},
                 {"action": "click", "x": 400, "y": 240, "element_desc": "日期"},
                 {"action": "select", "option_index": 15},
                 {"action": "click", "x": 500, "y": 400, "element_desc": "预订"},
                 {"action": "done", "success": True, "explanation": "已预订"},
             ],
             expected_page="confirmation"),
        Task(goal="为用户 alice@x.com 重置密码",
             plan=[
                 {"action": "click", "x": 50, "y": 50, "element_desc": "登录"},
                 {"action": "click", "x": 100, "y": 200, "element_desc": "忘记密码"},
                 {"action": "type",  "text": "alice@x.com", "x": 200, "y": 300},
                 {"action": "click", "x": 300, "y": 400, "element_desc": "提交"},
                 {"action": "done", "success": True, "explanation": "重置邮件已发送"},
             ],
             expected_page="reset_sent"),
    ]


def apply_action(state: BrowserState, action: dict) -> BrowserState:
    new = BrowserState(url=state.url, page=state.page, filled=dict(state.filled))
    act = action["action"]
    if act == "click":
        # element_id 是与显示语言无关的控制语义；element_desc 仍兼容原英文轨迹和中文 UI。
        element_id = str(action.get("element_id", "")).casefold()
        desc = str(action.get("element_desc", "")).casefold()
        if element_id in {"book", "submit"} or any(
            label in desc for label in ("book", "submit", "预订", "提交")
        ):
            new.page = "confirmation"
        elif element_id == "forgot_password" or any(
            label in desc for label in ("forgot", "忘记")
        ):
            new.page = "reset_sent"
        elif element_id == "login" or any(
            label in desc for label in ("login", "登录")
        ):
            new.page = "login"
        elif element_id == "search" or any(
            label in desc for label in ("search", "搜索")
        ):
            new.page = "search"
    elif act == "type":
        new.filled[action.get("x", 0)] = action.get("text", "")
    elif act == "select":
        new.filled["select_idx"] = action.get("option_index", 0)
    elif act == "done":
        # 仅作为终端信号；不要覆盖工作流页面状态
        pass
    return new


def run_task(task: Task) -> dict:
    state = BrowserState()
    trace = []
    for step, action in enumerate(task.plan, 1):
        trace.append((step, action["action"], action.get("element_desc", "")))
        state = apply_action(state, action)
    success = (state.page == task.expected_page)
    return {"goal": task.goal, "trace": trace, "final_page": state.page,
            "success": success}


def print_schema() -> None:
    print("\n动作 SCHEMA")
    print("-" * 60)
    for act, params in ACTION_SCHEMA.items():
        print(f"  {act:<18}{params}")


def run_benchmark() -> None:
    print("\n基准测试——2 个示例任务")
    print("-" * 60)
    tasks = mock_tasks()
    total = len(tasks)
    passed = 0
    for task in tasks:
        r = run_task(task)
        status = "通过" if r["success"] else "失败"
        print(f"  [{status}] {r['goal']}")
        for step, act, desc in r["trace"]:
            print(f"    步骤 {step}：{act:<10} {desc}")
        if r["success"]:
            passed += 1
    print(f"\n  得分：{passed}/{total}")


def benchmark_leaderboard() -> None:
    print("\n2026 年多模态智能体基准快照")
    print("-" * 60)
    rows = [
        ("ScreenSpot-Pro",  "Qwen2.5-VL-72B 85",  "Claude Opus 4.7 ~90"),
        ("VisualWebArena",  "开放模型约 20",       "Gemini 3 Pro 约 27"),
        ("WebArena",        "开放模型约 35",       "饱和水平约 60"),
        ("AgentVista",      "开放模型约 10-20",    "前沿水平 27-40"),
        ("Ferret-UI mobile","Qwen2.5-VL ~70",     "GPT-5 ~82"),
    ]
    print(f"  {'基准':<20}{'开放模型':<26}{'前沿水平'}")
    for r in rows:
        print(f"  {r[0]:<20}{r[1]:<26}{r[2]}")


def main() -> None:
    print("=" * 60)
    print("多模态智能体毕业项目（第 12 阶段，第 25 课）")
    print("=" * 60)

    print_schema()
    run_benchmark()
    benchmark_leaderboard()

    print("\n记忆压缩策略")
    print("-" * 60)
    print("  摘要链：定期文本摘要，丢弃旧截图")
    print("  跳帧  ：保留首尾及每隔第 3 张")
    print("  仅日志：上下文中仅保留动作日志（Claude 计算机操作）")
    print("  最优  ：日志 + 最近 2 张截图 + 摘要的混合方案")

    print("\n你现在已完成第 12 阶段")
    print("-" * 60)
    print("  从补丁到智能体。共 25 节课涵盖：")
    print("  感知 -> 融合 -> 生成 -> 音频 -> 机器人 -> RAG -> 智能体")
    print("  每个原语都可追溯到你可以阅读的某篇 arxiv 论文。")


if __name__ == "__main__":
    main()
