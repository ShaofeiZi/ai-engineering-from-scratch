"""阶段 13 第 01 课 - 工具接口、four-step 循环、无 LLM.

实现了每个 2026 tool-calling 技术栈（OpenAI、Anthropic、Gemini、MCP、A2A）都采用的 describe -> decide -> execute -> observe 循环。
其中 "decide" 步骤用关键词路由器模拟，以便循环可以离线运行；
在第 02 课中将其替换为任意真实的服务商即可。

该测试框架：
  - 注册三个工具（add、get_time、get_weather）
  - 根据最小 JSON Schema 子集校验 tool-call 参数
  - 打印每一步，方便你查看整个编排流程
  - 将迭代次数限制在 MAX_TURNS 以防止失控循环

运行：python code/main.py
"""

from __future__ import annotations

import datetime as dt
import json
import re
import time
import uuid
from dataclasses import dataclass
from typing import Any, Callable


MAX_TURNS = 5


@dataclass
class Tool:
    name: str
    description: str
    input_schema: dict
    executor: Callable[[dict], Any]
    consequential: bool = False


def tool_add(args: dict) -> dict:
    return {"sum": args["a"] + args["b"]}


def tool_get_time(args: dict) -> dict:
    tz = args.get("timezone", "UTC")
    now = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    return {"now": now, "timezone": tz}


def tool_get_weather(args: dict) -> dict:
    fake = {"Bengaluru": 28, "Tokyo": 12, "Zurich": 4, "Lagos": 31}
    city = args["city"]
    units = args.get("units", "celsius")
    temp = fake.get(city, 20)
    return {"city": city, "temp": temp, "units": units}


REGISTRY: list[Tool] = [
    Tool(
        name="add",
        description=(
            "当用户要求计算两个数字之和时使用。"
            "不要用于减法、乘法或符号代数。"
        ),
        input_schema={
            "type": "object",
            "properties": {
                "a": {"type": "number"},
                "b": {"type": "number"},
            },
            "required": ["a", "b"],
        },
        executor=tool_add,
    ),
    Tool(
        name="get_time",
        description=(
            "当用户询问当前时间时使用。"
            "不要用于历史日期或未来日程安排。"
        ),
        input_schema={
            "type": "object",
            "properties": {
                "timezone": {"type": "string"},
            },
            "required": [],
        },
        executor=tool_get_time,
    ),
    Tool(
        name="get_weather",
        description=(
            "当用户询问指定城市的当前天气状况时使用。"
            "不要用于天气预报或历史天气数据。"
        ),
        input_schema={
            "type": "object",
            "properties": {
                "city": {"type": "string"},
                "units": {"type": "string", "enum": ["celsius", "fahrenheit"]},
            },
            "required": ["city"],
        },
        executor=tool_get_weather,
    ),
]


def validate(schema: dict, value: Any) -> list[str]:
    errors: list[str] = []
    t = schema.get("type")
    if t == "object":
        if not isinstance(value, dict):
            return [f"期望 object，实际为 {type(value).__name__}"]
        for field in schema.get("required", []):
            if field not in value:
                errors.append(f"缺少必填字段 '{field}'")
        for key, sub in schema.get("properties", {}).items():
            if key in value:
                errors.extend(validate(sub, value[key]))
        return errors
    if t == "number" and not isinstance(value, (int, float)):
        errors.append(f"期望 number，实际为 {type(value).__name__}")
    if t == "string" and not isinstance(value, str):
        errors.append(f"期望 string，实际为 {type(value).__name__}")
    if "enum" in schema and value not in schema["enum"]:
        errors.append(f"值 {value!r} 不在 enum {schema['enum']} 中")
    return errors


def fake_decide(user_msg: str, history: list[dict]) -> dict:
    """供模型使用的 Stand-in。通过关键词路由，使循环可离线运行。

    生产环境替代方案：将其替换为 provider.chat.completions.create，并传入
    tools=[t.input_schema for t in REGISTRY]。返回结构相同。
    """
    last = history[-1] if history else {}
    if last.get("role") == "tool":
        return {"content": f"根据工具输出生成的最终答案：{last.get('content')}"}
    msg = user_msg.lower()
    if re.search(r"\b(add|sum|plus)\b", msg) or "加" in msg or "求和" in msg:
        nums = [float(n) for n in re.findall(r"-?\d+\.?\d*", msg)]
        if len(nums) >= 2:
            return {
                "tool_calls": [
                    {
                        "id": f"call_{uuid.uuid4().hex[:8]}",
                        "name": "add",
                        "arguments": {"a": nums[0], "b": nums[1]},
                    }
                ]
            }
    if "time" in msg or "几点" in msg or "时间" in msg:
        return {
            "tool_calls": [
                {
                    "id": f"call_{uuid.uuid4().hex[:8]}",
                    "name": "get_time",
                    "arguments": {"timezone": "UTC"},
                }
            ]
        }
    match = re.search(r"weather in (\w+)", msg) or re.search(r"(\w+) 的天气", user_msg)
    if match:
        city = match.group(1).title()
        return {
            "tool_calls": [
                {
                    "id": f"call_{uuid.uuid4().hex[:8]}",
                    "name": "get_weather",
                    "arguments": {"city": city, "units": "celsius"},
                }
            ]
        }
    return {"content": "无法将该请求路由到任何已注册的工具。"}


def run_loop(user_msg: str) -> None:
    print("=" * 72)
    print(f"用户：{user_msg}")
    print("-" * 72)
    tools_by_name = {t.name: t for t in REGISTRY}
    history: list[dict] = [{"role": "user", "content": user_msg}]
    for turn in range(1, MAX_TURNS + 1):
        decision = fake_decide(user_msg, history)
        if "content" in decision:
            print(f"第 {turn} 轮 决策：最终答案")
            print(f"模型：{decision['content']}")
            return
        for call in decision["tool_calls"]:
            tool = tools_by_name.get(call["name"])
            print(f"第 {turn} 轮 决策：调用 {call['name']} id={call['id']}")
            print(f"           参数 = {json.dumps(call['arguments'])}")
            if tool is None:
                print(f"           错误：未知工具 {call['name']}")
                return
            errs = validate(tool.input_schema, call["arguments"])
            if errs:
                print(f"           校验错误：{errs}")
                return
            if tool.consequential:
                print("           确认关卡：工具有副作用，将进行确认")
            start = time.perf_counter()
            result = tool.executor(call["arguments"])
            ms = (time.perf_counter() - start) * 1000
            print(f"第 {turn} 轮 执行：{tool.name} -> {json.dumps(result)}"
                  f" [{ms:.2f} ms]")
            history.append({
                "role": "tool", "id": call["id"],
                "name": tool.name, "content": json.dumps(result),
            })
        print(f"第 {turn} 轮 观察：历史记录长度 = {len(history)}")
    print("循环终止：触发 MAX_TURNS 断路器")


def describe_registry() -> None:
    print("工具注册表")
    print("-" * 72)
    for t in REGISTRY:
        kind = "有副作用" if t.consequential else "纯函数"
        print(f"  {t.name:14s} [{kind}] - {t.description}")
    print()


def main() -> None:
    print("=" * 72)
    print("第 13 阶段第 01 课 - 工具接口")
    print("=" * 72)
    describe_registry()
    for query in (
        "请计算 7 加 35",
        "现在几点？",
        "请告诉我 Bengaluru 的天气",
        "写一首关于茶的俳句",
    ):
        run_loop(query)
        print()


if __name__ == "__main__":
    main()
