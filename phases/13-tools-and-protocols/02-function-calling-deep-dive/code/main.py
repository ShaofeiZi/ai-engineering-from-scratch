"""Phase 13 Lesson 02 - 函数调用深度解析：跨三大服务商。

采用一个规范化工具，生成 OpenAI、Anthropic 和 Gemini 的声明
载荷，随后将每种形状的 hand-crafted 响应解析回 provider-agnostic 调用对象。仅使用标准库；无网络请求。

运行：python code/main.py
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from typing import Any


@dataclass
class Tool:
    name: str
    description: str
    input_schema: dict
    strict: bool = True


@dataclass
class Call:
    id: str
    name: str
    args: dict


@dataclass
class ToolChoice:
    mode: str
    tool_name: str | None = None


WEATHER = Tool(
    name="get_weather",
    description=(
        "当用户询问指定城市的当前天气状况时使用。"
        "不要用于天气预报或历史天气数据。"
    ),
    input_schema={
        "type": "object",
        "properties": {
            "city": {"type": "string"},
            "units": {"type": ["string", "null"], "enum": ["celsius", "fahrenheit"]},
        },
        "required": ["city", "units"],
        "additionalProperties": False,
    },
)


def to_openai(tool: Tool) -> dict:
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.input_schema,
            "strict": tool.strict,
        },
    }


def to_anthropic(tool: Tool) -> dict:
    return {
        "name": tool.name,
        "description": tool.description,
        "input_schema": tool.input_schema,
    }


def _gemini_schema(node: Any) -> Any:
    if isinstance(node, dict):
        out: dict = {}
        for k, v in node.items():
            if k == "additionalProperties":
                continue
            if k == "type" and isinstance(v, str):
                out["type"] = v.upper()
                continue
            out[k] = _gemini_schema(v)
        return out
    if isinstance(node, list):
        return [_gemini_schema(x) for x in node]
    return node


def to_gemini(tool: Tool) -> dict:
    return {
        "functionDeclarations": [
            {
                "name": tool.name,
                "description": tool.description,
                "parameters": _gemini_schema(tool.input_schema),
            }
        ]
    }


def tool_choice_openai(tc: ToolChoice) -> Any:
    if tc.mode == "auto":
        return "auto"
    if tc.mode == "none":
        return "none"
    if tc.mode == "required":
        return "required"
    if tc.mode == "force":
        return {"type": "function", "function": {"name": tc.tool_name}}
    raise ValueError(tc.mode)


def tool_choice_anthropic(tc: ToolChoice) -> dict:
    if tc.mode == "auto":
        return {"type": "auto"}
    if tc.mode == "none":
        return {"type": "none"}
    if tc.mode == "required":
        return {"type": "any"}
    if tc.mode == "force":
        return {"type": "tool", "name": tc.tool_name}
    raise ValueError(tc.mode)


def tool_choice_gemini(tc: ToolChoice) -> dict:
    mode_map = {"auto": "AUTO", "none": "NONE", "required": "ANY"}
    if tc.mode in mode_map:
        return {"function_calling_config": {"mode": mode_map[tc.mode]}}
    if tc.mode == "force":
        return {
            "function_calling_config": {
                "mode": "ANY",
                "allowed_function_names": [tc.tool_name],
            }
        }
    raise ValueError(tc.mode)


OPENAI_RESPONSE = {
    "choices": [
        {
            "message": {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_abc123",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"city":"Bengaluru","units":"celsius"}',
                        },
                    }
                ],
            },
            "finish_reason": "tool_calls",
        }
    ]
}

ANTHROPIC_RESPONSE = {
    "id": "msg_01",
    "type": "message",
    "role": "assistant",
    "content": [
        {"type": "text", "text": "Looking that up."},
        {
            "type": "tool_use",
            "id": "toolu_xyz789",
            "name": "get_weather",
            "input": {"city": "Bengaluru", "units": "celsius"},
        },
    ],
    "stop_reason": "tool_use",
}

GEMINI_RESPONSE = {
    "candidates": [
        {
            "content": {
                "role": "model",
                "parts": [
                    {
                        "functionCall": {
                            "id": "fc-9a3d",
                            "name": "get_weather",
                            "args": {"city": "Bengaluru", "units": "celsius"},
                        }
                    }
                ],
            },
            "finishReason": "STOP",
        }
    ]
}


def parse_openai(resp: dict) -> list[Call]:
    msg = resp["choices"][0]["message"]
    calls = []
    for tc in msg.get("tool_calls", []):
        fn = tc["function"]
        calls.append(Call(id=tc["id"], name=fn["name"], args=json.loads(fn["arguments"])))
    return calls


def parse_anthropic(resp: dict) -> list[Call]:
    calls = []
    for block in resp.get("content", []):
        if block.get("type") == "tool_use":
            calls.append(Call(id=block["id"], name=block["name"], args=block["input"]))
    return calls


def parse_gemini(resp: dict) -> list[Call]:
    calls = []
    for part in resp["candidates"][0]["content"].get("parts", []):
        if "functionCall" in part:
            fc = part["functionCall"]
            calls.append(Call(id=fc.get("id", ""), name=fc["name"], args=fc["args"]))
    return calls


def diff_line(a: str, b: str, c: str) -> None:
    print(f"  OpenAI    : {a}")
    print(f"  Anthropic : {b}")
    print(f"  Gemini    : {c}")


def main() -> None:
    print("=" * 72)
    print("第 13 阶段第 02 课 - 函数调用深度解析")
    print("=" * 72)
    print("\n规范化工具：")
    print(json.dumps(asdict(WEATHER), indent=2))

    print("\n--- 服务商声明 ---")
    print("\nOpenAI：")
    print(json.dumps(to_openai(WEATHER), indent=2))
    print("\nAnthropic：")
    print(json.dumps(to_anthropic(WEATHER), indent=2))
    print("\nGemini：")
    print(json.dumps(to_gemini(WEATHER), indent=2))

    print("\n--- tool_choice 转换 ---")
    for mode in ("auto", "none", "required", "force"):
        tc = ToolChoice(mode=mode, tool_name="get_weather" if mode == "force" else None)
        print(f"\nmode = {mode!r}")
        diff_line(
            json.dumps(tool_choice_openai(tc)),
            json.dumps(tool_choice_anthropic(tc)),
            json.dumps(tool_choice_gemini(tc)),
        )

    print("\n--- 解析服务商响应 ---")
    oa = parse_openai(OPENAI_RESPONSE)[0]
    an = parse_anthropic(ANTHROPIC_RESPONSE)[0]
    gm = parse_gemini(GEMINI_RESPONSE)[0]
    print(f"\nOpenAI     : {oa}")
    print(f"Anthropic  : {an}")
    print(f"Gemini     : {gm}")

    print("\n--- id 前缀 ---")
    print(f"  OpenAI     : {oa.id} (call_...)")
    print(f"  Anthropic  : {an.id} (toolu_...)")
    print(f"  Gemini     : {gm.id} (fc- / Gemini 3+ 起为 UUID)")

    print("\n--- 解析后 args 类型 ---")
    print(f"  OpenAI 原始 args 类型 : string -> {type(oa.args).__name__}")
    print(f"  Anthropic 原始 args   : object -> {type(an.args).__name__}")
    print(f"  Gemini 原始 args      : object -> {type(gm.args).__name__}")

    print("\n--- 等价性检查 ---")
    all_names = {oa.name, an.name, gm.name}
    all_args = {json.dumps(oa.args, sort_keys=True),
                json.dumps(an.args, sort_keys=True),
                json.dumps(gm.args, sort_keys=True)}
    print(f"  各服务商工具名称是否一致 : {len(all_names) == 1}")
    print(f"  各服务商 args 载荷是否一致 : {len(all_args) == 1}")


if __name__ == "__main__":
    main()
