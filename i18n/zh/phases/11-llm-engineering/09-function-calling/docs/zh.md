# 函数调用与工具使用

> 大语言模型本身什么也做不了。它们只会生成文本，这就是全部能力。它们无法查看天气、查询数据库、发送电子邮件、运行代码或读取文件。你见过的每个“AI 智能体”，本质上都是由大语言模型生成一段 JSON，说明要调用哪个函数——然后由你的代码真正调用它。模型是大脑，工具是双手，函数调用则是连接二者的神经系统。

**Type:** 构建
**Languages:** Python
**Prerequisites:** 阶段 11 第 03 课（结构化输出）
**Time:** 约 75 分钟
**Related:** 阶段 11 · 14（模型上下文协议）——当一个工具需要在多个宿主之间共享时，应从内联函数调用升级为 MCP 服务器。本课介绍内联场景，MCP 则介绍协议场景。

## 学习目标

- 实现函数调用循环：定义工具 Schema、解析模型生成的工具调用 JSON、执行函数并返回结果
- 设计带有清晰描述和类型化参数的工具 Schema，让模型能够可靠调用
- 构建多轮智能体循环，通过串联多个函数调用回答复杂查询
- 处理函数调用的边界情况：并行工具调用、错误传播，以及防止无限工具循环

## 问题

你构建了一个聊天机器人。用户问：“东京现在天气怎么样？”

模型回答：“我无法访问实时天气数据，但根据季节判断，东京可能在 15 摄氏度左右……”

这只是披着免责声明外衣的幻觉。模型不知道天气，也永远不会知道。天气每小时都在变化，而模型的训练数据已经是几个月前的了。

正确答案需要调用 OpenWeatherMap API，取得当前温度并返回真实数值。模型无法调用 API，但你的代码可以。缺少的一环是一套结构化协议：它让模型能够表达“我需要用这些参数调用天气 API”，也让你的代码能够执行调用，再把结果送回模型。

这就是函数调用。模型输出结构化 JSON，描述要调用哪个函数以及使用哪些参数；应用程序执行函数；结果回到对话中；模型再用该结果生成最终答案。

没有函数调用时，大语言模型只是百科全书；有了函数调用，它们才成为智能体。

## 概念

### 函数调用循环

每次工具使用交互都遵循相同的五步循环。

```mermaid
sequenceDiagram
    participant U as User
    participant A as Application
    participant M as Model
    participant T as Tool

    U->>A: "What's the weather in Tokyo?"
    A->>M: messages + tool definitions
    M->>A: tool_call: get_weather(city="Tokyo")
    A->>T: Execute get_weather("Tokyo")
    T->>A: {"temp": 18, "condition": "cloudy"}
    A->>M: tool_result + conversation
    M->>A: "It's 18C and cloudy in Tokyo."
    A->>U: Final response
```

第 1 步：用户发送消息。第 2 步：模型接收消息和工具定义（描述可用函数的 JSON Schema）。第 3 步：模型不直接回复文本，而是输出工具调用——包含函数名和参数的结构化 JSON 对象。第 4 步：你的代码执行该函数并捕获结果。第 5 步：结果返回模型，模型此时拥有真实数据，可以生成最终答案。

模型从不执行任何操作。它只决定调用什么，以及使用哪些参数。你的代码才是执行者。

### 工具定义：JSON Schema 契约

每个工具都由一份 JSON Schema 定义，它告诉模型该函数做什么、接受哪些参数，以及这些参数必须是什么类型。

```json
{
  "type": "function",
  "function": {
    "name": "get_weather",
    "description": "Get current weather for a city. Returns temperature in Celsius and conditions.",
    "parameters": {
      "type": "object",
      "properties": {
        "city": {
          "type": "string",
          "description": "City name, e.g. 'Tokyo' or 'San Francisco'"
        },
        "units": {
          "type": "string",
          "enum": ["celsius", "fahrenheit"],
          "description": "Temperature units"
        }
      },
      "required": ["city"]
    }
  }
}
```

`description` 字段至关重要。模型会读取它们，以决定何时以及如何使用工具。“gets weather”之类含糊的描述，会比“Get current weather for a city. Returns temperature in Celsius and conditions.”导致更差的工具选择。描述本身就是用于工具选择的提示词。

### 提供商对比

所有主流提供商都支持函数调用，但 API 表面有所不同。

| 提供商 | API 参数 | 工具调用格式 | 并行调用 | 强制调用 |
|----------|--------------|-----------------|---------------|----------------|
| OpenAI (GPT-5, o4) | `tools` | `tool_calls[].function` | 支持（每轮多个） | `tool_choice="required"` |
| Anthropic (Claude 4.6/4.7) | `tools` | `content[].type="tool_use"` | 支持（多个内容块） | `tool_choice={"type":"any"}` |
| Google (Gemini 3) | `function_declarations` | `functionCall` | 支持 | `function_calling_config` |
| 开放权重（Llama 4、Qwen3、DeepSeek-V3） | Llama 4 原生支持 `tools`；其他模型使用 Hermes 或 ChatML | 不一 | 取决于模型 | 基于提示词；支持时可用 `tool_choice` |

到 2026 年，三家闭源提供商已经收敛到几乎相同的、基于 JSON Schema 的格式。Llama 4 原生提供 `tools` 字段，其结构与 OpenAI 一致。开放权重微调模型仍各不相同——第三方微调中最常见的是 Hermes 格式。需要跨宿主共享工具时，应优先使用 MCP（阶段 11 · 14），而不是内联函数调用——同一个服务器可供所有宿主使用。

### 工具选择：自动、必选与指定

你可以控制模型何时使用工具。

**自动**（默认）：模型自行判断是调用工具还是直接回答。“2+2 等于多少？”会直接回答；“天气怎么样？”则会调用工具。

**必选**：模型必须调用至少一个工具。当你确定用户意图需要工具时使用此模式，防止模型凭空猜测而不查询真实数据。

**指定函数**：强制模型调用某个特定函数。`tool_choice={"type":"function", "function": {"name": "get_weather"}}` 保证无论查询内容如何，都会调用天气工具。它适合路由场景——上游逻辑已经确定需要哪个工具时使用。

### 并行函数调用

GPT-4o 和 Claude 可以在一轮中调用多个函数。用户问：“东京和纽约的天气怎么样？”模型会同时输出两个工具调用：

```json
[
  {"name": "get_weather", "arguments": {"city": "Tokyo"}},
  {"name": "get_weather", "arguments": {"city": "New York"}}
]
```

你的代码执行两个调用（最好并发执行），返回两份结果，再由模型综合成一个回答。这样可将往返次数从 2 次减为 1 次。对于每次查询需要调用 5～10 个工具的智能体，并行调用能把延迟降低 60%～80%。

### 结构化输出与函数调用

第 03 课介绍了结构化输出。函数调用使用相同的 JSON Schema 机制，但目的不同。

**结构化输出**：强制模型按特定形状生成数据，输出本身就是最终产物。例如：从文本中提取产品信息，得到 `{name, price, in_stock}`。

**函数调用**：模型声明要执行某个操作的意图，输出只是中间步骤。例如 `get_weather(city="Tokyo")`——模型请求执行操作，而不是生成最终答案。

需要提取数据时使用结构化输出；需要模型与外部系统交互时使用函数调用。

### 安全：不可妥协的规则

函数调用是你能赋予大语言模型的最危险能力。模型会选择执行什么。如果工具集中包含数据库查询，模型就会构造查询；如果包含 Shell 命令，模型就会编写命令。

**规则 1：绝不能把模型生成的 SQL 直接发送给数据库。** 模型可能而且确实会生成 DROP TABLE、UNION 注入，或返回所有行的查询。始终使用参数化查询，始终验证，始终通过操作白名单加以限制。

**规则 2：使用函数白名单。** 模型只能调用你显式定义的函数。绝不要构建一个“按名称执行任意函数”的通用工具。如果内部有 50 个函数，只暴露用户所需的 5 个。

**规则 3：验证参数。** 模型可能传入城市名 `"; DROP TABLE users; --"`。执行前，要根据预期类型、范围和格式验证每个参数。

**规则 4：净化工具结果。** 如果工具返回敏感数据（API 密钥、个人身份信息、内部错误），应先过滤，再发送给模型。模型可能把工具结果逐字写进回答。

**规则 5：限制工具调用频率。** 循环中的模型可能调用工具数百次。请设置上限（每次对话 10～20 次较为合理），打断无限循环。

### 错误处理

工具会失败：API 会超时，数据库会宕机，文件会不存在。模型需要知道工具何时失败，以及失败原因。

把错误作为结构化工具结果返回，而不是抛出异常：

```json
{
  "error": true,
  "message": "City 'Toky' not found. Did you mean 'Tokyo'?",
  "code": "CITY_NOT_FOUND"
}
```

模型读取错误后，可以调整参数并重试。模型很擅长根据结构化错误消息自我纠正，却不善于从空响应或笼统的“出了点问题”中恢复。

### MCP：模型上下文协议

MCP 是 Anthropic 提出的开放式工具互操作标准。它不让每个应用分别定义自己的工具，而是提供统一协议：工具由 MCP 服务器提供，MCP 客户端（如 Claude Code、Cursor 或你的应用）负责使用。

一个 MCP 服务器可以把工具暴露给任意兼容客户端。Postgres MCP 服务器能让任何兼容 MCP 的智能体访问数据库；GitHub MCP 服务器能让任何智能体访问仓库。工具只定义一次，即可随处使用。

MCP 之于函数调用，就像 HTTP 之于网络。它标准化了传输层，让工具可以移植。

```figure
mx-tool-call-loop
```

## 动手构建

### 第 1 步：定义工具注册表

构建一个注册表，存储工具定义及其实现。每个工具都有一份 JSON Schema 定义（模型看到的内容）和一个 Python 函数（代码实际执行的内容）。

```python
import json
import math
import time
import hashlib


TOOL_REGISTRY = {}


def register_tool(name, description, parameters, function):
    TOOL_REGISTRY[name] = {
        "definition": {
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": parameters,
            },
        },
        "function": function,
    }
```

### 第 2 步：实现五个工具

构建计算器、天气查询、Web 搜索模拟器、文件读取器和代码运行器。

```python
def calculator(expression, precision=2):
    allowed = set("0123456789+-*/.() ")
    if not all(c in allowed for c in expression):
        return {"error": True, "message": f"Invalid characters in expression: {expression}"}
    try:
        result = eval(expression, {"__builtins__": {}}, {"math": math})
        return {"result": round(float(result), precision), "expression": expression}
    except Exception as e:
        return {"error": True, "message": str(e)}


WEATHER_DB = {
    "tokyo": {"temp_c": 18, "condition": "cloudy", "humidity": 72, "wind_kph": 14},
    "new york": {"temp_c": 22, "condition": "sunny", "humidity": 45, "wind_kph": 8},
    "london": {"temp_c": 12, "condition": "rainy", "humidity": 88, "wind_kph": 22},
    "san francisco": {"temp_c": 16, "condition": "foggy", "humidity": 80, "wind_kph": 18},
    "sydney": {"temp_c": 25, "condition": "sunny", "humidity": 55, "wind_kph": 10},
}


def get_weather(city, units="celsius"):
    key = city.lower().strip()
    if key not in WEATHER_DB:
        suggestions = [c for c in WEATHER_DB if c.startswith(key[:3])]
        return {
            "error": True,
            "message": f"City '{city}' not found.",
            "suggestions": suggestions,
            "code": "CITY_NOT_FOUND",
        }
    data = WEATHER_DB[key].copy()
    if units == "fahrenheit":
        data["temp_f"] = round(data["temp_c"] * 9 / 5 + 32, 1)
        del data["temp_c"]
    data["city"] = city
    return data


SEARCH_DB = {
    "python function calling": [
        {"title": "OpenAI Function Calling Guide", "url": "https://platform.openai.com/docs/guides/function-calling", "snippet": "Learn how to connect LLMs to external tools."},
        {"title": "Anthropic Tool Use", "url": "https://docs.anthropic.com/en/docs/tool-use", "snippet": "Claude can interact with external tools and APIs."},
    ],
    "MCP protocol": [
        {"title": "Model Context Protocol", "url": "https://modelcontextprotocol.io", "snippet": "An open standard for connecting AI models to data sources."},
    ],
    "weather API": [
        {"title": "OpenWeatherMap API", "url": "https://openweathermap.org/api", "snippet": "Free weather API with current, forecast, and historical data."},
    ],
}


def web_search(query, max_results=3):
    key = query.lower().strip()
    for db_key, results in SEARCH_DB.items():
        if db_key in key or key in db_key:
            return {"query": query, "results": results[:max_results], "total": len(results)}
    return {"query": query, "results": [], "total": 0}


FILE_SYSTEM = {
    "data/config.json": '{"model": "gpt-4o", "temperature": 0.7, "max_tokens": 4096}',
    "data/users.csv": "name,email,role\nAlice,alice@example.com,admin\nBob,bob@example.com,user",
    "README.md": "# My Project\nA tool-use agent built from scratch.",
}


def read_file(path):
    if ".." in path or path.startswith("/"):
        return {"error": True, "message": "Path traversal not allowed.", "code": "FORBIDDEN"}
    if path not in FILE_SYSTEM:
        available = list(FILE_SYSTEM.keys())
        return {"error": True, "message": f"File '{path}' not found.", "available_files": available, "code": "NOT_FOUND"}
    content = FILE_SYSTEM[path]
    return {"path": path, "content": content, "size_bytes": len(content), "lines": content.count("\n") + 1}


def run_code(code, language="python"):
    if language != "python":
        return {"error": True, "message": f"Language '{language}' not supported. Only 'python' is available."}
    forbidden = ["import os", "import sys", "import subprocess", "exec(", "eval(", "__import__", "open("]
    for pattern in forbidden:
        if pattern in code:
            return {"error": True, "message": f"Forbidden operation: {pattern}", "code": "SECURITY_VIOLATION"}
    try:
        local_vars = {}
        exec(code, {"__builtins__": {"print": print, "range": range, "len": len, "str": str, "int": int, "float": float, "list": list, "dict": dict, "sum": sum, "min": min, "max": max, "abs": abs, "round": round, "sorted": sorted, "enumerate": enumerate, "zip": zip, "map": map, "filter": filter, "math": math}}, local_vars)
        result = local_vars.get("result", None)
        return {"success": True, "result": result, "variables": {k: str(v) for k, v in local_vars.items() if not k.startswith("_")}}
    except Exception as e:
        return {"error": True, "message": f"{type(e).__name__}: {e}"}
```

### 第 3 步：注册全部工具

```python
def register_all_tools():
    register_tool(
        "calculator", "Evaluate a mathematical expression. Supports +, -, *, /, parentheses, and decimals. Returns the numeric result.",
        {"type": "object", "properties": {"expression": {"type": "string", "description": "Math expression, e.g. '(10 + 5) * 3'"}, "precision": {"type": "integer", "description": "Decimal places in result", "default": 2}}, "required": ["expression"]},
        calculator,
    )
    register_tool(
        "get_weather", "Get current weather for a city. Returns temperature, condition, humidity, and wind speed.",
        {"type": "object", "properties": {"city": {"type": "string", "description": "City name, e.g. 'Tokyo' or 'San Francisco'"}, "units": {"type": "string", "enum": ["celsius", "fahrenheit"], "description": "Temperature units, defaults to celsius"}}, "required": ["city"]},
        get_weather,
    )
    register_tool(
        "web_search", "Search the web for information. Returns a list of results with title, URL, and snippet.",
        {"type": "object", "properties": {"query": {"type": "string", "description": "Search query"}, "max_results": {"type": "integer", "description": "Maximum results to return", "default": 3}}, "required": ["query"]},
        web_search,
    )
    register_tool(
        "read_file", "Read the contents of a file. Returns the file content, size, and line count.",
        {"type": "object", "properties": {"path": {"type": "string", "description": "Relative file path, e.g. 'data/config.json'"}}, "required": ["path"]},
        read_file,
    )
    register_tool(
        "run_code", "Execute Python code in a sandboxed environment. Set a 'result' variable to return output.",
        {"type": "object", "properties": {"code": {"type": "string", "description": "Python code to execute"}, "language": {"type": "string", "enum": ["python"], "description": "Programming language"}}, "required": ["code"]},
        run_code,
    )
```

### 第 4 步：构建函数调用循环

这是核心引擎。它模拟模型决定调用哪个工具、执行工具，再把结果送回模型的过程。

```python
def simulate_model_decision(user_message, tools, conversation_history):
    msg = user_message.lower()

    if any(word in msg for word in ["weather", "temperature", "forecast"]):
        cities = []
        for city in WEATHER_DB:
            if city in msg:
                cities.append(city)
        if not cities:
            for word in msg.split():
                if word.capitalize() in [c.title() for c in WEATHER_DB]:
                    cities.append(word)
        if not cities:
            cities = ["tokyo"]
        calls = []
        for city in cities:
            calls.append({"name": "get_weather", "arguments": {"city": city.title()}})
        return calls

    if any(word in msg for word in ["calculate", "compute", "math", "what is", "how much"]):
        for token in msg.split():
            if any(c in token for c in "+-*/"):
                return [{"name": "calculator", "arguments": {"expression": token}}]
        if "+" in msg or "-" in msg or "*" in msg or "/" in msg:
            expr = "".join(c for c in msg if c in "0123456789+-*/.() ")
            if expr.strip():
                return [{"name": "calculator", "arguments": {"expression": expr.strip()}}]
        return [{"name": "calculator", "arguments": {"expression": "0"}}]

    if any(word in msg for word in ["search", "find", "look up", "google"]):
        query = msg.replace("search for", "").replace("look up", "").replace("find", "").strip()
        return [{"name": "web_search", "arguments": {"query": query}}]

    if any(word in msg for word in ["read", "file", "open", "cat", "show"]):
        for path in FILE_SYSTEM:
            if path.split("/")[-1].split(".")[0] in msg:
                return [{"name": "read_file", "arguments": {"path": path}}]
        return [{"name": "read_file", "arguments": {"path": "README.md"}}]

    if any(word in msg for word in ["run", "execute", "code", "python"]):
        return [{"name": "run_code", "arguments": {"code": "result = 'Hello from the sandbox!'", "language": "python"}}]

    return []


def execute_tool_call(tool_call):
    name = tool_call["name"]
    args = tool_call["arguments"]

    if name not in TOOL_REGISTRY:
        return {"error": True, "message": f"Unknown tool: {name}", "code": "UNKNOWN_TOOL"}

    tool = TOOL_REGISTRY[name]
    func = tool["function"]
    start = time.time()

    try:
        result = func(**args)
    except TypeError as e:
        result = {"error": True, "message": f"Invalid arguments: {e}"}

    elapsed_ms = round((time.time() - start) * 1000, 2)
    return {"tool": name, "result": result, "execution_time_ms": elapsed_ms}


def run_function_calling_loop(user_message, max_iterations=5):
    conversation = [{"role": "user", "content": user_message}]
    tool_definitions = [t["definition"] for t in TOOL_REGISTRY.values()]
    all_tool_results = []

    for iteration in range(max_iterations):
        tool_calls = simulate_model_decision(user_message, tool_definitions, conversation)

        if not tool_calls:
            break

        results = []
        for call in tool_calls:
            result = execute_tool_call(call)
            results.append(result)

        conversation.append({"role": "assistant", "content": None, "tool_calls": tool_calls})

        for result in results:
            conversation.append({"role": "tool", "content": json.dumps(result["result"]), "tool_name": result["tool"]})

        all_tool_results.extend(results)
        break

    return {"conversation": conversation, "tool_results": all_tool_results, "iterations": iteration + 1 if tool_calls else 0}
```

### 第 5 步：参数验证

构建一个验证器，在执行前根据 JSON Schema 检查工具调用参数。

```python
def validate_tool_arguments(tool_name, arguments):
    if tool_name not in TOOL_REGISTRY:
        return [f"Unknown tool: {tool_name}"]

    schema = TOOL_REGISTRY[tool_name]["definition"]["function"]["parameters"]
    errors = []

    if not isinstance(arguments, dict):
        return [f"Arguments must be an object, got {type(arguments).__name__}"]

    for required_field in schema.get("required", []):
        if required_field not in arguments:
            errors.append(f"Missing required argument: {required_field}")

    properties = schema.get("properties", {})
    for arg_name, arg_value in arguments.items():
        if arg_name not in properties:
            errors.append(f"Unknown argument: {arg_name}")
            continue

        prop_schema = properties[arg_name]
        expected_type = prop_schema.get("type")

        type_checks = {"string": str, "integer": int, "number": (int, float), "boolean": bool, "array": list, "object": dict}
        if expected_type in type_checks:
            if not isinstance(arg_value, type_checks[expected_type]):
                errors.append(f"Argument '{arg_name}': expected {expected_type}, got {type(arg_value).__name__}")

        if "enum" in prop_schema and arg_value not in prop_schema["enum"]:
            errors.append(f"Argument '{arg_name}': '{arg_value}' not in {prop_schema['enum']}")

    return errors
```

### 第 6 步：运行演示

```python
def run_demo():
    register_all_tools()

    print("=" * 60)
    print("  Function Calling & Tool Use Demo")
    print("=" * 60)

    print("\n--- Registered Tools ---")
    for name, tool in TOOL_REGISTRY.items():
        desc = tool["definition"]["function"]["description"][:60]
        params = list(tool["definition"]["function"]["parameters"].get("properties", {}).keys())
        print(f"  {name}: {desc}...")
        print(f"    params: {params}")

    print(f"\n--- Argument Validation ---")
    validation_tests = [
        ("get_weather", {"city": "Tokyo"}, "Valid call"),
        ("get_weather", {}, "Missing required arg"),
        ("get_weather", {"city": "Tokyo", "units": "kelvin"}, "Invalid enum value"),
        ("calculator", {"expression": 123}, "Wrong type (int for string)"),
        ("unknown_tool", {"x": 1}, "Unknown tool"),
    ]
    for tool_name, args, label in validation_tests:
        errors = validate_tool_arguments(tool_name, args)
        status = "VALID" if not errors else f"ERRORS: {errors}"
        print(f"  {label}: {status}")

    print(f"\n--- Tool Execution ---")
    direct_tests = [
        {"name": "calculator", "arguments": {"expression": "(10 + 5) * 3 / 2"}},
        {"name": "get_weather", "arguments": {"city": "Tokyo"}},
        {"name": "get_weather", "arguments": {"city": "Mars"}},
        {"name": "web_search", "arguments": {"query": "python function calling"}},
        {"name": "read_file", "arguments": {"path": "data/config.json"}},
        {"name": "read_file", "arguments": {"path": "../etc/passwd"}},
        {"name": "run_code", "arguments": {"code": "result = sum(range(1, 101))"}},
        {"name": "run_code", "arguments": {"code": "import os; os.system('rm -rf /')"}},
    ]
    for call in direct_tests:
        result = execute_tool_call(call)
        print(f"\n  {call['name']}({json.dumps(call['arguments'])})")
        print(f"    -> {json.dumps(result['result'], indent=None)[:100]}")
        print(f"    time: {result['execution_time_ms']}ms")

    print(f"\n--- Full Function Calling Loop ---")
    test_queries = [
        "What's the weather in Tokyo?",
        "Calculate (100 + 250) * 0.15",
        "Search for MCP protocol",
        "Read the config file",
        "Run some Python code",
        "Tell me a joke",
    ]
    for query in test_queries:
        print(f"\n  User: {query}")
        result = run_function_calling_loop(query)
        if result["tool_results"]:
            for tr in result["tool_results"]:
                print(f"    Tool: {tr['tool']} ({tr['execution_time_ms']}ms)")
                print(f"    Result: {json.dumps(tr['result'], indent=None)[:90]}")
        else:
            print(f"    [No tool called -- direct response]")
        print(f"    Iterations: {result['iterations']}")

    print(f"\n--- Parallel Tool Calls ---")
    multi_city_query = "What's the weather in tokyo and london?"
    print(f"  User: {multi_city_query}")
    result = run_function_calling_loop(multi_city_query)
    print(f"  Tool calls made: {len(result['tool_results'])}")
    for tr in result["tool_results"]:
        city = tr["result"].get("city", "unknown")
        temp = tr["result"].get("temp_c", "N/A")
        print(f"    {city}: {temp}C, {tr['result'].get('condition', 'N/A')}")

    print(f"\n--- Security Checks ---")
    security_tests = [
        ("read_file", {"path": "../../etc/passwd"}),
        ("run_code", {"code": "import subprocess; subprocess.run(['ls'])"}),
        ("calculator", {"expression": "__import__('os').system('ls')"}),
    ]
    for tool_name, args in security_tests:
        result = execute_tool_call({"name": tool_name, "arguments": args})
        blocked = result["result"].get("error", False)
        print(f"  {tool_name}({list(args.values())[0][:40]}): {'BLOCKED' if blocked else 'ALLOWED'}")
```

## 投入使用

### OpenAI 函数调用

```python
# from openai import OpenAI
#
# client = OpenAI()
#
# tools = [{
#     "type": "function",
#     "function": {
#         "name": "get_weather",
#         "description": "Get current weather for a city",
#         "parameters": {
#             "type": "object",
#             "properties": {
#                 "city": {"type": "string"},
#                 "units": {"type": "string", "enum": ["celsius", "fahrenheit"]}
#             },
#             "required": ["city"]
#         }
#     }
# }]
#
# response = client.chat.completions.create(
#     model="gpt-4o",
#     messages=[{"role": "user", "content": "Weather in Tokyo?"}],
#     tools=tools,
#     tool_choice="auto",
# )
#
# tool_call = response.choices[0].message.tool_calls[0]
# args = json.loads(tool_call.function.arguments)
# result = get_weather(**args)
#
# final = client.chat.completions.create(
#     model="gpt-4o",
#     messages=[
#         {"role": "user", "content": "Weather in Tokyo?"},
#         response.choices[0].message,
#         {"role": "tool", "tool_call_id": tool_call.id, "content": json.dumps(result)},
#     ],
# )
# print(final.choices[0].message.content)
```

OpenAI 通过 `response.choices[0].message.tool_calls` 返回工具调用。每个调用都有一个 `id`，返回结果时必须带上它，模型借此把结果与调用对应起来。GPT-4o 可以在一个响应中返回多个工具调用——应遍历并执行全部调用。

### Anthropic 工具使用

```python
# import anthropic
#
# client = anthropic.Anthropic()
#
# response = client.messages.create(
#     model="claude-sonnet-5",
#     max_tokens=1024,
#     tools=[{
#         "name": "get_weather",
#         "description": "Get current weather for a city",
#         "input_schema": {
#             "type": "object",
#             "properties": {
#                 "city": {"type": "string"},
#                 "units": {"type": "string", "enum": ["celsius", "fahrenheit"]}
#             },
#             "required": ["city"]
#         }
#     }],
#     messages=[{"role": "user", "content": "Weather in Tokyo?"}],
# )
#
# tool_block = next(b for b in response.content if b.type == "tool_use")
# result = get_weather(**tool_block.input)
#
# final = client.messages.create(
#     model="claude-sonnet-5",
#     max_tokens=1024,
#     tools=[...],
#     messages=[
#         {"role": "user", "content": "Weather in Tokyo?"},
#         {"role": "assistant", "content": response.content},
#         {"role": "user", "content": [{"type": "tool_result", "tool_use_id": tool_block.id, "content": json.dumps(result)}]},
#     ],
# )
```

Anthropic 将工具调用作为 `type: "tool_use"` 的内容块返回。工具结果则放在一条 `type: "tool_result"` 的用户消息中。注意关键差异：Anthropic 使用 `input_schema` 定义工具参数，而 OpenAI 使用 `parameters`。

### MCP 集成

```python
# MCP servers expose tools over a standardized protocol.
# Any MCP-compatible client can discover and call these tools.
#
# Example: connecting to a Postgres MCP server
#
# from mcp import ClientSession, StdioServerParameters
# from mcp.client.stdio import stdio_client
#
# server_params = StdioServerParameters(
#     command="npx",
#     args=["-y", "@modelcontextprotocol/server-postgres", "postgresql://localhost/mydb"],
# )
#
# async with stdio_client(server_params) as (read, write):
#     async with ClientSession(read, write) as session:
#         await session.initialize()
#         tools = await session.list_tools()
#         result = await session.call_tool("query", {"sql": "SELECT count(*) FROM users"})
```

MCP 将工具实现与工具使用解耦。Postgres 服务器了解 SQL，GitHub 服务器了解 API，而你的智能体只需发现并调用工具——无须为每种集成编写提供商专用代码。

## 交付成果

本课会产出 `outputs/prompt-tool-designer.md`——一个用于设计工具定义的可复用提示词模板。向它描述希望工具完成的工作，它就会生成带有描述、类型与约束的完整 JSON Schema 定义。

本课还会产出 `outputs/skill-function-calling-patterns.md`——一套用于在生产环境中实现函数调用的决策框架，涵盖工具设计、错误处理、安全性和提供商特定模式。

## 练习

1. **添加第六个工具：数据库查询。** 实现一个带内存表的模拟 SQL 工具。该工具接受表名和过滤条件，而不是原始 SQL。验证表名位于白名单中，并将过滤运算符限制为 `=`、`>`、`<`、`>=`、`<=`。以 JSON 返回匹配的行。

2. **实现带错误反馈的重试。** 工具调用失败时（例如找不到城市），把错误消息送回模型决策函数，让它修正参数。记录每次调用的重试次数，并将每次工具调用的最大重试次数设为 3。

3. **构建多步骤智能体。** 有些查询需要串联多个工具调用：“读取配置文件，告诉我配置了哪个模型，再在 Web 上搜索该模型的价格。”实现一个循环，把累计结果传入每个决策步骤，直到模型判断不再需要工具。迭代次数限制为 10，防止无限循环。

4. **衡量工具选择准确率。** 创建 30 个测试查询及其预期工具名称。对全部查询运行决策函数，并计算正确选择工具的比例。找出最容易导致工具混淆的查询。

5. **实现工具调用缓存。** 如果相同工具在 60 秒内以完全相同的参数被调用，则返回缓存结果，不再重复执行。使用 `(tool_name, frozenset(args.items()))` 作为字典键。对包含 20 个查询的一次对话测量缓存命中率。

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|----------------|----------------------|
| 函数调用 | “工具使用” | 模型输出结构化 JSON，描述要以哪些参数调用哪个函数——执行函数的是你的代码，不是模型 |
| 工具定义 | “函数 Schema” | 描述工具名称、用途、参数与类型的 JSON Schema 对象——模型读取它来判断何时以及如何使用工具 |
| 工具选择 | “调用模式” | 控制模型必须调用工具（必选）、可以调用工具（自动），或必须调用某个指定工具（具名） |
| 并行调用 | “多工具调用” | 模型在一轮中输出多个工具调用，以减少往返次数——GPT-4o 和 Claude 都支持 |
| 工具结果 | “函数输出” | 执行工具得到的返回值；将其作为消息送回模型，模型就能在回答中使用真实数据 |
| 参数验证 | “输入检查” | 执行工具前，验证模型生成的参数是否符合预期类型、范围与约束 |
| MCP | “工具协议” | 模型上下文协议——Anthropic 的开放标准，通过服务器暴露工具，让任何兼容客户端都能发现并调用 |
| 智能体循环 | “ReAct 循环” | 模型决定工具、代码执行工具、结果反馈给模型的迭代循环，直到模型拥有足够信息作答 |
| 工具投毒 | “通过工具实施提示注入” | 一种攻击：工具结果中包含操纵模型行为的指令——必须净化所有工具输出 |
| 频率限制 | “调用预算” | 为每次对话设置工具调用次数上限，防止无限循环和失控的 API 成本 |

## 延伸阅读

- [OpenAI 函数调用指南](https://platform.openai.com/docs/guides/function-calling)——使用 GPT-4o 进行工具调用的权威参考，涵盖并行调用、强制调用与结构化参数
- [Anthropic 工具使用指南](https://docs.anthropic.com/en/docs/tool-use)——Claude 的工具使用实现，涵盖 input_schema、多工具响应与 tool_choice 配置
- [模型上下文协议规范](https://modelcontextprotocol.io)——AI 应用之间实现工具互操作的开放标准，采用服务器/客户端架构
- [Schick 等，2023——“Toolformer: Language Models Can Teach Themselves to Use Tools”](https://arxiv.org/abs/2302.04761)——训练大语言模型判断何时以及如何调用外部工具的奠基论文
- [Patil 等，2023——“Gorilla: Large Language Model Connected with Massive APIs”](https://arxiv.org/abs/2305.15334)——针对 1,645 个 API 对大语言模型进行准确调用微调，并减少幻觉
- [Berkeley 函数调用排行榜](https://gorilla.cs.berkeley.edu/leaderboard.html)——实时比较 GPT-4o、Claude、Gemini 与开放模型函数调用准确率的基准
- [Yao 等，“ReAct: Synergizing Reasoning and Acting in Language Models”（ICLR 2023）](https://arxiv.org/abs/2210.03629)——围绕每次工具调用的“思考—行动—观察”外层智能体循环；本课结束之处，正是阶段 14 开始之处。
- [Anthropic——Building effective agents（2024 年 12 月）](https://www.anthropic.com/research/building-effective-agents)——由单一工具使用原语构建的五种可组合模式（提示链、路由、并行化、编排器—工作器、评估器—优化器）。
