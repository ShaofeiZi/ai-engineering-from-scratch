"""第 13 阶段综合项目 - 无状态进程内研究与报告模拟。

在一个可读的演示中涵盖 Phase 13 的多个边界：
  - 网关形态的静态令牌查找与 RBAC
  - 每请求协议元数据与强制服务器发现
  - 返回任务扩展和 UI 形态数据的本地工具函数
  - 以嵌套 span 表示的 A2A 写作代理委派
  - 共享同一 trace id 的内存 trace 字典
  - 保护 description 变更的固定哈希 manifest

本文件不实现 MCP 或 A2A 传输、OAuth 交换、MCP App
桥接、遥测导出器或执行沙箱。仅使用标准库。

运行：python code/main.py
"""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from copy import deepcopy
from datetime import datetime, timezone


SPANS: list[dict] = []
TASKS: dict[str, dict] = {}

PROTOCOL_VERSION = "2026-07-28"
TASK_EXTENSION = "io.modelcontextprotocol/tasks"
SERVER_INFO = {"name": "research-simulator", "version": "1.0.0"}


def request_meta(*, tasks: bool = False) -> dict:
    extensions = {TASK_EXTENSION: {}} if tasks else {}
    return {
        "io.modelcontextprotocol/protocolVersion": PROTOCOL_VERSION,
        "io.modelcontextprotocol/clientCapabilities": {"extensions": extensions},
        "io.modelcontextprotocol/clientInfo": {
            "name": "capstone-client",
            "version": "1.0.0",
        },
    }


def _server_meta() -> dict:
    return {"io.modelcontextprotocol/serverInfo": deepcopy(SERVER_INFO)}


def complete_result(**fields: object) -> dict:
    return {"resultType": "complete", **fields, "_meta": _server_meta()}


def protocol_error(code: int, message: str, data: dict | None = None) -> dict:
    error = {"code": code, "message": message}
    if data is not None:
        error["data"] = data
    return {"error": error}


def validate_request_meta(meta: dict, *, require_tasks: bool = False) -> dict | None:
    if not isinstance(meta, dict):
        return protocol_error(-32602, "params._meta must be an object")
    requested = meta.get("io.modelcontextprotocol/protocolVersion")
    if not isinstance(requested, str):
        return protocol_error(-32602, "protocolVersion must be a string")
    if requested != PROTOCOL_VERSION:
        return protocol_error(
            -32022,
            "Unsupported protocol version",
            {"supported": [PROTOCOL_VERSION], "requested": requested},
        )
    capabilities = meta.get("io.modelcontextprotocol/clientCapabilities")
    if not isinstance(capabilities, dict):
        return protocol_error(-32602, "clientCapabilities must be an object")
    extensions = capabilities.get("extensions", {})
    if require_tasks and (
        not isinstance(extensions, dict) or TASK_EXTENSION not in extensions
    ):
        return protocol_error(
            -32021,
            "Missing required client capability",
            {
                "requiredCapabilities": {
                    "extensions": {TASK_EXTENSION: {}}
                }
            },
        )
    return None


def server_discover(meta: dict) -> dict:
    invalid = validate_request_meta(meta)
    if invalid:
        return invalid
    return complete_result(
        supportedVersions=[PROTOCOL_VERSION],
        capabilities={
            "tools": {"listChanged": False},
            "extensions": {TASK_EXTENSION: {}},
        },
        ttlMs=3_600_000,
        cacheScope="public",
    )


def _hex(n: int) -> str:
    return uuid.uuid4().hex[: n * 2]


def span(name: str, kind: str, trace_id: str | None, parent: str | None,
         attrs: dict) -> dict:
    tid = trace_id or _hex(16)
    sp = {"name": name, "kind": kind, "traceId": tid, "spanId": _hex(8),
          "parentSpanId": parent, "start": time.time_ns(), "attrs": attrs, "end": 0}
    SPANS.append(sp)
    return sp


def finish(sp: dict) -> None:
    sp["end"] = max(time.time_ns(), sp["start"] + 1)


TOOLS = [
    {"name": "arxiv_search", "description": "当用户希望按关键词搜索 arXiv 时使用。"},
    {"name": "generate_report", "description": "当用户需要完整报告时使用。"},
]

PAPERS = [
    {
        "arxiv_id": "2603.22489",
        "title": "MCP 部署中的工具投毒攻击",
        "canonical_title": "Tool poisoning attacks on MCP deployments",
    },
    {
        "arxiv_id": "2604.01055",
        "title": "Agent 间协作基准",
        "canonical_title": "Agent-to-agent coordination benchmarks",
    },
    {
        "arxiv_id": "2603.30016",
        "title": "通过 Tasks 执行长时间运行的工具调用",
        "canonical_title": "Long-running tool calls via Tasks",
    },
]

PINNED = {f"research::{t['name']}": hashlib.sha256(t["description"].encode()).hexdigest()
          for t in TOOLS}


def research_arxiv_search(args: dict) -> dict:
    q = args["query"].casefold()
    hits = [
        {"arxiv_id": paper["arxiv_id"], "title": paper["title"]}
        for paper in PAPERS
        if q in paper["title"].casefold()
        or q in paper["canonical_title"].casefold()
    ]
    return complete_result(
        content=[{"type": "text", "text": json.dumps(hits)}],
        isError=False,
    )


def research_generate_report(args: dict, trace_id: str, parent: str) -> dict:
    task_id = f"tsk_{uuid.uuid4().hex[:10]}"
    sp = span("mcp.task.working", "INTERNAL", trace_id, parent,
              {"gen_ai.operation.name": "execute_tool", "mcp.task.id": task_id})
    a2a = span("a2a.SendMessage", "CLIENT", trace_id, sp["spanId"],
               {"a2a.peer": "writer-agent", "a2a.skill": "summarize_papers"})
    finish(a2a)
    finish(sp)
    html = (
        "<!doctype html><html><body>"
        "<h1>代理协议 arXiv 报告</h1><ul>"
        + "".join(f"<li>{p['arxiv_id']}: {p['title']}</li>" for p in PAPERS)
        + "</ul><script>/* 此演示有意省略真实的 MCP App 桥接。 */</script></body></html>"
    )
    now = datetime.now(timezone.utc).isoformat()
    TASKS[task_id] = {
        "resultType": "complete",
        "taskId": task_id,
        "status": "completed",
        "createdAt": now,
        "lastUpdatedAt": now,
        "ttlMs": 900_000,
        "pollIntervalMs": 1_000,
        "result": complete_result(
            content=[
                {"type": "text", "text": "报告已生成：已总结 3 篇论文。"},
                {"type": "ui_resource", "uri": "ui://report/current"},
            ],
            ui={
                "resourceUri": "ui://report/current",
                "csp": {"default-src": "'self'"},
                "permissions": [],
            },
            html=html,
        ),
        "_meta": _server_meta(),
    }
    return {
        "resultType": "task",
        "taskId": task_id,
        "status": "working",
        "createdAt": now,
        "lastUpdatedAt": now,
        "ttlMs": 900_000,
        "pollIntervalMs": 1_000,
        "_meta": _server_meta(),
    }


def tasks_get(task_id: str, meta: dict) -> dict:
    invalid = validate_request_meta(meta, require_tasks=True)
    if invalid:
        return invalid
    if not isinstance(task_id, str):
        return protocol_error(-32602, "Unknown taskId")
    task = TASKS.get(task_id)
    if task is None:
        return protocol_error(-32602, "Unknown taskId")
    return deepcopy(task)


USERS = {
    "tok_alice": {"id": "alice", "scopes": {"research:read", "research:write"}},
    "tok_bob":   {"id": "bob",   "scopes": {"research:read"}},
}
REQUIRED_SCOPE = {"arxiv_search": "research:read",
                  "generate_report": "research:write"}

AUDIT: list[dict] = []


def pin_ok(tool_name: str, description: str) -> bool:
    return PINNED.get(f"research::{tool_name}") == hashlib.sha256(description.encode()).hexdigest()


def gateway_call(token: str, tool_name: str, args: dict,
                 trace_id: str, parent: str, meta: dict) -> dict:
    invalid = validate_request_meta(
        meta, require_tasks=tool_name == "generate_report"
    )
    if invalid:
        return invalid
    u = USERS.get(token)
    if not u:
        return {"error": "unauthenticated"}
    required = REQUIRED_SCOPE.get(tool_name)
    if required and required not in u["scopes"]:
        AUDIT.append({"user": u["id"], "tool": tool_name, "decision": "403"})
        return {"error": "insufficient_scope", "scope": required}
    tool = next((t for t in TOOLS if t["name"] == tool_name), None)
    if tool is None:
        return {"error": "unknown tool"}
    if not pin_ok(tool_name, tool["description"]):
        return {"error": "hash_mismatch"}
    sp = span("mcp.call", "CLIENT", trace_id, parent,
              {"gen_ai.operation.name": "execute_tool", "gen_ai.tool.name": tool_name,
               "gateway.user": u["id"], "mcp.server": "research"})
    if tool_name == "arxiv_search":
        result = research_arxiv_search(args)
    else:
        result = research_generate_report(args, trace_id, sp["spanId"])
    finish(sp)
    AUDIT.append({"user": u["id"], "tool": tool_name, "decision": "allow"})
    return result


def orchestrator(token: str, user_query: str) -> dict:
    trace_id = _hex(16)
    root = span("agent.invoke_agent", "INTERNAL", trace_id, None,
                {"gen_ai.operation.name": "invoke_agent",
                 "gen_ai.agent.name": "research-orchestrator"})

    llm1 = span("llm.chat", "CLIENT", trace_id, root["spanId"],
                {"gen_ai.operation.name": "chat", "gen_ai.provider.name": "openai",
                 "gen_ai.request.model": "gpt-4o", "gen_ai.usage.input_tokens": 24})
    finish(llm1)

    search = gateway_call(token, "arxiv_search",
                          {"query": "agent"}, trace_id, root["spanId"],
                          request_meta())
    report = gateway_call(token, "generate_report",
                          {"format": "html"}, trace_id, root["spanId"],
                          request_meta(tasks=True))
    task = None
    if report.get("resultType") == "task":
        task = tasks_get(report["taskId"], request_meta(tasks=True))

    llm2 = span("llm.chat", "CLIENT", trace_id, root["spanId"],
                {"gen_ai.operation.name": "chat", "gen_ai.provider.name": "openai",
                 "gen_ai.request.model": "gpt-4o", "gen_ai.usage.output_tokens": 85})
    finish(llm2)

    finish(root)
    return {"trace_id": trace_id, "search": search, "report": report, "task": task}


def demo() -> None:
    print("=" * 72)
    print("第 13 阶段综合项目 - 研究与报告生态系统")
    print("=" * 72)

    print("\n--- 无状态服务器发现 ---")
    discovery = server_discover(request_meta())
    print(f"  协议          : {discovery['supportedVersions'][0]}")
    print(f"  任务扩展      : {TASK_EXTENSION in discovery['capabilities']['extensions']}")

    print("\n--- 以 alice 身份运行编排器（read+write）---")
    out = orchestrator("tok_alice", "总结 2026 年引用量最高的三篇 arXiv 论文")
    print(f"  trace id       ：{out['trace_id']}")
    print(f"  搜索结果      : {out['search']['content'][0]['text']}")
    print(f"  报告句柄      : {out['report']['taskId']} ({out['report']['status']})")
    print(f"  任务状态      : {out['task']['status']}，通过 tasks/get")
    print(f"  UI 字节数     : {len(out['task']['result']['html'])}")

    print("\n--- 以 bob 身份运行编排器（只读）---")
    out = orchestrator("tok_bob", "生成一份报告")
    print(f"  generate_report 结果：{out['report']}")

    print("\n--- 审计日志 ---")
    for row in AUDIT:
        print(f"  {row}")

    print("\n--- OTel GenAI span ---")
    for sp in SPANS:
        dur_ms = round((sp['end'] - sp['start']) / 1_000_000, 2) if sp['end'] else 0
        parent = sp['parentSpanId'][:6] if sp['parentSpanId'] else "ROOT"
        print(f"  [{sp['traceId'][:6]}] {sp['name']:20s} {sp['kind']:8s} "
              f"父 span={parent}  耗时={dur_ms}ms")

    print("\n--- 基元覆盖情况 ---")
    covered = [
        "工具接口与直接函数分发",
        "server/discover 与每请求无状态元数据",
        "结构化内容字典",
        "任务扩展句柄与 tasks/get 轮询",
        "ui:// 形态的资源引用",
        "使用固定哈希检测 description 变更",
        "静态令牌权限范围与网关策略模拟",
        "A2A 形态的不透明委派边界",
        "内存中的 trace 标识符与父 span 标识符",
        "编排器在本地操作之间的路由",
    ]
    for c in covered:
        print(f"  + {c}")


if __name__ == "__main__":
    demo()
