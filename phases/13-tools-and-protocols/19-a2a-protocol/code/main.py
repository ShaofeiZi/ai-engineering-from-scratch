"""Phase 13 Lesson 19 - A2A agent-to-agent 协议。

研究代理通过 A2A 调用写作代理：
  1. 研究代理获取写作代理的 Agent Card
  2. 提交包含文本 + 文件 + 数据部分的 Task
  3. 写作代理状态转换：working -> input_required -> working -> completed
  4. 研究代理接收一个 Artifact

仅使用标准库；in-process 传输层替代 HTTP. 上的 JSON-RPC

运行：python code/main.py
"""

from __future__ import annotations

import base64
import json
import uuid
from dataclasses import dataclass, field


WRITER_AGENT_CARD = {
    "schemaVersion": "1.0",
    "name": "writer-agent",
    "description": "根据原始素材撰写技术摘要和报告。",
    "url": "https://writer.example.com/a2a",
    "version": "1.0.0",
    "skills": [
        {
            "id": "draft_report",
            "name": "起草报告",
            "description": "根据原始素材和目标长度生成报告。",
            "inputModes": ["text", "file", "data"],
            "outputModes": ["text", "artifact"],
        }
    ],
    "capabilities": {"streaming": True, "pushNotifications": False},
}


@dataclass
class Part:
    kind: str
    payload: dict


@dataclass
class Message:
    role: str
    parts: list[Part] = field(default_factory=list)


@dataclass
class Artifact:
    name: str
    mimeType: str
    parts: list[Part]


@dataclass
class Task:
    id: str
    state: str = "submitted"
    messages: list[Message] = field(default_factory=list)
    artifact: Artifact | None = None

    def append(self, m: Message) -> None:
        self.messages.append(m)


TASK_STORE: dict[str, Task] = {}


def writer_tasks_send(skill_id: str, message: Message) -> Task:
    task = Task(id=f"task_{uuid.uuid4().hex[:10]}")
    TASK_STORE[task.id] = task
    task.state = "working"
    task.append(message)
    print(f"    WRITER  : 已启动任务 {task.id} skill={skill_id}")
    # 需要 target_length
    data_parts = [p for p in message.parts if p.kind == "data"]
    if not data_parts or "targetLength" not in data_parts[0].payload:
        task.state = "input_required"
        task.append(Message(role="agent", parts=[
            Part("text", {"text": "请通过 data 部分指定 targetLength。"})
        ]))
        print(f"    WRITER  : 已暂停 input_required")
    else:
        finish(task, data_parts[0].payload["targetLength"])
    return task


def writer_tasks_reply(task_id: str, message: Message) -> Task:
    task = TASK_STORE[task_id]
    task.append(message)
    data_parts = [p for p in message.parts if p.kind == "data"]
    if task.state == "input_required" and data_parts:
        task.state = "working"
        finish(task, data_parts[0].payload.get("targetLength", "short"))
    return task


def finish(task: Task, length: str) -> None:
    text = f"[写作代理] {length} 摘要：" \
           f"已确定主题、提取关键要点并起草结论。"
    task.artifact = Artifact(
        name="summary",
        mimeType="text/markdown",
        parts=[Part("text", {"text": text})],
    )
    task.state = "completed"
    print(f"    WRITER  : 已完成任务 {task.id}")


def research_agent_flow() -> None:
    print("=" * 72)
    print("第 13 阶段第 19 课 - 研究代理通过 A2A 调用写作代理")
    print("=" * 72)

    print("\n--- 研究代理获取写作代理的 Agent Card ---")
    print(json.dumps({k: WRITER_AGENT_CARD[k] for k in ("name", "url", "skills")}, indent=2))

    skill = WRITER_AGENT_CARD["skills"][0]
    skill_id = skill["id"]
    print(f"\n  研究代理将调用 skill：{skill_id}")

    msg = Message(role="user", parts=[
        Part("text", {"text": "请总结附件中的论文。"}),
        Part("file", {"file": {"name": "paper.pdf", "mimeType": "application/pdf",
                                "bytes": base64.b64encode(b"fake-pdf").decode()}}),
    ])
    task = writer_tasks_send(skill_id, msg)
    print(f"  研究代理：任务状态 = {task.state}")

    if task.state == "input_required":
        print("\n--- 研究代理提供缺失的数据 ---")
        followup = Message(role="user", parts=[
            Part("data", {"targetLength": "3 段"}),
        ])
        task = writer_tasks_reply(task.id, followup)
        print(f"  研究代理：任务状态 = {task.state}")

    print("\n--- 研究代理读取 artifact ---")
    if task.artifact:
        print(f"  名称     : {task.artifact.name}")
        print(f"  mimeType : {task.artifact.mimeType}")
        print(f"  内容     : {task.artifact.parts[0].payload['text']}")

    print("\n--- 生命周期观察 ---")
    print(f"  最终状态 : {task.state}")
    print(f"  消息数量 : {len(task.messages)}")


if __name__ == "__main__":
    research_agent_flow()
