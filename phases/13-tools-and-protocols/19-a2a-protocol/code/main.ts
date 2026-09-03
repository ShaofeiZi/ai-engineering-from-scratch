// 第 13 阶段 第 19 课 — A2A 代理间协议，TypeScript 版本。
//
// 研究代理通过 A2A 调用写作代理：
//   1. 研究代理获取写作代理的 Agent Card
//   2. 提交一个包含文本、文件和数据部分的 Task
//   3. 写作代理状态流转：working -> input_required -> working -> completed
//   4. 研究代理接收一个 Artifact
//
// 仅使用标准库；进程内传输代替了基于 HTTP 的 JSON-RPC。
//
// 规范参考：
//   A2A 协议          https://a2aproject.github.io/A2A/specification
//   Agent Card schema  https://a2aproject.github.io/A2A/specification/#agent-card
//
// 运行：npx tsx code/main.ts

import { randomUUID } from "node:crypto";

type Capabilities = { streaming: boolean; pushNotifications: boolean };

type Skill = {
  id: string;
  name: string;
  description: string;
  inputModes: string[];
  outputModes: string[];
};

type AgentCard = {
  schemaVersion: string;
  name: string;
  description: string;
  url: string;
  version: string;
  skills: Skill[];
  capabilities: Capabilities;
};

const WRITER_AGENT_CARD: AgentCard = {
  schemaVersion: "1.0",
  name: "writer-agent",
  description: "根据原始素材撰写技术摘要和报告。",
  url: "https://writer.example.com/a2a",
  version: "1.0.0",
  skills: [
    {
      id: "draft_report",
      name: "起草报告",
      description: "给定原始素材和目标长度，生成一份报告。",
      inputModes: ["text", "file", "data"],
      outputModes: ["text", "artifact"],
    },
  ],
  capabilities: { streaming: true, pushNotifications: false },
};

type TextPart = { kind: "text"; payload: { text: string } };
type FilePart = {
  kind: "file";
  payload: { file: { name: string; mimeType: string; bytes: string } };
};
type DataPart = { kind: "data"; payload: Record<string, unknown> };
type Part = TextPart | FilePart | DataPart;

type Message = { role: "user" | "agent"; parts: Part[] };

type Artifact = { name: string; mimeType: string; parts: Part[] };

type TaskState =
  | "submitted"
  | "working"
  | "input_required"
  | "completed"
  | "failed"
  | "canceled";

type Task = {
  id: string;
  state: TaskState;
  messages: Message[];
  artifact: Artifact | null;
};

const TASK_STORE = new Map<string, Task>();

function newTask(): Task {
  const id = `task_${randomUUID().replace(/-/g, "").slice(0, 10)}`;
  const task: Task = { id, state: "submitted", messages: [], artifact: null };
  TASK_STORE.set(id, task);
  return task;
}

function findDataPart(message: Message): DataPart | undefined {
  return message.parts.find((p): p is DataPart => p.kind === "data");
}

function finish(task: Task, length: string): void {
  const text =
    `[写作代理] ${length} 摘要：已确定主题，提取关键要点，生成结论。`;
  task.artifact = {
    name: "summary",
    mimeType: "text/markdown",
    parts: [{ kind: "text", payload: { text } }],
  };
  task.state = "completed";
  console.log(`    WRITER  : 已完成任务 ${task.id}`);
}

function writerTasksSend(skillId: string, message: Message): Task {
  const task = newTask();
  task.state = "working";
  task.messages.push(message);
  console.log(`    WRITER  : 已启动任务 ${task.id} skill=${skillId}`);

  const data = findDataPart(message);
  if (!data || !("targetLength" in data.payload)) {
    task.state = "input_required";
    task.messages.push({
      role: "agent",
      parts: [
        {
          kind: "text",
          payload: { text: "请通过 data 部分指定 targetLength。" },
        },
      ],
    });
    console.log(`    WRITER  : 已暂停，状态为 input_required`);
  } else {
    finish(task, String(data.payload.targetLength));
  }
  return task;
}

function writerTasksReply(taskId: string, message: Message): Task {
  const task = TASK_STORE.get(taskId);
  if (!task) throw new Error(`unknown task ${taskId}`);
  task.messages.push(message);
  const data = findDataPart(message);
  if (task.state === "input_required" && data) {
    task.state = "working";
    finish(task, String(data.payload.targetLength ?? "short"));
  }
  return task;
}

function researchAgentFlow(): void {
  console.log("=".repeat(72));
  console.log("第 13 阶段 第 19 课 — 研究代理通过 A2A 调用写作代理（TypeScript 版本）");
  console.log("=".repeat(72));

  console.log("\n--- 研究代理获取写作代理的 Agent Card ---");
  console.log(
    JSON.stringify(
      {
        name: WRITER_AGENT_CARD.name,
        url: WRITER_AGENT_CARD.url,
        skills: WRITER_AGENT_CARD.skills,
      },
      null,
      2,
    ),
  );

  const skill = WRITER_AGENT_CARD.skills[0];
  const skillId = skill.id;
  console.log(`\n  研究代理将调用 skill：${skillId}`);

  const fakePdfBytes = Buffer.from("fake-pdf").toString("base64");
  const initialMessage: Message = {
    role: "user",
    parts: [
      { kind: "text", payload: { text: "请总结附件中的论文。" } },
      {
        kind: "file",
        payload: {
          file: { name: "paper.pdf", mimeType: "application/pdf", bytes: fakePdfBytes },
        },
      },
    ],
  };
  let task = writerTasksSend(skillId, initialMessage);
  console.log(`  研究代理 : 任务状态 = ${task.state}`);

  if (task.state === "input_required") {
    console.log("\n--- 研究代理补充缺失的数据 ---");
    const followup: Message = {
      role: "user",
      parts: [{ kind: "data", payload: { targetLength: "3 段" } }],
    };
    task = writerTasksReply(task.id, followup);
    console.log(`  研究代理 : 任务状态 = ${task.state}`);
  }

  console.log("\n--- 研究代理读取 artifact ---");
  if (task.artifact) {
    const firstPart = task.artifact.parts[0];
    console.log(`  名称     : ${task.artifact.name}`);
    console.log(`  mimeType : ${task.artifact.mimeType}`);
    if (firstPart.kind === "text") {
      console.log(`  内容     : ${firstPart.payload.text}`);
    }
  }

  console.log("\n--- 生命周期观察 ---");
  console.log(`  最终状态 : ${task.state}`);
  console.log(`  消息数量 : ${task.messages.length}`);
}

researchAgentFlow();
