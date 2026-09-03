# 多智能体软件团队（TypeScript 骨架）

用于多智能体软件团队毕业项目的多文件 TypeScript 骨架。
Planner、Coder 和 Reviewer 智能体共享一个工作区，并通过协调器轮转。
工作树桩通过 execFile 启动子进程，带有拒绝名单和 shell 元字符拒绝机制。

## 布局

- `src/index.ts` — 演示运行器。
- `src/agent.ts` — 基类 `Agent` 以及 `PlannerAgent`、`CoderAgent`、`ReviewerAgent`。
- `src/coordinator.ts` — 轮询循环和轮转跟踪。
- `src/workspace.ts` — 共享的内存文件系统和消息日志。
- `src/runtime.ts` — 使用拒绝名单的 `child_process.execFile` 工作树桩。
- `src/types.ts` — 共享类型。
- `tests/*.test.ts` — `node --test` 风格测试，通过 `tsx` 运行。

## 安装

```bash
npm install
```

## 运行

```bash
npm start
```

## 验证

```bash
npm run typecheck
npm test
```

## 规格参考

- 源课程：`phases/19-capstone-projects/10-multi-agent-software-team/docs/en.md`
- [MetaGPT](https://github.com/FoundationAgents/MetaGPT) 基于角色的多智能体框架。
