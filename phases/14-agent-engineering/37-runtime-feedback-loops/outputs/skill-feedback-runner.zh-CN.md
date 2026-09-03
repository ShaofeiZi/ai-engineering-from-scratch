---
name: feedback-runner
description: 包装 shell 命令，确定性捕获 stdout/stderr/退出码/执行时长，为每条命令持久化一条 JSONL 记录，并在缺少反馈时拒绝推进智能体循环。
version: 1.0.0
phase: 14
lesson: 37
tags: [feedback, subprocess, runner, jsonl, loop-control]
---

给定一个在智能体循环中运行 shell 命令的项目，产出一个反馈运行器及其写入的 JSONL 文件。

需要产出：

1. `tools/run_with_feedback.py`，暴露 `run_with_feedback(command: list[str], agent_note: str, timeout_s: float) -> FeedbackRecord`。
2. `feedback_record.jsonl` 位于工作台下，每行一条记录。
3. `tools/feedback_loader.py`，返回当前任务最近的 N 条记录。
4. 一个 `loop_can_advance(record) -> bool` 辅助函数，智能体循环在声明成功之前调用它。
5. 测试覆盖：成功路径、非零退出码、超时、缺失二进制文件、确定性的头部/尾部截断。

硬性拒绝：

- 运行器中任何位置出现 `shell=True`。仅允许 Argv 模式。
- 依赖于挂钟时间或随机采样的截断。相同输入必须产出相同记录。
- 缺少 `duration_ms` 的记录。缓慢的探针是工作台卡死的第一个征兆。
- 返回无界列表的加载器。必须限制为最近 N 条或进行分页。

拒绝规则：

- 如果项目通过 stdout 传递密钥，在没有脱敏步骤的情况下拒绝交付运行器。展示本会被捕获的那些行。
- 如果项目存在可能无限挂起的命令，在没有默认超时和显式覆盖列表的情况下拒绝交付。
- 如果运行器在具有共享状态的 worker 中运行，拒绝跳过 JSONL 追加时的文件锁。多个写入者会撕裂文件。

输出结构：

```
<repo>/
├── feedback_record.jsonl
└── tools/
    ├── run_with_feedback.py
    ├── feedback_loader.py
    └── test_feedback_runner.py
```

以"接下来阅读什么"结尾，指向：

- 第 38 课，关于消费这些记录的验证关卡。
- 第 39 课，关于在给运行评分时读取反馈的审查智能体。
- 第 23 课，关于 OTel GenAI 规范，以便在反馈稳固后添加到遥测侧。
