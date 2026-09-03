---
name: handoff-generator
description: 从工作台产物生成会话结束交接包，同时产出人类可读的 Markdown 和机器可读的 JSON，后者以七个规范字段为键。
version: 1.0.0
phase: 14
lesson: 40
tags: [handoff, generator, session-end, packet, next-action]
---

给定一个工作台（state、verdict、review、feedback log、diff），产出一个接入智能体运行时的会话结束交接生成器。

产出：

1. `tools/generate_handoff.py`，暴露 `generate_handoff(snapshot) -> (markdown, payload)`。
2. `outputs/handoff/<session_id>/handoff.md` 和 `handoff.json`。
3. `handoff.schema.json`，覆盖七个必需字段以及 feedback tail 格式。
4. 会话结束钩子脚本，运行生成器并在任何字段缺失时拒绝关闭会话。
5. `docs/handoff.md`，列出七个字段、其来源以及裁剪策略。

硬性拒绝：

- 缺少 `next_action` 的交接。伪装成交接的状态报告会毒害下一个会话。
- 手写摘要的生成器。智能体的职责是让工作台保持在可生成状态。
- 与 JSON 不一致的 Markdown 包。JSON 是来源；Markdown 是 JSON 的渲染。
- 长于 30 条的 feedback tail。完整日志在版本控制中；交接包必须保持精简。

拒绝规则：

- 如果验证报告缺失，拒绝生成交接包。没有 verdict 的交接只是空想。
- 如果审查报告缺失且预期需要人工审查者，拒绝并要求先完成审查。
- 如果 diff 摘要为空但会话运行时间超过 5 分钟，在生成前暴露该异常；怀疑是卡住的会话而非真正的空操作。

输出结构：

```
<repo>/
├── outputs/handoff/<session_id>/
│   ├── handoff.md
│   └── handoff.json
├── tools/generate_handoff.py
├── handoff.schema.json
└── docs/handoff.md
```

以“下一步阅读”结尾，指向：

- 第 41 课，在真实风格示例应用上的端到端练习。
- 第 42 课，将生成器打包进 capstone 工作台包。
- 第 29 课（生产运行时），将会话结束接入队列、事件和 cron 触发器。
