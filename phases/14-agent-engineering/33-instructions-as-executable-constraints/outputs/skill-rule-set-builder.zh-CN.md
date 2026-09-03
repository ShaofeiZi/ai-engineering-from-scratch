---
name: rule-set-builder
description: 采访项目所有者，将其现有的散文式指令划分为五个操作性类别，并输出带版本号的 agent-rules.md 以及 Python 检查器桩代码。
version: 1.0.0
phase: 14
lesson: 33
tags: [rules, instructions, constraints, checker, workbench]
---

给定一个仓库以及任何现有的散文式指令（`AGENTS.md`、`CONTRIBUTING.md`、入职文档），生成一个可由工作台执行的五类规则集。

五个类别：

1. `startup` — 工作开始前必须满足的条件。
2. `forbidden` — 绝不允许发生的事情。
3. `definition_of_done` — 用以证明任务已完成的标准。
4. `uncertainty` — 智能体不确定时的行为准则。
5. `approval` — 需要人工签字确认的事项。

产出内容：

1. `docs/agent-rules.md`，每条规则对应一个 `##` 标题。每条规则包含 `category`、`check` 以及一行描述。
2. `tools/rule_checker.py`，包含一个 `RuleChecker` 类，为每个 `check` 暴露一个方法。每个方法接收一个 `TurnTrace` 数据类并返回 `bool`。
3. `tools/rule_report.py` 运行器，加载规则、对一条 trace 运行检查器、输出 `rule_report.json`。
4. 一份迁移说明文件：记录哪些散文行变成了哪条规则，哪些作为愿景性内容被舍弃，以及原因。

硬性拒绝：

- 没有 `check` 字段的规则。仅具愿景性的规则应放入入职文档，而非工作台规则集。
- 单一的“要小心”规则。请指定一个类别和一个 check，否则将其移除。
- 需要 LLM 调用的检查。规则检查必须是确定性的且成本低廉，以便每轮都能运行。
- 超过 200 行的规则文件。按类别拆分为 `agent-rules.{startup,forbidden,done,uncertainty,approval}.md` 并通过一个父索引路由。

拒绝规则：

- 如果智能体产品无法提供 `TurnTrace`（无插桩），则拒绝接入检查器，直到至少记录了 `read_state_file`、`edited_files` 和 `tests_exit_code`。
- 如果现有指令大多为愿景性内容（>50%），则在输出规则之前先呈现该发现。规则集会显得单薄；这是正确的。
- 如果某条规则是因某一次历史事件而添加的，请附上事件 id，以便未来的审查决定其是否仍然需要。

输出结构：

```
<repo>/
├── docs/
│   └── agent-rules.md
├── tools/
│   ├── rule_checker.py
│   └── rule_report.py
└── docs/migration-notes.md
```

最后以“延伸阅读”结尾，指向：

- 第 36 课，讲解扩展 forbidden 类别的按任务范围契约。
- 第 38 课，讲解消费规则报告的验证门。
- 第 39 课，讲解对规则合规性进行评分的审查者智能体。
