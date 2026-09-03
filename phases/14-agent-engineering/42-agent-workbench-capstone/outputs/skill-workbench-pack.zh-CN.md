---
name: workbench-pack
description: 生成项目调优的即插即用智能体工作台产物包——规则根据团队历史进行强化，作用域 glob 匹配仓库，评审维度扩展一个领域特定条目。
version: 1.0.0
phase: 14
lesson: 42
tags: [capstone, workbench-pack, installer, schemas, drop-in]
---

给定一个仓库、团队的故障历史以及运行其中的智能体产品，产出一个调优过的 agent-workbench-pack 和一个安装器。

产出内容：
1. `agent-workbench-pack/` 目录，匹配规范布局：AGENTS.md、docs/、schemas/、scripts/、bin/、README.md、VERSION。
2. 一个 `bin/install.sh`，在没有 `--force` 的情况下拒绝覆盖已有产物包，并将 `.workbench-version` 写入目标仓库。
3. 项目调优版的 `agent-rules.md`（每个类别至少有一条源自团队最近六次故障的规则）、`reviewer-rubric.md`（新增第六个领域维度）以及 `scope_contract.schema.json`（包含项目特定的 glob）。
4. 一个 `lint_pack.py` 脚本，在脚本与 schema 之间或 VERSION 与 schema 的 `schema_version` 之间出现漂移时失败。
5. 可选的 CI 集成，在演示分支上安装产物包并针对已知良好的任务运行验证关卡。

硬性拒绝：

- 产物包包含项目特定的任务。任务存在于目标仓库的看板上。
- 产物包绑定到单一厂商 SDK。仅限框架无关；SDK 接线是目标仓库的职责。
- 安装器修改状态文件。安装器是幂等的、仅操作表面层；状态属于智能体和人类。
- 规则没有对应的检查函数。愿景性规则属于入职指南，而非产物包。

拒绝规则：

- 如果故障历史为空，拒绝交付调优过的 `agent-rules.md`。使用规范默认值并暴露该缺口。
- 如果目标仓库的 CI 与安装不兼容（没有 `.github/workflows/`，也没有等价物），拒绝可选的 CI 步骤并记录手动路径。
- 如果团队使用产物包的私有 fork，拒绝编写公开安装器。私有安装器承载私有不变量。

输出结构：

```
agent-workbench-pack/
├── AGENTS.md
├── docs/
├── schemas/
├── scripts/
├── bin/install.sh
├── lint_pack.py
├── VERSION
└── README.md
```

以「接下来阅读什么」结尾，指向：

- 第 41 课，该产物包所改进的前后对比基准。
- 第 30 课（评估驱动的智能体开发），消费产物包裁定的评估循环。
- [SkillKit](https://github.com/rohitg00/skillkit)，跨 32 个 AI 智能体分发产物包。
