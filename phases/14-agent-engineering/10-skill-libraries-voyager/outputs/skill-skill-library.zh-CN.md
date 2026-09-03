---
name: skill-library
description: 生成一个 Voyager 形态的技能库，支持注册、基于相似度的检索、组合式执行以及由失败驱动的迭代优化。
version: 1.0.0
phase: 14
lesson: 10
tags: [voyager, skills, library, composition, refinement]
---

给定目标运行时和一个领域，产出一个支持 Voyager 三大组件的技能库：课程钩子、可检索的技能存储、迭代优化。

产出内容：

1. `Skill` 类型，包含 `name`、`description`、`code`、`version`、`tags`、`depends_on`、`history`。每次写入都记录之前的代码。
2. `SkillLibrary`，包含 `register(skill, dedup=True)`（新增或版本号升级）、`search(query, top_k, tag_filter)`、`get(name)`、`topo_order(name)`（依赖解析）、`execute(name, context)`（拓扑执行）。
3. 检索必须使用 embedding 相似度或 BM25，而非对整个技能库进行 LLM 打分。允许在 top-k 候选列表上进行 LLM 重排。
4. 执行必须按每个技能捕获异常，并将其呈现到 trace 中，作为优化循环可消费的反馈。
5. 优化钩子：在 `execute` 失败后，运行时收集 (task, skill_name, error, env_state)，将其传递给模型，并对重写后的技能调用 `register`。版本号升级；history 保留旧代码。

硬性拒绝条件：

- 技能库中的技能是散文式文本而非代码。技能必须是可执行的。散文内容应放在 `description` 中。
- 没有拓扑排序的组合。不带环检测的深度优先遍历在技能 DAG 上会崩溃。
- 静默的版本覆盖。每次优化必须升级 `version` 并将旧代码推入 `history` 以供审计。

拒绝规则：

- 如果目标运行时没有用于技能执行的沙箱，则当技能会触及生产系统时，拒绝执行。在上线前要求具备沙箱（Lesson 09 原则）。
- 如果用户要求"每次失败都自动重试而不做优化"，拒绝。没有优化的重试只会放大 bug，并不能修复它。
- 如果技能库超过约 200 个技能且采用扁平检索，拒绝称其为"生产就绪"。应先添加 tag 过滤和分层命名空间。

输出：`skill.py`、`library.py`、`execute.py`、`refine.py`，以及一个 `README.md`，说明去重规则、检索后端、优化 prompt 和版本策略。最后以"延伸阅读"结尾，指向 Lesson 17（Claude Agent SDK 集成）、Lesson 16（OpenAI Agents SDK 工具翻译）或 Lesson 30（评估技能库质量）。
