---
name: state-graph
description: 构建一个 LangGraph 形态的状态机，包含类型化状态、条件边、按节点检查点持久化以及可恢复续跑。
version: 1.0.0
phase: 14
lesson: 13
tags: [langgraph, state-machine, durable, checkpointing, human-in-the-loop]
---

给定目标运行时、一个状态形状、一组节点函数和一个 checkpointer 后端，生成一个有状态的智能体图。

产出：

1. 一个类型化的 `State`（dict 或 Pydantic）。为每个字段编写文档。节点读取状态；它们返回更新。
2. 一个 `StateGraph`，使用 `add_node`、`add_edge`、`add_conditional_edges`、`set_entry`，以及 `START`/`END` 哨兵。
3. 一个 `Checkpointer` 接口，包含 `save(session_id, node, state)` 和 `load_latest(session_id)`。默认使用 SQLite；允许 Postgres/Redis/自定义。
4. 一个 `Runner`，逐步遍历图，在每个节点之后序列化状态，捕获 `PausedAtNode` 以支持人工介入（human-in-the-loop），并支持带可选 `state_override` 的 `resume_from`。
5. 三个拓扑辅助工具：supervisor（中心路由器）、swarm（共享工具交接）、hierarchical（子图）。

坚决拒绝：

- 没有显式随机种子或时钟捕获的非确定性节点。续跑假设在给定输入状态下，节点输出是可复现的。
- 只保存“摘要”状态的 checkpointer。必须序列化完整状态，否则续跑会中断。
- 每条边都是条件边的图。优先使用线性链，辅以少量分支。

拒绝规则：

- 如果用户要求一个没有持久化的状态图，拒绝。其全部意义就在于可持久化的续跑；如果不需要续跑，请使用第 12 课中的工作流模式。
- 如果用户要求“仅在成功时检查点”，拒绝。失败也需要状态——这正是调试的起点。
- 如果图包含超过约 30 个节点，拒绝扁平布局并要求使用嵌套子图。30 个节点的扁平图无法审阅。

输出：`state.py`、`graph.py`、`checkpointer.py`、`runner.py`、`README.md`，其中 README 需说明状态模式、checkpointer 的选择以及续跑语义。以“延伸阅读”结尾，指向第 14 课了解 actor 模型替代方案、第 16 课了解交接/护栏层，或第 23 课了解图步骤的 OTel spans。
