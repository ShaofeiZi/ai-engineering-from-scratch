# 任务 - 最小智能体工作台

## 目标
将三文件最小工作台（路由器、状态、任务板）落地到一个全新的 `workdir/` 中，并证明单轮智能体交互能够读取状态、拉取任务、写入作用域，并持久化更新后的状态。

## 输入
- 课程代码旁的一个空 `workdir/` 目录
- 对三个文件的了解：`AGENTS.md`、`agent_state.json`、`task_board.json`

## 交付物
- `code/main.py`，创建这三个文件并运行一轮交互
- `workdir/AGENTS.md`，简短的路由器，指向状态、任务板和验证命令
- `workdir/agent_state.json`，包含活动任务 id、触及的文件、下一步动作
- `workdir/task_board.json`，包含一个小型待办列表及状态

## 验收标准
- `python3 code/main.py` 在首次和第二次运行时均以零退出
- 第二次运行从第一次结束处继续，而非从头开始
- 脚本打印的 Diff 显示该轮交互触及的那一个文件

## 不在范围内
- 作用域契约、验证门、审查者智能体。这些将在后续课程中叠加其上。
- 冗长的单体 `AGENTS.md`。路由器刻意保持简短。

## 参考资料
- `docs/en.md` - 完整课程
- `code/main.py` - 参考实现
- `outputs/skill-minimal-workbench.md` - 提取的技能
