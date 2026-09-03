# 任务 - 仓库记忆与持久化状态

## 目标
为 `agent_state.json` 和 `task_board.json` 编写 JSON Schema，构建一个能够加载、校验、修改并原子化写入的 `StateManager`，并证明跨两个回合的往返可成功。

## 输入
- 来自第 32 课的三文件工作台结构
- 一个仅使用标准库的校验器，覆盖 required、type、enum、pattern 和 items

## 交付物
- 与代码同目录的 `agent_state.schema.json` 和 `task_board.schema.json`
- `StateManager.load`、`StateManager.update`、`StateManager.commit`，采用先写临时文件再重命名的写入方式
- 一个跨两个回合修改状态并能干净重载的演示运行

## 验收标准
- `python3 code/main.py` 以零退出码退出
- 一次错误写入（缺少必填字段、枚举值非法）会被拒绝，不会被持久化
- 运行后的 `workdir/agent_state.json` 能通过 schema 校验

## 范围之外
- SQLite 或外部存储后端。本地文件就是本课的核心。
- LangGraph checkpointers、Letta memory块。理念相同，存储不同；此处不在范围内。

## 参考资料
- `docs/en.md` - 完整课程
- `code/main.py` - 参考实现
- `outputs/skill-state-schema.md` - 提取的技能
