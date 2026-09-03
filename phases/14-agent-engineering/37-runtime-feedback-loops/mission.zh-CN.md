# 任务 - 运行时反馈循环

## 目标
构建 `run_with_feedback`，它封装 `subprocess.run`，捕获 stdout、stderr、退出码和执行时长，以确定性方式截断输出，并追加一条 JSONL 记录，供下一轮对话和验证关卡共同读取。

## 输入
- 三个用于演练运行器的演示命令：一个成功、一个失败、一个慢速
- token 预算：确定性头部加尾部，并带有 `...truncated N lines...` 标记

## 交付物
- `run_with_feedback(command, agent_note)` 写入 `feedback_record.jsonl`
- 一个将 JSONL 流式加载为 Python 列表的加载器
- 一个展示每个命令最后一条记录的打印机

## 验收标准
- `python3 code/main.py` 退出码为零
- `feedback_record.jsonl` 在多次重跑中为每个命令累积一条记录
- 一条 `exit_code: null` 的命令不能被循环标记为成功

## 范围之外
- 遥测管道（OTel、Langfuse）。反馈面向下一轮对话；遥测面向运维人员。
- 脱敏处理与轮转策略。课程练习提示涵盖了这些内容。

## 参考资料
- `docs/en.md` - 完整课程
- `code/main.py` - 参考实现
- `outputs/skill-feedback-runner.md` - 提取的技能
