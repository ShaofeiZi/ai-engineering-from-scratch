# 任务 - 多会话交接

## 目标
在会话结束时，从工作台产物生成 `handoff.md` 和 `handoff.json`，使下一个会话在第一分钟内即可高效开始。两种格式携带相同的七个字段；二者不一致时以 JSON 为准。

## 输入
- 来自先前课程的 `agent_state.json`、`verification_report.json`、`review_report.json`、`feedback_record.jsonl`
- 七个字段：summary、changed_files、commands_run、failed_attempts、open_risks、next_action、verdict_pointer

## 交付物
- 一个 `WorkbenchSnapshot` 加载器，打包上述四个产物
- `generate_handoff(snapshot) -> (markdown, payload)`
- 一个反馈过滤器，选取最后 K 条记录以及所有退出码非零的记录
- 将 `handoff.md` 和 `handoff.json` 写入脚本所在目录

## 验收标准
- `python3 code/main.py` 退出码为零
- 两个文件都包含全部七个字段，且 `next_action` 非空
- 使用相同输入重新运行脚本，生成的产物完全一致

## 不在范围内
- 压缩策略（Codex compact 端点、Claude Code 五阶段压缩）。交接用于关闭会话；压缩用于延长会话。
- PR 模板化。该 markdown 可作为 PR 正文复用，但课程止步于文件本身。

## 参考资料
- `docs/en.md` - 完整课程
- `code/main.py` - 参考实现
- `outputs/skill-handoff-generator.md` - 提取的技能
