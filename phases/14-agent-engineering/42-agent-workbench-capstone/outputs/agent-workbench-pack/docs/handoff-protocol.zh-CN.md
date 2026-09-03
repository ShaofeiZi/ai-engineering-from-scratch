# 交接协议

每次会话结束时都会生成一份交接包，包含：

- summary
- changed_files
- commands_run
- failed_attempts
- open_risks（严重程度 + 详情）
- next_action（一个具体步骤）
- verdict_pointer（指向验证与评审报告的路径）

该包同时以 handoff.md（面向人类）和 handoff.json（面向下一个智能体）的形式输出。
缺失字段将中止会话结束钩子。
