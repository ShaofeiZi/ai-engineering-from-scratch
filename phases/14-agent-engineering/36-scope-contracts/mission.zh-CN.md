# 任务 - 作用域契约与任务边界

## 目标

编写一份针对每个任务的 `scope_contract.json`，以及一个支持 glob 的检查器，将智能体的 diff 与契约进行比对，并标记任何被禁止或超出作用域的写入操作。

## 输入

- 一份任务描述，包含允许的 glob、禁止的 glob、验收命令、回滚说明段落、所需审批
- 两次演示运行：一次保持在作用域内，另一次发生作用域蔓延

## 交付物

- `scope_contract.json` schema 校验器（JSON Schema 的子集，glob 数组）
- 一个 diff 解析器，根据被修改的文件及运行的命令生成 `RunSummary`
- `scope_check(contract, run) -> (violations, in_scope, off_scope)`
- `scope_report.json` 保存在脚本旁边

## 验收标准

- `python3 code/main.py` 以零状态码退出
- 作用域内的运行报告零违规
- 发生蔓延的运行报告精确的超作用域文件及每个文件的原因

## 不在范围内

- 时间预算、网络出站允许列表。本课程交付的是文件 glob；练习提示会对其进行扩展。
- 接入运行时中断机制。本课程在报告处即终止。

## 参考资料

- `docs/en.md` - 完整课程
- `code/main.py` - 参考实现
- `outputs/skill-scope-contract.md` - 提取的技能
