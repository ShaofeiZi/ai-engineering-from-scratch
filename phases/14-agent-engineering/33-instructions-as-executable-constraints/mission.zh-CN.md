# 任务 - 将智能体指令转化为可执行约束

## 目标
将散文式指令转化为跨五个类别的机器可检查规则，并输出一份评审人员可打分的规则报告。

## 输入
- `docs/agent-rules.md`，每个标题对应一条规则，每条规则包含 slug、category、description 以及一个 `check` 字段
- 一次故意违反两条规则的演示智能体运行

## 交付物
- 解析器，将 `agent-rules.md` 加载为 dataclass
- `rule_checker.py` 风格的函数，每个 `check` 引用对应一个
- `rule_report.json`，包含每条规则的通过/失败及汇总严重级别

## 验收标准
- `python3 code/main.py` 退出码为零
- 输出打印解析后的规则集、运行轨迹以及每条规则的通过/失败
- `rule_report.json` 捕获两条故意违规

## 不在范围内
- 将检查器接入 CI。本课程止步于一份书面报告。
- 框架护栏（OpenAI SDK、LangGraph interrupts）。规则集是这些框架所实现的人类可读契约。

## 参考资料
- `docs/en.md` - 完整课程
- `code/main.py` - 参考实现
- `outputs/skill-rule-set-builder.md` - 提取的技能
