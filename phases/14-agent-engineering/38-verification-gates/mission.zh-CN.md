# 任务 - 验证关卡

## 目标
将 `verify(task_id, artifacts)` 实现为基于范围报告、规则报告、反馈日志和 diff 的纯确定性函数，在每次任务收尾时输出一个 `verification_report.json`。

## 输入
- `scope_report.json`、`rule_report.json`、`feedback_record.jsonl` 以及 diff 的桩加载器
- 检查表：验收已运行、验收以零退出码退出、范围干净、无 `null` 退出、所有阻断级严重性的规则通过

## 交付物
- 一个纯函数 `verify(task_id, artifacts) -> VerdictReport`
- 一个打印器，展示每项检查的结果以及最终的通过/失败
- 三个写入磁盘的演示场景：干净通过、范围蔓延、验收缺失

## 验收标准
- `python3 code/main.py` 以零退出码退出
- 干净通过场景报告 `passed: true`；另外两个场景报告 `passed: false`
- 每个场景在 `outputs/verification/` 下写入一个独立的 `verification_report.json`

## 范围之外
- LLM 作为评判者的逻辑。验证关卡保持确定性；定性判断属于第 39 课中的评审者。
- 已签名的覆盖审计日志。练习提示会引导以该方式扩展验证关卡。

## 参考
- `docs/en.md` - 完整课程
- `code/main.py` - 参考实现
- `outputs/skill-verification-gate.md` - 提取的技能
