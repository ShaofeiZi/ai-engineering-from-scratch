---
name: benchmark-harness
description: 构建一个 SWE-bench 风格的评测框架，包含 FAIL_TO_PASS / PASS_TO_PASS 门控、污染检查和步数指标。
version: 1.0.0
phase: 14
lesson: 19
tags: [swe-bench, gaia, agentbench, harness, evaluation]
---

给定一个代码库和一组 (bug, fix) 对，构建一个以真实单元测试为门控并记录运营指标的基准评测框架。

产出：

1. 每个任务的定义：`(tid, description, state_before, fail_to_pass_tests, pass_to_pass_tests, solution)`。
2. 一个运行器，应用智能体的补丁，在沙箱中运行仓库的测试套件，并记录：FTP 通过数、PTP 通过数、步数、token 数、挂钟时间、成本。
3. 一个污染检查：将 issue 文本与产出的补丁进行模式匹配；重叠率 >=30% 时标记。
4. 一个报告器，按任务和汇总分数输出 JSON，并附带 P50/P75/P95 的步数和成本。
5. 一个 CI 作业，在每个 PR 上运行该框架，并在回归 >=5% 时失败。

硬性拒绝：

- 只报告单一汇总数字的框架。要求提供每任务结果 + 分布。
- 不在沙箱中运行测试的框架。智能体提供的补丁是不可信代码。
- 没有 PASS_TO_PASS 门控的框架。破坏其他测试的补丁会静默导致产品回归。

拒绝规则：

- 如果用户要求“只要 FAIL_TO_PASS 分数”，拒绝。添加 PASS_TO_PASS；破坏已有测试比遗漏修复是更严重的回归。
- 如果测试未锚定到特定 commit，拒绝。测试的漂移会使得各次运行的分数不可比较。
- 如果任务与训练期间见过的 issue 文本重叠，明确标记。

输出：`tasks.py`、`harness.py`、`contamination.py`、`report.py`、`README.md`，解释沙箱、门控和污染策略。最后以“接下来读什么”结尾，指向 Lesson 30，介绍在框架之上进行评估驱动开发。
