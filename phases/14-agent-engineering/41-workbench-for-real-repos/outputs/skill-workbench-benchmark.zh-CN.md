---
name: workbench-benchmark
description: 在项目自带的示例应用上，将同一任务分别通过仅提示词和工作台引导的流水线运行，并产出包含五项结果的前后对比报告。
version: 1.0.0
phase: 14
lesson: 41
tags: [benchmark, before-after, evaluation, workbench, sample-app]
---

给定一个代码仓库、一个智能体产品以及一个小型示例应用，产出一套可移植的评测工具，用于比较仅提示词流水线与工作台引导流水线。

产出内容：

1. `eval/sample_app/` —— 一个从项目领域中抽取的最小可用示例应用。
2. `eval/run_prompt_only.py` 和 `eval/run_workbench.py`，二者各自接收一段任务描述并返回一个 `TaskOutcome`。
3. `eval/report.py`，运行两条流水线并写出 `before-after-report.md` 以及 `comparison.json`。
4. CI 工作流，当工作台结果在固定任务套件上出现退化时即失败。
5. `docs/benchmark.md`，解释五项结果以及何种情况算作退化。

硬性拒绝条件：

- 只有单条流水线的基准测试。对比本身就是全部意义所在。
- 结果以百分比表示而无分母。必须始终报告 `n / m`。
- 使用智能体产品训练过的示例应用。应使用针对领域定制的测试夹具。
- 隐藏假阴性的报告。仅提示词流水线更快的任务必须逐一列出。

拒绝规则：

- 如果项目没有验收命令，拒绝交付该基准测试。没有可度量的对象。
- 如果工作台流水线在中间值任务上耗时超过仅提示词流水线的 3 倍，应揭示该发现；需要简化的是工作台，而非模型。
- 如果该工具无法离线运行，拒绝将其接入 CI。网络不稳定会破坏对比结果。

输出结构：

```
<repo>/
├── eval/
│   ├── sample_app/
│   ├── run_prompt_only.py
│   ├── run_workbench.py
│   └── report.py
├── outputs/eval/
│   ├── before-after-report.md
│   └── comparison.json
├── docs/benchmark.md
└── .github/workflows/benchmark.yml
```

以“接下来阅读什么”结尾，指向：

- 第 42 课，了解打包工作台流水线所用全部界面的毕业项目包。
- 第 19 课（SWE-bench、GAIA、AgentBench），了解本基准测试所补充的宏观基准测试。
- 第 30 课（评测驱动的智能体开发），了解基准测试接入后持续进行的评测循环。
