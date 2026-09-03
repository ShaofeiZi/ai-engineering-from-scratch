---
name: reviewer-agent
description: 搭建一个审查者智能体角色，配备五维度评分量表，读取构建者产物，生成结构化审查报告，并让人工审查从一份已写就的页面开始，而非从空白页开始。
version: 1.0.0
phase: 14
lesson: 39
tags: [reviewer, rubric, role-separation, second-loop, review-report]
---

给定一个已在产出工作台产物的构建者智能体，搭建一个审查者来读取这些产物并撰写结构化报告。

产出：

1. `agents/reviewer.md`，包含审查者的系统提示词：只读访问、五维度评分量表、每个评分必须引用产物路径。
2. `tools/reviewer.py`，从工作台加载 `ReviewerInputs`，并按维度运行 LLM 评分器。
3. `outputs/review/<task_id>.json`，作为标准审查报告路径。
4. `docs/reviewer-rubric.md`，列出五个维度、每个维度回答的问题，以及 0-1-2 的锚点描述。
5. CI 步骤，在构建者任务关闭时将审查报告作为 PR 评论发布。

硬性拒绝：

- 审查者对 diff 拥有写权限。构建者与审查者之间的差距正是全部信号来源；抹平这一差距会破坏可靠性。
- 评分量表缺少每个分数的锚点描述。没有锚点的“0 到 2 打分”会退化为凭感觉。
- 审查报告省略引用。每个评分必须指向某个文件或追踪条目。
- 与构建者共享系统提示词。同一模型可以；同一提示词不行。

拒绝规则：

- 如果构建者未产出验证报告，拒绝运行审查者。验收必须先成立，评判才有价值。
- 如果项目已关闭任务少于三个，拒绝声称评分量表已校准。将首批报告保存为校准集。
- 如果被要求在最低置信度以下打分，审查者应拒绝，并将不确定的维度上报给人工。

输出结构：

```
<repo>/
├── agents/reviewer.md
├── tools/reviewer.py
├── outputs/review/
│   └── <task_id>.json
├── docs/reviewer-rubric.md
└── .github/workflows/review.yml
```

以“后续阅读”结尾，指向：

- 第 40 课，介绍结合验证 + 审查的交接包。
- 第 41 课，介绍端到端演练构建者/审查者分离的真实风格任务。
- 第 05 课（Self-Refine 与 CRITIC），介绍本课所改进的单智能体自审查基线。
