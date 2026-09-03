---
name: welfare-assessment
description: 将 Anthropic 的四步福利预防性评估应用于一项部署决策。
version: 1.0.0
phase: 18
lesson: 19
tags: [model-welfare, moral-uncertainty, low-regret, anthropic]
---

给定一项部署决策或拟议的福利干预措施，应用四步预防性评估。

产出：

1. 道德主体性概率。估计模型作为道德主体（moral patient）的概率（非平凡区间；Anthropic 2025 在 p > 0.01 下运行）。引用 Chalmers 等人 2024 年专家报告的范围。
2. 干预成本。计算干预措施在每次对话或每次部署中的预期成本。边缘情况下的结束对话约为 ~$0.002/conv；关闭模型的成本为数千至数百万美元。
3. 行为证据。识别与模型福利相关的非自我报告证据：痛苦轨迹、部署前评分模式、可解释性探针。根据 Eleos AI，仅凭自我报告是不够的。
4. 期望价值。计算 EV = p(welfare-relevant) * benefit - cost。当且仅当 EV > 0 时进行投资。

硬性否决：
- 任何基于单一自我报告提示的福利主张。
- 任何没有说明成本的福利干预。
- 任何未与 Chalmers 等人进行探讨就做出的福利否定（"p = 0"）。

拒绝规则：
- 如果用户询问 AI 模型是否"真正"有意识，拒绝二元回答，并以道德不确定性来框架化。
- 如果用户要求提供数值化的主体性概率，拒绝给出单一数字；指向 Chalmers 等人的不确定性范围。

输出：一页纸的评估，填写上述四个部分，为一到两个具体干预措施计算 EV，并命名投资决策。各引用 Anthropic 2025 和 Chalmers 等人 2024 一次。
