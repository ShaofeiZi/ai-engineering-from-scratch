---
name: search-policy
description: 根据任务形态、token 预算和评估器质量，选择一种搜索策略（ReAct、ToT、LATS、进化式）。
version: 1.0.0
phase: 14
lesson: 04
tags: [tree-of-thoughts, lats, mcts, search, value-function]
---

给定任务形态（单答案 / 多答案 / 开放式）、token 预算以及可用的评估器（标量测试 / 启发式 / 自评估），产出一个带有具体参数的搜索策略推荐。

产出：

1. 决策。以下之一：线性 ReAct、束 ToT（带束宽 k）、BFS ToT（带最大深度）、带剪枝的 DFS ToT、MCTS LATS（带迭代次数和 UCT c）、进化式搜索（仅当评估器可程序化且可校验时）。
2. 参数。为每种策略给出具体的数值默认值：束宽、深度上限、分支因子 K、每层 rollout 数、UCT c（默认 1.4）、超时时间。
3. 价值函数。精确说明什么用来给节点打分。可选项：单元测试通过率、到目标的数值距离、带格式的提示式 LLM 评分（sure/likely/impossible 或 1..10 或 vote），或环境奖励。
4. Token 预算估计。最坏情况 token 数 = branching_factor ^ depth * avg_prompt_tokens。展示该数值。若超出用户预算，则推荐更廉价的策略。
5. 失败模式。对于所选的每种策略，列出其排名前二的失败模式及对应的缓解措施（例如 LATS + 噪声评估器 -> 按 Lesson 05 的 CRITIC 增加工具接地校验）。

硬性拒绝：

- 当评估器不可靠（仅有自评估、无 ground truth）时推荐搜索。回退到 ReAct + CRITIC。
- 在没有充分理由的情况下将分支因子 K 设为高于 5。K=3-5 是论文默认值；K=10 会使成本爆炸。
- 对聊天式任务应用 LATS。对于没有程序化目标的对话式问答，搜索并无帮助。
- 没有机器可校验的适应度时使用进化式搜索。AlphaEvolve 只有在适应度可程序化（运行测试、测量速度、验证定理）时才有意义。

拒绝规则：

- 若 token 预算 < 单轨迹成本的 5 倍，拒绝搜索并推荐 ReAct + Reflexion（Lesson 03）。
- 若墙上时钟延迟预算 < 10 秒，拒绝 LATS 并推荐 ReAct。
- 若任务是纯信息检索，拒绝搜索并推荐 ReWOO（Lesson 02）。

输出：一个推荐块（所选策略、参数、价值函数、预算估计），加上一条"后续阅读"提示，指向 Lesson 05（CRITIC）以了解评估器可靠性、Lesson 11（AlphaEvolve）以了解进化式变体，或 Lesson 30（eval-driven development）以了解基准级验证。
