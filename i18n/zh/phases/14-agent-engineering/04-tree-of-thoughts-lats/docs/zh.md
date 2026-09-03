# Tree of Thoughts 与 LATS：审慎搜索

> 单条思维链轨迹没有回溯空间。ToT（Yao 等，2023）把推理转化为一棵树，并在每个节点进行自我评估。LATS（Zhou 等，2024）以蒙特卡洛树搜索为框架，统一了 ToT、ReAct 和 Reflexion。Game of 24 的准确率从 CoT 的 4% 提升到 ToT 的 74%；LATS 在 HumanEval 上达到 92.7% pass@1。

**Type:** 构建
**Languages:** Python（标准库）
**Prerequisites:** 阶段 14 · 01（智能体循环）、阶段 14 · 03（Reflexion）
**Time:** 约 75 分钟

## 学习目标

- 将推理表达为搜索：节点是“思路”，边是“扩展”，价值表示“有多大希望”。
- 使用标准库实现带自我评估评分的 ToT 风格 BFS 树搜索。
- 扩展为玩具版 LATS MCTS 循环，包含选择／扩展／模拟／反向传播。
- 判断何时值得为搜索付出成倍的 token 成本（Game of 24、代码生成），何时单条轨迹已经足够（简单问答）。

## 问题

思维链是一条线性路径。如果第一步错了，之后每一步都会建立在错误前提之上。在 Game of 24（用四个数字和 + − × ÷ 得到 24）中，GPT-4 CoT 的准确率只有 4%。模型会过早选错子表达式，之后便无法恢复。

推理真正需要的是提出多个候选方案、评估这些方案、选择有希望的方向，并在遇到死路时回溯的能力。这就是搜索。Tree of Thoughts 与 LATS 是两种经典形式。

## 核心概念

### Tree of Thoughts（Yao 等，NeurIPS 2023）

每个节点都是一个连贯的中间步骤（“一条思路”），且可以扩展成 K 个子思路。LLM 使用评分提示对每个节点进行自我评估。搜索会探索整棵树，可以采用 BFS、DFS 或 beam search。

```
                     (root: "find 24 from 4 6 4 1")
                    /               |            \
           ("6 - 4 = 2")    ("4 + 1 = 5")    ("4 * 6 = 24")  <- Score: HIGH
              /   \              |                  |
          ...    ...          ...                finish
```

自我评估是整个方法的承重部件。论文展示了三种变体：`sure / likely / impossible` 分类、`1..10` 数值评分，以及在候选项之间投票。三者在 Game of 24 上都大幅超过 CoT（使用 GPT-4 时从 4% 提升到 74%）。

### LATS（Zhou 等，ICML 2024）

LATS 在 MCTS 框架下统一了 ToT、ReAct 与 Reflexion。LLM 扮演三种角色：

- **Policy**：提出候选的下一步行动（ReAct 风格）。
- **Value function**：为部分轨迹评分（ToT 风格的自我评估）。
- **Self-reflector**：失败时写下自然语言反思（Reflexion 风格），并用它为未来 rollout 重新播种。

环境反馈（观察）会混入 Value function，使搜索依据真实工具结果，而不只是模型意见。论文发表时的结果为：使用 GPT-4 在 HumanEval 上达到 92.7% pass@1（当时的 SOTA），使用 GPT-3.5 在 WebShop 上获得 75.9 的平均分（接近基于梯度的微调）。

### 最简 MCTS

每轮迭代有四个阶段：

1. **选择（Select）**——使用 UCT（树的置信上限）从根节点向下走到叶节点。
2. **扩展（Expand）**——通过 Policy 生成 K 个子节点。
3. **模拟（Simulate）**——使用 Policy 从某个子节点 rollout 到叶节点，再由 Value function（或环境奖励）为叶节点评分。
4. **反向传播（Backpropagate）**——沿路径向上更新访问次数与价值估计。

UCT 公式：`Q(s, a) + c * sqrt(ln N(s) / N(s, a))`。第一项是利用，第二项是探索。应针对不同任务调节 `c`。

### 成本现实

搜索会让 token 用量爆炸。ToT 在 Game of 24 上使用的 token 是 CoT 的 100–1000 倍，LATS 也与之类似。这绝非免费，应只为以下情况保留搜索：

- 单条轨迹已被证明确实不够的任务（Game of 24、复杂代码）。
- 正确性比挂钟时间更重要的任务。
- 具有便宜而可靠的 Value function 的任务（代码单元测试、数学中的明确目标）。

如果任务只有一个正确答案，但 Evaluator 噪声很大，搜索往往会让结果更糟——它会找到一个“得分很高”的错误答案。

### 2026 年的定位

大多数生产智能体不会运行 LATS，而是运行带有工具落地验证的 ReAct（CRITIC，第 05 课）。搜索主要出现在专业细分场景：

- 以测试作为 Value function 的编码智能体（HumanEval 风格）。
- 探索多条查询路径的深度研究智能体。
- LangGraph 子图中的规划密集型工作流。

AlphaEvolve（第 11 课）是 2025 年的极端案例：对代码进行进化搜索，使用机器可检查的适应度，并取得前沿突破（56 年来首次改进 4x4 矩阵乘法）。

```figure
tree-of-thoughts
```

## 动手构建

`code/main.py` 实现了：

- 在一个风格化“选择算术运算”任务上运行的小型 ToT BFS。
- 在同一任务上运行的玩具版 LATS MCTS 循环，包含选择／扩展／模拟／反向传播，并使用 UCT 选择。
- 将符号评分与自我评估分数组合起来的 Value function。

运行：

```
python3 code/main.py
```

追踪会展示 ToT 如何通过 BFS 为每个节点扩展三个候选项，并与 LATS 通过 MCTS 收敛到最佳 rollout 的过程进行比较。两者的 token 数都会打印出来。

## 实际使用

LangGraph 以子图模式提供 ToT 风格探索；LangChain 团队 2024 年 5 月关于 LATS 的博客是参考教程。LlamaIndex 提供 `TreeOfThoughts` 智能体。对于大多数 2026 年生产智能体，这种模式位于一个 `if task_complexity > threshold: use_search()` 门控之后——参见第 05 课的 Evaluator–Optimizer 模式。

## 交付成果

`outputs/skill-search-policy.md` 根据任务形态、预算和 Evaluator 保真度，在线性 ReAct、ToT、LATS 与进化搜索之间作出选择。

## 练习

1. 分别使用 UCT c=0.1 和 c=2.0 运行玩具版 LATS。追踪发生了什么变化？
2. 将 Value function 换成噪声更大的评分器（加入随机抖动）。MCTS 还能找到最佳叶节点吗？它能容忍的最低信噪比是多少？
3. 实现 beam-search ToT（每一层保留 top-k），并与 BFS 比较。在 token 预算紧张时哪一种更好？
4. 阅读 LATS 第 5.1 节，复现 HumanEval 的轨迹数量：需要多少次 rollout 才能达到论文报告的 pass@1？
5. 阅读 LATS 论文中关于“LATS 何时帮助较少”的讨论。写出一段决策规则，将任务形态映射到搜索策略。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------------|------------------------|
| Tree of Thoughts | “分支式 CoT” | Yao 等——由思路节点与自我评估组成的树 |
| LATS | “用于 LLM 的 MCTS” | Zhou 等——在 MCTS 下统一 ToT + ReAct + Reflexion |
| UCT | “置信上限” | 平衡利用（Q）和探索（ln N / n）的选择公式 |
| Value function | “这个状态有多好” | 通过提示得到的 LLM 分数或环境奖励；用于反向传播 |
| Policy | “行动提议器” | ReAct 风格生成器；输出候选的下一条思路／行动 |
| Rollout | “模拟轨迹” | 使用 Policy 从节点走到叶节点，再用 Value 评分 |
| 反向传播 | “更新祖先节点” | 沿路径向上传递叶节点奖励，并更新访问次数与 Q |
| 搜索成本 | “Token 爆炸” | Game of 24 上为 CoT 的 100–1000 倍；采用前先做预算 |

## 延伸阅读

- [Yao 等，Tree of Thoughts（arXiv:2305.10601）](https://arxiv.org/abs/2305.10601)——奠基论文
- [Zhou 等，LATS（arXiv:2310.04406）](https://arxiv.org/abs/2310.04406)——融合 Reflexion 反馈的 MCTS
- [LangGraph 概览](https://docs.langchain.com/oss/python/langgraph/overview)——用于搜索的子图模式
- [AlphaEvolve（arXiv:2506.13131）](https://arxiv.org/abs/2506.13131)——使用程序化 Evaluator 的进化搜索
