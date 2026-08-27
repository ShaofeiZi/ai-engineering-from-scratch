# 使用 HTN 与进化搜索进行规划

> 符号规划适合处理方案正确性可以证明的问题。进化式代码搜索适合处理 fitness function 可由机器检查的问题。ChatHTN（2025）与 AlphaEvolve（2025）展示了各自与 LLM 结合后能够释放的能力。

**Type:** 构建
**Languages:** Python（标准库）
**Prerequisites:** 阶段 14 · 02（ReWOO 与 Plan-and-Execute）
**Time:** 约 75 分钟

## 学习目标

- 解释 Hierarchical Task Network：task、method、operator、precondition 与 effect。
- 描述 ChatHTN 的混合循环——符号搜索加 LLM 回退分解。
- 解释 AlphaEvolve 的进化循环，以及为什么它只能与程序化 evaluator 配合工作。
- 仅用标准库实现玩具版 HTN planner 与玩具版进化搜索。

## 问题

ReWOO（第 02 课）、Plan-and-Execute 和 ReAct 可以覆盖大多数智能体规划，但不擅长以下两类问题：

1. **要求可证明正确的方案。** 调度、航路规划、合规工作流——方案必须在构造时就保证可靠。一个表达流畅、却偶尔虚构步骤的 LLM 方案不可接受。
2. **具有机器可检查 fitness function 的优化。** 矩阵乘法、调度启发式、compiler pass——目标不是“一个正确方案”，而是“最优方案”。

HTN 规划与 AlphaEvolve 分别解决这两个不同问题。二者都把 LLM 视为放大器，而不是替代品。

## 概念

### Hierarchical Task Network

HTN 包含：

- **Task**——compound task（有待分解）和 primitive task（可直接执行）。
- **Method**——在满足 precondition 时，把 compound task 分解成 subtask 的方式。
- **Operator**——带有 precondition 和 effect 的 primitive action。
- **State**——一组事实。

规划过程是：给定目标 task 与初始 state，找到由 primitive operator 构成的分解结果，使各 operator 的 precondition 按顺序得到满足。

HTN 的历史早于 LLM，至今仍是构造可证明正确方案的参考方法。

### ChatHTN（Gopalakrishnan 等，2025）

ChatHTN（arXiv:2505.11814）交替执行符号 HTN 与 LLM 查询：

1. 尝试使用已有 method 分解当前 compound task。
2. 如果没有适用 method，则询问 LLM：“你会如何分解 `task`，其 state 为 `s`？”
3. 把 LLM 响应转换成候选 subtask。
4. 根据 operator schema 验证；拒绝无效分解。
5. 递归执行。

论文的核心主张是：每个生成方案都可证明 sound，因为 LLM 的建议只作为候选分解进入系统，绝不会直接编辑方案。符号层掌握正确性；LLM 则扩展 method library。

在线 method learning（OpenReview `gwYEDY9j2x`，2025 年后续工作）增加一个 learner，通过回归泛化 LLM 生成的分解——最多可将 LLM 查询频率降低 75%。

### AlphaEvolve（Novikov 等，2025）

AlphaEvolve（arXiv:2506.13131，DeepMind，2025 年 6 月）是另一类系统：由 Gemini 2.0 Flash/Pro ensemble 编排的进化式代码搜索。

循环如下：

1. 从一个 seed program + 一个 programmatic evaluator（返回 fitness score）开始。
2. LLM ensemble 提出 mutation。
3. 用 evaluator 运行 mutation。
4. 保留最优者，再次 mutation。

已公布的成果包括：

- 56 年来首次改进 4x4 复数矩阵乘法的 Strassen 方法（48 次标量乘法）。
- 通过 Borg 调度启发式，回收 0.7% 的 Google 计算资源。
- 在一项前沿工作负载上把 FlashAttention 加速 32%。

硬性约束是：fitness function 必须可由机器检查。对散文答案进行进化搜索不会收敛。

### 各自适用的时机

| 问题类别 | 使用方案 | 原因 |
|---------------|-----|-----|
| 带硬约束的调度 | HTN + ChatHTN | 可证明 soundness |
| Compiler 优化 | AlphaEvolve | 可由机器检查的 fitness |
| 多步骤任务执行 | ReAct / ReWOO | LLM 参与循环，不提供形式保证 |
| 带测试的代码改进 | AlphaEvolve | 测试就是 evaluator |
| 受策略约束的自动化 | HTN | precondition 编码策略 |

### 这种模式会在哪里出错

- **没有 operator 的 HTN。** 如果缺少 precondition/effect schema，soundness 主张就会崩塌。ChatHTN 的“LLM 建议分解”必须依靠 schema 拒绝无效动作。
- **没有真实 evaluator 的 AlphaEvolve。** “询问 LLM 代码是否更好”不是 fitness function。evaluator 必须确定且快速。
- **过度工程化。** 大多数智能体任务两者都不需要。应先考虑 ReAct 或 ReWOO。

```figure
htn-tree-expand
```

## 构建它

`code/main.py` 实现两个玩具系统：

- 一个仅依赖标准库的 HTN planner，包含 operator、method、precondition、effect，以及在没有 method 匹配 compound task 时启动的 `LLMFallback`。“LLM”是脚本化 decomposer，因此 planner 可以离线运行。
- 一个对算术程序进行的标准库进化搜索：生成 expression，使其在测试集上的输出最小化 `|f(x) - target|`。evaluator 是确定性的。

运行：

```
python3 code/main.py
```

trace 会展示 HTN planner 如何分解 compound task（方案中途会使用一次 LLM fallback），以及进化循环如何收敛到目标 expression。

## 使用它

- **HTN planner**——`pyhop`、`SHOP3`，或针对领域专属策略执行自行构建。
- **ChatHTN**——研究代码；其“符号 + LLM 回退”模式可以直接移植到任意 HTN planner。
- **AlphaEvolve**——DeepMind 论文；其“ensemble + evaluator”模式可以复现。OpenEvolve 等开源 fork 正在出现。
- **智能体框架**——目前都没有一等的 HTN 或 AlphaEvolve 支持。可将其构建为子智能体或后台 worker。

## 交付它

`outputs/skill-hybrid-planner.md` 会生成混合 planner 脚手架（HTN 或 evolutionary），并明确限定 LLM 的角色。

## 练习

1. 为 HTN planner 增加 backtracking：当 operator 的 postcondition 在运行时失败，回滚并尝试下一个 method。
2. 为 ChatHTN 添加 LLM method cache：当 LLM 分解 task `T`，且 state pattern 为 `P` 时，保存结果。下次调用时先重新检查 method library。
3. 将进化搜索 evaluator 替换为真实测试套件。进化出一个能够通过 20 个测试用例的排序函数，并报告收敛所需 generation 数。
4. 阅读 AlphaEvolve 的 evaluator 设计说明。为你关心的领域设计 evaluator（SQL 查询优化、测试套件最小化、部署 YAML）。
5. 组合两种方法：使用 HTN 把 compound task 分解成 subtask，再对每个 subtask 的 primitive operator 使用进化搜索。它在哪里大放异彩，在哪里又属于过度工程化？

## 关键术语

| 术语 | 人们通常怎么说 | 它的实际含义 |
|------|----------------|------------------------|
| HTN | “分层 planner” | 使用 operator、precondition 和 effect 进行 task 分解 |
| Method | “分解规则” | 把 compound task 拆成 subtask 的方法 |
| Operator | “Primitive action” | 带 precondition 与 effect 的具体步骤 |
| ChatHTN | “LLM + HTN” | 没有 method 匹配时，符号 planner 询问 LLM |
| AlphaEvolve | “进化式代码搜索” | LLM ensemble 改变代码；确定性 evaluator 负责选择 |
| Fitness function | “Evaluator” | 对输出执行确定、机器可检查的评分 |
| Online method learning | “缓存的 LLM 分解” | 保存并泛化 LLM 方案，以降低查询成本 |

## 延伸阅读

- [Gopalakrishnan 等，ChatHTN（arXiv:2505.11814）](https://arxiv.org/abs/2505.11814)——符号 + LLM 混合 planner
- [Novikov 等，AlphaEvolve（arXiv:2506.13131）](https://arxiv.org/abs/2506.13131)——使用 LLM mutation 的进化式代码搜索
- [Anthropic，构建有效智能体](https://www.anthropic.com/research/building-effective-agents)——何时应使用 planner，何时简单循环更合适
