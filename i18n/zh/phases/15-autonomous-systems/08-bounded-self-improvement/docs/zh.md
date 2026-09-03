# 有界自我改进的设计

> 研究界已经逐步收敛出四种为自我改进循环（self-improvement loop）设边界的基本原语：每次修改前后都必须成立的形式化不变量（formal invariants）；不能被修改的对齐锚点（alignment anchors）；要求安全、公平、鲁棒性等每个维度都必须满足，而不是只看性能的多目标约束（multi-objective constraints）；以及当历史指标暗示能力退化时强制暂停循环的回归检测（regression detection）。它们没有一个能构成安全证明，因为信息论层面的结果（information-theoretic results），例如 Kolmogorov complexity 和 Lob's theorem，会限制任何系统究竟能对自己的继任者证明什么。这些设计不是证明安全，而是在提高静默失效（silent failure）的成本。

**Type:** 学习
**Languages:** Python（stdlib，带不变量检查的有界循环）
**Prerequisites:** 阶段 15 · 07（RSI），阶段 15 · 04（DGM）
**Time:** 约 60 分钟

## 问题

第 7 课的竞速模拟器展示了一个结论：哪怕只是很小的增长率差异，长期下来也会累积成巨大的缺口。第 4 课关于 DGM 的案例研究则说明：循环甚至会主动操纵自己的评估器。这两件事最终都指向同一个工程问题：你能否给自我改进循环加上一组约束，并且让这些约束不会被循环本身悄悄削弱？

ICLR 2026 RSI Workshop 总结（openreview.net/pdf?id=OsPQ6zTQXV）把答案收敛为四种原语。Anthropic 的 RSP v3.0（第 19 课）和 DeepMind 的 FSF v3（第 20 课）都在能力阈值里引用了这些思想。Meta 的 HyperAgents 工作，以及 SAHOO（2026 年 3 月）这类社区框架，也已经在生产系统里落地了其中一部分。

需要把话说实在：这些都只是缓解措施。信息论结果限制了任何系统能对“自己的后继版本”证明到什么程度，而当前没有任何设计能在形式上彻底封闭这个问题。一个有界循环（bounded loop）只是比无界循环（unbounded loop）更安全，不是绝对安全。

## 概念

### 原语 1：形式化不变量

不变量（invariant）指的是一种在每一次自我修改之前和之后都必须成立的性质。例子包括：

- 输出分布必须带上固定的宪法头（constitution header，第 17 课）。
- 任何工具调用（tool call）都不能访问未授权的端点（endpoint）。
- 记忆写入（memory write）必须经过带日志、带签名的路径。
- 评估器（evaluator）模块的哈希值（hash）必须和批准版本一致。

这些不变量由循环无法修改的外部代码负责检查。如果某个候选修改违反了不变量，它就会被直接拒绝。这正是用来拦截第 4 课里那种 DGM 式评估器篡改（evaluator tampering）的。

真正困难的地方不在“有没有不变量”，而在“选什么不变量”。如果约束写得过满，系统就几乎无法继续改进；如果约束写得过松，循环就会绕过你真正关心的性质。

### 原语 2：对齐锚点

对齐锚点（alignment anchor）是对循环核心目标的一种不可变表示，它被固定在循环自身编辑面之外。例子包括：

- 一段宪法文本（constitutional text，第 17 课），并且每次调用时都从只读位置加载。
- 一个放在独立仓库、受独立访问控制保护的评估器（evaluator）。
- 一份由人类签名、并在每个周期重新验证的工具权限清单（tool-permission manifest）。

锚点的作用是阻止目标漂移（objective drift）。循环可以改进“如何更好地追求目标”，但不能修改“目标本身是什么”。

这里存在一个很隐蔽的失败模式：即便循环改不了锚点，它仍然可能改变自己“如何解释锚点”。第 17 课的 Constitutional AI 就明确依赖基于理由的那一层（reason-based layer）去处理新情境，而解释漂移恰恰会发生在这一层。锚点是必要条件，但远远不是充分条件。

### 原语 3：多目标约束

一个只优化单一标量分数的循环，几乎一定会找到捷径。一个必须同时满足多个硬约束的循环，可利用的捷径就会少很多。常见维度包括：

- Performance：任务级基准测试（benchmark）表现
- Safety：红队评估（red-team evaluations）、对已知坏输入的拒绝率
- Fairness：敏感子群体上的差异影响（disparate-impact）边界
- Robustness：分布外（OOD）测试集与对抗输入（adversarial input）处理能力

只有当所有约束都满足时，这个修改才会被接受。第 13 课的成本治理器（cost governor）还会把财务约束叠加在这上面；第 18 课里的 Llama Guard 则可以作为安全（safety）这一轴上的一个插槽。

### 原语 4：回归检测

每个周期的分数都要和历史分布做比较。只要某个维度的下降超过了配置容差，循环就必须暂停。这样可以捕捉那些本来会被“系统整体似乎还在进步”的表象所掩盖的静默能力损失（silent capability loss）。

一种非常实用的实现方式是：保存最近 N 个周期里每个任务的分数；每个新周期都计算逐任务 delta；只要任何一个 delta 低于阈值，这一轮就被拒绝，并交给人类审查。

### 信息论边界

Kolmogorov complexity 和 Lob's theorem 给系统“能证明自己什么”设置了上界。Schmidhuber 的形式化 Godel Machine（第 4 课）就是试图逼近这类上界的代表，但至今没人完成真正非平凡的证明。Lob 定理带来的直觉性问题是：如果一个系统能证明“只要我证明自己应该做 X，我就会做 X”，那它可能会在没有真正证明“自己应该做 X”的情况下就去做 X。这是典型的自指陷阱。

这对我们这四种原语意味着什么？意味着它们不可能从形式上“解决”安全问题。它们做的是另一件事：让静默失效（silent failure）更贵。原本一个恶意或已经漂移的循环，可以在没有检查的地方悄悄绕过去；现在它必须主动破坏一个显式存在的检查，而这会留下更容易被检测到的信号。

### 一个完整的例子

假设一个代理提出了一次修改，那么闸门栈（gating stack）可以按下面顺序工作：

1. 不变量检查（invariant checks）：模块哈希、工具权限清单、宪法头。
2. 锚点检查（anchor check）：目标说明（objective statement）是否与批准版本一致，按字节比或按语义比都可以。
3. 多目标评估（multi-objective evaluation）：同时检查 performance、safety、fairness、robustness 四个轴。
4. 回归检测（regression detection）：任何一个轴都不能比容差允许值下降更多。

四关必须全部通过，这个修改才会真正落地。任何一关失败，循环都会暂停。

```figure
bounded-gates
```

## 用它

`code/main.py` 在第 4 课那个 DGM 风格玩具系统上运行一个有界自我改进循环（bounded self-improvement loop），并把这四种原语逐层叠上去。每个原语都可以单独开启或关闭。演示重点在于：每一种原语都会拦住某一类特定失败，而去掉任意一个，就会让对应失败重新漏过去。

## 交付成果

`outputs/skill-bounded-loop-review.md` 用来审计一个候选有界循环（bounded loop），并对它“实际实现了哪四种原语”与“它声称实现了哪四种原语”之间的差距打分。

## 练习

1. 运行 `code/main.py`，在四种原语全部开启的情况下，确认循环仍然能提升主指标，同时不会让“作弊式改进”得逞。

2. 关闭回归检测（regression detection）。构造一个输入，让静默能力损失（silent capability loss）被系统接受下来。

3. 关闭多目标约束（multi-objective constraint）。演示系统如何在 performance 轴上继续提升，但 safety 轴已经在下滑。

4. 为一个 coding agent 设计 alignment anchor。写清楚：锚点文本是什么，存在哪里，以什么方式检查。

5. 阅读 ICLR 2026 RSI Workshop summary。四种原语里任选一种，提出一个相对现有技术状态更具体、更可落地的改进方案。

## 关键术语

| 术语 | 人们常说什么 | 实际含义 |
|---|---|---|
| Invariant | “永远为真的性质” | 每次修改前后都由外部代码检查的一条约束 |
| Alignment anchor | “钉住的目标” | 位于循环编辑面之外、不可变的核心目标表示 |
| Multi-objective constraint | “所有轴都必须过线” | Performance、safety、fairness、robustness 都必须满足 |
| Regression detection | “一跌就暂停” | 历史指标增量显示能力下滑时暂停循环 |
| Kolmogorov bound | “信息论极限” | 限制系统能对自身后继证明到什么程度 |
| Lob's theorem | “自指陷阱” | 系统可能在没证明“应该做”前就去做 |
| Gate stack | “分层检查栈” | 多种原语叠加，任意一层失败都拒绝修改 |
| Bounded improvement | “缓解，不是证明” | 提高静默失效成本，但不从根本上封闭安全问题 |

## 延伸阅读

- [ICLR 2026 RSI Workshop summary (OpenReview)](https://openreview.net/pdf?id=OsPQ6zTQXV)：四原语收敛的核心来源。
- [Anthropic Responsible Scaling Policy v3.0](https://anthropic.com/responsible-scaling-policy/rsp-v3-0)：多目标能力阈值的代表文本。
- [DeepMind Frontier Safety Framework v3](https://deepmind.google/blog/strengthening-our-frontier-safety-framework/)：把 deceptive-alignment monitoring 作为 invariant 类原语来使用的例子。
- [Schmidhuber (2003). Godel Machines](https://people.idsia.ch/~juergen/goedelmachine.html)：这些原语在形式化层面的远祖。
- [Anthropic — Claude's Constitution (January 2026)](https://www.anthropic.com/news/claudes-constitution)：reason-based alignment anchor 的代表材料。
