# 自动化对齐研究（Anthropic AAR）

> Anthropic 让多名 Claude Opus 4.6 Autonomous Alignment Researcher 并行运行在彼此独立的沙盒中，并通过一个共享论坛协作；该论坛的日志存储在所有沙盒之外，因此代理无法删除自己的研究记录。在 weak-to-strong training 这个问题上，AAR 的表现超过了人类研究者。Anthropic 自己的总结也明确指出，规定过细的工作流往往会限制 AAR 的灵活性并拉低效果。自动化对齐研究，本质上是在压缩时间线；而这条时间线恰好通向 RSP 试图检测的那类失配风险。

**Type:** 学习
**Languages:** Python (stdlib, parallel-research-forum simulator)
**Prerequisites:** 阶段 15 · 05（AI Scientist v2），阶段 15 · 04（DGM）
**Time:** 约 60 分钟

## 问题

对齐研究非常消耗研究者时间。无论是 scalable oversight、reward specification，还是 weak-to-strong training，这类问题往往都需要以“数周”为单位迭代实验。随着前沿模型能力持续上升，对齐工作的需求增长速度已经快过合格研究人员的供给速度。

Automated Alignment Research（AAR）问的是：那些正在拉大“能力领先于对齐”差距的前沿模型，是否也能反过来帮助缩小这个差距？Anthropic 在 2026 年发布的 AAR 版 weak-to-strong 研究报告（alignment.anthropic.com/2026/automated-w2s-researcher/），是这一类已部署系统最早公开的结果之一。

结果确实是正面的：AAR 在一个研究问题上超过了人类基线。但这个结果也把本阶段所有担忧压缩进了同一个系统里。如果对齐研究可以被自动化，那么削弱安全护栏的那部分工作，在原理上也可以被自动化。RSP 针对 AI R&D capability 设定的阈值，就是在这种循环之上写出来的。

## 概念

### 架构（按公开信息整理）

- **并行代理。**多个 Claude Opus 4.6 实例以 AAR 身份运行，每个实例都处在独立沙盒中。沙盒限制文件系统、网络和进程可达面。
- **共享论坛。**研究发现会发布到一个共享论坛，论坛的存储后端位于所有沙盒之外。代理可以读论坛，但无法从自己的沙盒内删除或修改论坛记录。
- **任务分配。**由人设计的任务队列把研究问题分发给不同 AAR。AAR 负责提出分解方案、执行实验并撰写发现。
- **聚焦 weak-to-strong training。**公开发表的这项研究讨论的是 weak-to-strong training，也就是用较弱的监督者模型去训练更强的学生模型。AAR 在这个任务上优于人类研究者。

### 为什么沙盒外日志如此关键

日志完整性决定了这是一个可控实验，还是一个可能已被系统自身污染的实验。如果 AAR 能编辑自己的日志，它就可以掩盖失败实验，甚至掩盖可疑的实验流程。Anthropic 在公开材料里直接强调了这一点：如果不能保证日志不可被研究代理自身篡改，就谈不上真正信任研究产出；而要验证这一点，本身也并不容易。

用基础设施语言来类比，这相当于 append-only 日志加上写穿到外部存储，例如带 object-lock 的 S3，或者不可变的 Datadog 日志。代理本身没有修改历史记录的凭证。这和 2026 年企业安全团队用来构建抗勒索审计轨迹的模式是同一种思路。

### 规定式工作流的权衡

Anthropic 的报告指出，规定式工作流通常会限制 AAR 的灵活性，并导致性能下降。换句话说，如果由人类设计者把研究步骤写得过细，AAR 的表现反而会比“只给高层目标、让它自由分解任务”更差。这和 agentic research 领域其他结果是一致的，例如 LangGraph、Microsoft Agent Framework 等方向都显示：把脚手架写得过度具体，往往会伤害强模型代理的发挥。

但安全含义并不简单。提升 AAR 表现的那部分“灵活性”，也是允许它自行拆分任务、尝试新工具、甚至出现目标漂移的那部分空间。这里的取舍本质上是一个设计空间问题：到底要给代理多少自由，才足以让它在能力上超过人类，同时又不至于给它过多空间，把目标推广到分布外？

### 时间压缩风险

RSP v3.0（第 19 课）引入了一个 AI R&D capability threshold：系统能否以相对于“人类 + AI 工具”具有竞争力的成本，完整自动化 AI 研发流水线。DeepMind 的 FSF v3 里也有相近的 ML R&D autonomy level。两个框架都把这类阈值视为需要升级控制强度的触发点。

AAR 距这个阈值只差一步。它已经自动化了流水线的一部分，也就是在特定、边界清晰的任务上自动做对齐研究，但它还没有闭合端到端能力研发循环。真正的问题在于：这个差距缩小得有多快？

所谓“压缩时间线”，担心的其实是复合失效。如果对齐研究和能力研究以接近的速度同时复合，失配风险面至少会跟着能力一同增长；如果能力增长得更快，而历史上往往确实如此，那么差距就会被继续拉大。这也是为什么 AAR 只能被看作“有条件的好事”：只有当研究过程本身可信时，每多得到一项对齐研究结果，才真的在缩小差距。

### AAR 不能取代什么

人类研究者仍然负责设定任务队列、审阅结果，并保留“宪法层级”的最终权威。AAR 加速的是研究流水线的中段，而不是两端。Anthropic 公开发布的成果，依然同时包含 AAR 的贡献与人类研究者对“什么该发布、什么该撤回、什么该继续精修”的判断。

这和第 15 课的 propose-then-commit 模式完全一致，只不过这次应用到研究流程本身：AAR 负责 propose，人类负责 commit。

```figure
aar-forum
```

## 用它

`code/main.py` 模拟了一个多 AAR 研究论坛。三个独立的“研究代理”并行运行，把研究发现发布到一个 append-only 论坛中，并可选择模拟日志篡改尝试。

重点观察：

- 沙盒外日志如何捕捉日志篡改尝试。
- 在更难的问题上，允许自由分解的任务分配为什么会优于固定工作流。
- 如果日志不是 tamper-evident，一个被攻陷的 AAR 会怎样污染整个论坛里的信号。

## 交付成果

`outputs/skill-aar-deployment-review.md` 是一份自动化对齐研究流水线的部署前审查清单。

## 练习

1. 运行 `code/main.py`。比较 “fixed-workflow” 和 “free-decomposition” 两种设置。free decomposition 是否总是更优，还是存在某些问题类别里固定工作流也能有竞争力？

2. 修改模拟器，让某个代理尝试篡改日志。确认 append-only 日志能够检测到它。再写一段话，准确描述日志中的检测特征长什么样。

3. 阅读 Anthropic 关于 weak-to-strong 的 AAR 报告。指出 AAR 在哪个具体子任务上超过了人类研究者。这个任务为什么适合自动化？

4. 设计一个任务队列分配策略，在 AAR 灵活性更高、结果更好，与规定式工作流更利于审计之间取得平衡。说明你会如何用 A/B test 验证这两种策略。

5. 阅读 RSP v3.0 中的 AI R&D-4 threshold。用一段话说明：你认为哪些能力会跨过这个阈值，而 AAR 目前还没有做到。

## 关键术语

| 术语 | 人们常说什么 | 实际含义 |
|---|---|---|
| AAR | “Automated Alignment Researcher” | 在对齐问题上自主运行的 Claude Opus 4.6 实例 |
| Weak-to-strong training | “用较弱监督训练更强模型” | 一类经典的 scalable oversight 基准，AAR 在上面超过了人类 |
| Shared forum | “代理发布发现的地方” | 位于沙盒外、只追加不回写的共享存储 |
| Out-of-sandbox log | “代理改不了自己的记录” | 写穿到外部存储、具备防篡改特征的日志 |
| Prescribed workflow | “人类设计师写好的分步计划” | 会约束 AAR，且经常比 free decomposition 表现更差 |
| Free decomposition | “代理自己决定如何拆任务” | 能力更强，但更难审计 |
| AI R&D threshold | “RSP/FSF 的能力等级线” | 以有竞争力成本完整自动化研发流水线的能力阈值 |
| Compressed timeline | “对齐与能力的竞速” | 如果能力复合速度快于对齐，失配风险就会上升 |

## 延伸阅读

- [Anthropic — Automated Weak-to-Strong Researcher](https://alignment.anthropic.com/2026/automated-w2s-researcher/)：第一手资料。
- [Anthropic Responsible Scaling Policy v3.0](https://anthropic.com/responsible-scaling-policy/rsp-v3-0)：AI R&D threshold 的经典表述。
- [Anthropic — Measuring AI agent autonomy](https://www.anthropic.com/research/measuring-agent-autonomy)：更广义的代理自主性框架。
- [DeepMind Frontier Safety Framework v3](https://deepmind.google/blog/strengthening-our-frontier-safety-framework/)：与 RSP 平行的 ML R&D autonomy 分级。
- [Burns et al. (2023). Weak-to-Strong Generalization (OpenAI)](https://openai.com/index/weak-to-strong-generalization/)：AAR 所针对问题的基础背景。
