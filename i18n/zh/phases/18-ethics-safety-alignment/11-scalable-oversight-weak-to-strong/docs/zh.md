# 可扩展监督与弱到强泛化

> Burns 等人（OpenAI Superalignment，"Weak-to-Strong Generalization"，2023）提出了一个可作为超级对齐代理问题的实验设定：用较弱模型生成的标签去微调强模型。如果强模型能够从这种不完美的弱监督中正确泛化，那么当前以人类为尺度的对齐方法也许能够延伸到超人系统。可扩展监督与 W2SG 是互补关系。可扩展监督，例如辩论、递归奖励建模、任务分解，会提升监督者的有效能力，使其能跟上被监督模型；W2SG 则保证强模型能够从监督者给出的不完美监督中正确泛化。Debate Helps W2SG（arXiv:2501.13124，2025 年 1 月）将这两者结合起来。

**Type:** 学习
**Languages:** Python (stdlib, W2SG gap simulator)
**Prerequisites:** 阶段 18 · 01（指令遵循作为对齐信号）、阶段 18 · 10（AI Control：在破坏尝试下保障安全）、阶段 09（强化学习基础）
**Time:** 约 60 分钟

## 学习目标

- 定义可扩展监督与弱到强泛化，并解释它们为何互补。
- 描述 Burns 等人 2023 年的实验设定：用 GPT-2 产生的标签微调 GPT-4。
- 解释 Performance Gap Recovered（PGR）指标及其衡量对象。
- 说明三种主要的可扩展监督机制，即 debate、recursive reward modeling、task decomposition，并各举一个优势。

## 问题

到目前为止，第 18 阶段中的所有对齐技术都默认监督者有能力评估模型行为。但一旦模型达到超人水平，监督者就成了系统中的薄弱环节。超级对齐提出的问题是：一个更弱的监督者，是否能够可靠地产出一个更强且仍然对齐的模型？

Burns 等人把这个问题缩减成一个可操作、可实证的实验设定：用弱监督去监督强模型，然后测量在弱监督之下，强模型原本的能力还保留下来多少。这不是超级对齐的完整解法，而是一个能让我们衡量进展的代理任务。

## 概念

### W2SG：Burns 等人的实验设定

- 弱模型：GPT-2 级别。
- 强模型：GPT-4 级别。
- 目标：该任务上强 GPT-4 的性能上限。

步骤如下：
1. 获取弱模型在某个任务上的 zero-shot 预测。
2. 用这些弱标签数据去微调强模型。
3. 测量微调后的强模型准确率。

基线包括：弱模型自身的准确率，以及强模型在金标准标签上监督训练时的性能上限。

差距指标是 Performance Gap Recovered（PGR）：
(fine-tuned - weak) / (ceiling - weak)。当 PGR = 1.0 时，表示弱监督完全弥合了差距；当 PGR = 0 时，表示弱监督没有提供帮助。

### Burns 等人的经验发现

在 NLP、棋类谜题和 reward modeling 任务上，PGR 始终为正：强模型能够泛化到弱监督者出错之外的部分。不同任务的 PGR 大约在 20% 到 80% 之间波动。强模型会从不完美标签中“理解”任务本意，并用自己预训练得到的先验进行补全。

Burns 等人也指出了几个局限：
- 这里的 weak-to-strong gap 是能力差距，不是对齐差距。弱模型本身是已对齐的；而真正的超人模型，也许是由人类来对齐，但能力差距会更大。
- 强模型的泛化，可能更多来自任务先验，而不是真正恢复了监督者的真实意图。
- 真正困难的问题，是如何把强模型的潜在知识有效激发出来；PGR 只是对此的一种特定操作化测量。

### 可扩展监督：三种机制

- **Debate（Irving 等人，2018）。** 两个 U 实例分别站在对立面进行论证，由较弱的裁判做判断。核心假设是：找到一个有说服力的真实答案，比编造一个同样有说服力的假答案更容易。2024 到 2025 年的最新工作（如 Khan 等人、Michael 等人）表明，debate 有时有用，有时无效，是否有效取决于任务结构。
- **Recursive Reward Modeling（Leike 等人，2018）。** U 帮助人类训练用于 U+1 的奖励模型。监督者的有效能力会随模型一起增长。
- **Task Decomposition（Christiano、Shlegeris、Amodei，2018）。** 把一个难任务递归拆成若干个人类可以验证的子任务。它依赖任务本身可被拆解。

每一种机制，都对任务结构或中间组件的对齐状态作了某种假设。

### 为什么可扩展监督与 W2SG 是互补的

可扩展监督提升的是监督者信号的有效质量。
W2SG 则负责弥合监督者所能提供的那个不完美信号与强模型能力之间的差距。

Lang 等人的 Debate Helps Weak-to-Strong Generalization（arXiv:2501.13124）把两者结合起来：先用 debate 协议生成更好的弱标签，再让强模型在这些标签上训练。论文报告称，这能在 NLP 任务上提升 PGR。

### 组织层面的波动

2024 年 5 月，在 Jan Leike 离开 OpenAI 转投 Anthropic 后，OpenAI 的 Superalignment 团队解散了。但这条研究议程，也就是可扩展监督、W2SG 与自动化对齐研究，并没有消失，而是在 Anthropic 和各类学术实验室继续推进，例如 MATS（第 28 课）、Redwood（第 10 课）、Apollo（第 8 课）、METR（第 28 课）。组织结构变了，研究问题没有变。

### 它在第 18 阶段中的位置

第 6 到第 10 课描述了威胁，以及在 U 不可信这一前提下的防御式范式。第 11 课转向进攻式范式：设法让监督者强到足以验证 U 的对齐状态。第 12 到第 16 课随后进入对抗评估的实际工具链。

```figure
scalable-oversight
```

## 用它

`code/main.py` 会在一个合成任务上模拟 W2SG 微调。弱标签器的准确率是 70%，并且带有结构性错误；强模型在金标准标签上的性能上限是 95%。你会在弱标签上微调强模型，测量 PGR，并把它和“强模型用金标签训练”以及“单独使用弱模型”的结果进行比较。

## 交付成果

本课会产出 `outputs/skill-w2sg-pgr.md`。给定一段监督设置描述，它会识别弱监督者、强模型以及监督质量，并计算或要求提供 PGR。它还会标记这项主张到底是在说“弱者可以监督强者”，还是在说“弱者加上监督机制可以监督强者”。

## 练习

1. 运行 `code/main.py`。分别报告 weak_accuracy = 0.60、0.70、0.80 时的 PGR，并解释 PGR 曲线为什么呈现这种形状。

2. 修改弱标签器，让它带有结构化错误，例如总是在某类特定输入上出错。此时 PGR 会升高、降低，还是保持不变？解释原因。

3. 阅读 Burns 等人 2023 年论文第 4.3 节（NLP tasks）。复现 “confidence auxiliary loss” 的直觉：当强模型比弱标签更自信时，最后是谁占上风？

4. 为一个软件工程任务设计一个结合 debate 与 task decomposition 的可扩展监督协议。分别指出每个组件的一个失效模式，并解释这种组合能否解决这些问题，或者为什么仍然解决不了。

5. 说明什么样的证据会推翻 “weak-to-strong generalization 是通向 superalignment 的可行路径” 这一主张。请给出具体的经验特征，而不是笼统表态。

## 关键术语

| 术语 | 人们常说什么 | 它真正指的是什么 |
|------|-----------------|------------------------|
| Scalable oversight | “让监督者更强” | 能提高监督者评估更强模型能力的各种机制 |
| W2SG | “弱者监督强者” | 在弱标签上微调强模型，并测量其恢复了多少能力 |
| PGR | “performance gap recovered” | (fine-tuned - weak) / (ceiling - weak)；1.0 表示完全弥合，0 表示没有帮助 |
| Debate | “两个 U 实例辩论” | 一种可扩展监督机制，由弱裁判在两个 U 辩手间做选择 |
| RRM | “recursive reward modeling” | 用 U 帮助训练 U+1 的奖励模型；监督能力随着 U 增长 |
| Task decomposition | “把任务拆成人能检查的子任务” | 递归把难任务分解成可验证的子任务 |
| Superalignment | “对齐超人 AI” | 研究如何对齐那些人类已无法直接评估的模型 |

## 进一步阅读

- [Burns et al. — Weak-to-Strong Generalization (OpenAI 2023)](https://openai.com/index/weak-to-strong-generalization/) — W2SG 的代表性论文
- [Irving, Christiano, Amodei — AI safety via debate (arXiv:1805.00899)](https://arxiv.org/abs/1805.00899) — 辩论机制的原始论文
- [Leike et al. — Scalable agent alignment via reward modeling (arXiv:1811.07871)](https://arxiv.org/abs/1811.07871) — 递归奖励建模
- [Khan et al. — Debating with More Persuasive LLMs Leads to More Truthful Answers (arXiv:2402.06782)](https://arxiv.org/abs/2402.06782) — 2024 年关于更强辩手参与时辩论效果的实证研究
- [Lang et al. — Debate Helps Weak-to-Strong Generalization (arXiv:2501.13124)](https://arxiv.org/abs/2501.13124) — 2025 年将辩论与 W2SG 结合的研究
