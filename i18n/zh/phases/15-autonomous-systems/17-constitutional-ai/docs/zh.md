# Constitutional AI 与规则覆盖

> Anthropic 于 2026 年 1 月 22 日公开的 Claude Constitution 长达 79 页，并以 CC0 发布。它把对齐方法从基于规则（rule-based）推向基于推理（reason-based），并明确给出四层优先级：（1）安全与支持人类监督；（2）伦理；（3）Anthropic 指南；（4）有帮助性。行为约束被分成两类：一类是硬编码禁令（hardcoded prohibitions），比如提升生物武器能力和 CSAM，运营方和用户都无法覆盖；另一类是软编码默认值（soft-coded defaults），运营方可以在规定边界内调节。其 2022 年原始路线（Bai 等）是通过自我批评和基于宪章的 RLAIF 来训练无害性。需要坦白的是：基于推理的对齐能否成立，取决于模型能否把原则泛化到未曾预料的情境中。Anthropic 自己 2023 年的参与式实验显示，公众原则与公司原则之间大约有 50% 的分歧，而 2026 版宪章（Constitution）并没有吸收这部分结果。

**Type:** 学习
**Languages:** Python（stdlib，四层优先级解析器）
**Prerequisites:** 阶段 15 · 06（自动化对齐研究），阶段 15 · 10（权限模式）
**Time:** 约 60 分钟

## 问题

部署到真实世界的 agent，总会遇到设计者从未见过的输入。规则清单永远不可能长到覆盖所有情况，也不可能短到在算力紧张时还足够快地判断完毕。真正的工程问题是：怎样把 agent 对齐到一套既能穿越长尾场景、又能支持快速推理的原则上？

基于规则的对齐（rule-based alignment，RBA）的思路是：把所有不允许做的事尽量列举出来。它检查快、容易审计，但几乎不可能始终保持最新，而且往往会对一些它没明确列到、但形式相近的情况过度拒绝。基于推理的对齐（reason-based alignment），也就是 2026 版 Claude Constitution 的路线，则是把原则编码进去，让模型自己做推理。它更容易扩展到未见场景，但更难审计，失败时出错的地方也从“漏了哪条规则”变成“把原则理解错了”。

2026 版 Constitution 选择的是一种明确的中间立场。那些错误性不依赖上下文的事，例如 bioweapons uplift、CSAM，使用 hardcoded prohibitions，也就是典型的 RBA：无论 operator 还是 user 如何要求，统统不允许。除此之外的部分，则按四层优先级做 reason-based 判断：安全与支持人类监督优先，其次是伦理，再其次是 Anthropic 明示的 guidelines，最后才是 helpfulness。operator 可以在 soft-coded 区域内调整默认行为，但碰不到 hardcoded prohibitions。

## 概念

### 四层优先级层级

1. **安全与支持人类监督。** 这是最高优先级。模型要优先避免削弱人类和 Anthropic 监督、纠正 AI 的能力。它不是宽泛的“尽量谨慎”，而是更具体的“不要以会让人类监督更困难的方式行动”。
2. **伦理。** 包括诚实、避免伤害个体、不欺骗、不操纵。当它与 Anthropic 自己的指南冲突时，伦理优先。
3. **Anthropic 指南。** 这是 Anthropic 在操作层面认定重要的规范，例如产品边界、交互方式、该用什么工具以及何时使用。
4. **有帮助性。** 最低优先级。在不违反更高层原则的前提下，尽可能有帮助。

当不同层发生冲突时，高层覆盖低层。它的结构和 Unix priority 或网络 QoS 很像，目标不是让每个单一维度都做到最好，而是让冲突时的分辨规则足够稳定、可预测。

### 硬编码禁令与软编码默认值

**Hardcoded：**
- 生物武器 / CBRN 能力提升
- CSAM
- 针对关键基础设施的攻击
- 在用户直接询问时，欺骗用户关于模型自身身份

这些都不能被运营方（operator）覆盖，也不能被用户（user）覆盖。能在模型权重层执行的，会通过 RLHF / Constitutional AI 训练固化进去；不能只靠权重完成的，则要在推理层（inference layer）继续拦住。

**软编码默认值（operator 可调）：**
- 响应长度默认值
- 主题范围，也就是模型是否要拒绝超出部署范围的话题
- 风格，例如 formal 还是 casual
- 工具使用模式

这些运营方调整都必须发生在明确声明过的边界内部。运营方不能靠“换个名字”来实质性移除 hardcoded prohibitions。

### 2022 年的 CAI 训练路线

最初版 Constitutional AI（Bai et al., 2022）训练 harmlessness 的方法大致是：

1. 先让模型对一组 prompts 生成回答。
2. 再让模型依据一份 constitution，也就是一组明确写出的原则，对每个回答做 critique。
3. 根据 critique 修订回答。
4. 最后在这些修订前后的样本对上做 RLAIF（reinforcement learning from AI feedback）。

得到的结果，是模型会用“基于原则的解释”来拒绝有害请求，而不是简单粗暴的一刀切拒绝。2026 版 Constitution 则是在这条训练脉络的后继基础上，再加上针对显式四层优先级的额外后训练。

### 基于推理的对齐能抓住什么，又会漏掉什么

**能抓住的：**
- 一些由允许的基本元素组合出来、但原则明显仍然适用的新情况。
- 与禁止请求高度类似、但措辞更新颖的请求。
- 依赖“你又没明确说 X 不允许”的社会工程攻击。

**容易漏掉的：**
- 利用原则歧义的攻击，例如“用户既然要求了，那 helpfulness 不就应该答应吗”。
- 两条原则在某个意外情境中发生冲突，而层级解释本身又不够明确。
- 训练迭代中对同一条原则产生缓慢的解释漂移，也就是原则漂移（principle drift）。

### 2023 年的参与式实验

Anthropic 在 2023 年做过一次实验，把公司自己撰写的 constitution 和通过公众输入整理出的 constitution 做比较，后者大约来自 1,000 名美国受访者。两份版本的原则只在约 50% 的内容上达成一致。它们分歧的地方，有些问题上公众版更严格，例如政治内容处理；另一些问题上则更宽松，例如是否应主动披露 AI 身份。2026 版 Constitution 没有吸收这部分公众来源的发现，这也是该路线一个被明文记录的张力点。

### 为什么硬编码禁令仍然必需

单靠 reason-based alignment，没有办法真正封住长尾。一个攻击者如果能让模型接受某个前提，比如“我们是一家持牌的生物武器研究实验室”，往往就能绕过那些依赖具体情境推理的原则。hardcoded prohibitions 不会因为前提包装而弯曲。它们对应的是第 14 课里所说的“hard constitutional limit”，只是这次它位于对齐层（alignment layer）。

### Constitution 在整套栈里的位置

Constitution 不是第 14 课的 kill switch。它位于模型层（model layer），也就是模型权重被训练成更偏好什么。kill switch 和 canary tokens 则位于运行时层（runtime layer），决定运行时实际允许什么。两层都需要，而且解决的不是同一类问题。如果模型权重过于宽松，运行时就必须拦下模型提出的错误动作；如果运行时限制过严，连模型提出的正确动作也会被拒绝。前者需要运行时补足模型层，后者则是运行时策略本身的问题。不同层解决不同类风险。

```figure
mx-priority-tiers
```

## 用起来

`code/main.py` 实现了一个最小化的四层优先级解析器（resolver）。它接收一个拟议动作，以及该动作在 safety、ethics、guidelines、helpfulness 四个维度上的评估结果，然后返回允许执行、拒绝执行，或返回一个修改后的动作。驱动程序包含几个小案例：明确允许、明确拒绝、硬编码禁令（hardcoded prohibition），以及跨层冲突的模糊案例。

## 交付物

`outputs/skill-constitution-review.md` 用来审计某个部署的宪法层（constitutional layer）：哪些部分是硬编码的（hardcoded），哪些是软编码的（soft-coded），运营方（operator）可以调哪些默认值，以及真实的冲突解决顺序是否真的遵守这四层层级。

## 练习

1. 运行 `code/main.py`。确认即便 helpfulness 很高，hardcoded prohibition 仍然会触发。然后把 resolver 改成让 helpfulness 高于 ethics，观察失败模式。

2. 阅读公开发布的 Claude Constitution（79 页，CC0）。找出一条你认为表述不够具体的原则，写两段话说明它究竟模糊在哪，以及你会怎样把它改得更精确。

3. 为一个 customer-support agent 设计一组软编码默认值（soft-coded defaults）。哪些是运营方（operator）可以调的？哪些绝不能碰？分别说明边界理由。

4. 阅读 Bai et al. 2022 的 CAI 论文。描述一种场景：在这种场景里，Constitutional AI 的 critique-and-revise 循环会比一条简单明确的硬规则产生更糟的结果。指出它属于哪一类问题。

5. Anthropic 2023 年的参与式实验显示，公众原则与公司原则之间大约有 50% 分歧。选择一个对生产部署真正重要的类别，例如政治中立，设计一种机制，让运营方（operator）可以表达自己的价值取向，同时硬编码禁令（hardcoded prohibitions）仍然完全不可动。

## 关键术语

| 术语 | 人们通常怎么说 | 实际含义 |
|---|---|---|
| Constitutional AI | “Anthropic 的宪章式对齐方法” | 依据成文宪章进行自我批评，再基于此开展 RLAIF |
| Reason-based alignment | “按原则推理” | 让模型围绕原则推理，从而处理未见场景 |
| Hardcoded prohibition | “绝不做 X” | 一条任何运营方或用户都无法覆盖的规则型禁令 |
| Soft-coded default | “可调默认值” | 落在声明边界内、由运营方调整的行为默认值 |
| 四层优先级 | “优先级顺序” | 安全与监督 > 伦理 > 指南 > 有帮助性 |
| RLAIF | “AI 反馈强化学习” | 奖励信号来自模型生成的批评意见 |
| Participatory constitution | “公众参与制定的宪章” | Anthropic 2023 年实验中的公众来源原则；与公司版约有 50% 分歧 |
| Principle drift | “原则理解漂移” | 模型对固定原则文本的理解随训练周期缓慢漂移 |

## 延伸阅读

- [Anthropic — Claude's Constitution (January 2026)](https://www.anthropic.com/news/claudes-constitution) — 公开的 79 页 CC0 文档。
- [Bai et al. — Constitutional AI: Harmlessness from AI Feedback](https://www.anthropic.com/research/constitutional-ai-harmlessness-from-ai-feedback) — 2022 年的原始论文。
- [Anthropic — Collective Constitutional AI (2023)](https://www.anthropic.com/research/collective-constitutional-ai-aligning-a-language-model-with-public-input) — 参与式 constitutional AI 实验。
- [Anthropic — Responsible Scaling Policy v3.0](https://anthropic.com/responsible-scaling-policy/rsp-v3-0) — 了解 Constitution 在 RSP 栈中的位置。
- [Anthropic — Measuring agent autonomy in practice](https://www.anthropic.com/research/measuring-agent-autonomy) — 看 Constitution 在长时程 agent 部署中的角色。
