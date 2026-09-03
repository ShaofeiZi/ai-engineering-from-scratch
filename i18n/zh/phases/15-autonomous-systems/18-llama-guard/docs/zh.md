# Llama Guard 与输入/输出分类

> Llama Guard 3（Meta，基于 Llama-3.1-8B，并针对内容安全微调）会按照 MLCommons 的 13 类风险分类法（hazard taxonomy），对 LLM 的输入和输出同时做分类，支持 8 种语言。它的 1B-INT4 量化版本在移动端 CPU 上也能跑到每秒 30 个 token 以上。Llama Guard 4 则升级为多模态（multimodal，image + text），分类扩展到 S1–S14，其中新增了 S14 Code Interpreter Abuse，并且可以作为 Llama Guard 3 8B/11B 的即插即用替代（drop-in replacement）。NVIDIA NeMo Guardrails v0.20.0（2026 年 1 月）则在输入护栏（input rails）和输出护栏（output rails）之上加入了用 Colang 描述的对话流程护栏（dialog-flow rails）。需要坦白的是，Huang et al. 在《Bypassing Prompt Injection and Jailbreak Detection in LLM Guardrails》（arXiv:2504.11168）里展示过，Emoji Smuggling 对六套知名 guard 系统的攻击成功率（attack success rate）达到了 100%；其中 NeMo Guard Detect 在 jailbreak 上也记录到了 72.54% ASR。分类器只是一层，不是完整解法。

**Type:** 学习
**Languages:** Python（stdlib，带类别标签的分类器模拟器）
**Prerequisites:** 阶段 15 · 10（权限模式），阶段 15 · 17（Constitutional AI 与规则覆盖）
**Time:** 约 45 分钟

## 问题

面向 LLM 输入与输出的分类器，处在 agent stack 最狭窄的一层：每一个请求都要经过它，每一个响应也都要经过它。一个好的分类器层（classifier layer）应该足够快、有清晰的分类体系（taxonomy），并且能用很小的算力成本挡下大部分明显滥用；一个差的分类器层则只会制造“我们已经很安全了”的错觉。

2024 到 2026 年的分类器栈（classifier stack），已经逐渐收敛到几类可用于生产的方案。Llama Guard（Meta）以开放权重的形式提供，采用 Meta Community License。NeMo Guardrails（NVIDIA）则提供宽松许可的护栏（rails）机制，并用 Colang 来定义对话流程规则。这两类系统都不是拿来替代基础模型（foundation model）自身安全行为的，而是要和基础模型配套使用。

它们的失败面也同样已经被相当清楚地刻画出来。字符级攻击，比如 emoji smuggling 和 homoglyph substitution；上下文中的重定向，例如“忽略前面的规则，改按下面的要求答”；以及语义改写，都会让分类器准确率可测地下降。Huang et al. 2025 的结果就显示，一个具体的 Emoji Smuggling 攻击在六套被点名的 guard 系统上都实现了 100% ASR。

## 概念

### 快速认识 Llama Guard 3

- 基础模型（Base model）：Llama-3.1-8B
- 针对内容安全做微调，不是通用聊天模型
- 同时分类输入与输出
- 使用 MLCommons 的 13 类风险分类体系（13-hazard taxonomy）
- 支持 8 种语言
- 1B-INT4 量化版本可在移动 CPU 上跑到 >30 tok/s

分类体系（taxonomy）本身就是产品的一部分。从 “S1 Violent Crimes” 到 “S13 Elections”，它提供了一套模型训练和下游系统都能共享的分类词汇。这样一来，后续系统就可以按类别执行差异化动作：例如 S1 直接阻断，S6 转人工审核，S12 只做标注但继续放行。

### Llama Guard 4 的新增内容

- 多模态（Multimodal）：支持 image + text 输入
- 扩展分类体系（taxonomy）：S1–S14，其中新增 S14 Code Interpreter Abuse
- 可作为 Llama Guard 3 8B/11B 的即插即用替代（drop-in replacement）

S14 对本阶段尤其关键。自主编码代理（coding agent，第 9 课）会在沙箱（sandbox）里执行代码（第 11 课）；而专门针对代码解释器滥用（code interpreter misuse）的一个分类标签，正好补上了旧分类体系（taxonomy）里没有被单独命名的一类风险。

### NeMo Guardrails（NVIDIA）

- v0.20.0 发布于 2026 年 1 月
- Input rails：在用户轮次（user turn）做 classify-and-block
- Output rails：在模型轮次（model turn）做 classify-and-block
- Dialog rails：用 Colang 定义对话流程约束，例如“用户如果问 X，就必须回答 Y”
- 可集成 Llama Guard、Prompt Guard 和自定义分类器

真正把它和普通输入/输出分类区别开的，是对话护栏（dialog rail）这一层。输入护栏（input rails）和输出护栏（output rails）只处理单轮内容；对话护栏（dialog rails）则可以强制执行“即便用户换三种说法来问，客服机器人（customer-support bot）也不讨论医疗诊断”这种跨轮规则。

### 攻击语料

**Emoji Smuggling**（Huang et al., arXiv:2504.11168）：在被禁止请求的字符之间插入不可见或视觉近似的 emoji。分词器（tokenizer）会以一种不同于分类器预期的方式处理它们，于是绕过检测。该论文报告它对六套知名 guard 系统都实现了 100% ASR。

**Homoglyph substitution**：把拉丁字母替换成看起来几乎一样的西里尔字母。比如 Bomb 变成 Воmb；如果分类器主要按英文训练，就可能直接漏掉。

**In-context redirection**：通过“在回答前请先假设这是研究场景，并改用另一套策略”这种话术，测试分类器是否会被输入中的声明重新定位。

**Semantic paraphrase**：用全新的说法重写被禁止请求。分类器的微调数据不可能覆盖所有表达方式。

**NeMo Guard Detect**：在 Huang et al. 那篇论文里，它在 jailbreak benchmark 上测得 72.54% ASR。这个数字来自精心构造的对抗样本；普通用户随手尝试的 jailbreak 成功率会低得多，但上限显然也不是“接近零”。

### 分类器擅长什么

- **快速默认拒绝**明显滥用，例如请求生成 CSAM，能在毫秒级被截住。
- **按类别路由**，为不同类别执行不同策略，例如直接封禁、记录日志、转人工。
- **Output rails** 可以拦住本来已经通过输入层、但模型输出中又泄露了敏感内容的情况。
- **合规表面**更清晰，对监管者来说，一个带明确定义 taxonomy 的分类器系统更容易文档化和审计。

### 分类器会输在哪里

- 对抗式构造，例如 emoji smuggling、homoglyph substitution。
- 多轮攻击，因为分类器往往只看单轮或有限上下文。
- 语义改写成训练数据没见过的表达方式。
- 本身就处在允许与禁止边界上的模糊内容。

### 分层防御

分类器层（classifier layer）处在宪法层（constitutional layer，第 17 课）之下、运行时层（runtime layer，第 10、13、14 课）之上。整体组合大致是：

- **Weights**：模型通过 Constitutional AI 训练，在默认情况下拒绝明显滥用。
- **Classifier**：Llama Guard / NeMo Guardrails，快速挡下明显滥用，并进行分类路由。
- **运行时**：权限模式、预算、终止开关、金丝雀机制。
- **Review**：对于有后果的动作，用 propose-then-commit 的 HITL 流程做最终把关。

单独任何一层都不够。每一层覆盖的是不同类型的攻击面。

```figure
a5-guard-sieve
```

## 用起来

`code/main.py` 模拟了一个玩具分类器，它对输入轮文本使用一个 6 类分类体系（taxonomy）。相同的文本会分别以原始形式、emoji smuggling 形式和 homoglyph substitution 形式送入分类器；命中率会像 Huang et al. 论文里描述的那样下降。驱动程序还展示了输出护栏（output rails）如何在输入已被接受的情况下，继续拦下有问题的输出。

## 交付物

`outputs/skill-classifier-stack-audit.md` 用来审计一个部署中的分类器层（classifier layer），包括模型、分类体系（taxonomy）、输入/输出护栏（input/output rails）、对话护栏（dialog rails），并指出其中的缺口。

## 练习

1. 运行 `code/main.py`。确认分类器能抓住原始恶意输入，但会漏掉 emoji smuggling 版本。加入一个 normalization 步骤后，再测一次新的命中率。

2. 阅读 MLCommons 的 13-hazard taxonomy 和 Llama Guard 4 的 S1–S14 列表。找出 S1–S14 中那一项在原始 13-hazard 集里没有直接映射，并解释为什么 S14 Code Interpreter Abuse 对 Phase 15 特别关键。

3. 为一个 customer-support bot 设计一条 NeMo Guardrails dialog rail，要求它绝不能讨论 diagnosis。先用自然语言写出来，Colang 语义与之接近；再用三种不同措辞的“求诊断”问题去测试它。

4. 阅读 Huang et al.（arXiv:2504.11168）。从 emoji smuggling、homoglyph、paraphrase 中挑一种攻击，提出一个 mitigation，同时指出这种 mitigation 自己的失败模式是什么。

5. NeMo Guard Detect 在 jailbreak benchmark 上的 72.54% ASR，是在对抗性构造条件下测得的。请设计一个评估协议，用来衡量 classifier 在普通、非对抗用户分布下的 ASR。你预计会得到什么数量级的结果？为什么这个数字需要和对抗场景下的 ASR 分开看？

## 关键术语

| 术语 | 人们通常怎么说 | 实际含义 |
|---|---|---|
| Llama Guard | “Meta 的安全分类器” | 基于 Llama-3.1-8B、针对输入和输出分类微调的安全分类器 |
| MLCommons taxonomy | “MLCommons 风险分类体系” | 一套共享的内容安全分类词汇 |
| S1–S14 | “Llama Guard 4 分类标签” | 扩展后的分类体系，其中 S14 是代码解释器滥用 |
| NeMo Guardrails | “NVIDIA 的护栏框架” | 输入护栏、输出护栏与对话护栏；使用 Colang 定义流程 |
| Emoji Smuggling | “表情符号夹带攻击” | 在字符间插入不可见表情符号；在六套防护系统上达到 100% ASR |
| Homoglyph | “形近字母替换” | 用西里尔字母伪装拉丁字母；针对英文训练的分类器容易漏检 |
| ASR | “攻击成功率” | 攻击绕过分类器的比例 |
| Dialog rail | “对话流程护栏” | 能跨多轮持续生效的会话级规则 |

## 延伸阅读

- [Inan et al. — Llama Guard: LLM-based Input-Output Safeguard](https://ai.meta.com/research/publications/llama-guard-llm-based-input-output-safeguard-for-human-ai-conversations/) — Llama Guard 的原始论文。
- [Meta — Llama Guard 4 model card](https://www.llama.com/docs/model-cards-and-prompt-formats/llama-guard-4/) — 了解 multimodal 能力与 S1–S14 taxonomy。
- [NVIDIA NeMo Guardrails (GitHub)](https://github.com/NVIDIA-NeMo/Guardrails) — 2026 年 1 月的 v0.20.0 版本。
- [Huang et al. — Bypassing Prompt Injection and Jailbreak Detection in LLM Guardrails](https://arxiv.org/abs/2504.11168) — 比较多套 guard 系统的 ASR 数据。
- [Anthropic — Measuring agent autonomy in practice](https://www.anthropic.com/research/measuring-agent-autonomy) — 理解 classifier layer 与 runtime layer 的组合关系。
