# ASCII 艺术与视觉越狱

> Jiang, Xu, Niu, Xiang, Ramasubramanian, Li, Poovendran, "ArtPrompt: ASCII Art-based Jailbreak Attacks against Aligned LLMs" (ACL 2024, arXiv:2402.11753). 将有害请求里与安全相关的 token 遮蔽掉，再用同一组字母的 ASCII 艺术图案替换这些 token，然后把伪装后的 prompt 发出去。GPT-3.5、GPT-4、Gemini、Claude、Llama-2 都无法稳定识别 ASCII 艺术 token。该攻击可以绕过 PPL（困惑度过滤）、Paraphrase defenses 和 Retokenization。相关工作还包括：ViTC 基准用来衡量模型识别非语义视觉提示的能力；StructuralSleight 则把 Uncommon Text-Encoded Structures（树、图、嵌套 JSON）推广为一整类编码攻击。

**Type:** 构建
**Languages:** Python (stdlib, ArtPrompt token-masking harness)
**Prerequisites:** 阶段 18 · 12（PAIR 自动化攻击）、阶段 18 · 13（多样本越狱）
**Time:** 约 60 分钟

## 学习目标

- 描述 ArtPrompt 攻击的流程：识别目标词、替换成 ASCII-art、生成最终伪装 prompt。
- 解释为什么常见防御（PPL、Paraphrase、Retokenization）对 ArtPrompt 无效。
- 定义 ViTC，并说明它测量的能力是什么。
- 说明 StructuralSleight 如何把攻击推广到任意 Uncommon Text-Encoded Structures。

## 问题

第 12 课中的改写与角色扮演攻击，以及第 13 课中的长上下文攻击，都依赖文本层面的模式。ArtPrompt 攻击的则是识别层：模型并不是把被禁止的 token 当作普通文本来解析，而是在解析一个由字符拼成的图像。安全过滤器看到的是无害的标点和空格；模型看到的却是一个词。

## 概念

### ArtPrompt，分两步

步骤 1：识别目标词。给定一个有害请求，攻击者先用 LLM 找出其中与安全相关的词，例如在 "how to make a bomb" 里识别出 "bomb"。

步骤 2：生成伪装 prompt。把每个识别出的词替换成对应的 ASCII 艺术字形，也就是用一个 7x5 或 7x7 的字符块勾出字母轮廓。模型接收到的是一块由标点和空格组成的网格；只要模型有足够强的识别能力，就能把它读成原词，而安全过滤器看到的只是网格。

结果是：GPT-4、Gemini、Claude、Llama-2、GPT-3.5 都会失守。在论文使用的基准子集上，攻击成功率超过 75%。

### 为什么常见防御会失效

- **PPL（困惑度过滤）**。ASCII 艺术的困惑度确实很高，但所有新颖输入的困惑度都可能很高。只要阈值设置到足以挡住 ArtPrompt，也会一起挡住合法的结构化输入。
- **Paraphrase**。一旦对 prompt 做改写，ASCII 艺术理论上会被破坏。但在实践里，负责改写的 LLM 往往会把这段图案保留下来，甚至重建出来。
- **Retokenization**。换一种 tokenizer 来切分 token，并不会改变模型实际上是在“看形状”这件事。它识别的是字母轮廓，不是 token 切分方式。

根本问题在于，安全过滤通常停留在 token 或语义层；ArtPrompt 攻击的却是视觉识别层。

### ViTC 基准

ViTC 测的是模型识别非语义视觉提示的能力，例如读取 ASCII-art、wingdings 以及其他不以普通文本语义表达的信息。ArtPrompt 的攻击效果与 ViTC 准确率高度相关：模型越擅长读这种“视觉文本”，ArtPrompt 在它身上就越有效。这本质上是能力与安全之间的张力。

### StructuralSleight

StructuralSleight 可以看作是 ArtPrompt 的一般化：把攻击扩展到 Uncommon Text-Encoded Structures（UTES）。树结构、图结构、嵌套 JSON、CSV-in-JSON、diff 风格代码块都属于这一类。只要某种结构在安全训练数据里足够少见、但模型又能成功解析，它就可能被用来隐藏有害内容。

这对防御的含义是：安全机制必须覆盖模型能够解析的各种结构化表示，而这个集合既很大，还在持续增长。

### 图像模态的对应物

视觉 LLM（GPT-5.2、Gemini 3 Pro、Claude Opus 4.5、Grok 4.1）让这类攻击面进一步扩大。使用真实图片实现的 ArtPrompt 式攻击，通常比 ASCII 艺术这种文本模拟版本更强，因为图像编码器能提供更丰富的信号。

### 它在 Phase 18 中的位置

第 12 到 14 课覆盖了三条彼此正交的攻击向量：迭代优化（PAIR）、上下文长度利用（MSJ）以及编码攻击（ArtPrompt / StructuralSleight）。第 15 课会把视角从模型内部攻击切换到系统边界攻击（indirect prompt injection）。第 16 课则介绍相应的防御工具链。

```figure
al-ascii-cloak
```

## 用它

`code/main.py` 构建了一个 toy ArtPrompt。你可以把有害查询中的特定词替换成 ASCII 艺术字形，验证伪装后的字符串能够绕过关键词过滤器，并且在可选情况下，用一个简单识别器把它重新解码回来。

## 交付成果

这一课产出 `outputs/skill-encoding-audit.md`。给定一份 jailbreak-defense 报告，它会枚举其中覆盖到的编码攻击家族（ASCII art、base64、leet-speak、UTF-8 homoglyph、UTES），以及每一类攻击分别由哪一层防御拦截。

## 练习

1. 运行 `code/main.py`。验证伪装后的字符串能够通过一个简单的关键词过滤器，并报告所需的字符级改动。

2. 为同一个目标词实现第二种编码方式：base64。比较它相对于 ArtPrompt 的绕过率和恢复难度。

3. 阅读 Jiang et al. 2024 的 Section 4.3（五个模型的结果）。提出一个理由，解释为什么 Claude 在同一基准上的 ArtPrompt 抵抗力高于 Gemini。

4. 设计一个生成前防御，用来检测 prompt 中形似 ASCII 艺术的区域。测量它在合法代码、表格和数学符号上的误报率。

5. StructuralSleight 列出了 10 种编码结构。请草拟一个能够覆盖这 10 种结构的通用防御，并估算每个受保护 prompt 的计算成本。

## 关键词

| 术语 | 常见说法 | 实际含义 |
|------|-----------------|------------------------|
| ArtPrompt | "the ASCII-art attack" | 通过 ASCII 艺术渲染来遮蔽安全词的两步式越狱攻击 |
| Cloaking | "hide the word" | 把禁用 token 替换成模型能读懂、过滤器却识别不到的视觉表示 |
| UTES | "uncommon structure" | Uncommon Text-Encoded Structure，例如树、图、嵌套 JSON，可用于夹带内容 |
| ViTC | "visual-text capability" | 衡量模型读取非语义视觉编码能力的基准 |
| Perplexity filter | "PPL defense" | 通过高困惑度拒绝 prompt 的防御；会误伤合法结构化输入 |
| Retokenization | "tokenizer 位移防御" | 先用另一种 tokenizer 重新处理 prompt；之所以失效，是因为攻击依赖视觉识别 |
| Homoglyph | "lookalike characters" | 视觉上与拉丁字母相同的 Unicode 字符，可绕过子串检查 |

## 进一步阅读

- [Jiang et al. — ArtPrompt (ACL 2024, arXiv:2402.11753)](https://arxiv.org/abs/2402.11753) — ASCII 艺术越狱论文
- [Li et al. — StructuralSleight (arXiv:2406.08754)](https://arxiv.org/abs/2406.08754) — 对 UTES 攻击的推广
- [Chao et al. — PAIR (Lesson 12, arXiv:2310.08419)](https://arxiv.org/abs/2310.08419) — 可与之互补的迭代攻击
- [Anil et al. — Many-shot Jailbreaking (Lesson 13)](https://www.anthropic.com/research/many-shot-jailbreaking) — 可与之互补的长度攻击
