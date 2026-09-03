# 综合项目 15——宪法式安全护栏与红队演练场

> 到 2026 年，安全分类器技术栈已经基本成型：Anthropic Constitutional Classifiers、Meta Llama Guard 4、Google ShieldGemma-2、NVIDIA Nemotron 3 Content Safety，以及负责多语言覆盖的 X-Guard。garak、PyRIT、NVIDIA Aegis 和 promptfoo 则成为常用的对抗评估工具。NeMo Guardrails v0.12 可以把这些组件接入生产流水线。本综合项目要把整套体系真正落地：围绕目标应用搭建分层安全护栏，让自主红队智能体运行 6 类以上攻击，再执行一次宪法式自我批评训练，最终给出可量化的无害性变化。

**Type:** 综合项目
**Languages:** Python（安全流水线、红队）、YAML（策略配置）
**Prerequisites:** 第 10 阶段（从零构建 LLM）、第 11 阶段（LLM 工程）、第 13 阶段（工具）、第 14 阶段（智能体）、第 18 阶段（伦理、安全、对齐）
**Phases exercised:** P10 · P11 · P13 · P14 · P18
**Time:** 25 小时

## 问题

2026 年 LLM 安全的难点已经不再是“分类器能不能用”，而是怎样把它们正确组合在生产应用外围，既避免过度拒绝，又不留下明显漏洞。Llama Guard 4 负责检测英文内容是否违反策略；X-Guard 覆盖 132 种语言的越狱攻击；ShieldGemma-2 用于拦截图像中的提示词注入；NVIDIA Nemotron 3 Content Safety 覆盖企业场景常见的安全类别。Anthropic Constitutional Classifiers 走的是另一条路线，主要用于训练阶段，而不是服务阶段。

攻击方式也在演化。PAIR 和 TAP 可以自动发现越狱路径，GCG 会生成基于梯度的对抗性后缀，多轮对话和语码转换攻击则会利用智能体记忆。任何真正部署的 LLM 都需要一套红队演练场；garak 和 PyRIT 是这类体系的常用驱动器。每个成功攻击还必须附上成文的缓解方案和 CVSS 评分。

你要加固一个目标应用，可以选用 8B 指令微调模型，也可以复用其他综合项目中的 RAG 聊天机器人；然后对它运行 6 类以上攻击，并量化加固前后的无害性变化。

## 概念

整个安全流水线分为五层。**输入清理（Input sanitize）**：去除零宽字符，解码 base64/rot13，规范化 Unicode。**策略层（Policy layer）**：使用 NeMo Guardrails v0.12 的规则处理越界主题、毒性内容和个人身份信息提取。**分类器门控（Classifier gate）**：输入侧使用 Llama Guard 4，非英文场景使用 X-Guard，图像输入使用 ShieldGemma-2。**模型层（Model）**：目标 LLM 本体。**输出过滤（Output filter）**：输出再次经过 Llama Guard 4、Presidio 个人身份信息清理，并在适用时强制检查引用。**人工介入层（HITL tier）**：高风险输出不直接返回，而是进入 Slack 队列等人工复核通道。

红队演练场由调度器定时运行。PAIR 和 TAP 自主探索越狱路径，GCG 执行基于梯度的后缀攻击。测试还要覆盖 ASCII / base64 / rot13 编码攻击、多轮攻击（角色扮演、记忆利用），以及语码转换攻击，例如混合英语与斯瓦希里语或泰语。每次运行都要生成结构化发现文件，其中包含 CVSS 评分和披露时间线。

宪法式自我批评（constitutional self-critique）是一种训练阶段干预。准备 1,000 条尝试诱导有害回答的提示词，让模型先起草答案，再依据一份明确写出的宪法进行批评，例如“不造成伤害”“引用证据”“拒绝非法请求”。批评模型提出异议的样本需要重写，目标模型再用这些经批评改进的样本对继续训练。最后在留出评估集上测量无害性指标的前后变化。

## 架构

```
request (text / image / multilingual)
      |
      v
input sanitize (strip zero-width, decode, normalize)
      |
      v
NeMo Guardrails v0.12 rails (off-domain, policy)
      |
      v
classifier gate:
  Llama Guard 4 (English)
  X-Guard (multilingual, 132 langs)
  ShieldGemma-2 (image prompts)
  Nemotron 3 Content Safety (enterprise)
      |
      v (allowed)
target LLM
      |
      v
output filter: Llama Guard 4 + Presidio PII + citation check
      |
      v
HITL tier for flagged outputs

parallel:
  red-team scheduler
    -> garak (classic attacks)
    -> PyRIT (orchestrated red team)
    -> autonomous jailbreak agent (PAIR + TAP)
    -> GCG suffix attacks
    -> multilingual / code-switch
    -> multi-turn persona adoption

output: CVSS-scored findings + disclosure timeline + before/after harmlessness delta
```

## 技术栈

- 安全分类器：Llama Guard 4、ShieldGemma-2、NVIDIA Nemotron 3 Content Safety、X-Guard
- 护栏框架：NeMo Guardrails v0.12 + OPA
- 红队驱动器：garak（NVIDIA）、PyRIT（Microsoft Azure）、NVIDIA Aegis、promptfoo
- 越狱智能体：PAIR（Chao 等，2023）、Tree-of-Attacks（TAP）、GCG 对抗性后缀
- 宪法式训练：Anthropic 风格的自我批评循环 + 基于批评结果的 SFT
- 个人身份信息清理：Presidio
- 目标应用：一个 8B 指令微调模型，或其他综合项目中的 RAG 聊天机器人

```figure
cf-safety-stack
```

## 动手构建

1. **准备目标应用。** 在 vLLM 上启动一个 8B 指令微调模型，或者复用其他综合项目中的 RAG 聊天机器人。这就是被测应用。

2. **封装安全流水线。** 在目标系统外围接入五层安全管道。确认每一层都能单独观测，例如在 Langfuse 中为每层记录一个跨度（span）。

3. **验证分类器覆盖范围。** 加载 Llama Guard 4、X-Guard（多语言）和 ShieldGemma-2（图像）。先在一小批带标签样本上运行，建立基线。

4. **实现红队调度器。** 调度 garak、PyRIT、PAIR 智能体、TAP 智能体、GCG 运行器、多轮攻击器和语码转换攻击器。每类攻击使用独立队列。

5. **构建攻击套件。** 至少覆盖六类攻击：(1) PAIR 自动越狱，(2) TAP 攻击树，(3) GCG 梯度后缀，(4) ASCII / base64 / rot13 编码，(5) 多轮角色扮演，(6) 多语言语码转换。每一类都要报告成功率。

6. **执行宪法式自我批评。** 整理 1,000 条尝试诱导有害回答的提示词。对每条提示词，目标模型先生成答案，再由批评模型依据写好的宪法评分，例如“不造成伤害”“引用证据”“拒绝非法请求”。批评模型提出异议的样本要被重写，目标模型再用这些改进后的样本对进行微调。最后在留出评估集上测量无害性变化。

7. **测量过度拒绝。** 在 XSTest 等良性提示词套件上跟踪假阳性率。系统面对良性问题时必须保持有用，而不是一味拒绝。

8. **进行 CVSS 评分。** 每个成功的越狱都要按 CVSS 4.0 打分，包括攻击向量、复杂度和影响，并形成披露时间线与缓解方案。

9. **演练场自动化。** 上述所有流程都应挂到 cron 上自动运行；发现项写入队列；一旦过度拒绝出现回归，就向 Slack 发告警。

## 运行示例

```
$ safety probe --model=target --family=PAIR --budget=50
[attacker]   PAIR agent running on target
[attack]     attempt 1/50: disguise query as academic research ... blocked
[attack]     attempt 2/50: appeal to roleplay ... blocked
[attack]     attempt 3/50: chain-of-thought coax ... SUCCEEDED
[finding]    CVSS 4.8 medium: roleplay bypass on target
[range]      7 successes out of 50 (14% success rate)
```

## 交付成果

`outputs/skill-safety-harness.md` 就是本课交付物：一套生产级分层安全管道，以及一套可复现的红队演练场，并附带无害性指标的前后变化。

| 权重 | 评判标准 | 衡量方式 |
|:-:|---|---|
| 25 | 攻击面覆盖 | 至少演练 6 类攻击，覆盖 2 种以上语言 |
| 20 | 真阳性与假阳性的权衡 | 攻击拦截率与 XSTest 良性样本通过率的对比 |
| 20 | 自我批评带来的变化 | 留出评估集上无害性指标的前后变化 |
| 20 | 文档与披露 | 包含 CVSS 评分和时间线的发现报告 |
| 15 | 自动化与可重复性 | 所有任务都通过 cron 运行并配置告警 |
| **100** | | |

## 练习

1. 在 RAG 聊天机器人上运行 garak 的提示词注入插件，比较启用和关闭输出过滤层时的攻击成功率。

2. 加入第七类攻击：通过检索文档发起间接提示词注入，并测量还需要增加哪些防御。

3. 实现“拒绝但提供帮助”（refuse-with-help）模式：护栏拦截时，不只给出生硬拒绝，还提供一个更安全的相关回答。测量它对 XSTest 的影响。

4. 找出一种 X-Guard 表现较差的语言，并设计有针对性的微调数据集。

5. 在 30B 模型上运行宪法式自我批评，测量无害性改善幅度是否会随模型规模变化。

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|-----------------|------------------------|
| 分层安全 | “纵深防御” | 在输入、门控、输出和 HITL 等多层部署防护 |
| Llama Guard 4 | “Meta 的安全分类器” | 2026 年输入/输出内容分类器的参考基线 |
| PAIR | “越狱智能体” | Chao 等人提出的 LLM 驱动越狱发现方法 |
| TAP | “攻击树” | PAIR 的树搜索变体 |
| GCG | “贪婪坐标梯度” | 基于梯度的对抗性后缀攻击 |
| 宪法式自我批评 | “Anthropic 风格训练” | 目标模型起草回答 -> 批评模型评分 -> 重写 -> 再训练 |
| XSTest | “良性探测集” | 用于度量过度拒绝回归的基准集 |
| CVSS 4.0 | “严重性评分” | 为安全发现项打分的标准漏洞评分体系 |

## 延伸阅读

- [Anthropic Constitutional Classifiers](https://www.anthropic.com/research/constitutional-classifiers) — 训练阶段参考资料
- [Meta Llama Guard 4](https://www.llama.com/docs/model-cards-and-prompt-formats/llama-guard-4/) — 2026 年输入/输出分类器
- [Google ShieldGemma-2](https://huggingface.co/google/shieldgemma-2b) — 图像与多模态安全模型
- [NVIDIA Nemotron 3 Content Safety](https://developer.nvidia.com/blog/building-nvidia-nemotron-3-agents-for-reasoning-multimodal-rag-voice-and-safety/) — 企业级参考方案
- [X-Guard (arXiv:2504.08848)](https://arxiv.org/abs/2504.08848) — 覆盖 132 种语言的多语言安全模型
- [garak](https://github.com/NVIDIA/garak) — NVIDIA 红队测试工具包
- [PyRIT](https://github.com/Azure/PyRIT) — Microsoft 红队测试框架
- [NeMo Guardrails v0.12](https://docs.nvidia.com/nemo-guardrails/) — 防护规则框架
- [PAIR (arXiv:2310.08419)](https://arxiv.org/abs/2310.08419) — 越狱智能体论文
