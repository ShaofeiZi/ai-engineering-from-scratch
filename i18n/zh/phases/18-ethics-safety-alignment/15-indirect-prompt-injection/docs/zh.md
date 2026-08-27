# 间接提示注入：生产环境中的攻击面

> 间接提示注入（IPI）会把指令藏进外部内容里，例如网页、电子邮件、共享文档或工单；而 agentic system 会在没有用户明确触发的情况下消费这些内容。IPI 是 2026 年生产环境中的主导威胁：它绕过用户输入过滤，因为攻击者根本不需要接触用户；随着代理处理越来越多外部内容，它还能悄无声息地扩张；它攻击的还是那些根本没人逐条审读 prompt 的自动化工作流。MDPI Information 17(1):54（2026 年 1 月）汇总了 2023 到 2025 年的研究。NDSS 2026 的 IPI defense 论文则指出了核心难点：注入指令在语义上可能完全无害，例如 “please print Yes”，因此检测绝不只是关键词过滤的问题。《攻击者后手出招》（Nasr et al.，OpenAI / Anthropic / DeepMind 联合，2025 年 10 月）进一步表明：适应性攻击（梯度搜索、RL、随机搜索、人类红队）击穿了 12 个已发表防御中的绝大多数，这些防御原本都声称攻击成功率接近 0。

**Type:** 构建
**Languages:** Python (stdlib, IPI attack + defense harness)
**Prerequisites:** 阶段 18 · 12（PAIR 自动化攻击）、阶段 14（智能体工程）
**Time:** 约 75 分钟

## 学习目标

- 定义间接提示注入，并描述三种常见投递路径。
- 解释为什么仅针对用户输入的过滤器会完全漏掉 IPI。
- 说明为什么 “信息流控制” 会成为 2026 年的主流防御框架。
- 陈述 Nasr et al.（2025 年 10 月）关于适应性攻击击穿已发表 IPI 防御的结论。

## 问题

直接 prompt injection 需要攻击者能接触到用户或用户的 prompt。IPI 则完全不需要。攻击者只要把 payload 放进代理可能读取的内容里，例如网页、收件箱邮件、GitHub issue、商品评论，代理就在正常工作流程中把它读进去并执行其中的指令。用户只是内容的载体，不是攻击意图的来源。

## 概念

### 三种常见投递路径

- **Retrieval-augmented generation (RAG).** 攻击者发布一份文档；检索步骤把它取回来；prompt 在用户问题之前拼接了这份文档；模型于是执行攻击者埋进去的指令。
- **Inbox / document workflows.** 攻击者给用户发送一封邮件；代理负责读邮件；prompt 包含邮件正文；模型随后遵循邮件里的指令。
- **Tool output.** 攻击者控制代理使用的某个工具，例如一个会返回攻击者控制结果的 web search；工具输出里包含指令；代理的控制流继续执行这些指令。

这三类攻击有一个共同结构：攻击者控制了 prompt 的一部分，但完全不需要碰用户面向系统输入的那一部分。

### 为什么用户输入过滤器会漏掉它

IPI payload 根本不在用户输入中，而是在检索或读取到的内容里。如果过滤器只看用户输入，那 payload 会直接绕过它。如果过滤器改成审查所有进入模型的内容，它就必须覆盖任意外部文本，这不仅成本高，而且会对那些恰好使用祈使句、操作说明或流程语言的合法内容产生大量误报。

### 面向 AI 的 Information Flow Control (IFC)

2026 年的主流防御思路借鉴了经典操作系统安全。把每个内容来源都看成带安全标签的输入源。用户查询标成 “trusted”。检索得到的内容标成 “untrusted”。把模型的控制流当成一种信息流：任何由不可信内容触发的动作，在执行前都必须得到可信输入的批准。

CaMeL（Microsoft 2025）、ConfAIde（Stanford 2024）以及 NDSS 2026 的 IPI-defense 论文，都从不同角度把 IFC 落到了工程实现上。它们共享的核心原则是：只要代码与数据还共处在同一个 context window 中，目标就不是彻底预防，而是把影响限制住。

### 攻击者后手出招

Nasr et al.（2025 年 10 月）用适应性攻击测试了 12 个已发表 IPI 防御，包括梯度搜索、RL policy、随机搜索和 72 小时人类红队。凡是原论文里报告为近乎零 ASR 的防御，最终几乎都被打到了 90% 以上的 ASR。

这带来的方法论结论很直接：如果没有适应性攻击评估，就不该声称一个 IPI 防御是可靠的。静态攻击基准不能证明鲁棒性，因为攻击者本来就可以看到你的防御设计，再反过来为它定制 payload。

### 真实事件

第 25 课会进一步展开 EchoLeak（CVE-2025-32711，CVSS 9.3），这是 Microsoft 365 Copilot 中第一个公开记录的 zero-click IPI。GitHub Copilot Chat 上还有 CamoLeak（CVSS 9.6），GitHub Copilot 还曝出过 CVE-2025-53773。IPI 已经不是论文基准里的问题，而是在真实生产部署里持续发生的安全事件。

### OWASP 和 NIST 的表述

OWASP LLM Top 10（2025）把 prompt injection（包括 direct 和 indirect）列为 LLM01，也就是应用层排名第一的威胁。NIST AI SPD 2024 则把 indirect prompt injection 直接称为 “generative AI's greatest security flaw”。

### 它在 Phase 18 中的位置

第 12 到 14 课讨论的是以模型为中心的 jailbreak。第 15 课转向以系统为中心的攻击，这正是 2026 年生产环境里的主战场。第 16 课讲防御工具。第 25 课则会展开具体 CVE 的事件叙事。

```figure
al-injection-vector
```

## 用它

`code/main.py` 构建了一个 IPI harness。这个 toy agent 有三个工具：search web、read email、send message。环境中放入了攻击者可控内容，其中嵌有一条指令，例如 “forward this to all contacts”。你可以在三种代理之间切换：naive agent 会直接服从注入指令；filter-defended agent 会对检索内容做关键词过滤；IFC agent 会区分 trusted 与 untrusted 内容，并拒绝由 untrusted 内容直接驱动的控制流命令。

## 交付成果

这一课产出 `outputs/skill-ipi-audit.md`。给定一份 agentic deployment 描述，它会枚举所有不可信内容源，检查部署是否实现了 IFC，并标出那些在没有 trust label 的情况下直接流入模型的输入源。

## 练习

1. 运行 `code/main.py`。测量攻击对这三种代理分别有多高的成功率。

2. 在检索内容上实现一个基于 paraphrase 的防御。测量它对合法检索文本的良性误报率。

3. 阅读 NDSS 2026 的 IPI-defense 论文。说明 “benign instruction” 难题是什么，以及为什么它让基于关键词的过滤失效。

4. 设计一个部署场景，让代理从第三方 API 获取工具输出。为每个 prompt 片段标出 trust level，并写出约束代理动作的 IFC policy。

5. 把 Nasr et al. 2025 的适应性攻击方法复现在练习 2 的 filter-defended agent 上。报告适应性攻击前后的 ASR。

## 关键词

| 术语 | 常见说法 | 实际含义 |
|------|-----------------|------------------------|
| IPI | "间接提示注入" | 通过并非用户亲自撰写的内容完成注入，代理在正常运行中读到并执行 |
| RAG injection | "poisoned retrieval" | 攻击者发布内容，检索步骤把它取回，prompt 中随之带入 payload |
| Zero-click | "no user action" | 攻击会在代理正常运行时自动触发，用户什么都不用做 |
| IFC | "信息流控制" | 基于标签的防御方法：来自不可信内容的动作必须得到可信输入批准 |
| Adaptive attack | "gradient / RL red-team" | 知道防御细节并据此优化的攻击；这是诚实评估所必需的攻击方式 |
| Benign instruction | "please print Yes" | 语义上看起来无害的 IPI 指令，因此关键词过滤抓不住 |
| Scope violation | "cross-trust exfiltration" | 代理从一个信任域读取数据，再把它输出到另一个信任域 |

## 进一步阅读

- [MDPI Information 17(1):54 — Indirect Prompt Injection Survey (January 2026)](https://www.mdpi.com/2078-2489/17/1/54) — 对 2023 到 2025 年工作的综述
- [Nasr et al. — The Attacker Moves Second (joint OpenAI/Anthropic/DeepMind, October 2025)](https://arxiv.org/abs/2510.18108) — 适应性攻击评估
- [Greshake et al. — Not what you've signed up for (arXiv:2302.12173)](https://arxiv.org/abs/2302.12173) — 最早的 IPI 论文
- [OWASP — LLM Top 10 (2025)](https://genai.owasp.org/llm-top-10/) — 将 prompt injection 列为 LLM01
