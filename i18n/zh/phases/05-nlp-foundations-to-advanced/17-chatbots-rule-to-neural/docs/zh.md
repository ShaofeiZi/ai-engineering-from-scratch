# 聊天机器人——从规则、神经网络到大语言模型智能体

> ELIZA 用模式匹配作答，DialogFlow 映射意图，GPT 从权重中回答，Claude 调用工具并验证结果。每个时代都解决了上一个时代最严重的失败。

**Type:** 学习
**Languages:** Python
**Prerequisites:** 阶段 5 · 13（问答系统）、阶段 5 · 14（信息检索）
**Time:** 约 75 分钟

## 问题

用户说“我想改签航班。”系统必须判断用户想做什么、还缺少哪些信息、如何获得这些信息，以及如何完成操作。接着用户又说“等等，那如果我改成取消呢？”，系统必须记住上下文、切换任务并保留状态。

对于机器学习系统而言，对话很难。输入是开放式的，输出必须在多轮之间保持连贯，系统还可能需要对现实世界执行操作（更改航班、信用卡扣款）。每一个错误步骤都会直接暴露给用户。

聊天机器人架构经历了四种范式的更迭，每一种都是因为前一种失败得太过明显而出现。本课将按顺序讲解它们。2026 年的生产格局是后两种范式的混合。

## 概念

![聊天机器人的演进：基于规则 → 检索 → 神经网络 → 智能体](../../../../../../phases/05-nlp-foundations-to-advanced/17-chatbots-rule-to-neural/assets/chatbot.svg)

### 脚本统治的半个世纪：1950～2001

第一个范式并非只持续了五年，而是持续了五十年。理解其演变历程十分重要，因为其中每个系统本质上都是同一台机器——匹配输入、发出预制响应、更新少量状态——而为这台机器不断增加规则，五十年间始终没有产生通用能力。这个上限正是第二至第四种范式存在的原因。

**1950 年。** 图灵没有正面回答“机器能思考吗？”，而是提出一种可操作的替代方式：如果讯问者通过电传打字机无法分辨机器和人，那么哲学问题便无关紧要。在这个领域尚未得到命名之前，对话就已成为它的基准。

**1956 年。** 名称出现了——达特茅斯夏季研讨会创造了“人工智能”一词，其猜想是智能的每个特征“原则上都可以得到如此精确的描述，以至于机器可以模拟它”。提案预计用两个月取得实质性进展。

**1966 年。** ELIZA 发布了你将在第 1 步构建的反射技巧：分解规则从输入中提取片段，重组规则则把片段以问题形式回应。总共约 200 个模式，没有状态，没有理解——用户却仍向它倾诉心事。此后余生，Weizenbaum 都对如此少的机制竟能产生这种效果深感不安。

**1972 年。** 斯坦福为模拟偏执心理而构建的 PARRY，加入了 ELIZA 所缺少的部分：内部状态。恐惧、愤怒和不信任三个数值变量会在每轮更新，并决定接下来触发哪段脚本，因此相同输入会根据此前对话得到不同响应。在盲测对话记录中，精神科医生区分 PARRY 与真人患者的准确率仅相当于随机猜测。它是角色设定的直接祖先——用三个浮点数实现的系统提示。同年，这两个机器人通过 ARPANET 被安排彼此对话：治疗师脚本采访偏执状态机，成就了网络上的首次机器人对话。

**1995 年。** ALICE 使用 AIML 扩展 ELIZA 的方法。AIML 是一种用于模式—模板对的 XML 方言，包含约 4 万个人工编写的类别，并三次赢得 Loebner Prize。它证明了规则系统的扩展定律：增加规则只能扩大覆盖范围，无法带来通用性。每条规则都会成为需要维护的负担。

**2001 年。** SmarterChild 把这种方法带给 3000 万即时通信用户，并加入后端查询——天气、股票、电影场次——再把结果填进模板。仔细看，它就是披着 2001 年外衣的工具调用：解析意图、调用服务、把结果渲染进回复。

五十年，始终只有一种机制，规则数量不断增长。这个范式终结，不是因为有人从理论上否定了它，而是因为人工状态机的维护成本随覆盖范围线性增长，用户预期却随着他们上周看到的新产品不断抬高。

```figure
chatbot-lineage
```

**基于规则（ELIZA、AIML、DialogFlow）。** 人工编写的模式匹配用户输入并生成响应。意图分类器把请求路由到预定义流程，槽位填充状态机负责收集所需信息。它在设计好的狭窄范围内表现极佳，超出范围便会立即失败。在不允许幻觉的安全关键领域（银行身份验证、航班预订）中，它至今仍在使用。

**基于检索。** FAQ 风格的系统。编码每个（话语、响应）对；运行时编码用户消息，再检索最接近的已存响应。可以把它理解为 Zendesk 经典的“相关文章”功能。它比规则更能处理释义，又不会生成内容，因此不会产生幻觉。

**神经网络式（序列到序列）。** 在对话日志上训练编码器—解码器，从零生成响应。输出流畅，却容易产生泛泛的回答（“我不知道”）和事实漂移，始终无法可靠地围绕主题。这就是 Google、Facebook 和 Microsoft 在 2016～2019 年间的聊天机器人都令人失望的原因。

**大语言模型智能体。** 在语言模型外部包裹一层循环，让它规划、调用工具并验证结果。它不是拥有一段长提示的聊天机器人，而是一个智能体循环：规划 → 调用工具 → 观察结果 → 决定下一步。检索优先的落地方式（RAG）防止它产生幻觉，工具调用则让它真正执行操作。这是 2026 年的架构。

四种范式并非依次完全取代彼此。2026 年的生产聊天机器人会把请求路由给全部四类机制：身份验证和破坏性操作走规则流程，FAQ 走检索，自然措辞由神经生成完成，含糊的开放式请求则交给大语言模型智能体。

## 动手构建

### 第 1 步：基于规则的模式匹配

```python
import re


class RulePattern:
    def __init__(self, pattern, response_template):
        self.regex = re.compile(pattern, re.IGNORECASE)
        self.template = response_template


PATTERNS = [
    RulePattern(r"my name is (\w+)", "Nice to meet you, {0}."),
    RulePattern(r"i (need|want) (.+)", "Why do you {0} {1}?"),
    RulePattern(r"i feel (.+)", "Why do you feel {0}?"),
    RulePattern(r"(.*)", "Tell me more about that."),
]


def rule_based_respond(user_input):
    for pattern in PATTERNS:
        m = pattern.regex.match(user_input.strip())
        if m:
            return pattern.template.format(*m.groups())
    return "I don't understand."
```

20 行代码实现 ELIZA。反射技巧（“I feel sad”→“Why do you feel sad”）就是 Weizenbaum 在 1966 年展示的经典心理治疗师演示，至今仍有启发性。

### 第 2 步：基于检索（FAQ）

这段示意代码需要 `pip install sentence-transformers`（它会连带安装 torch）。本课可运行的 `code/main.py` 改用标准库实现的 Jaccard 相似度，因此课程无须外部依赖即可运行。

```python
from sentence_transformers import SentenceTransformer
import numpy as np


FAQ = [
    ("how do i reset my password", "Go to Settings > Security > Reset Password."),
    ("how do i cancel my order", "Go to Orders, find the order, click Cancel."),
    ("what is your return policy", "30-day returns on unused items, original packaging."),
]


encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
faq_questions = [q for q, _ in FAQ]
faq_embeddings = encoder.encode(faq_questions, normalize_embeddings=True)


def faq_respond(user_input, threshold=0.5):
    q_emb = encoder.encode([user_input], normalize_embeddings=True)[0]
    sims = faq_embeddings @ q_emb
    best = int(np.argmax(sims))
    if sims[best] < threshold:
        return None
    return FAQ[best][1]
```

基于阈值的拒答是关键设计。如果最佳匹配还不够接近，就返回 `None`，让系统升级处理。

### 第 3 步：神经生成（基线）

使用小型指令微调编码器—解码器（FLAN-T5）或微调后的对话模型。2026 年，它单独使用时无法达到生产要求（会自相矛盾、偏离主题、生成事实谬误），但可以作为混合系统中的自然表达模块。DialoGPT 风格的仅解码器模型需要显式的轮次分隔符和 EOS 处理，才能产生连贯回复；FLAN-T5 的 text2text 流水线则可直接用于教学示例。

```python
from transformers import pipeline

chatbot = pipeline("text2text-generation", model="google/flan-t5-small")

response = chatbot("Respond politely to: Hi there!", max_new_tokens=40)
print(response[0]["generated_text"])
```

### 第 4 步：大语言模型智能体循环

2026 年的生产形态如下：

```python
def agent_loop(user_message, tools, llm, max_steps=5):
    history = [{"role": "user", "content": user_message}]
    for _ in range(max_steps):
        response = llm(history, tools=tools)
        tool_call = response.get("tool_call")
        if tool_call:
            tool_name = tool_call.get("name")
            args = tool_call.get("arguments")
            if not isinstance(tool_name, str) or tool_name not in tools:
                history.append({"role": "assistant", "tool_call": tool_call})
                history.append({"role": "tool", "name": str(tool_name), "content": f"error: unknown tool {tool_name!r}"})
                continue
            if not isinstance(args, dict):
                history.append({"role": "assistant", "tool_call": tool_call})
                history.append({"role": "tool", "name": tool_name, "content": f"error: arguments must be a dict, got {type(args).__name__}"})
                continue
            fn = tools[tool_name]
            result = fn(**args)
            history.append({"role": "assistant", "tool_call": tool_call})
            history.append({"role": "tool", "name": tool_name, "content": result})
        else:
            return response["content"]
    return "I could not complete the task in the step budget."
```

这里有三点需要明确。工具是大语言模型可以调用的函数；当大语言模型返回最终答案而不是工具调用时，循环结束；步骤预算则防止智能体在含糊任务上无限循环。

真实生产系统还会增加：检索优先的落地（每次调用大语言模型前注入相关文档）、护栏（破坏性操作必须确认）、可观测性（记录每一步），以及评估（自动检查智能体行为是否仍符合规范）。

### 第 5 步：混合路由

```python
def hybrid_chat(user_input):
    if is_destructive_action(user_input):
        return structured_flow(user_input)

    faq_answer = faq_respond(user_input, threshold=0.6)
    if faq_answer:
        return faq_answer

    return agent_loop(user_input, tools, llm)


def is_destructive_action(text):
    danger_words = ["delete", "cancel", "charge", "refund", "transfer"]
    return any(w in text.lower() for w in danger_words)
```

模式如下：任何破坏性操作都交给确定性规则，预制 FAQ 交给检索，其他请求交给大语言模型智能体。这就是 2026 年客服系统实际采用的架构。

## 学以致用

2026 年的技术栈：

| 用例 | 架构 |
|---------|---------------|
| 预订、支付、身份验证 | 基于规则的状态机 + 槽位填充 |
| 客户支持 FAQ | 在精选答案上检索 |
| 开放式帮助对话 | 带 RAG 与工具调用的大语言模型智能体 |
| 内部工具/IDE 助手 | 带工具调用（搜索、读取、写入）的大语言模型智能体 |
| 陪伴型/角色型聊天机器人 | 通过角色系统提示调优的大语言模型，并在知识库上检索 |

生产环境始终应使用混合路由。没有任何单一架构能妥善处理所有请求。路由层本身通常是一个小型意图分类器。

## 仍会进入生产环境的失败模式

- **自信地捏造。** 大语言模型智能体声称完成了某项实际并未完成的操作。缓解方法：验证结果、记录工具调用；如果工具没有返回成功结果，绝不允许模型声称已经完成操作。
- **提示注入。** 用户插入试图覆盖系统提示的文本。在 OWASP 2025 年大语言模型应用十大风险中排名 LLM01。它分为两类：直接注入（粘贴到聊天中）和间接注入（隐藏在智能体读取的文档、电子邮件或工具输出中）。

  攻击成功率会随场景变化。在通用工具使用和编码基准中，前沿模型的实测成功率约为 0.5%～8.5%；在某些高风险设置中（针对 AI 编码智能体的自适应攻击、存在漏洞的编排机制），成功率曾达到约 84%。真实生产 CVE 包括 EchoLeak（CVE-2025-32711，CVSS 9.3）：Microsoft 365 Copilot 中的零点击数据外泄漏洞，可由攻击者控制的电子邮件触发。

  缓解方法：在整个循环中始终把用户输入视为不可信；调用工具前进行净化；将工具输出与主提示隔离；使用 **Plan-Verify-Execute（规划—验证—执行）规划模式**，让智能体先制定计划，再在执行前对照该计划验证每个动作（这样可以阻止工具结果注入计划外的新动作）；破坏性操作必须由用户确认；对工具权限采用最小权限原则。在本课中，PVE 指 Plan-Verify-Execute。它不同于 Phase 14, Lesson 27 中的 **Prompt-Validator-Executor（提示—验证器—执行器）验证模式**；后者会在工具执行前插入一个独立的验证器。

  再多的提示工程也无法彻底消除这一风险。必须使用外部运行时防御层（LLM Guard、允许列表验证、语义异常检测）。
- **范围蔓延。** 工具调用返回了旁支信息，智能体因而偏离任务。缓解方法：缩小工具契约；保持系统提示聚焦；增加偏题率评估。
- **无限循环。** 智能体反复调用同一个工具。缓解方法：设置步骤预算、对工具调用去重，并让大语言模型裁判判断“是否仍在取得进展”。
- **上下文窗口耗尽。** 长对话把最早的轮次挤出上下文。缓解方法：总结早期轮次、通过相似度检索相关历史，或使用长上下文模型。

## 交付成果

保存为 `outputs/skill-chatbot-architect.md`：

```markdown
---
name: chatbot-architect
description: Design a chatbot stack for a given use case.
version: 1.0.0
phase: 5
lesson: 17
tags: [nlp, agents, chatbot]
---

Given a product context (user need, compliance constraints, available tools, data volume), output:

1. Architecture. Rule-based, retrieval, neural, LLM agent, or hybrid (specify which paths go where).
2. LLM choice if applicable. Name the model family (Claude, GPT-4, Llama-3.1, Mixtral). Match to tool-use quality and cost.
3. Grounding strategy. RAG sources, retrieval method (see lesson 14), tool contracts.
4. Evaluation plan. Task success rate, tool-call correctness, off-task rate, hallucination rate on held-out dialogs.
5. Execution control. For a tool-using agent, apply the **Plan-Verify-Execute planning pattern**: agree on a plan, verify each proposed action against it, then execute. In this lesson, PVE means Plan-Verify-Execute; do not confuse it with Phase 14's **Prompt-Validator-Executor validation pattern**, which places a separate validator before tool execution.

Refuse to recommend a pure-LLM agent for any destructive action (payments, account deletion, data modification) without a structured confirmation flow. Refuse to skip the prompt-injection audit if the agent has write access to anything.
```

## 练习

1. **简单。** 为咖啡店点单机器人实现上面的规则响应，编写 10 个模式。测试边缘情况：重复点单、修改、取消、意图不明确。
2. **中等。** 构建 FAQ + 大语言模型回退的混合系统。为一个 SaaS 产品准备 50 条预制 FAQ，再让大语言模型结合文档站点检索作为回退。在 100 个真实支持问题上测量拒答率和准确率。
3. **困难。** 使用三个工具（搜索、读取用户数据、发送电子邮件）实现上面的智能体循环。运行包含 50 个测试场景的评估，其中包括提示注入尝试。报告偏题率、任务失败率和所有成功的注入。

## 关键术语

| 术语 | 人们通常怎么说 | 实际含义 |
|------|-----------------|-----------------------|
| 意图 | 用户想做什么 | 分类标签（book_flight、reset_password），用于路由到处理器。 |
| 槽位 | 一条信息 | 机器人需要的参数（日期、目的地）；槽位填充就是依次询问这些信息。 |
| RAG | 检索加生成 | 检索相关文档，再以它们为依据生成大语言模型响应。 |
| 工具调用 | 函数调用 | 大语言模型发出包含名称与参数的结构化调用；运行时执行并返回结果。 |
| 智能体循环 | 规划、行动、验证 | 控制器交替运行大语言模型调用与工具调用，直到任务完成。 |
| 提示注入 | 用户攻击提示 | 试图覆盖系统提示的恶意输入。 |

## 延伸阅读

- [Turing（1950），计算机器与智能](https://academic.oup.com/mind/article/LIX/236/433/986238)——让对话成为这一领域基准的论文。
- [Weizenbaum（1966），ELIZA——研究自然语言交流的计算机程序](https://web.stanford.edu/class/cs124/p36-weizenabaum.pdf)——最初的规则式聊天机器人论文。
- [Colby、Weber、Hilf（1971），人工偏执](https://doi.org/10.1016/0004-3702(71)90002-6)——PARRY 的情感变量架构，也是第一个有状态聊天机器人。
- [Thoppilan 等（2022），LaMDA：用于对话应用的语言模型](https://arxiv.org/abs/2201.08239)——Google 在大语言模型智能体接管之前推出的后期神经聊天机器人论文。
- [Yao 等（2022），ReAct：协同语言模型中的推理与行动](https://arxiv.org/abs/2210.03629)——为智能体循环模式命名的论文。
- [Anthropic 构建高效智能体指南](https://www.anthropic.com/research/building-effective-agents)——2024 年的生产指南，到 2026 年仍然适用。
- [Greshake 等（2023），这不是你预想的结果：通过间接提示注入攻破现实世界中的大语言模型集成应用](https://arxiv.org/abs/2302.12173)——提示注入论文。
- [OWASP 2025 年大语言模型应用十大风险——LLM01 提示注入](https://genai.owasp.org/llmrisk/llm01-prompt-injection/)——让提示注入成为首要安全问题的排名。
- [AWS——保护 Amazon Bedrock Agent 免受间接提示注入](https://aws.amazon.com/blogs/machine-learning/securing-amazon-bedrock-agents-a-guide-to-safeguarding-against-indirect-prompt-injections/)——实用的编排层防御，包括 Plan-Verify-Execute（规划—验证—执行）规划模式和用户确认流程。
- [EchoLeak（CVE-2025-32711）](https://www.vectra.ai/topics/prompt-injection)——由间接提示注入导致的经典零点击数据外泄 CVE，说明拥有写权限的智能体为何需要运行时防御。
