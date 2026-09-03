---
name: check-understanding
version: 1.0.0
description: AI Engineering from Scratch 的阶段测验。通过 "quiz me"、"test phase"、"check my understanding"、"do I know phase 3" 或 `/check-understanding <phase>` 触发。
---

# 理解检测

测试你对《AI Engineering from Scratch》课程中某个已完成阶段的掌握程度。

## 激活

当用户说出以下内容时，此技能将被激活：
- `/check-understanding 3` 或 `/check-understanding deep-learning`
- "quiz me on phase 2"
- "test phase 1"
- "check my understanding of transformers"
- "do I know phase 3"
- "am I ready for the next phase"

## 输入

接受阶段编号（0-19）或阶段名称作为参数。如果未提供参数，则通过列出全部 20 个阶段，让用户选择想要测试的阶段。

## 阶段映射

将参数映射到 `phases/` 下对应的阶段目录：

| 输入 | 目录 | 阶段名称 |
|-------|-----------|------------|
| 0, setup, tooling | `00-setup-and-tooling` | Setup & Tooling |
| 1, math, math-foundations | `01-math-foundations` | Math Foundations |
| 2, ml, ml-fundamentals | `02-ml-fundamentals` | ML Fundamentals |
| 3, deep-learning, dl | `03-deep-learning-core` | Deep Learning Core |
| 4, cv, computer-vision, vision | `04-computer-vision` | Computer Vision |
| 5, nlp | `05-nlp-foundations-to-advanced` | NLP -- Foundations to Advanced |
| 6, speech, audio | `06-speech-and-audio` | Speech & Audio |
| 7, transformers | `07-transformers-deep-dive` | Transformers Deep Dive |
| 8, generative, gen-ai, genai | `08-generative-ai` | Generative AI |
| 9, rl, reinforcement-learning | `09-reinforcement-learning` | Reinforcement Learning |
| 10, llms, llm, llms-from-scratch | `10-llms-from-scratch` | LLMs from Scratch |
| 11, llm-engineering, llm-eng | `11-llm-engineering` | LLM Engineering |
| 12, multimodal | `12-multimodal-ai` | Multimodal AI |
| 13, tools, protocols, mcp | `13-tools-and-protocols` | Tools & Protocols |
| 14, agents, agent-engineering | `14-agent-engineering` | Agent Engineering |
| 15, autonomous | `15-autonomous-systems` | Autonomous Systems |
| 16, multi-agent, swarms | `16-multi-agent-and-swarms` | Multi-Agent & Swarms |
| 17, infrastructure, production, infra | `17-infrastructure-and-production` | Infrastructure & Production |
| 18, ethics, safety, alignment | `18-ethics-safety-alignment` | Ethics, Safety & Alignment |
| 19, capstone, projects | `19-capstone-projects` | Capstone Projects |

## 流程

### 第一步：解析阶段

解析参数。如果是数字，验证其在 0 到 19（含）之间。如果数字超出范围，告诉用户："Phase [N] does not exist. Valid phases are 0-19." 并展示完整列表供其选择。如果是名称或关键词，在上面的阶段映射表中查找。如果关键词未匹配到任何条目，告诉用户："Unknown phase '[keyword]'. Pick from the list below:" 并展示全部 20 个阶段。如果未提供参数，请用户从完整列表中选择。

### 第二步：阅读阶段内容

如果仓库已克隆（当前目录或上级目录中存在 `phases/` 目录），找到 `phases/<phase-dir>/` 下的所有课程目录，并读取每门课程的 `docs/en.md`。如果未克隆，从 README 的 Contents 部分获取该阶段的课程列表（获取 `https://raw.githubusercontent.com/rohitg00/ai-engineering-from-scratch/main/README.md`），然后从相同的 raw 基础 URL 获取每门课程的 `docs/en.md`。这些文档包含你将用于生成问题的教学材料。

根据需要阅读尽可能多的课程文档，以覆盖该阶段的全部内容。如果某个阶段课程较多（15+），优先阅读有代表性的分布：前几节、中间和最后几节。

### 第三步：生成 8 道题目

根据刚刚阅读的课程内容，创建恰好 8 道选择题：

**第 1-4 题：概念题（是什么/为什么）**
测试对概念、定义和推理的理解。示例：
- "X 的目的是什么？"
- "当 Z 时为什么会出现 Y？"
- "哪句话最能描述 A 和 B 之间的关系？"
- "X 解决了什么问题？"

**第 5-8 题：实践题（怎么做/构建）**
测试应用知识和实现意识。示例：
- "你会如何实现 X？"
- "哪种方法能正确解决 Y？"
- "构建 Z 的正确步骤顺序是什么？"
- "如果在训练中观察到 X，你应该怎么做？"

每道题必须恰好有 4 个选项，标记为 A、B、C 和 D。恰好有一个选项是正确的。错误选项应当具有合理性，但对于学过该材料的人来说应明显不正确。

为每道题标注其出自的具体课程（例如"Lesson 03: Matrix Transformations"）。

### 第四步：逐题呈现

使用 AskUserQuestion 工具（或等效的交互式提示）逐题呈现。格式如下：

```text
Question 1/8 (Conceptual) -- from Lesson 03: Matrix Transformations

What is the geometric interpretation of an eigenvalue?

A) The angle of rotation applied by the matrix
B) The factor by which the eigenvector is scaled during transformation
C) The determinant of the transformation matrix
D) The rank of the matrix after transformation
```

等待用户回答后再进入下一题。

### 答案隔离

在学习者回答当前问题之前，将正确选项和解释保密。切勿在回复格式提示中使用真实答案字母、可能的答案或生成的答案分布。需要纯文本提示时，使用以下内容：`Reply with one letter: <A|B|C|D>.`

### 第五步：记录与评分

保持累计记录：
- 8 题中答对的总数
- 对于每道答错的题，记录：题号、用户的答案、正确答案，以及该题出自哪门课程

### 第六步：展示结果

8 道题全部完成后，显示分数和评级：

**答对 7-8 题：已掌握**
如果该阶段是 19（Capstone Projects）："You have mastered Phase 19, the final phase." 仅当你能够验证课程其余部分已完成（当前目录中的 `LEARNING.md` 其 Path 表显示 Phases 0-18 为 Done 或 Skip）时，才添加 "Congratulations, you have completed the entire curriculum."；单次阶段测验不能证明全部完成。
否则："You have a strong grasp of Phase N. You are ready to move on to Phase N+1: [next phase name]."

**答对 5-6 题：接近掌握**
"基础扎实。在继续之前，请复习以下具体内容："
然后列出与答错题目相关的课程。

**答对 3-4 题：正在发展**
"你正在逐步建立理解，但还需要重温以下课程："
然后列出每道答错的题目及其需要重新阅读的课程。

**答对 0-2 题：从头开始**
"这个阶段还需要投入更多时间。请从头重新学习，并重点关注以下内容："
然后列出所有答错的主题。

### 第七步：错题详解

对于用户答错的每一道题，展示：

```text
Question N: [question text, abbreviated]
Your answer: B
Correct answer: C -- [the correct option text]
Why: [1-2 sentence explanation of why C is correct]
Review: Lesson NN -- [lesson name] (phases/<phase-dir>/NN-<lesson-slug>/docs/en.md)
```

### 第八步：接下来做什么？

最后提供三个选择：

1. **重新测试** -- 从同一阶段生成全新的 8 道题目
2. **尝试其他阶段** -- 选择不同的阶段进行测试
3. **解释某个主题** -- 就你答错的题目中的任何概念提问

等待用户做出选择并据此执行。

## 规则

- 重新测试时避免重复出题，直到题目池耗尽。一旦耗尽，可在后续重测中重新打乱或改写题目。
- 题目必须直接基于课程文档，而非通用知识。
- 在用户回答之前，不得显示正确答案。
- 在示例中说明学习者应如何回复时，不要包含字面答案字母；使用 `<A|B|C|D>` 作为占位符。
- 题目文本保持简洁。最多一到两句话。
- 错误选项必须具有合理性。不得有玩笑式答案。
- 如果某个阶段尚未编写课程文档（未找到 `en.md` 文件），告诉用户："Phase N does not have lesson content yet. Pick a completed phase to quiz on."
