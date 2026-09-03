# 综合项目 17——个人人工智能导师（自适应、多模态、带记忆）

> 到 2026 年，Khanmigo（Khan Academy）、Duolingo Max、Google LearnLM / Gemini for Education、Quizlet Q-Chat 和 Synthesis Tutor 都已实现大规模自适应多模态辅导。它们采用相似的结构：以苏格拉底式策略引导学生，而不是直接给出答案；维护每次互动后都会更新的学习者模型，通常基于贝叶斯知识追踪；支持语音、文本和数学题照片输入；通过课程图谱检索内容；安排间隔重复；并严格过滤不适合相应年龄的内容。本综合项目要交付一套面向特定学科的导师系统，可以选择 K-12 代数或 Python 入门；随后用 10 名学习者开展两周效果研究，并通过内容安全审计。

**Type:** 综合项目
**Languages:** Python（后端、学习者模型）、TypeScript（Web 应用）、SQL（通过 Postgres + Neo4j 实现课程图谱）
**Prerequisites:** 第 5 阶段（NLP）、第 6 阶段（语音）、第 11 阶段（LLM 工程）、第 12 阶段（多模态）、第 14 阶段（智能体）、第 17 阶段（基础设施）、第 18 阶段（安全）
**Phases exercised:** P5 · P6 · P11 · P12 · P14 · P17 · P18
**Time:** 30 小时

## 问题

自适应辅导过去只是教育科技研究中的小众方向，到 2026 年已经成为消费级产品。Khanmigo 已部署到美国大多数学区；Duolingo Max 达到数千万月活跃用户；Google LearnLM / Gemini for Education 为 Google Classroom 提供辅导能力；Quizlet Q-Chat 与单词卡配合使用；Synthesis Tutor 则以“面向好奇儿童的导师”走红。它们有几个共性：支持打字、语音和拍照识别方程等多模态输入；采用先提问、后解释的苏格拉底式教学；每次互动后更新学习者模型；严格过滤不适合相应年龄的内容。

本综合项目的验收标准不是“做出一个演示”，而是真正开展效果研究：让 10 名学习者参与两周实验，并进行前测和后测。语音交互必须自然，可复用综合项目 03 的语音子栈。记忆系统必须保护隐私，安全过滤器则必须通过面向 K-12、符合 COPPA 要求的红队测试。

## 概念

系统由四个核心部件构成。**导师策略（Tutor policy）**采用苏格拉底式循环：学习者直接索要答案时，系统改为提出引导性问题；回答正确时，进入下一个概念；卡住时，提供分层提示。**学习者模型（Learner model）**使用贝叶斯知识追踪或简化变体，在每次互动后更新各课程节点的掌握概率。**课程图谱（Curriculum graph）**是一张 Neo4j 图，节点代表概念，边表示先修关系；导师策略在图上选择接下来要学的概念。**记忆（Memory）**采用类似 agentmemory 的情景记忆与语义记忆存储，保存过往互动、常见错误和个人偏好。

交互层必须支持多种模态。文本用于键盘作答；语音输入可通过 LiveKit + Whisper 实现，直接复用综合项目 03；数学题照片由 dots.ocr 或 PaliGemma 2 识别；语音输出使用 Cartesia Sonic-2。安全侧接入 Llama Guard 4，再叠加适龄过滤器，阻断成人内容、暴力和自伤；同时实现符合 COPPA 的记忆保留策略。

真正的交付物是效果研究：10 名学习者参与两周实验，完成前测和后测，并报告学习增益和置信区间。还要设置非自适应基线作为对照，即以固定顺序提供相同内容，不启用导师策略。

## 架构

```
learner device
  |
  +-- text         -> web app
  +-- voice        -> LiveKit Agents (ASR + TTS)
  +-- photo math   -> dots.ocr / PaliGemma 2
       |
       v
  tutor policy (LangGraph)
       - Socratic decision head
       - next-concept chooser (curriculum graph walk)
       - hint scaffolder
       - mastery update
       |
       v
  learner model (BKT / item-response theory)
       - per-concept mastery probability
       - spaced-repetition scheduler (SM-2 or FSRS)
       |
       v
  memory (agentmemory-style)
       - episodic: every interaction
       - semantic: learned mistakes, preferences
       - retention policy: COPPA / GDPR aware
       |
       v
  curriculum graph (Neo4j)
       - prerequisite edges
       - OER content attached
       |
       v
  safety:
    Llama Guard 4 + age-appropriate filter
    memory access guarded by learner ID scope
```

## 技术栈

- 学科选择：K-12 代数或 Python 入门，二选一，重在深度而非广度
- 导师策略：基于 Claude Sonnet 4.7 的 LangGraph，并启用提示缓存
- 学习者模型：经典贝叶斯知识追踪，或使用 FSRS 安排间隔重复
- 课程图谱：Neo4j，保存概念节点、先修关系边和 OER 内容
- 记忆：agentmemory 风格的持久化向量、情景记忆和语义记忆存储
- 语音：LiveKit Agents 1.0 + Cartesia Sonic-2，复用综合项目 03 的子栈
- 拍照识题：dots.ocr 或 PaliGemma 2，负责识别方程
- 安全：Llama Guard 4 + 自定义适龄过滤器
- 评估：按 Bloom 认知层级生成题目、前后测框架与效果研究工具

```figure
cf-tutor-loop
```

## 动手构建

1. **构建课程图谱。** 建一张包含 50～150 个概念节点的 Neo4j 图。例如 K-12 代数可以从“数轴”一路覆盖到“二次方程求根公式”；边表示先修关系。每个节点都绑定 OER 内容，例如 Open Textbook 或 OpenStax。

2. **实现学习者模型。** 初始化贝叶斯知识追踪的先验参数，例如猜测概率、失误概率和学习转移概率。每次互动后更新对应概念的掌握概率，并按学习者分别持久化。

3. **实现导师策略。** 用 LangGraph 搭建明确的节点图，至少包括：`read_signal`（判断学习者回答是正确、部分正确还是卡住）、`select_concept`（从课程图谱中选择优先级最高的概念）、`scaffold`（生成苏格拉底式提示）和 `update_mastery`。

4. **实现记忆。** 每次互动都写入情景记忆存储。错误模式和偏好经过提炼后进入语义记忆。保留策略必须符合 COPPA：默认一年后自动删除，并允许家长访问和删除数据。

5. **实现语音路径。** 把 LiveKit Agents 工作进程接到导师策略。ASR 使用 Whisper-v3-turbo，TTS 使用 Cartesia Sonic-2，并支持插话，复用综合项目 03 的机制。

6. **实现拍照数学路径。** 允许上传或拍摄图片，用 dots.ocr 或 PaliGemma 2 识别方程，再把结果作为结构化输入传给导师系统。

7. **落实安全控制。** 所有模型输出都必须经过 Llama Guard 4 和适龄过滤器，阻断自伤、成人内容和暴力内容。记忆访问必须限定在学习者 ID 范围内，并提供供家长删除数据的界面。

8. **开展效果研究。** 招募 10 名学习者，先完成一份标准化 30 题前测；随后与导师系统互动两周，每周 3 次；最后完成后测。再与一组同样为 10 人的非自适应基线组比较：学习内容相同，但不启用自适应导师策略。

9. **生成每周进度报告。** 为每位学习者自动生成 PDF 周报，总结已练习主题、掌握程度变化和下一步建议。

## 运行示例

```
learner: "I don't understand why 3x + 6 = 12 means x = 2"
[signal]   stuck
[concept]  'isolating variables' (prerequisite: addition-subtraction-equality)
[scaffold] "what number would you subtract from both sides to start?"
learner: "6"
[signal]   correct
[mastery]  addition-subtraction-equality: 0.62 -> 0.77
[concept]  continue 'isolating variables'
[scaffold] "great. now what is 3x / 3 equal to?"
```

## 交付成果

`outputs/skill-ai-tutor.md` 是本课交付物：一个面向特定学科的自适应导师系统，具备多模态输入、学习者模型、记忆和安全机制，并经过学习效果实测。

| 权重 | 评判标准 | 衡量方式 |
|:-:|---|---|
| 25 | 学习增益 | 10 名学习者参与的两周研究中，前测与后测的差值 |
| 20 | 苏格拉底式教学保真度 | 对话文本样本的量表评分 |
| 20 | 多模态体验 | 语音、照片与文本的端到端一致性 |
| 20 | 安全与隐私保障 | Llama Guard 4 通过率与符合 COPPA 的留存策略 |
| 15 | 课程广度与图谱质量 | 概念覆盖度与先修关系图的一致性 |
| **100** | | |

## 练习

1. 运行两轮效果研究：一轮启用自适应学习者模型，另一轮随机排列概念顺序。报告两者差值。通常自适应方案会更好，但真正有价值的是量化它领先多少。

2. 增加多模态对照测试：同一道概念题分别通过文本、语音和照片呈现，测量学习者使用自己偏好的模态时是否掌握得更快。

3. 构建家长仪表板，显示已练习主题、掌握程度变化、即将学习的概念和安全事件，例如护栏命中。整体必须符合 COPPA。

4. 增加语言切换模式：接受西班牙语输入，并使用西班牙语教学。测量 X-Guard 在这个场景下的覆盖情况。

5. 专门压测记忆隐私：验证学习者 A 即使发起重新摄取语音片段的攻击，也不能看到学习者 B 的数据。所有访问尝试都必须记录并告警。

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|-----------------|------------------------|
| 苏格拉底式策略 | “提问，不灌输” | 导师通过引导性提问教学，而不是直接给答案 |
| 贝叶斯知识追踪 | “BKT” | 用于估计各概念掌握概率的经典学习者模型方程 |
| FSRS | “Free Spaced Repetition Scheduler” | 2024 年的间隔重复调度器，优于 SM-2 |
| 课程图谱 | “概念 DAG” | 用 Neo4j 表示的概念图，边表示先修关系 |
| 情景记忆 | “逐次互动日志” | 保存每次互动，供之后检索 |
| 语义记忆 | “学习到的模式库” | 从情景记忆中提炼并转入长期存储的错误模式与偏好 |
| COPPA | “儿童隐私法” | 限制收集美国 13 岁以下儿童数据的法律 |

## 延伸阅读

- [Khanmigo (Khan Academy)](https://www.khanmigo.ai) — 面向消费者的 K-12 导师参考产品
- [Duolingo Max](https://blog.duolingo.com/duolingo-max/) — 语言学习导师参考产品
- [Google LearnLM / Gemini for Education](https://blog.google/technology/google-deepmind/learnlm) — 托管式参考模型
- [Quizlet Q-Chat](https://quizlet.com) — 另一种参考产品
- [Synthesis Tutor](https://www.synthesis.com) — 初创公司参考产品
- [FSRS algorithm](https://github.com/open-spaced-repetition/fsrs4anki) — 间隔重复调度器
- [Bayesian Knowledge Tracing](https://en.wikipedia.org/wiki/Bayesian_knowledge_tracing) — 学习者模型中的经典方法
- [LiveKit Agents](https://github.com/livekit/agents) — 语音技术栈
