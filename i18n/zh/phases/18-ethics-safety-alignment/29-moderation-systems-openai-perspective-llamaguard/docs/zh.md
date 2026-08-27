# 内容审核系统：OpenAI、Perspective 与 Llama Guard

> 生产级内容审核系统把第 12～16 课定义的安全策略真正落实到用户接触产品的界面上。OpenAI Moderation API 的 `omni-moderation-latest`（2024）基于 GPT-4o，一次调用即可同时分类文本和图像；它在多语言测试集上比上一版本提高了 42%，响应模式返回 13 个类别布尔值：harassment、harassment/threatening、hate、hate/threatening、illicit、illicit/violent、self-harm、self-harm/intent、self-harm/instructions、sexual、sexual/minors、violence、violence/graphic；大多数开发者可以免费使用。常见部署采用三层结构：输入审核（生成前）、输出审核（生成后）和自定义审核（领域规则）。同一层内可以异步并行调用多个分类器来隐藏延迟；内容被标记后，系统可以返回占位响应或拒答。Llama Guard 3/4（第 16 课）覆盖 14 类 MLCommons 危害，还支持 Code Interpreter Abuse、8 种语言（v3）和多图像输入（v4）。Perspective API（Google Jigsaw）早于“大语言模型充当审核器”的浪潮，主要输出毒性分数，并提供严重毒性、侮辱和脏话等维度，至今仍是内容审核研究的经典基线。Azure Content Moderator 已于 2024 年 2 月弃用，将于 2027 年 2 月退役，由 Azure AI Content Safety 取代。

**Type:** 构建
**Languages:** Python（标准库，三层审核工具）
**Prerequisites:** 阶段 18 · 16（红队工具：Llama Guard、Garak、PyRIT）
**Time:** 约 60 分钟

## 学习目标

- 描述 OpenAI Moderation API 的类别体系，以及它与 Llama Guard 3 的 MLCommons 类别集合有何不同。
- 描述审核系统的三层模式（输入、输出、自定义），并分别指出一种失效模式。
- 说明 Perspective API 作为大语言模型时代之前的研究基线处于什么位置，以及它为何沿用至今。
- 说明 Azure 产品的弃用时间线。

## 问题

第 12～16 课讨论攻击与防御工具；第 29 课讨论如何在用户真正接触产品的界面上部署这些防御。到 2026 年，三层审核模式已经成为默认配置。

## 概念

### OpenAI 内容审核 API

`omni-moderation-latest`（2024）基于 GPT-4o，可在一次调用中同时分类文本和图像。大多数开发者可以免费使用。

类别（响应模式中的 13 个布尔值）包括：
- harassment, harassment/threatening
- hate, hate/threatening
- self-harm, self-harm/intent, self-harm/instructions
- sexual, sexual/minors
- violence, violence/graphic
- illicit, illicit/violent

多模态支持覆盖 `violence`、`self-harm` 和 `sexual`，但不包括 `sexual/minors`；其余类别仅支持文本。

在 `code/main.py` 的教学实现里，我们会把 `/threatening`、`/intent`、`/instructions` 和 `/graphic` 这些子类折叠回上层父类，以降低讲解复杂度。真正的生产代码应保留完整的 13 类响应模式。

与上一代审核端点相比，它在多语言测试集上提高了约 42%。应用通常根据各类别的分数自行设定阈值。

### Llama Guard 3/4

第 16 课已经介绍过 Llama Guard。它使用 14 类 MLCommons 危害分类，这套组织方式与 OpenAI 响应模式中的 13 个类别布尔值并不相同。v3 支持 8 种语言；Llama Guard 4（2025 年 4 月）原生支持多模态，参数规模为 12B。

OpenAI 与 Llama Guard 的分类体系有重叠，但并不一致。例如，OpenAI 把 illicit 作为一个较宽泛的类别，而 Llama Guard 会把“暴力犯罪”和“非暴力犯罪”分开。实际部署时，团队通常选择与自身策略分类体系更贴合的一套。

### Perspective API（Google Jigsaw）

Perspective API（Google Jigsaw）是“大语言模型充当审核器”兴起之前就存在的毒性评分系统。它的类别包括 TOXICITY、SEVERE_TOXICITY、INSULT、PROFANITY、THREAT 和 IDENTITY_ATTACK。它以单一主分数 TOXICITY 为核心，其余类别作为细分维度。

它至今仍被广泛用作内容审核研究基线，是因为 API 稳定、文档充分，而且积累了多年的校准数据。对于现代、紧贴大语言模型的使用场景，OpenAI Moderation 或 Llama Guard 往往更合适；但在研究中，Perspective 依然很有参考价值。

### 三层审核模式

1. **输入审核（Input moderation）。** 在生成前对用户提示词分类，若被标记则拒绝。延迟成本是一次分类器调用。
2. **输出审核（Output moderation）。** 在交付前对模型输出分类，若被标记则替换为拒答。延迟成本是生成后的一次分类器调用。
3. **自定义审核（Custom moderation）。** 运行领域特定规则，例如正则表达式、允许列表和业务策略；既可用于输入，也可用于输出。

这三层在设计上按顺序执行：输入审核必须在生成前完成，输出审核则发生在生成之后。并行性主要体现在同一层内，例如同时运行多个分类器（OpenAI Moderation、Llama Guard、Perspective），以隐藏单个分类器的延迟。作为可选优化，系统也可以先显示占位响应（例如“请稍候，正在检查……”），等输入审核完成后再开始流式输出首个 token。内容被标记后的处理方式可以配置为拒答、净化，或升级给人工审核。

### 失败模式

- **只有输入审核。** 无法捕获输出幻觉；第 12～14 课中的编码攻击也可能绕过输入分类器。
- **只有输出审核。** 任何输入都能先到达模型，成本更高，而且可能向攻击者暴露更多内部推理。
- **只有自定义审核。** 正则表达式一类规则难以跨类别泛化，而且十分脆弱。

因此，分层审核才是默认方案，通过多道防线相互兜底。

### Azure Content Moderator 弃用

Azure Content Moderator 已于 2024 年 2 月弃用，并将在 2027 年 2 月退役。它的替代方案是 Azure AI Content Safety，后者基于大语言模型，并与 Azure OpenAI 集成。对于 Azure 生态中的部署团队，这是一项横跨 2024～2027 年的字段级迁移工程。

### 本课在第 18 阶段中的位置

第 16 课讨论红队场景中的审核工具；第 29 课讨论真正部署到产品中的审核系统；第 30 课则进一步收束到当前的双重用途能力证据。

```figure
an-moderation-layers
```

## 用它

`code/main.py` 会搭建一个三层审核工具：输入审核器（关键词 + 类别分数）、输出审核器（对输出运行同一个分类器）和自定义审核器（领域规则）。你可以输入不同内容，观察究竟由哪一层拦截。

## 交付成果

本课产出 `outputs/skill-moderation-stack.md`。给定一个具体部署，它会推荐审核栈配置：输入层和输出层分别使用什么分类器、加入哪些自定义规则，以及由什么评判器处理边缘情况。

## 练习

1. 运行 `code/main.py`。把无害、边界和有害三类输入依次送入三层审核系统，记录分别由哪一层触发。

2. 扩展这个工具，为某个具体类别加入类似 Perspective API 的毒性评分。比较其阈值行为与类别分数有何不同。

3. 阅读 OpenAI Moderation API 文档和 Llama Guard 3 的类别列表。把每个 OpenAI 类别映射到最接近的 Llama Guard 类别，并找出 3 个无法直接对应的类别。

4. 为一个代码助手部署（例如 GitHub Copilot）设计审核栈。指出哪些类别最相关、哪些最不相关，并提出自定义规则。

5. Azure Content Moderator 将于 2027 年 2 月退役。请为迁移到 Azure AI Content Safety 设计一份计划，并指出风险最高的环节。

## 关键词

| 术语 | 人们通常怎么说 | 实际含义 |
|------|-----------------|------------------------|
| OpenAI Moderation | “omni-moderation-latest” | 基于 GPT-4o 的分类器；文本覆盖 13 个类别，并提供部分多模态支持 |
| Perspective API | “Google Jigsaw 的毒性检测” | 大语言模型时代之前的毒性评分基线 |
| Llama Guard | “MLCommons 的 14 类分类法” | Meta 的危害分类器（v3：8B 文本模型、支持 8 种语言；v4：12B 多模态模型） |
| 输入审核（Input moderation） | “生成前过滤器” | 在调用模型之前对用户提示词进行分类 |
| 输出审核（Output moderation） | “生成后过滤器” | 在交付内容之前对模型输出进行分类 |
| 自定义审核（Custom moderation） | “领域规则” | 针对具体部署编写的规则，例如正则表达式、允许列表和策略 |
| 分层审核（Layered moderation） | “三层全上” | 标准的生产环境部署模式 |

## 进一步阅读

- [OpenAI Moderation API docs](https://platform.openai.com/docs/api-reference/moderations) - omni-moderation 端点
- [Meta PurpleLlama + Llama Guard](https://github.com/meta-llama/PurpleLlama) - Llama Guard 仓库
- [Google Jigsaw Perspective API](https://perspectiveapi.com/) - 毒性评分
- [Azure AI Content Safety](https://learn.microsoft.com/en-us/azure/ai-services/content-safety/) - Azure 替代方案
