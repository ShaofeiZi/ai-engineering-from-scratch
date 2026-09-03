# 技能调用与路由

> 调用先是一项权限决策，随后才是一项相关性决策。好的描述帮助模型作出选择；好的策略决定是否允许这一选择。

**Type:** 构建
**Languages:** Python (stdlib)
**Prerequisites:** 第 13 阶段 · 第 24 课（技能发现与渐进式披露）
**Time:** 约 105 分钟

## 学习目标

- 区分用户显式调用、模型隐式调用、应用调用和技能间调用。
- 将人类可见性与模型可选资格建模为彼此独立的策略维度。
- 编写同时包含正向触发条件与近似但不匹配边界的路由描述。
- 在追踪记录和测试中区分资格判断、选择、激活、参数绑定与执行。
- 适配运行时特有的调用字段，同时不把它们冒充为可移植 frontmatter。

## 问题

你安装了一个 `database-migration` 技能。用户可以按名称运行它，但模型也会看到它的描述，并在有人询问一般数据库问题时选择它。于是，这个技能为一个原本只需要解释的问题提出了模式变更建议。

你添加 `user-invocable: false`，希望阻止用户手动运行它；另一个运行时却忽略了这个字段。你又添加 `disable-model-invocation: true`，希望让该技能彻底消失；但在理解这个字段的运行时里，用户仍可显式调用它。

字段名称本身没有问题，错误在于背后的模型。“用户能看到它”“模型能选择它”“应用能预加载它”和“其中的工具能执行”是彼此独立的事实。一个名为 `invocable` 的布尔值无法表达这些差异。

路由还有第二种失败模式。描述含糊时，多个技能看起来都可能适用；描述堆满关键词时，无关任务也会触发它们。目录是一种概率性接口：既要足够紧凑以装入上下文，也要足够具体以正确路由。

## 概念

### 五种通道都能启动生命周期

| 参与者 | 调用形式 | 典型用途 | 主要风险 |
|---|---|---|---|
| 人类用户 | 在 UI 或提示中点名技能 | 有意选择工作流 | 用户预期的可用性或权限超出宿主实际授予范围 |
| 模型或自主智能体 | 根据任务上下文选择目录项 | 自动采用专家流程 | 误报路由 |
| 应用 | 通过运行时代码激活或预加载技能 | 固定的产品工作流 | 与某个宿主形成隐藏耦合 |
| 另一个技能或子智能体 | 请求一个准确技能作为工作流依赖 | 组合 | 循环、依赖缺失或上下文泄漏 |
| 评估框架 | 在固定场景下激活准确技能 | 可重复测量 | 测试技能时意外绕过了待评估的生产策略 |

可移植 Agent Skills 规范定义的是包格式。它并未标准化一种通用的斜杠命令 UI、隐式路由标志、应用 API 或子智能体生命周期。

### 调用的五个阶段

```figure
skill-invocation-stages
```

请准确使用以下术语：

- **具备资格（Eligible）**：策略允许这个参与者请求该技能。
- **已选择（Selected）**：用户点名了该技能，或路由器判断它与任务相关。
- **已激活（Activated）**：技能指令已经进入工作上下文。
- **执行中（Executing）**：智能体已经根据这些指令开始模型或工具工作。
- **已完成（Completed）**：输出通过了独立的成功检查。

只记录 `skill_used=true` 的追踪信息，会掩盖故障究竟发生在哪个边界。

### 人类调用与模型调用构成 2×2 矩阵

| 人类可调用 | 模型可调用 | 模式 | 适用示例 |
|:---:|:---:|---|---|
| 是 | 是 | 共享 | 代码解释、测试规划、文档审阅 |
| 是 | 否 | 仅限人类 | 发布准备、账单导出、破坏性清理计划 |
| 否 | 是 | 仅限模型 | 内部风格指南、领域参考、自动支持流程 |
| 否 | 否 | 已禁用或仅限应用 | 分阶段发布、已弃用包、程序化预加载 |

这个矩阵是一种策略模型，不是标准 YAML。

目前有一种宿主使用 `disable-model-invocation: true` 表示“仅限人类”这一行，使用 `user-invocable: false` 表示“仅限模型”这一行，默认则是两者都允许。另一种宿主使用 `agents/openai.yaml`，并通过其中的 `allow_implicit_invocation: false` 保留显式调用，同时禁用隐式选择。这些都是运行时适配器；未知宿主可能会忽略它们。

这个容易混淆的细节非常重要：`user-invocable: false` 并不意味着“模型不能使用此技能”，它只会在定义该字段的宿主中移除用户直接调用。`disable-model-invocation: true` 也不意味着“技能已禁用”，它只移除模型发起的选择，同时保留用户显式访问。

### 显式调用以身份为先

显式调用会直接提供技能身份：

```text
/release-readiness v2.4.0
```

或者：

```text
release-readiness check v2.4.0 without publishing
```

当前 Codex 接口文档说明，可以用 `/skills` 进行选择，也可以在请求中直接写出技能名称来显式调用。Claude Code 文档则说明了 `/skill-name` 和宿主特有的参数展开。确切语法、菜单可见性、引号规则与变量展开都属于宿主行为。

显式请求仍然必须通过策略检查。点名技能不能绕过缺失的权限、工作区约束、审批门或运行时隔离。

### 隐式调用以描述为先

在隐式路由中，模型最初看到的是目录元数据，而非技能全文。因此，描述就是技能的路由接口。

较弱的描述：

```yaml
description: Helps with releases.
```

过于宽泛的描述：

```yaml
description: Use for release, version, package, build, deploy, publish, tag, changelog, GitHub, CI, or software tasks.
```

边界明确的描述：

```yaml
description: Inspect an already prepared release candidate and produce a readiness report. Use when the user asks whether a version, tag, package, or image is ready to publish; do not use for ordinary build failures or feature development.
```

边界明确的版本包含四项信息：

1. **能力：** 检查已准备好的候选版本。
2. **输出：** 就绪度报告。
3. **正向边界：** 用户询问发布产物是否已准备就绪。
4. **负向边界：** 普通构建和功能开发不在范围内。

当两个相近技能共享词汇时，负向边界非常有用，但不能代替近似但不匹配用例的评估。

### 路由是一种带弃权选项的分类

对于技能 `s` 和请求 `x`，可以设想路由器计算如下分数：

```text
score(s, x) = capability_match + trigger_match + context_match - exclusion_match - ambiguity_penalty
```

实际评分可能由 LLM 决策，而不是算术运算，但工程原则不变：选择结果应超过阈值，也应胜过竞争技能。证据不足时，应当弃权。

```figure
skill-routing-abstention
```

对于高影响技能，即使描述非常出色，隐式路由也可能并不合适。当误报成本高于自动选择带来的便利时，应采用仅限人类的策略。

### 必须先判断资格，再进行排名

不要对所有已发现技能评分、选出最强匹配，然后才检查该技能的策略。这样一来，被禁止的最高匹配项会错误地阻止分数稍低但有资格的候选项进入考虑范围。

隐式路由应按以下顺序进行：

1. 根据请求参与者和当前宿主适配器，筛选已发现技能。
2. 只对具备资格的候选项评分。
3. 如果最强的合格匹配超过阈值并满足歧义规则，就选择它。
4. 没有候选项具备资格，或合格项分数都不够高时，选择弃权。

假设 `incident-triage` 得分为 `0.80`，但其宿主扩展禁用了模型调用；`incident-review` 得分为 `0.55`，且允许模型调用。路由器应把 `incident-review` 作为最佳合格候选项进行评估，而不是选择 `incident-triage`、拒绝它，然后停止。

这种顺序也能防止策略变更改变相关性分数的含义。资格定义候选集合，相关性只负责对该集合排序。

### 路由评估需要近似但不匹配用例

正向用例用于证明召回能力：

```json
{"prompt":"Is version 2.4.0 ready to publish?","expected":"release-readiness"}
```

明显的负向用例用于证明基本精确率：

```json
{"prompt":"Explain rotary position embeddings.","expected":null}
```

近似但不匹配用例用于暴露边界质量：

```json
{"prompt":"Why did today's package build fail?","expected":"build-diagnostics"}
```

这个近似用例与发布技能共享 `package` 和 `build` 两个词，但实际属于另一个工作流。只包含明显正例与毫不相关负例的路由集合，会夸大质量。

### 参数有三种表示形式

调用参数会跨越多个边界：

```figure
skill-argument-boundaries
```

在每个边界，都应保留意图，同时不要把文本当作代码。

- 宿主解析器决定命令语法与引号规则。
- 技能按照宿主规则接收已经绑定的文本或变量。
- 指令负责验证必需值与默认值。
- 工具调用把值转换为类型化模式，并再次验证。

不要把原始参数插值进 shell 命令。应优先使用通过参数向量调用的脚本，或类型化 MCP 工具。

### 应用调用属于显式编排

当产品工作流已经知道任务类型时，应用可以直接激活技能。例如，拉取请求审阅服务可以在用户按下 Review 后预加载 `pull-request-risk-review`。

这样可以消除路由不确定性，但会产生对运行时 API 的依赖。应把适配器放在可移植正文之外：

```figure
skill-host-adapter
```

换用其他兼容客户端打开技能时，技能本身仍应清晰易懂。

### 技能间调用是一条类似工具调用的边

假设依赖文件发生变化时，`release-readiness` 会请求 `security-change-review`。

调用方应提供：

- 目标技能身份；
- 有边界的任务与产物路径；
- 预期响应契约；
- 调用理由；
- 目标不可用时的回退方案；
- 最大深度或循环规则。

```json
{
  "target_skill": "security-change-review",
  "task": "Review dependency changes in the candidate diff",
  "inputs": ["artifacts/release.diff"],
  "expected": "risk-report.json",
  "max_depth": 2
}
```

不能简单地把第二个技能粘贴进第一个技能。宿主负责决定如何激活它、是否共享上下文、是否在分支中运行，或是否通过工具结果返回。

### 上下文生命周期取决于宿主

技能激活后，其正文可能留在对话中，也可能在上下文压缩时被总结，或者在委派的上下文中运行。工具许可可能只持续一轮，而指令会保留更久。子智能体可能收到技能，却拿不到父级的完整历史。

不要编写依赖不可见生命周期假设的技能。应把持久输出放入文件或类型化状态，确保重新进入时安全，并说明中断后必须重新加载哪些内容。

```markdown
On resume, read `artifacts/release-readiness.json` if it exists.
Revalidate the candidate commit before continuing.
Do not repeat an external write whose idempotency key is already recorded.
```

## 构建它

`code/main.py` 将策略与路由实现为彼此分离的适配器。

该模型包括：

- `Actor`：表示人类、模型、自主智能体、应用、技能和评估框架调用方；
- `SkillMetadata`：表示路由身份；
- `InvocationPolicy`：表示人类/模型矩阵；
- `InvocationRequest` 和 `InvocationDecision`：表示可追踪的输入与结果；
- `CorePolicyAdapter`：表示不使用宿主扩展的可移植行为；
- `ExtensionPolicyAdapter`：表示已识别的运行时字段；
- `build_invocation_matrix(policy)`：生成 2×2 视图；
- `route_request(skills, request, adapter)`：先筛选资格，再完成相关性排名、选择和拒绝。

运行方法：

```bash
cd phases/13-tools-and-protocols/25-skill-invocation-and-routing
python3 code/main.py
python3 -m unittest discover -s code/tests -v
```

演示会打印一份矩阵，以及显式人类调用、隐式模型调用、自主智能体调用、应用调用、技能组合调用和评估框架调用的决策。扩展适配器的结果会展示：先移除被禁止但词法匹配最高的候选项，再对有资格的替代项进行排名。演示还包含精确名称允许名单，无需模型 API。确定性路由器用于让策略边界可供检查，并不是要声称词法匹配能复现生产环境中的模型路由。

### 为什么核心适配器与扩展适配器要分开

如果一个解析器为观察到的所有 frontmatter 字段都赋予含义，它就会在不知不觉中把运行时约定冒充成标准。拆分适配器会迫使调用方明确指出当前启用的是哪一种宿主语义。

`CorePolicyAdapter` 只使用应用提供的策略。`ExtensionPolicyAdapter` 识别一组显式列出的宿主字段，并记录是哪个字段改变了决策。

## 使用它

发布技能之前，先编写调用契约：

```yaml
actors:
  human: allow
  model: deny
  application: allow
  skill: deny
explicit_name: release-readiness
arguments:
  candidate: required
  publish: fixed_false
ambiguity: ask_user
missing_dependency: stop
context:
  durable_state: artifacts/release-readiness.json
  max_composition_depth: 2
```

这份契约是适配器与测试的设计文档。除非标准明确采纳，否则它并不是可移植的 `SKILL.md` frontmatter。

## 交付成果

本课会生成 `skill-invocation-router` 软件包。它包含调用模型参考、宿主策略示例和一个不执行实际操作的 CLI。该 CLI 会评估一次人类、模型、自主智能体、应用、技能组合或评估框架请求，并返回包含通道、适配器、分数与理由的 JSON 决策。

单请求 CLI 是一个策略探针，不是完整的触发评估。请使用第 27 课中带标签的正向用例与近似但不匹配用例设计，计算混淆计数、精确率、召回率和重复运行稳定性。

## 练习

1. 创建人类/模型矩阵的全部四行，并为每一行编写一个合理用例。
2. 为 `CorePolicyAdapter` 添加仅限应用的激活方式，并证明人类和模型调用方仍会被拒绝。
3. 为部署技能编写十个近似但不匹配用例。每条提示都必须与该技能共享词汇，但实际属于另一个工作流。
4. 在排名前两位的路由分数之间添加歧义差值。差值过小时返回 `ask`。
5. 为技能间请求添加最大组合深度，并检测由两个技能组成的循环。
6. 使用核心适配器和扩展适配器分别运行同一组带标签数据，并解释每一项决策变化。

## 关键术语

| 术语 | 人们通常怎么说 | 实际含义 |
|---|---|---|
| 显式调用 | “斜杠命令” | 参与者直接提供技能身份，但仍受策略约束 |
| 隐式调用 | “由模型选择” | 路由器根据任务上下文，从具备资格的目录元数据中选择 |
| 用户可调用 | “人类可以使用” | 宿主特有的菜单或直接调用属性，并非核心字段 |
| 模型可调用 | “智能体可以使用” | 在宿主策略下参与隐式模型选择的资格 |
| 调用适配器 | “Frontmatter 解析器” | 把宿主字段和 API 映射到已声明策略模型的代码 |
| 近似但不匹配用例 | “困难负例” | 与技能预期输入相似、但不应触发技能的请求 |
| 弃权 | “未选择技能” | 证据缺失或含义模糊时，有意作出的路由结果 |

## 延伸阅读

- [优化技能描述](https://agentskills.io/skill-creation/optimizing-descriptions)：了解正向触发、具体性与评估。
- [评估技能](https://agentskills.io/skill-creation/evaluating-skills)：了解触发评估与输出评估的设计。
- [OpenAI：构建技能](https://learn.chatgpt.com/docs/build-skills)：了解当前 Codex 的显式与隐式调用控制。
- [Claude Code 技能](https://code.claude.com/docs/en/skills)：了解一种宿主的 `user-invocable`、`disable-model-invocation`、参数与委派上下文。
