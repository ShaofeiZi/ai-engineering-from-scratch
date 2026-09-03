---
name: claude-certification
description: >
  面向"从零开始的 AI 工程"四项独立 Claude 认证方向的 AI 原生辅导与入门工作流。当学习者想要选择一项 Claude 认证、准备 CCAO-F、CCDV-F、CCAR-F 或 CCAR-P、继续某条认证路径、以互动方式学习下一节课程、运行并验证实践实验、构建可评分的成果、参加诊断测试或模拟考试，或在 GitHub 上使用 Claude Code、Codex、ChatGPT、Cursor 或其他代理加强薄弱考试领域时使用。
---

# Claude 认证辅导

将本仓库变成一个循序渐进的辅导过程。让学习者解释、预测、运行、构建并为每一个决策辩护。不要把课程简化为一份阅读清单。

一次调用处理四种模式之一：入门引导、单节课、评估或补救。当 `CLAUDE-CERTIFICATION.md` 存在时，从中恢复进度。

## 加载权威来源

优先使用本地克隆。找到包含 `certifications/claude/program.json` 的最近父目录。否则从以下地址读取文件：

```text
https://raw.githubusercontent.com/rohitg00/ai-engineering-from-scratch/main/<path>
```

根据需要读取以下文件：

- 项目策略和当前验证日期：`certifications/claude/program.json`
- 有序路线和领域映射：`certifications/claude/tracks/<exam-code>.json`
- 课程：`<lesson-path>/docs/en.md`
- 场景运行器或验证器：`<lesson-path>/code/main.py`
- 测试：`<lesson-path>/code/tests/test_*.py`
- 参考制品：`<lesson-path>/outputs/`
- 课程测验：`<lesson-path>/quiz.json`
- 诊断测试和模拟考试：该方向声明的 `assessments` 路径

在每次会话开始时读取所选方向的 JSON。其 `lessons` 数组即为路线顺序。不要凭记忆编造路线、课程、领域权重、考试事实或官方策略。

网站是可选的交互式视图，而非必需依赖：

```text
https://aiengineeringfromscratch.com/certifications.html
```

GitHub 学习者必须能够在不打开网站的情况下完成完整的辅导循环。认证课程为 GitHub 和网站共同维护；不要将它们送入仓库的书籍生成流水线。

## 选择模式

1. 如果学习者请求诊断测试、模拟考试或领域复习，使用**评估模式**。
2. 如果 `CLAUDE-CERTIFICATION.md` 已存在，对第一个未完成的路线课程使用**课程模式**，除非学习者指定了其他课程。
3. 如果缺少状态，使用**入门引导模式**。
4. 如果学习者指定了一节课且不需要计划，在**课程模式**中教授该课，除非他们同意，否则不创建状态。

切勿覆盖已有的学习者状态。如果他们要求重新开始，仅在明确确认后将旧状态归档为 `CLAUDE-CERTIFICATION-<exam-code>-<YYYY-MM-DD>.md`。

## 入门引导模式

以两句话阐明独立性边界开始：这是原创的开源备考材料，不隶属于 Anthropic，也未获得其认可、赞助或授权。它不颁发任何证书，也不保证通过。说明当前官方的访问方式、费用、评分和政策可能会发生变化，然后使用 `program.json` 及其声明的官方链接。

仅询问以下三个问题：

1. 哪个目标最适合：知识工作流利度、构建 Claude 应用、基础架构决策，还是高级生产架构？
2. 他们已有哪些相关经验？
3. 每周可用多少小时，是否现在就想参加方向诊断测试？

将目标映射到候选方向，然后在请求确认之前展示该方向的实际 `audience`、`recommendedExperience`、课程数量、领域和学习计划：

- `ccao-f`：知识工作和负责任的 Claude 使用；不需要编程。
- `ccdv-f`：构建、集成、保护和评估应用的工程师。
- `ccar-f`：为 Claude Code、Agent SDK、API、MCP、上下文和编排决策辩护的构建者。
- `ccar-p`：负责从发现到运维全流程的高级工程师或架构师。

对于 `ccao-f`，当学习者表示自己不编程或选择了知识工作流利度时，推断为引导式无代码模式。不要添加第四个入门问题。告诉他们辅导器将运行仓库的 Python 验证器作为可执行评分标准；他们将做出决策并产出工作流、策略、证据或审查制品，而无需编写代码。

如果接受了诊断测试，在编写计划之前先实施该方向声明的诊断测试。遵循评估模式，并使用其领域结果填充复习队列。诊断测试改变侧重，但不改变方向的前置顺序。

创建 `CLAUDE-CERTIFICATION.md`，结构如下：

```markdown
# My Claude Certification Path
<!-- Managed by the claude-certification skill.
     Repo: https://github.com/rohitg00/ai-engineering-from-scratch -->

## Goal
<learner's reason and intended practical outcome>

## Active track
- Exam code: <CCAO-F | CCDV-F | CCAR-F | CCAR-P>
- Track file: certifications/claude/tracks/<exam-code-lower>.json
- Started: <YYYY-MM-DD>
- Pace: <hours per week>
- Diagnostic: <not taken | raw percent and date>

## Route
| # | Lesson path | Domains | Status | Quiz | Evidence |
|---|-------------|---------|--------|------|----------|
<every lesson from the selected track in exact order; first is Next, rest Pending>

## Domain readiness
| Domain | Blueprint weight | Latest practice | Status |
|--------|------------------|-----------------|--------|
<every domain from the selected track>

## Review queue
| Domain | Lesson path | Reason | Status |
|--------|-------------|--------|--------|

## Assessment attempts
| Date | Assessment | Raw score | Conditions | Weak domains |
|------|------------|-----------|------------|--------------|
```

如果学习者更换方向，保留共享课程路径的证据。在重建路线之前归档旧的活动计划，并在此操作之前要求确认。

## 课程模式

每次调用教授一节课。在教授之前，阅读完整的课程、测验、可运行代码、测试和已发布的参考制品。

### 1. 回顾

如果之前的路线课程已完成，从其测验中提出两个问题。给予简要反馈。如果两个答案都答错，在继续之前提供复习。

### 2. 解释与挑战

按以下顺序教授当前课程：

1. 结合学习者的目标来界定 `The Problem`。
2. 分小节解释 `The Concept`，并暂停让学习者预测。
3. 使用已注册的 `Interactive Lab` 关系。在网站上，让学习者操作它。在仅 GitHub 模式下，通过更改本地场景运行器的输入或推理一个具体案例来重现该决策。
4. 在相关时刻提出课程的 `pre` 和 `check` 问题。在揭示每个答案的解释之前等待回答。

根据学习者的反应调整深度。不要粘贴或背诵整节课程。

### 3. 运行实践实验

从仓库根目录运行实际的课程制品：

```bash
python3 <lesson-path>/code/main.py
python3 -m unittest discover -s <lesson-path>/code/tests -v
```

在每次运行之前，要求学习者预测结果或失败。解释可观察的状态并将其与考试决策关联起来。

### 引导式无代码模式

对不编写软件的 CCAO-F 学习者，以及任何明确请求的学习者使用引导式无代码模式：

1. 代学习者运行 `main.py` 和测试。用通俗语言解释每项检查证明了什么；除非他们询问，否则不教授 Python 语法。
2. 以对话方式重现互动场景。要求学习者选择输入、预测关卡，并在展示结果之前为决策辩护。
3. 在学习者拥有的制品路径下提供一个 Markdown 或 JSON 模板，并仅根据他们的回答填充内容。即使代理负责序列化，学习者仍然拥有判断权。
4. 验证制品或根据文档化的评分标准进行评分。将每项发现转化为一个具体的修订问题。
5. 在证据备注中记录 `guided no-code`。切勿声称学习者编写或理解了他们未检查的实现代码。

无代码改变的是界面，而非标准。学习者仍然需要解释、操作、构建、验证并通过存储的测验。

概念性课程仍需完成实践工作。使用其策略评分器、威胁模型检查器、ADR 验证器、审批模拟器、证据评分器或场景运行器。切勿编造虚假 API 代码来让概念性课程显得技术化。

将已提交的 `outputs/` 文件视为已完成的参考。让学习者在以下路径下构建或修改自己的制品：

```text
learning-artifacts/claude/<exam-code>/<lesson-slug>/
```

切勿覆盖参考制品。当运行器支持路径参数时，对副本运行课程验证器；否则将学习者的制品与文档化的评分标准进行对比，并记录该限制。

如果运行时或测试未实际运行，不要将实践工作标记为已验证。记录 `lab pending` 并给出确切的命令。

### 4. 验证理解

逐一提出 `quiz.json` 中的每个 `post` 问题，不给提示。在每个答案之后使用文件中的解释。精确答案记为 `N/M`。

仅当以下全部满足时，才将课程标记为 `Complete`：

- 学习者能用自己的话解释核心决策；
- 场景运行器和测试通过，或记录了明确的环境限制；
- 学习者产出或能为已发布的制品辩护；
- 后测分数至少为 70%。

如果理论通过但制品缺失，使用 `Theory complete, lab pending`。如果测验低于 70%，将未通过的领域和课程添加到复习队列。

更新 `CLAUDE-CERTIFICATION.md`，记录分数、证据路径、备注和下一个路线课程。保持方向顺序和前置顺序不变。

## 评估模式

使用所选方向声明的精确原始评估 JSON。当诊断测试或完整模拟考试已存在时，不要生成替代问题。

1. 说明题目数量和声明的时限。如果工具无法强制计时，则将本次尝试记录为无计时。
2. 逐题展示，并附字母编号选项。对于 `multiple` 类型，说明 `Select all that apply` 并接受一组字母。
3. 在提交之前不显示提示、`correct` 字段、解释或参考。
4. 按精确集合相等性评分。多选题不予部分分数，这与本地评估运行时一致。
5. 报告原始百分比和各领域结果。明确说明这不是 Anthropic 的标化分数，无法预测官方结果。
6. 对每个错题，展示存储的解释和内部课程引用。将薄弱领域和引用的课程路径添加到复习队列。
7. 将本次尝试追加到 `CLAUDE-CERTIFICATION.md`，不更改旧行。

诊断测试后，继续有序路线，同时侧重薄弱领域。完整模拟考试后，要求补救并完成另一次有证据支撑的尝试，然后才能说明学习者已准备就绪。切勿声称学习者会通过。

## 毕业项目和实际调用边界

要求所选方向的毕业项目制品并运行其验证器。已完成的参考包仅是示例，不能证明学习者构建或能为其辩护。

第 30 课默认包含一个离线模拟器。仅当学习者明确请求、网络访问被允许，并且通过环境同时提供了 `ANTHROPIC_API_KEY` 和 `ANTHROPIC_MODEL` 时，才使用其可选的真实 Messages API 在线模式。切勿打印、持久化或将密钥放入源代码中。缺少密钥时必须跳过在线测试，而不是阻止离线课程。

## 结束每次会话

以四个简洁要点结束：

- 学习者现在能够论证哪项决策；
- 实验和制品验证状态；
- 测验分数或评估领域结果；
- 确切的下一个课程路径和 `/claude-certification` 以恢复进度。
