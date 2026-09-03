# Agent Skill：可移植契约与运行时边界

> Skill 并不是换了一个更好文件名的长 prompt。它是一个可发现的软件包，由指令、资源和可执行辅助程序组成，并通过运行时契约进入智能体上下文。

**Type:** 构建
**Languages:** Python (stdlib)
**Prerequisites:** 第 13 阶段 · 第 01 课（工具接口）、第 13 阶段 · 第 05 课（工具 Schema 设计）
**Time:** 约 90 分钟

## 学习目标

- 定义 Agent Skill，并且不把它与 prompt、仓库指令、工具、hook、子智能体或插件混为一谈。
- 阅读可移植的 `SKILL.md` 契约，并将其与运行时专属扩展区分开。
- 将发现、选择、激活、资源加载、工具使用和验证解释为不同的生命周期阶段。
- 在运行时把 Skill 软件包放入智能体目录前，对其进行验证。
- 针对具体任务，在 Skill、MCP 工具、hook、子智能体和普通代码之间做出选择。

## 十分钟获得首次成功

请先完成这里的操作，再阅读后面的长篇解释。你将创建一个小型
Skill，把完整的审查器软件包安装到真实智能体宿主中，调用它，验证
结果，然后卸载它。这样便能通过可观察结果证明整个生命周期。

### 真实宿主实验的预检

真实宿主检查点需要 Node.js、`npx`、Python 3、一个选定的
支持 Skill 的宿主，以及安装程序所选项目或用户作用域的写入权限。
先验证本地命令：

```bash
node --version
npx --version
python3 --version
```

安装前先确定要使用的宿主和作用域。如果缺少任一要求，可以在网站上
阅读本课，或继续完成下面的手动软件包练习。该回退方案会讲解契约，
但无法证明宿主发现、调用、捆绑脚本执行或卸载行为。请把这些观察项
保留为待验证状态。

### 1. 从空工作目录开始

在任何用于存放学习项目的父目录中运行以下命令：

```bash
mkdir -p agent-skills-first-run
cd agent-skills-first-run
TARGET_ROOT="$(pwd -P)"
printf 'TARGET_ROOT=%s\n' "$TARGET_ROOT"
ls -A
```

最后一条命令应该没有任何输出。如果它列出了文件，请换一个空目录，
让审查范围保持清晰。

为你的第一个 Skill 创建目录：

```bash
mkdir -p my-first-skill
```

创建 `my-first-skill/SKILL.md`，内容如下：

```markdown
---
name: my-first-skill
description: Turn rough meeting notes into a compact decision record when the user asks to capture a technical decision.
---

# Decision record

Extract the decision, context, alternatives, owner, and next review date.
If the notes do not contain a decision, ask one clarifying question instead
of inventing one.
```

验证文件确实创建在预期目录：

```bash
test -f my-first-skill/SKILL.md
```

没有输出且退出码为 0，表示文件存在。

### 2. 安装完整的审查器软件包

保持在 `agent-skills-first-run` 中并运行：

```bash
npx skills add rohitg00/ai-engineering-from-scratch --skill skill-contract-reviewer --full-depth
```

选择你正在使用的智能体宿主和作用域。安装程序应列出
`skill-contract-reviewer` 及其写入目标。必须使用 `--full-depth`，
因为本课的 Skill 是一个嵌套软件包，包含参考资料、脚本和资源文件。

把 `SKILL_ROOT` 设为安装程序报告的绝对目录。它必须是包含已安装
`SKILL.md` 的目录，不能是课程源码目录，也不能是当前工作区：

```bash
# Replace the placeholder with the destination printed by the installer.
SKILL_ROOT="$(cd "/absolute/path/to/skill-contract-reviewer" && pwd -P)"
test -f "$SKILL_ROOT/SKILL.md"
printf 'SKILL_ROOT=%s\n' "$SKILL_ROOT"
```

如果智能体会话此前已经打开，请启动新会话，或使用该宿主的 Skill
重新扫描命令。不要假设所有宿主都会热重载目录。

### 3. 显式调用

在已经安装该软件包的智能体中，将 `agent-skills-first-run` 设为工作
目录，并使用该宿主支持的语法：

| 宿主 | 显式调用方式 |
|---|---|
| Codex | 输入 `skill-contract-reviewer`，或从 `/skills` 中选择它，然后提供审查请求 |
| Claude Code | 输入 `/skill-contract-reviewer`，后接审查请求 |
| 可移植回退 | 输入 `Use skill-contract-reviewer to review the target package.` |

在请求中使用打印出来的 `SKILL_ROOT` 和 `TARGET_ROOT` 绝对路径。
要求宿主先展开它们再执行，并展示完整解析后的命令，不要提供依赖进程
工作目录的命令：

```text
Use skill-contract-reviewer to review <TARGET_ROOT>/my-first-skill. The installed bundle root is <SKILL_ROOT>. Run python3 <SKILL_ROOT>/scripts/check_skill.py <TARGET_ROOT>/my-first-skill. Before running it, show the fully resolved argv. Return the validation report, selected primitives, and one sentence for each selection. Include the resolved script path, resolved target path, cwd, argv, and exit code as execution evidence.
```

解析后的命令应具有以下形状，且不再包含任何占位符：

```bash
python3 "/absolute/install/path/skill-contract-reviewer/scripts/check_skill.py" \
  "/absolute/workspace/path/agent-skills-first-run/my-first-skill"
```

成功结果必须同时具备以下三个属性：

1. 宿主能按名称找到 `skill-contract-reviewer`。
2. 审查器会读取软件包契约并运行其捆绑的验证器。
3. 响应包含验证报告；样例没有结构错误，并且原语选择有充分依据。

执行证据还必须列出脚本路径、目标路径、cwd、确切的参数向量和退出码。
仅有一份行文流畅、却缺少这些字段的报告，并不能证明已安装的配套脚本
确实运行过。

如果宿主报告该 Skill 不可用，请检查安装目标，重新扫描或重启一次，
再重试显式请求。不要通过改写 Skill 描述掩盖安装失败。

### 4. 探测隐式选择

开始一个全新的智能体轮次，不点名该 Skill，输入同一个任务：

```text
Review <TARGET_ROOT>/my-first-skill as a reusable agent package and tell me whether its package contract is valid.
```

如果宿主会展示已选 Skill，请记录它是否选择了
`skill-contract-reviewer`。如果宿主不展示路由过程，则把隐式选择标记为
未验证。显式调用是可移植的回退方式。

### 5. 清理

只删除已经安装的审查器软件包：

```bash
npx skills remove skill-contract-reviewer
```

选择安装时使用的同一个宿主和作用域。重新扫描或开启新会话后，显式
请求 `skill-contract-reviewer` 应报告它不可用。保留
`my-first-skill` 供后续课程使用，也可以在完成本学习路径后删除整个
实验目录。

## 问题

假设你的团队有一套可靠的发布工作流。它会查找已合并变更、检查迁移说明、更新变更日志、运行打包命令，并生成审查清单。

把整套工作流塞进一个 prompt，复制起来容易，真正运行起来却很困难。这个 prompt 没有稳定身份、发现规则、资源边界或可测试的软件包结构，也无法回答几个基本问题：谁可以调用它？模型应在何时选择它？它可以运行哪些脚本？哪些文件可信？上下文压缩之后还能保留什么？

另一个极端是把所有可复用指令都视为 Skill。仓库约定、确定性自动化、外部工具、事件 hook 和被委派的智能体解决的是不同问题。把它们全塞进 `SKILL.md`，只会得到一个表面上可移植、实际上依赖某个宿主未公开行为的目录。

第一项工程任务是分类。先判断产物究竟是什么，再决定如何打包。

## 概念

### Skill 封装过程性知识

Agent Skill 是一个以 `SKILL.md` 为入口的目录。入口文件包含 YAML frontmatter，后接 Markdown 指令。目录还可以包含 reference、script 和 asset。

```figure
skill-package-anatomy
```

可部署单元是整个目录，而不只是 Markdown 文件。即使 frontmatter 能够解析，缺少配套引用文件的 `SKILL.md` 副本仍然是一个损坏的软件包。

### 相邻抽象

| 产物 | 主要职责 | 加载或运行时机 | 不应冒充的对象 |
|---|---|---|---|
| Prompt | 塑造一次模型交互 | 由应用或用户包含进上下文 | 带资源的版本化软件包 |
| 仓库指令 | 说明一个代码库的常驻规则 | 编码运行时进入该作用域 | 可复用的任务工作流 |
| Agent Skill | 提供可复用的过程性知识 | 显式或隐式激活 | 硬性授权边界 |
| MCP 工具 | 暴露有类型的远程能力 | 模型或应用调用它时 | 详细的操作流程 |
| Hook | 在事件发生时运行确定性逻辑 | 声明的事件发生时 | 概率式模型路由 |
| 子智能体 | 使用独立上下文与状态委派工作 | 编排器创建或调用它时 | 静态指令包 |
| 插件 | 分发更大的运行时扩展 | 宿主安装或启用它时 | 可移植 Skill 契约本身 |
| 学习型 Skill 库 | 保存通过经验发现的行为 | 策略检索先前程序或轨迹时 | 基于标准的 `SKILL.md` 软件包 |

发布 Skill 可以告诉智能体如何检查一次发布。MCP 服务器可以暴露发布注册表。hook 可以禁止直接推送。子智能体可以独立审计候选版本。这些组件之所以能够组合，是因为它们各自承担不同职责。

### “Skill”一词指代两种不同概念

研究系统有时也会把学习得到的程序、成功轨迹或环境专属策略片段称为 Skill。智能体可以在探索过程中创建这些产物，按任务相似性检索和执行，再根据反馈修订库。阶段 14 · 10 会构建这种终身学习库。

本小型学习路径中的 Agent Skill 与之不同。它是一个由作者编写的软件包，拥有明确的文件系统契约、目录元数据、渐进式披露、运行时介导的调用和宿主控制的工具。它可以由智能体生成或改进，但这种格式并不要求经过学习。

| 维度 | Agent Skill 软件包 | 学习型 Skill 库 |
|---|---|---|
| 主要单元 | `SKILL.md` 目录 | 程序、策略、轨迹或记忆记录 |
| 创建方式 | 编写、生成或策展 | 通常从环境经验中发现 |
| 选择方式 | 目录描述加运行时策略 | 根据任务状态进行检索或决策 |
| 执行方式 | 模型遵循指令并调用宿主工具 | 环境运行已保存的行为或代码产物 |
| 可移植性 | 软件包契约可以跨兼容宿主 | 通常绑定到一种环境和动作空间 |
| 评估方式 | 路由、产物、安全性和宿主兼容性 | 奖励、成功率、迁移能力和库增长 |

两种概念都在封装可复用能力。不能仅仅因为名称相同，就让它们共享实现层面的断言。

### 可移植核心

Agent Skills 规范要求两个 frontmatter 字段：

```yaml
---
name: release-readiness
description: Inspect a release candidate when the user asks whether a version is ready to publish.
---
```

`name` 是稳定标识符。它必须满足规范中的命名规则，并与父目录名称一致。`description` 同时承担文档说明与路由元数据的职责，应说明该 Skill 做什么以及何时适用。

可移植的可选字段如下：

| 字段 | 用途 | 可移植性说明 |
|---|---|---|
| `license` | 声明软件包的许可条款 | 核心规范 |
| `compatibility` | 声明环境要求 | 核心规范 |
| `metadata` | 携带字符串值的扩展数据 | 核心规范 |
| `allowed-tools` | 建议预先批准的工具 | 实验性；宿主支持不一 |

Markdown 正文承载操作指令。它应定义工作流、决策点、失败行为，以及指向配套资源的直接路径。

```markdown
# Release readiness

Use this workflow for a release candidate, not for ordinary development builds.

1. Read `references/release-policy.md`.
2. Run `python3 scripts/inspect_release.py --format json`.
3. Stop if the report contains a blocking failure.
4. Produce the checklist from `assets/release-checklist.md`.
5. Ask for approval before any publish or tag action.
```

### 运行时扩展是第二层

部分宿主接受额外 frontmatter 或配套配置。这些字段可能很有用，但不会自动具备可移植性。

| 行为 | 宿主扩展示例 | 属于可移植核心？ |
|---|---|:---:|
| 对模型路由隐藏 Skill，但保留用户直接调用 | `disable-model-invocation` | 否 |
| 从用户命令菜单隐藏 Skill，但允许模型路由 | `user-invocable` | 否 |
| 在命令菜单中显示参数帮助 | `argument-hint` | 否 |
| 在委派上下文中运行 Skill | `context`、`agent` | 否 |
| 固定模型或推理设置 | `model`、`effort` | 否 |
| 注册生命周期自动化 | `hooks` | 否 |
| 在 Codex 中禁用隐式调用 | `agents/openai.yaml` policy | 否 |

把每种扩展都视为适配器。确保核心工作流在没有扩展时仍然有效，记录回退方式，并测试实际消费它的宿主。运行时可能忽略未知字段、拒绝它，也可能保留它却不实现相应行为。

### Frontmatter 是可执行元数据

元数据会在读取 Skill 正文前改变系统行为。

- 格式错误的 `name` 会让发现失败。
- 含糊的 `description` 会把错误请求路由过来。
- 仅限人工调用的标志会把 Skill 从模型目录中移除。
- 工具许可可能改变宿主是否请求批准。
- 上下文设置可能把执行转移到另一个智能体会话。

应像审查配置代码一样审查 frontmatter：验证它、对它做版本控制，并在 eval 中覆盖它的行为。

### Skill 生命周期

```figure
skill-runtime-lifecycle
```

每条箭头都是一处边界，拥有各自的故障模式。

1. **发现（Discovery）**在已配置的位置中查找候选软件包。
2. **验证（Validation）**在发布到目录前拒绝格式错误或不安全的软件包。
3. **编目（Cataloging）**只暴露精简的 `name` 和 `description`，而不是完整软件包。
4. **选择（Selection）**判断该 Skill 是否相关。
5. **激活（Activation）**把正文加载到模型可见上下文中。
6. **披露（Disclosure）**只在某个分支确实需要时读取 reference 或 asset。
7. **执行（Execution）**在宿主的权限与隔离规则下使用宿主工具。
8. **验证结果（Verification）**独立检查产出的结果，而不依赖模型自己的成功声明。

混淆这些阶段会形成错误的心智模型。已发现的 Skill 不等于已激活。已激活的 Skill 不等于有权执行它描述的一切。获准的工具调用也不能证明结果正确。

### Skill 与工具彼此正交

MCP 回答：“该应用可以调用哪些能力，它们的 schema 是什么？”Skill 回答：“智能体应如何处理这一类任务？”

```figure
skill-tool-orthogonality
```

Skill 可以点名某个工具，但真正的能力注册表归宿主所有。如果工具不存在，Skill 应明确给出回退方案或清晰失败。它绝不能暗示：只要在文字中写出一个能力，该能力就会凭空出现。

### Skill 与仓库指令作用域不同

仓库指令描述你当前所处环境：命令、约定、生成文件与边界。Skill 提供可跨多个仓库复用的任务流程。

两者同时适用时，当前用户请求与仓库规则会约束 Skill。通用重构 Skill 不得推翻仓库中禁止编辑生成文件的规则。

### Skill 不会互相导入

一个 Skill 可以指示智能体调用另一个 Skill，但这不是语言级 import。第二个 Skill 仍然要经过运行时发现、资格判断、激活、权限和上下文处理。

应把跨 Skill 依赖写成可观察的工作流边：

```markdown
After producing the candidate changelog, invoke the `release-risk-review` skill.
Pass the candidate path and require a blocking or non-blocking verdict.
If that skill is unavailable, stop and report the missing dependency.
```

这样依赖便可以测试，宿主也有机会执行策略。

## 构建它

`code/main.py` 实现了一个面向标准的小型验证器和一个产物选择器。它只依赖标准库，因此每条规则都清晰可见。

验证器暴露：

- `parse_frontmatter(text)`，用于分离元数据与正文。
- `validate_skill_text(text, directory_name, allowed_runtime_extensions=())`，用于检查必填字段、命名、未知扩展、正文是否存在以及可移植限制。
- `ValidationIssue` 和 `SkillReport`，用于返回结构化证据，而不是一个不透明的布尔值。
- `FrontmatterSyntaxError`，表示无法安全解释的输入。

选择器暴露 `TaskShape` 和 `select_primitives(task)`。它根据任务需求选择普通代码、仓库指令、Skill、hook、子智能体或 MCP 工具。

运行实验：

```bash
cd "$(git rev-parse --show-toplevel)"
cd phases/13-tools-and-protocols/22-skills-and-agent-sdks
python3 code/main.py
python3 -m unittest discover -s code/tests -v
```

该命令块要求本地克隆，并且可以从克隆内的任何目录开始，
这样 `git rev-parse --show-toplevel` 才能解析仓库根目录。

演示会打印以下内容的 JSON：一个有效的可移植 Skill、一个带宿主扩展的 Skill、一个无效软件包，以及多组任务形状决策。请检查问题代码。软件包验证器应告诉作者如何修复产物，而不是替作者猜测意图。

### 验证顺序很重要

先验证成本低的结构事实，再验证更深层的内容规则：

```figure
skill-validation-order
```

这一顺序可防止次生错误遮蔽第一个被破坏的不变量。

## 使用它

编写 Skill 前，先填写这张决策卡：

| 问题 | 如果回答“是” | 可能使用的原语 |
|---|---|---|
| 是否需要在多个步骤中复用模型判断？ | 流程稳定，但决策会变化 | Skill |
| 是否必须在每次事件触发时执行？ | 漏执行一次都不可接受 | Hook 或应用代码 |
| 模型是否需要一个带类型输入的外部能力？ | 操作位于模型上下文之外 | 工具或 MCP 服务器 |
| 工作是否需要隔离的上下文、状态或所有权？ | 单独的 worker 返回有边界的结果 | 子智能体 |
| 指引是否只适用于一个仓库？ | 它描述本地命令与约束 | 仓库指令 |
| 一次交互是否足够？ | 不需要软件包生命周期 | Prompt |

许多生产工作流会同时命中多行。该决策卡可以防止某一种产物假装提供所有属性。

## 交付它

本课产出 `skill-contract-reviewer` 软件包，位于 `outputs/` 下，其中包含：

- 用于审查候选 Skill 软件包的可移植 `SKILL.md`；
- 可移植契约和原语选择的参考检查清单；
- 确定性的验证脚本；
- 覆盖 prompt、Skill、工具、hook、普通代码和子智能体的任务形状夹具。

请安装完整软件包，而不是只安装入口文件：

```bash
cd "$(git rev-parse --show-toplevel)"
python3 scripts/install_skills.py /tmp/aiefs-skills --phase 13 --type skill
```

课程安装程序会报告复制的每一个阶段 13 Skill，并写入
`/tmp/aiefs-skills/manifest.json`。这个干净的目标目录用于检查软件包形状；
前面的首次成功闭环则负责检查真实宿主中的发现与调用。

后续课程会深入讲解生命周期的每个阶段。第 24 课构建发现与渐进式披露。第 25 课构建调用策略与路由。第 26 课区分权限和沙箱。第 27 课把整个软件包变成经过评估的发布产物。

## 练习

1. 使用 `TaskShape` 对你团队的五个工作流进行分类。凡是选择了多个原语，都要为每项选择给出理由。
2. 添加边界测试，证明长度为 500 个字符的 `compatibility` 值能通过，而 501 个字符会因违反规范而失败。
3. 向 allowlist 添加一个运行时扩展。编写测试，证明同一文件仍然能与纯可移植 Skill 区分开。
4. 把一个 400 行 prompt 拆分为 `SKILL.md`、一份 reference、一个 script 契约和一个输出模板。让每个文件只负责一种信息。
5. 为引用了不可用 MCP 工具的 Skill 设计失败响应。不要静默替换成权限更广的工具。
6. 审查一个现有 Skill，把每个句子标记为路由、流程、策略、reference 指针或输出契约。移走所有不属于当前位置的内容。

## 关键术语

| 术语 | 人们通常怎么说 | 它的实际含义 |
|---|---|---|
| Agent Skill | “保存下来的 prompt” | 由过程性指令和可选资源组成的可发现目录 |
| Portable core | “每个运行时共享的字段” | Agent Skills 规范定义的契约 |
| Runtime extension | “额外 frontmatter” | 宿主专属配置，其行为需要兼容的适配器 |
| Activation | “Skill 已运行” | Skill 正文进入模型可见上下文；执行可能稍后才发生 |
| Skill dependency | “导入另一个 Skill” | 由运行时介导、包含可用性与策略检查的调用边 |
| Tool contract | “函数 schema” | 一项能力的输入、输出、权限、副作用、错误与证据 |

## 延伸阅读

- [Agent Skills 规范](https://agentskills.io/specification)：可移植目录与 frontmatter 契约。
- [Agent Skills 最佳实践](https://agentskills.io/skill-creation/best-practices)：作用域、指令与资源组织方式。
- [OpenAI：构建 Skill](https://learn.chatgpt.com/docs/build-skills)：当前 Codex 的发现与调用行为。
- [Claude Code Skill](https://code.claude.com/docs/en/skills)：一种运行时提供的调用、参数、工具及委派上下文扩展。
