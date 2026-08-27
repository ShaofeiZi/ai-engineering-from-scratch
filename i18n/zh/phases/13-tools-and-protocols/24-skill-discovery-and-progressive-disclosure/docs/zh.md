# Skill 发现与渐进披露

> Skill 在加载正文之前就已经开始发挥作用。名称与描述让它获得进入目录的资格；只有当任务真正需要时，更深层的文件才值得占用上下文。

**Type:** 构建
**Languages:** Python (stdlib)
**Prerequisites:** 第 13 阶段 · 第 22 课（Agent Skills：可移植契约与运行时边界）
**Time:** 约 105 分钟

## 学习目标

- 构建文件系统发现流水线，把作用域、验证、冲突策略与目录发布彼此分离。
- 解释三种披露层级：目录元数据、已激活指令与任务专用资源。
- 设计引用，使智能体可以直接抵达必需细节，而无须加载整个包。
- 分别管理目录空间预算与已激活 Skill 的上下文预算。
- Skill 读取自身资源时，拒绝路径穿越与符号链接逃逸。

## 问题

你的智能体安装了 200 个 Skill。如果在会话开始时加载每个 `SKILL.md`、参考文件、脚本与模板，当前任务会被无关规程淹没。什么都不加载，又会迫使用户记住精确的文件系统路径。

常见折中方案是目录：为每个符合条件的 Skill 向模型展示紧凑的身份与路由描述，只在选中后加载完整正文。这会带来两个新的工程问题。

第一，发现并不只是递归搜索文件。Skill 可能存在于项目、用户、管理员、插件或内置作用域中。两个包可能同名，符号链接可能指向受信根目录之外，格式错误的包可能消耗目录空间，或根本无法调用。

第二，渐进披露可能变成渐进困惑。如果 `SKILL.md` 写着“阅读相关指南”，而包内有十二份指南，模型只能猜。如果每份指南又指向三个文件，加载过程就会变成无界图遍历。

优秀的运行时会让发现具有确定性，让披露成为有意为之的选择。

## 概念

### 发现是一条编译器流水线

把文件系统视为源输入。不要把原始路径直接发布给模型。

```figure
skill-discovery-pipeline
```

每个阶段都应产生结构化数据与结构化失败。发现日志应当能够回答：

- 搜索了哪些根目录？
- 找到了哪些候选项？
- 哪些候选项被拒绝，原因是什么？
- 发生冲突时哪个包胜出？
- 哪些目录项因预算不足而被缩短或省略？

没有这些证据，“模型为什么没有使用我的 Skill”几乎无法诊断。

### 作用域属于运行时策略

可移植规范定义 Skill 包，却没有规定唯一安装路径或优先顺序。宿主决定搜索位置。

通用运行时可能使用以下作用域：

| 作用域 | 示例根目录 | 预期所有者 |
|---|---|---|
| 工作区 | `<repo>/.agents/skills/` | 项目维护者 |
| 用户 | `<user-data>/skills/` | 单个开发者 |
| 管理员 | `<system>/skills/` | 机器或组织策略 |
| 插件 | 已签名的插件包 | 插件发布者与安装器 |
| 内置 | 运行时包 | 运行时供应商 |

截至 2026 年 8 月，Codex 记录的项目发现路径从 `$CWD/.agents/skills` 开始，沿祖先目录一直搜索到仓库根目录，并包含用户、管理员与内置位置。它支持通过符号链接引用 Skill 目录。同名 Skill 可能同时出现，而不是被合并。这些是 Codex 行为，不是 `SKILL.md` 的要求；编写适配器时，应核对最新的 [Codex Skill 文档](https://learn.chatgpt.com/docs/build-skills)。

绝不要根据目录名称臆测优先级。应把优先级声明为策略，并为其编写测试。本课实验为每个 `Scope` 使用显式整数排名，因此相同候选集合总会得到相同解析结果。

### 冲突需要超越 `name` 的身份信息

两个名为 `release-readiness` 的包都可能合法：一个是工作区覆盖，另一个是用户默认。因此，目录项至少需要包含：

```json
{
  "name": "release-readiness",
  "description": "Inspect a release candidate for this repository.",
  "scope": "workspace",
  "source": "/repo/.agents/skills/release-readiness",
  "selected": true
}
```

常见冲突策略包括：

| 策略 | 优点 | 风险 |
|---|---|---|
| 保留所有候选项 | 不隐藏任何内容 | 模型会看到含义不明确的同名项 |
| 最高优先级作用域胜出 | 调用简单 | 本地包可能遮蔽受信包 |
| 拒绝重复项 | 不会静默遮蔽 | 合法覆盖也无法工作 |
| 按来源限定名称 | 身份明确 | 面向用户的名称会变长 |

为宿主选择一种策略。即使被拒绝或遮蔽的候选项不出现在模型目录中，也要保留在诊断信息里。

### 三种披露层级

Agent Skills 规范描述了分阶段加载。关键在于，每一层都有不同目的。

```figure
skill-disclosure-levels
```

#### 第 1 层：目录元数据

模型需要足够的信息，才能把 Skill 与邻近选项区分开。规范估算每个目录项约为 100 个词元，但实际序列化与分词方式由宿主决定。

有用的描述包含两个分句：

```yaml
description: Validate a release candidate and produce a readiness report. Use when the user asks whether a version, tag, or package is ready to publish.
```

第一句说明能力，第二句说明触发边界。第 25 课会使用正例与近似但不匹配的提示词评估这条边界。

#### 第 2 层：已激活指令

激活后，正文应同时充当地图与规程。规范建议把 `SKILL.md` 控制在 500 行以内。这是设计信号，不是要填满的目标。

正文应包含：

- 任务边界；
- 默认工作流；
- 分支条件；
- 指向更深层文件的直接引用；
- 工具与脚本契约；
- 失败与停止行为；
- 预期输出及其验证方式。

不要仅仅为了缩短入口文件，就把核心工作流移动到参考文件。激活后，模型必须获得足以正确开始任务的上下文。

#### 第 3 层：支持资源

参考资料提供文字或数据。脚本提供确定性计算。资产会被复制、填充或转换为交付物，而不应被当作指令。

| 目录 | 模型会读取吗？ | 模型会执行吗？ | 典型内容 |
|---|:---:|:---:|---|
| `references/` | 需要时读取 | 否 | Schema、策略、领域指南 |
| `scripts/` | 可能检查 | 通过获准工具执行 | 验证器、转换器、收集器 |
| `assets/` | 有用时读取 | 否 | 模板、夹具、图像、起始文件 |

这些名称是约定，不是魔法能力。宿主仍然需要文件访问权限与执行工具。

### 分支专用引用优于主题堆砌

应把入口文件写成决策地图：

```markdown
## Choose the path

- For a Python package, read `references/python-release.md`.
- For a container image, read `references/container-release.md`.
- For a documentation-only release, read `references/docs-release.md`.
- If the release combines artifact types, read only the guides for those artifacts.
```

这样，每个引用都有可观察的加载条件。“阅读 `references/` 获取更多信息”则没有。

引用图应保持浅层。官方指南建议从 `SKILL.md` 直接链接，并避免深层链式引用。一跳即可测试可达性，也能降低必要约束永远无法进入上下文的风险。

```figure
skill-reference-map
```

### 目录预算与活动上下文是两份不同预算

设 `c_i` 为 Skill `i` 的序列化目录成本，`B_c` 为目录预算，`b_j` 为活动正文成本，`r_k` 为实际加载的资源成本。

```text
catalog_cost = sum(c_i for every published skill)
active_cost = sum(b_j for every activated skill) + sum(r_k for every disclosed resource)
```

降低一项预算不会自动降低另一项。短描述可以节省目录空间，但激活后的 900 行正文仍可能淹没任务。把正文拆为参考文件，只有在运行时与指令真正避免加载无关分支时，才能降低活动成本。

当上下文窗口大小已知时，Codex 目前把初始 Skill 列表预算设为上下文窗口的 2%。8,000 字符只是在大小未知时的后备值，并不是与 2% 规则叠加的第二重上限。目录超过适用预算时，描述可能被缩短或省略。应把这些数字视为当前 Codex 策略，而不是 Agent Skills 标准的属性。

### 资源路径是一条信任边界

Skill 只能读取自身包内的文件。仅使用字符串前缀检查是不够的：

```text
references/../../../../.ssh/config
references/external-link -> /private/company-secrets
```

应按文件系统语义解析包根目录与候选路径，拒绝绝对路径输入，并验证解析后的候选项仍位于解析后的根目录之内。发现前要先决定是否允许符号链接；若允许，则每次都要检查解析后的目标。

路径包含关系并不能证明内容可信。位于包内的合法参考文件仍可能包含恶意指令。第 26 课会处理这种威胁。

```figure
skill-resource-containment
```

### 加载必须可观察

记录披露事件，但不要记录机密：

```json
{
  "event": "skill.resource.loaded",
  "skill": "release-readiness",
  "resource": "references/python-release.md",
  "reason": "candidate contains pyproject.toml",
  "bytes": 2840
}
```

原因字段会把一次上下文选择变成可审查证据，也能帮助识别那些导致智能体“以防万一”加载所有文件的指令。

## 动手构建

`code/main.py` 会构建确定性的发现与披露引擎。

发现界面包括：

- `Scope`：来源与优先级元数据；
- `SkillCandidate`：尚未验证的文件系统候选项；
- `discover_scope(scope)`：枚举直接子 Skill 目录；
- `resolve_collisions(candidates, precedence)`：应用一项已声明策略；
- `CatalogEntry` 与 `build_catalog(...)`：发布有界元数据；
- `CatalogBudget`：统计序列化条目成本，不假装字符就是通用词元。

披露界面包括：

- `load_skill_body(entry, ...)`：第 2 层激活；
- `validate_reference(skill_dir, reference)`：路径包含验证；
- `load_reference(...)`：有界的第 3 层读取。

运行实验：

```bash
cd "$(git rev-parse --show-toplevel)"
cd phases/13-tools-and-protocols/24-skill-discovery-and-progressive-disclosure
python3 code/main.py
python3 -m unittest discover -s code/tests -v
```

这段命令要求本地克隆，并能从克隆内部任意工作目录解析仓库根目录。

演示会创建临时项目与用户作用域，插入一项冲突，在刻意设置得很小的预算下构建目录，激活一个 Skill，再分别尝试读取有效参考和路径穿越目标。不会安装永久文件。

### 为什么发现是浅层的

`discover_scope` 只检查直接子目录中的 `SKILL.md`，不会把每个嵌套 `SKILL.md` 都递归视为独立包。这样可以保留包边界，避免误把已安装 Skill 内的示例或夹具发布出去。

### 为什么实验不解析任意 YAML

实验只支持构建目录所需的标量 frontmatter。生产运行时应使用安全 YAML 解析器，配合显式 Schema、大小限制，并禁用自定义对象构造。“仅标准库”是教学限制，不代表可以悄悄发明一个残缺 YAML 方言。

## 投入使用

将这份检查清单应用到任意发现适配器：

1. 列出每个已配置根目录及其可写者。
2. 说明是否允许符号链接包。
3. 验证包名称、目录名称、必需元数据与入口正文大小。
4. 在内部身份中保留来源与作用域。
5. 声明并测试同名重复项行为。
6. 测量发送给模型的确切序列化目录。
7. 记录加载正文或资源的原因。
8. 保证资源读取留在解析后的包根目录中。
9. 引用文件缺失时明确失败。
10. 安装项或策略发生变化时重新构建目录。

## 交付成果

本课会产出 `skill-catalog-builder` 软件包。它扫描按显式顺序排列的根目录，拒绝符号链接入口文件与名称—目录不匹配，解决跨作用域冲突，拒绝同优先级重复项，并使选中元数据适配已声明的条目数量、描述长度与序列化字符预算。

其 JSON 报告包含已选条目、被遮蔽候选项、省略条目、验证错误、优先级与预算用量。正文与参考文件加载仍属于独立运行时操作，因此目录构建器不会执行脚本，也不会把整个包纳入上下文。

## 练习

1. 添加插件作用域，并把优先级放在用户与内置作用域之间。使用测试证明冲突结果。
2. 把冲突策略从最高优先级改为限定名称，在目录中保留两个条目。
3. 为 `load_reference` 添加字节大小限制。测试恰好位于上限的文件，以及超出一字节的文件。
4. 创建两条听起来几乎相同的描述，重写它们，使触发边界不重叠。
5. 添加清单，为每个参考文件与脚本记录哈希。在加载前检测被修改的资源。
6. 为演示添加插桩，分别报告第 1、2、3 层的字节数。

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|---|---|---|
| Skill 发现 | “查找所有 SKILL.md” | 搜索已配置作用域、验证包、附加出处并应用策略 |
| Skill 目录 | “已安装 Skill 列表” | 面向模型的紧凑路由元数据，只包含符合条件的包 |
| 冲突策略 | “哪个重复项胜出” | 针对不同来源同名候选项的已声明规则 |
| 渐进披露 | “延迟加载” | 从目录到正文，再到分支专用资源的分阶段上下文准入 |
| 引用图 | “Skill 链接的文件” | 可达资源结构及其加载条件 |
| 路径包含 | “留在包内” | 验证解析后的资源目标仍位于解析后的包根目录内 |

## 延伸阅读

- [Agent Skills 规范](https://agentskills.io/specification)——包形态与渐进披露层级。
- [优化 Skill 描述](https://agentskills.io/skill-creation/optimizing-descriptions)——目录路由元数据。
- [Agent Skills 最佳实践](https://agentskills.io/skill-creation/best-practices)——直接引用与入口文件大小。
- [OpenAI：构建 Skill](https://learn.chatgpt.com/docs/build-skills)——当前 Codex 发现作用域与目录限制。
