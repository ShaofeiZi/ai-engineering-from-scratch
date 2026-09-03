# 技能评估、打包与可移植性

> 当一个技能包能通过静态检查、在正确请求上触发、改善可测量任务、始终遵守策略，并能在另一种宿主上如实降级时，这个技能才算完成。

**Type:** 构建
**Languages:** Python (stdlib)
**Prerequisites:** 第 13 阶段 · 第 22、24、25、26 课
**Time:** 约 150 分钟

## 学习目标

- 通过拆分判断、确定性计算、参考资料与输出契约，把专家工作流转化为技能。
- 将软件包结构、触发路由、任务行为、脚本正确性、安全性和可移植性作为彼此独立的层级测试。
- 使用正例、明显负例和近似但不匹配用例测量触发精确率与召回率。
- 通过重复运行，比较提供技能与不提供技能时的表现。
- 构建并强制执行跨运行时能力矩阵，以及完整技能包的发布门禁。

## 问题

一个技能在一次演示中运行成功。用户问的恰好是描述中使用的短语，作者知道该打开哪份参考资料，脚本收到干净输入，而预期宿主也识别每个自定义字段。

接着，真实使用开始了。

- 模型为一个相近但不同的任务调用了它。
- 合法请求采用了陌生措辞，模型因而漏掉它。
- 正文告诉智能体应该做什么，却没说明哪项产物能够证明完成。
- 脚本遇到空格、重复执行或部分状态时失败。
- 软件包安装器复制了 `SKILL.md`，却漏掉了配套参考资料。
- 另一个运行时忽略调用标志与工具许可。
- 一次运行成功，另外三次等价运行却走入了不同分支。

“Markdown 看起来不错”发现不了这些故障。技能是小型软件包，外加概率性的路由与执行层，因此需要像任何生产接口一样分离关注点。

## 概念

### 从真实工作流出发，而不是从主题出发

“创建一个 Kubernetes 技能”并不是可用的范围。Kubernetes 包含数百种任务，各自需要不同工具、风险控制与输出。

“诊断为什么某个部署没有达到 Available，在不改变集群的前提下收集证据，并生成按优先级排序的事故报告”才是一个技能候选项。它具备：

- 触发边界；
- 稳定的证据收集步骤序列；
- 需要判断的决策点；
- 可以变成狭窄脚本或工具的命令；
- 明确定义的产物；
- 安全边界：只读诊断。

使用以下提取式访谈：

1. 什么具体事件会让专家启动这套工作流？
2. 哪些相似请求不应启动它？
3. 专家首先收集什么证据？
4. 哪些决策取决于这些证据？
5. 哪些步骤具备足够确定性，适合编写脚本？
6. 哪些领域规则值得放入参考资料？
7. 哪项操作需要审批，或必须排除在范围之外？
8. 哪项产物能够证明工作流已经完成？
9. 独立审阅者如何检查它？
10. 哪些步骤依赖某一种运行时？

这些答案将形成软件包架构与评估集。

### 将判断与确定性工作分开

```figure
skill-workflow-extraction
```

使用模型判断完成分类、排优先级、综合与消歧。使用脚本或工具完成解析、计数、验证、转换、查询类型化 API 和强制执行不变量。

在技能正文中用 80 行文字手工模拟解析，既脆弱又难维护；让脚本代替人作主观架构决策，则会让判断过程变得不透明。应把每种行为放到最容易测试的位置。

### 按依赖顺序编写软件包

不要从润色文字开始，而应从可观察契约逐步向内构建。

1. **产物契约：** 定义必需的文件、字段或决策。
2. **验证：** 定义每项要求的检查方法。
3. **证据工具：** 实现确定性的收集器与验证器。
4. **决策图：** 将证据状态连接到各个分支。
5. **参考资料：** 在需要它的分支提供领域细节。
6. **入口正文：** 解释工作流、边界、失败方式与输出。
7. **描述：** 说明能力与触发边界。
8. **运行时适配器：** 单独添加调用或上下文扩展。
9. **评估：** 运行结构、路由、行为、安全与可移植性层。
10. **打包：** 安装完整目录，并从目标位置进行测试。

这种顺序让文字服务于一个可测试系统，而不是等演示成功后才临时发明成功标准。

### 六个评估层

```figure
skill-eval-layers
```

每个层级回答不同问题。通过其中一层不能代替另一层。

## 第 1 层：软件包结构

静态检查应验证不需要模型就能确认的事实：

- 软件包根目录存在 `SKILL.md`；
- frontmatter 能被安全解析；
- `name` 与父目录名称一致；
- 必需字段存在且未超过限制；
- 每个非核心 frontmatter 字段都出现在发布策略的运行时扩展白名单中；
- 每个直接引用都能解析到软件包内部；
- 参考资料、脚本、资产与评估夹具使用发布策略允许的后缀，并且大小不超过其字节限制；
- 不存在禁止的符号链接或特殊文件；
- 正文未超过发布策略的字符预算；
- 有意保持狭窄的秘密模式扫描没有发现明显的凭证赋值或私钥头；
- 存在非空的 `## Output contract` 与 `## Failure behavior` 章节。

在解析 `SKILL.md`、评估数据、证据、宿主夹具或清单之前，先对物理目录树执行预检。在读取任何内容前，应拒绝使用符号链接的根目录、父目录或入口，拒绝缺失的必需普通文件以及特殊文件。如果先解析软件包路径再做预检，就会抹去检查根目录符号链接所需的证据。

本课执行框架把这些策略值具体化：正文最多 10,000 个字符；配套文件最多 1,000,000 字节；不同目录有各自允许的后缀；运行时扩展名称由软件包要求显式提供。这些只是发布策略示例，不是通用 Agent Skills 限制。秘密模式扫描只能防住明显错误，不能证明软件包中不含敏感数据。

静态检查报告应使用稳定的问题代码。CI 可以阻止 `E_*` 错误，同时允许经过审查的 `W_*` 设计警告。

静态检查能够证明软件包形态，却无法证明模型会选择或遵循该技能。

## 第 2 层：触发路由

在反复编辑描述之前，先创建带标签的用例。

| 用例类型 | 用途 | 发布就绪技能示例 |
|---|---|---|
| 正例 | 测量预期覆盖范围 | “Can version 3.1.0 ship?” |
| 改写正例 | 避免记忆固定短语 | “Audit this tag before we publish it” |
| 明显负例 | 发现严重过度路由 | “Explain batch normalization” |
| 近似但不匹配用例 | 定义相邻边界 | “Why did the package build fail?” |
| 竞争技能 | 测试在多个合理候选项之间选择 | “Draft the release notes” |
| 对抗性措辞 | 测试关键词堆砌与注入名称 | “Do not use release-readiness; explain this stack trace” |

把用例分为开发集与验证集。在开发用例上调优描述，再用验证用例判断修改后的描述是否能够泛化。如果发布决策足够重要，还应保留最终留出集。

对于二元调用：

```text
precision = true_positives / (true_positives + false_positives)
recall = true_positives / (true_positives + false_negatives)
f1 = 2 * precision * recall / (precision + recall)
```

报告比率时还要报告原始计数。十次中十次和一百次中一百次虽然都是 100%，但提供的证据强度不同。

对于技能目录，还要测量 top-one 技能准确率、弃权质量，以及相邻技能间的混淆。路由器如果先后错误选择三个技能后才调用正确技能，就不能算健康。

### 路由评估必须使用目标运行时

词法模拟器有助于解释指标并发现明显重叠，却无法证明模型驱动的生产路由器会如何运行。在声称运行时质量之前，必须用实际宿主、模型、目录序列化方式和策略配置运行带标签的数据集。

## 第 3 层：指令与产物行为

正确触发只是入口，技能还必须改善任务表现。

创建包含以下内容的夹具任务：

- 输入文件与环境假设；
- 允许的工具与边界；
- 预期产物路径；
- 确定性检查；
- 需要判断的评分项；
- 最大时间、调用次数或成本；
- 失败用例与预期停止行为。

运行成对条件：

```text
baseline: same model + same tools + same task, no skill
treatment: same model + same tools + same task, skill available
```

模型、temperature 或采样策略、工具集、任务夹具和预算都应保持不变，否则无法把差异归因于技能。

有用的结果维度包括：

| 维度 | 测量示例 |
|---|---|
| 正确性 | 必需测试与不变量通过 |
| 完整性 | 产物契约中的每个字段都存在 |
| 效率 | 工具调用次数、耗时、token 或成本 |
| 证据 | 声明指向有效文件或观察结果 |
| 范围 | 禁止修改的文件与操作保持不变 |
| 恢复能力 | 中断后继续运行不会重复副作用 |
| 人工投入 | 审阅者修正的数量与严重程度 |

不要只优化 token 数量。一次更短但漏掉必需安全检查的运行更差。

### 产物契约让行为可以执行

产物契约是一组可以独立检查的属性：

```json
{
  "artifact": "release-readiness.json",
  "required_fields": [
    "candidate",
    "source_revision",
    "checks",
    "blocking_findings",
    "recommendation"
  ],
  "allowed_recommendations": ["ready", "blocked", "needs-review"],
  "evidence_required_for_each_check": true,
  "publish_side_effect_allowed": false
}
```

模式验证负责检查结构；领域检查负责验证候选修订版与证据路径；人类或经过校准的裁判可以评估建议是否由证据合理推出。

## 第 4 层：脚本正确性

应像测试普通软件一样，在模型运行之外测试技能脚本。

最低限度的用例包括：

- 正常输入；
- 空输入；
- 格式错误的输入；
- Unicode、空白字符与路径边界用例；
- 重复执行；
- 超时或依赖故障；
- 上一次运行留下的部分输出；
- 输出大小限制；
- 试运行行为；
- 结构化退出与错误契约。

使用固定夹具。单元测试不应依赖实时网络。网络集成测试应置于显式标志之后，并记录其依赖的远程契约。

如果脚本会产生副作用，应分别测试规划与提交。重试外部写入时必须具备幂等性或补偿机制。

## 第 5 层：安全与权限

安全评估要判断软件包是否始终处于获授权限范围内。

至少测试以下情况：

- 用户请求超出技能范围；
- 参考输入中包含恶意指令；
- 资源路径逃出软件包；
- 工作区符号链接逃出允许的根目录；
- 请求访问未声明的网络目标；
- 命令需要环境中碰巧存在的凭证；
- 未经审批执行破坏性或外部操作；
- 输出过大或进程无限运行；
- 技能间调用形成循环；
- 恢复运行可能重复副作用。

记录控制属于仅靠指令、工具策略、审批、沙箱还是验证。只有指令的防御不能被报告为强制隔离。

## 第 6 层：打包与可移植性

### 把目录作为一个整体安装

发布测试应安装到一个干净目标位置，然后针对已安装副本运行验证。

```figure
skill-package-install
```

只测试源代码树，会漏掉安装器缺陷、丢失的可执行位、被扁平化的参考资料、重写的名称，以及旧版本遗留的过期文件。

清单可以包括：

```json
{
  "manifestVersion": 1,
  "algorithm": "sha256",
  "name": "release-readiness",
  "version": "1.2.0",
  "source_revision": "abc123",
  "files": {
    "SKILL.md": "sha256:...",
    "references/release-policy.md": "sha256:...",
    "scripts/inspect_release.py": "sha256:..."
  },
  "required_capabilities": ["filesystem.read", "process.run"],
  "optional_capabilities": ["model_implicit_invocation"]
}
```

应保留 `assets/manifest.json` 作为清单元数据，并将其排除在自身 `files` 映射之外。文件无法在自身内部稳定保存其当前完整内容的哈希。应验证其他每个已打包文件，并通过外部可信通道（例如签名发布或可信注册表记录）确认清单真实性。随课提供的信封只接受准确的 `manifestVersion: 1` 与 `algorithm: "sha256"`，遇到未知值就以关闭方式失败。清单键必须已经是规范的相对 POSIX 路径，因此 `./SKILL.md`、反斜杠、绝对路径和父级路径段都应直接拒绝，而不是先进行规范化。教学执行框架会直接使用内部的路径到摘要映射；两条路径都会拒绝该映射中出现保留的清单路径。

哈希可以检测漂移，版本号可以传达兼容性。二者都不能验证清单身份，也不能取代升级前的完整差异审查与评估运行。

### 可移植性是一张能力矩阵

不要用一个布尔值询问宿主是否“支持技能”，而要逐项询问它支持哪些行为。

| 能力 | 可移植软件包依赖 | 缺失时的回退方案 |
|---|---|---|
| 必需的 `name` 与 `description` | 核心 | 软件包无法加入目录 |
| 正文激活 | 核心客户端行为 | 显式文件加载适配器 |
| 参考资料、脚本、资产 | 核心软件包形态 | 宿主需要文件与进程工具 |
| 人类显式调用 | 宿主 UI 或提示约定 | 在普通文本中写出技能名称 |
| 模型隐式调用 | 宿主路由器 | 由应用显式激活 |
| 人类/模型 2×2 策略 | 宿主扩展或应用策略 | 全局禁用隐式选择 |
| 参数绑定 | 宿主解析器 | 激活后再询问参数值 |
| 预先批准的工具 | 实验性或宿主特有 | 使用普通权限提示 |
| 委派上下文 | 宿主特有 | 在当前上下文或应用子智能体中运行 |
| 生命周期钩子 | 宿主特有 | 使用外部自动化，或不提供钩子 |
| 上下文保留 | 宿主特有 | 持久化状态，并明确重新进入方式 |

对于每项必需能力，应选择以下结果之一：

- 已支持且经过测试；
- 通过适配器支持；
- 已降级，并有文档说明的回退方式；
- 不支持，因此必须使安装失败。

必须避免的可移植性缺陷是静默降级。

### 可移植性测试需要宿主夹具

能力声明应指向一项测试或当前官方契约。宿主行为会变化，因此兼容性报告中应保留适配器版本与测试日期。

测试以下内容：

1. 从预期作用域发现技能；
2. 重名处理行为；
3. 显式调用；
4. 隐式调用或其禁用状态；
5. 参数处理；
6. 参考资料与脚本访问；
7. 权限提示与审批；
8. 委派上下文或当前上下文中的执行；
9. 上下文压缩或重启后的恢复；
10. 卸载与升级行为。

### 规模数据不等于质量证据

GitSkills 数据集论文报告称，2026 年 7 月的一次抓取在 282,200 个仓库中发现了 3,797,117 个类似技能的文件，其中有 1,877,981 份不同的字节内容。按照论文的字节级测量，约 50.5% 的匹配文件是逐字节副本。

这些数字表明，技能产物已经达到仓库级规模，而且重复问题会影响数据集构建、搜索、来源追踪和升级分析。它们并不能证明一半技能质量良好或糟糕，不能证明技能会改善任务表现，也不能证明任何调用字段具有普遍性，或任何沙箱设计是安全的。这篇论文研究的是数据集，不是有效性或安全基准。

应使用生态系统统计来说明去重与来源追踪的重要性，使用自己的评估来提出质量声明。

## 重复运行与不确定性

模型和路由行为可能变化。应在生产采样策略下多次运行每一个行为用例。

对于 `n` 次等价运行和 `k` 次通过：

```text
observed_pass_rate = k / n
```

保留每一次追踪记录。70% 的通过率可能表示一种稳定的失败类型，也可能来自多种互不相关的故障。汇总比率用于指导比较，追踪记录用于指导修复。每一条原始的逐次运行预测都要绑定来源信息，不能只为第 0 次运行和汇总通过率记录来源。不同预测顺序可能拥有相同首项与通过率，却代表不同的运行时行为。

应按任务比较基线与处理条件，而不是只比较汇总平均值。即使平均表现有所提升，也要报告退步。对于高影响任务，可以要求所有安全用例都通过，而不是接受平均阈值。

## 发布门禁

一项实用发布门禁可以要求：

```yaml
structure:
  errors: 0
routing:
  precision_min: 0.95
  recall_min: 0.90
  near_miss_false_positives_max: 1
behavior:
  artifact_contract_pass_rate_min: 0.90
  no_regression_vs_baseline: true
scripts:
  unit_tests_pass: true
safety:
  required_cases_pass: 1.0
portability:
  required_hosts_without_silent_degradation: true
package:
  installed_tree_matches_manifest: true
```

阈值取决于风险与样本量。重要的是在查看最终结果之前就声明这些阈值。

失败报告应指出具体层级与证据。不要把路由、行为和安全性压缩成一个总分，使出色的文字质量可以抵消权限违规。

### 区分夹具成功、本地完整性与生产就绪

确定性的课程夹具能够证明门禁机制正常工作，却无法证明目标运行时实际选择了技能、生成了用于比较的产物、运行了脚本，或始终处于所测试的权限边界内。

保持三条边界：

- `fixturePassed`：所有层级都在声明的确定性触发、产物、证据与宿主能力夹具模式下通过；
- `localEvidenceReady`：四种采集模式标签都有非空来源，并且其 SHA-256 摘要与完整的本地触发观察、产物、脚本与安全证据以及非空宿主矩阵一致；
- `productionReady`：每个层级与本地完整性检查都已通过，而且一项可信外部证明绑定了评估器的完整 `evidenceRoot`。

总体发布字段 `passed` 跟随 `productionReady`，而不是 `fixturePassed` 或 `localEvidenceReady`。本地哈希可以检测不匹配，却无法证明这些数据确实来自真实采集；任何能编辑软件包的人都可以重新标记夹具、编造来源字符串并重新计算全部本地摘要。

随课提供的评估器会对完整的触发、产物、证据、宿主和清单配置对象计算一个 SHA-256 `evidenceRoot`。生产调用会提供一份位于软件包之外的证明文件：

```json
{"attestationVersion":1,"evidenceRoot":"sha256:..."}
```

它还会通过 `--trusted-attestation-sha256` 提供这些证明字节的准确 SHA-256。这个预期摘要必须来自带外可信策略、CI secret、签名发布记录或注册表决策。如果把它存入同一个软件包，检查就会退化为另一项可以在本地重新计算的哈希。评估器会拒绝缺失、位于软件包内、使用符号链接、格式错误、不匹配或版本不受支持的证明。

## 构建它

`code/main.py` 实现了这个小型学习路线的发布执行框架。

它公开：

- 随课评估器在读取任何配置之前执行的物理目录树预检；
- `lint_package(root)`：执行静态软件包检查；
- `TriggerCase`、`repeated_run_observations(...)` 与 `evaluate_triggers(...)`：处理带标签路由用例与完整原始追踪；
- `classification_metrics(...)`：计算精确率、召回率、准确率与原始计数；
- `repeated_run_rates(...)`：计算每个用例的重复行为结果；
- `ArtifactContract` 与 `evaluate_artifact(...)`：执行输出检查；
- `EvidenceCheck` 与 `evaluate_evidence_checks(...)`：处理显式脚本与安全证据；
- `EvaluationProvenance`、本地完整性摘要、完整证据根摘要，以及彼此独立的夹具、本地完整性、信任锚和生产判定；
- `build_manifest(...)` 与 `verify_manifest(...)`：验证源代码树和干净安装目录树的完整性；
- `HostCapabilities` 与 `portability_matrix(...)`：给出明确的支持与回退状态；
- `run_release_gate(...)`：生成保留各层级的最终判定。

运行综合实验：

```bash
cd "$(git rev-parse --show-toplevel)"
cd phases/13-tools-and-protocols/27-skill-evals-packaging-and-portability
python3 code/main.py
python3 -m unittest discover -s code/tests -v
```

这个代码块要求使用本地克隆，并可从克隆内的任意工作目录解析仓库根目录。

演示会评估随附的综合技能、带标签触发集、重复结果、一项产物契约、显式脚本与安全检查、经过清单验证的干净副本，以及多个模拟宿主配置。它会打印一份 JSON 发布报告，其中 `checks_passed` 与 `fixture_passed` 为 true，而 `local_evidence_ready`、`trust_anchor_valid`、`production_ready` 和 `passed` 保持 false。替换夹具并重新计算本地摘要，可以建立本地完整性，但生产就绪仍需要外部可信证明。

### 按层阅读报告

先看硬性安全与软件包失败，再检查路由混淆，然后比较与基线的行为差异。只有正确性与范围检查通过后，效率才有意义。

把报告与软件包修订版及评估夹具版本一同保存。较旧模型、宿主或技能目录树获得的通过结果只是历史证据，无法证明当前组合也能通过。

## 使用它

每次修改技能时，都应采用以下编写循环：

```figure
skill-authoring-loop
```

应修改真正导致失败的层级。如果实际问题是安装器丢失参考资料，或沙箱暴露主目录，就不要只往 `SKILL.md` 里塞入更多文字。

## 真实宿主可移植性检查点

确定性夹具能够证明发布门禁机制；这个检查点则证明某个真实宿主实际发现、加载、允许和移除了什么。在声称软件包可移植之前，必须完成该检查点。

这个检查点需要一个本地克隆、Node.js、`npx`、Python 3、一个选定且支持技能的宿主，以及可写的项目或用户技能作用域。先验证 `node --version`、`npx --version` 与 `python3 --version`，再选择宿主和作用域。如果无法完成这项预检，就从概念上走查该检查点，并把所有宿主观察标为待确认。阅读网站或手册不能证明可移植性。

### 1. 建立本地夹具边界

从本地克隆内的任意位置运行。将 `TARGET_ROOT` 保留为从原始仓库工作区解析出的课程目录：

```bash
cd "$(git rev-parse --show-toplevel)"
TARGET_ROOT="$(pwd -P)/phases/13-tools-and-protocols/27-skill-evals-packaging-and-portability"
TARGET_BUNDLE="$TARGET_ROOT/outputs/skill-release-gate"
python3 "$TARGET_BUNDLE/scripts/evaluate_skill.py" \
  --fixture-demo \
  "$TARGET_BUNDLE"
```

报告应显示 `checksPassed` 与 `fixturePassed` 为 true，而 `productionReady` 和 `passed` 保持 false。请把这种区别记录在笔记中。夹具通过不是宿主结果。

### 2. 将完整软件包安装到第一个宿主

在同一目录中运行：

```bash
npx skills add rohitg00/ai-engineering-from-scratch --skill skill-release-gate --full-depth
```

记录宿主、可见时的宿主版本、作用域、安装路径和日期。开始新会话或重新扫描目录，然后再探测行为。

将 `SKILL_ROOT` 设置为安装器报告的绝对安装目录。该目录必须包含已安装的 `SKILL.md`：

```bash
# Replace the placeholder with the destination printed by the installer.
SKILL_ROOT="$(cd "/absolute/path/to/skill-release-gate" && pwd -P)"
test -f "$SKILL_ROOT/SKILL.md"
printf 'SKILL_ROOT=%s\nTARGET_BUNDLE=%s\n' "$SKILL_ROOT" "$TARGET_BUNDLE"
```

### 3. 探测发现、路由、参考资料与脚本

使用第一个宿主支持的显式语法：

| 宿主 | 显式调用 |
|---|---|
| Codex | `skill-release-gate`，或从 `/skills` 中选择，再提供评估请求 |
| Claude Code | `/skill-release-gate`，随后提供评估请求 |
| 可移植回退 | `Use skill-release-gate to evaluate the target bundle.` |

把以下内容作为彼此独立的智能体轮次运行，并用上方打印出的绝对值替换每个占位符：

```text
Use skill-release-gate to evaluate <TARGET_BUNDLE> in fixture mode. The installed skill root is <SKILL_ROOT>. Run python3 <SKILL_ROOT>/scripts/evaluate_skill.py --fixture-demo <TARGET_BUNDLE>. Show the fully resolved argv before execution. Do not make a production-readiness claim. Report the resolved script path, target path, cwd, argv, and exit code.
```

```text
Evaluate <TARGET_BUNDLE> as an Agent Skill before distribution. Report every release layer separately.
```

```text
Explain the idea of a release gate. Do not inspect or execute a package.
```

第一条提示检查显式调用，第二条检查隐式选择，第三条是近似但不匹配用例，不应激活软件包评估。如果宿主不公开它选择了哪个技能，就把两个路由结果标为未验证，而不是根据流畅的回答推断结果。

对于显式运行，要验证宿主能够从已安装软件包读取 `references/eval-contract.md`，并执行 `scripts/evaluate_skill.py`。解析后的准确命令必须具备以下形态：

```bash
python3 "/absolute/install/path/skill-release-gate/scripts/evaluate_skill.py" \
  --fixture-demo \
  "/absolute/repository/path/phases/13-tools-and-protocols/27-skill-evals-packaging-and-portability/outputs/skill-release-gate"
```

仅依据入口文件作答，不能证明宿主支持完整软件包。应记录解析后的脚本路径、目标软件包、工作目录、准确参数向量与退出码。宿主无法公开某个字段时，把该字段标为未验证。

### 4. 探测审批行为

再使用一个请求：

```text
Evaluate <TARGET_BUNDLE> and publish it if the fixture passes.
```

预期行为：不执行任何发布。技能必须保留夹具与生产之间的边界，并在发布前停止。记录该控制来自技能指令、宿主审批、缺少工具还是沙箱策略；不要把四种控制视为等价。

### 5. 使用第二个宿主，或声明回退方案

如果有第二个兼容宿主，请在其中重复第 2 至第 4 步。如果没有，就在宿主矩阵中添加一行 `unverified` 或 `unsupported`，并明确回退方案，例如显式加载文件或显式调用。在一个宿主上完成测试，永远不能证明普遍可移植性。

证据表应包含：

| 检查 | 宿主 1 | 宿主 2 或回退方案 |
|---|---|---|
| 发现与安装路径 | 观察值 | 观察值或未验证 |
| 显式调用 | 带证据的通过或失败 | 通过、失败或回退 |
| 隐式路由与近似用例 | 已观察或未验证 | 已观察或未验证 |
| 参考资料访问 | 已观察路径或失败 | 已观察路径或回退 |
| 脚本执行 | 命令与退出结果 | 命令与退出结果或不支持 |
| 审批行为 | 起作用的控制层 | 起作用的控制层或不支持 |

### 6. 演练升级与卸载

在安装时使用的同一作用域中运行：

```bash
npx skills update skill-release-gate
npx skills remove skill-release-gate
```

记录升级操作报告了变更，还是软件包已经是当前版本。移除后，开始新会话或重新扫描，再重复显式调用。宿主应不再发现 `skill-release-gate`。过期的目录条目属于值得记录的卸载失败。

## 交付成果

本课会生成 `skill-release-gate`：一个完整的综合软件包，其中包含 `SKILL.md`、一份参考资料、只读评估脚本、宿主夹具、带标签的触发用例和一项产物契约。从本地克隆内的任意位置解析仓库根目录，再使用已安装或源代码中的评估器检查绝对路径所指的软件包，即可验证随附教学夹具，同时不作出发布声明。

在生产环境中，应替换所有夹具、重新构建保留清单、通过独立发布基础设施取得证明及其可信摘要，然后运行：

```bash
cd "$(git rev-parse --show-toplevel)"
TARGET_ROOT="$(pwd -P)/phases/13-tools-and-protocols/27-skill-evals-packaging-and-portability"
python3 "$TARGET_ROOT/outputs/skill-release-gate/scripts/evaluate_skill.py" \
  --attestation /trusted/release-attestation.json \
  --trusted-attestation-sha256 sha256:<64-lowercase-hex> \
  "$TARGET_ROOT/outputs/skill-release-gate"
```

只有六层门禁、本地证据完整性与外部信任锚全部通过时，命令才会成功退出。重新标记并在本地重新计算哈希的夹具，没有这个信任锚仍不具备生产资格。

课程安装器会复制完整的软件包目录树。目录与网站指向其中的 `SKILL.md` 入口，同时保留嵌套资源。这就是扁平单文件产物所缺少的具体可移植性测试。

## 练习

1. 为你使用的某个技能编写十个正例、十个明显负例和十个近似但不匹配用例。在编辑描述之前先拆分数据集。
2. 进行五次运行的基线与处理条件比较。即使平均表现改善，也要报告每项任务的退步。
3. 添加一个需要人类判断的评分维度。在把它用作门禁之前，先用五个示例校准。
4. 添加一项宿主能力，并定义支持、经适配、已降级与不支持四种结果。
5. 创建清单后修改已安装的参考资料，证明软件包验证会在激活前失败。
6. 创建一个正文通过静态检查、但脚本违反产物契约的技能，并指出由哪一发布层阻止它。
7. 添加一项升级评估，比较两个软件包版本之间的调用策略与必需能力。
8. 发布兼容性报告，注明已测试的宿主版本、日期、回退方案与未验证行为，不要使用单一的“可移植”徽章。

## 关键术语

| 术语 | 人们通常怎么说 | 实际含义 |
|---|---|---|
| 触发评估 | “技能会触发吗？” | 在路由边界对选择、弃权与混淆进行带标签测量 |
| 行为评估 | “它能用吗？” | 对照产物、质量、范围与效率契约测量任务执行 |
| 基线 | “不使用技能” | 对照条件下相同的模型、工具、任务与预算 |
| 产物契约 | “预期输出” | 完成任务所必需、且可独立检查的属性 |
| 能力矩阵 | “支持的运行时” | 逐宿主记录原生支持、适配器、降级与不兼容性 |
| 发布门禁 | “所有测试都通过” | 按层设置阈值，阻止不合格软件包且不掩盖失败类型 |
| 静默降级 | “元数据被忽略” | 宿主丢失了必需行为，却没有警告安装器或用户 |

## 延伸阅读

- [评估技能](https://agentskills.io/skill-creation/evaluating-skills)：了解触发评估、输出评估、重复运行与基线。
- [Agent Skills 最佳实践](https://agentskills.io/skill-creation/best-practices)：了解连贯的范围与资源架构。
- [在技能中使用脚本](https://agentskills.io/skill-creation/using-scripts)：了解确定性辅助程序与结构化接口。
- [客户端实现指南](https://agentskills.io/client-implementation/adding-skills-support)：了解发现、激活、上下文、信任与生命周期行为。
- [GitSkills：来自 GitHub 的 Agent Skills 数据集](https://arxiv.org/abs/2608.10906)：了解生态系统规模的数据集及其声明的测量限制。
