# 失败模式：MAST、群体思维、单一文化与级联错误

> 到 2026 年，最有代表性的参考分类法是 **MAST**（Cemri et al., NeurIPS 2025, arXiv:2503.13657）。它基于 7 个最先进开源 MAS 的 1642 条执行轨迹，总结出 **41–86.7% 的失败率**。三类根因分别是：**Specification Problems**（41.77%），也就是角色含糊、任务定义不清；**Coordination Failures**（36.94%），也就是通信失效、状态不同步；以及 **Verification Gaps**（21.30%），也就是缺少验证、质量检查缺位。**Groupthink** 这一组问题（arXiv:2508.05687）则进一步补充了：单一文化崩溃、从众偏差、心智理论不足、混合动机动态，以及级联可靠性故障。典型级联例子是重试风暴：一次支付失败触发订单重试，订单重试又触发库存重试，最终在几秒内把库存服务打到 10 倍负载，因此需要 circuit breaker。另一个问题是 memory poisoning：一个代理的幻觉进入共享记忆后，下游代理把它当成事实继续推理；准确率不是瞬间崩掉，而是缓慢衰减，因此根因诊断会尤其痛苦。**STRATUS**（NeurIPS 2025）报告称，依靠专门的 detection、diagnosis 与 validation agent，缓解成功率可提升 1.5 倍。本课会把失败模式当作一等工程对象来处理。

**Type:** 学习
**Languages:** Python（标准库）
**Prerequisites:** 阶段 16 · 13（共享记忆），阶段 16 · 14（共识与 BFT），阶段 16 · 15（投票与辩论拓扑）
**Time:** 约 75 分钟

## 问题

多智能体系统在真实任务上的失败率高达 41–86.7%（Cemri et al. 2025 在 7 个开源 MAS 上测得）。这类问题不可能靠“再多加几个 agent”就调通。失败背后有结构性原因，而 MAST 正是用来给这些原因分门别类的。本课会把每个类别映射到具体的检测、诊断和缓解模式，让这些数字不再像一团随机噪声。

到 2026 年，生产实践的共识是：失败模式必须作为设计输入。只有当你能指着每个 MAST 类别，说清自己部署了什么缓解手段时，你的架构才算“足够好”。

## 概念

### MAST 分类

**规格问题（占失败的 41.77%）。** 智能体的任务定义不够严格。典型例子：

- 角色含糊：两个智能体都以为自己是审查者。
- 任务定义不足：用户只说“总结一下”，但实际需要特定视角。
- 成功标准隐含：智能体无法判断任务是否完成。

缓解方式：

- 写明角色契约。每个智能体的提示既要说明它做什么，也要说明它**不做什么**。
- 为每项任务编写验收测试。智能体开始前，先定义“完成应是什么样子”。
- 做执行前规格检查：由另一个智能体在真正派发任务前审查任务定义。

**协调失败（36.94%）。** 这是通信或状态层面的失效。

例子：

- 两个 agent 在没有同步机制的情况下同时更新共享状态。
- agent 之间消息丢失（队列故障、超时）。
- 状态漂移：agent A 认为任务已完成，agent B 仍在执行。

缓解方式：

- 用带版本号的共享状态，并配合乐观并发控制。
- 对关键消息采用显式 acknowledgment（没 ack 就持续重试）。
- 定期做 state-sync checkpoint，尽早检测漂移。

**验证缺口（21.30%）。** 输出缺少独立校验。

例子：

- 一个 agent 宣称成功，但没有任何人验证。
- 一串 agent 流水线层层信任前一个 agent 的输出。
- 测试覆盖没有覆盖到涌现出来的组合行为。

缓解方式：

- 引入独立 verifier agent（见 Lesson 13）。它应只读，并拥有独立信息来源。
- 写明确的 handoff contract，例如：“A 的输出必须先通过 checker C，B 才能开始。”
- 对结果做 outcome logging，便于事后分析。

### 群体思维家族（arXiv:2508.05687）

当 agent 开始同质化、彼此模仿时，常见的五类相关失败是：

**Monoculture collapse。** 相同的基础模型或训练数据会带来相关性错误。三个 agent 如果共用同一个 LLM，它们也会共享那套幻觉。

**Conformity bias。** agent 会向最响亮、最自信的同伴靠拢，即便对方是错的。

**Deficient ToM。** agent 无法正确建模彼此的信念，因此协调会瓦解（见 Lesson 18）。

**Mixed-motive dynamics。** 当 agent 的激励只是部分对齐时，它们会滑向折中但谁都不满意的中间解。

**Cascading reliability failures。** 一个组件的错误模式会触发下游依赖组件的错误模式。

### 级联案例：重试风暴

一个典型的 2026 事故模式如下：

```
payment service fails 10% of requests
   ↓
order agent retries payment (exponential backoff but naive)
   ↓
each retry is a new order-inventory check
   ↓
inventory service sees 2x normal load
   ↓
inventory service starts timing out
   ↓
every order retries inventory check
   ↓
inventory service sees 10x normal load
   ↓
cluster goes down
```

修复方式很传统：**circuit breaker**。当下游错误率超过阈值时，直接短路，返回缓存结果或默认结果。同时还要给每个请求设置有上限的 retry budget。

Circuit breaker 是少数几种几乎可以原封不动从分布式系统领域借到多智能体系统里的缓解手段。

### 记忆污染（复习）

呼应 Lesson 13：一个 agent 的幻觉会被写入共享记忆，随后下游 agent 把这条被污染的内容当作事实继续推理。从 MAST 的角度看，这属于共享记忆层上的 verification gap。

它的症状是准确率缓慢衰退。你不会看到一次明确的 crash，而是会看到系统慢慢漂移，这也是它最难做根因定位的地方。

缓解方式：append-only log、provenance、不可写 verifier。这些在 Lesson 13 已经覆盖过。

### STRATUS：专门负责故障检测的智能体

STRATUS（NeurIPS 2025）报告称，当你部署以下三类 agent 时，缓解成功率可以提升 1.5 倍：

- **Detection agent。** 监控症状模式，例如高分歧、重试激增、准确率漂移。
- **Diagnosis agent。** 基于症状推断最可能的 MAST 根因类别。
- **Validation agent。** 在缓解动作执行后，确认症状是否真正清除。

这本质上就是把 SRE 风格的 incident response 套进 agent system。三个角色都可以是带专门 prompt 的 LLM agent。

### 失败模式审计

到 2026 年，一个常见最佳实践是按年或按大版本做一次 failure-mode audit：

1. **抽样轨迹。** 收集大约 1000 条真实执行轨迹。
2. **归类。** 把每条轨迹中的失败映射到 MAST 与 Groupthink 类别。
3. **计算分类型失败率。** 看哪些类别在你的系统中占主导。
4. **缓解手段排序。** 哪种修复方式能消灭最多失败？
5. **挑选 2 到 3 项缓解手段。** 实施它们，并在下一季度重新审计。

真正重要的不是某次具体选择，而是这套纪律本身。没有审计，失败就会混进噪声里，永远得不到系统性治理。

### 当系统悄悄失败

最危险的失败类型是静默正确性失败。一个系统如果是显性失败，也就是崩溃、异常或告警，至少还能被监控到；一个系统如果持续生成“看起来合理但其实错误”的输出，异常日志根本抓不住它。这也是为什么 verification gap 虽然只占 21.30%，但单次失败成本往往最高。

你需要投入在：

- 抽样人工复核。
- golden dataset 回归测试。
- 对关键输出做 cross-agent cross-check。

### 快失败与慢失败

有些失败是立刻出现的，有些则是慢慢积累出来的。立刻出现的失败，例如超时、schema mismatch、auth error，检测成本低；慢失败，例如 memory poisoning、monoculture drift、role ambiguity，检测和预防都更贵。

2026 年的工程动作是：为慢失败布置代理指标，在它变成显性错误之前先抓到漂移。agreement rate、retry rate、output-length distribution，以及连续 agent 版本之间的 edit-distance，都是有价值的 proxy。

```figure
a5-retry-cascade
```

## 动手构建

`code/main.py` 实现了：

- `FailureTaxonomy`：把模拟 incident 归类到 MAST 与 Groupthink 类别。
- `CircuitBreaker`：经典模式；当错误率超过阈值就打开。
- `RetryStormSimulator`：演示级联失败，并支持切换 circuit breaker 的开与关。
- `DetectionAgent`：一个脚本化的 STRATUS 风格症状匹配器。

运行：

```
python3 code/main.py
```

预期输出：

- 不启用 circuit breaker 时，retry storm 会让 inventory error 爆炸增长（模拟）。
- 启用 circuit breaker 时，错误会被限制在阈值附近，并进入 degraded mode。
- detection agent 会识别该模式，并给出对应的 MAST 类别。

## 实际使用

`outputs/skill-mast-auditor.md` 用来对多智能体系统执行一次 MAST 风格的 failure-mode audit：从 trace 出发，完成归类，再输出缓解手段排序。

## 交付成果

生产环境中的失败模式纪律：

- **每季度做一次 MAST 审计。** 不是每年一次。系统长大后，主导类别会变化。
- **到处都要有 circuit breaker。** 每个对下游依赖服务的外呼都应有；默认开启阈值可从 5–10% 错误率开始。
- **维护 golden dataset。** 数据量可以小，但必须高质量、人工审核过，并按周做回归测试。
- **部署 STRATUS 三件套。** Detection、Diagnosis、Validation agent 负责盯生产；最开始可以只上 detection，等症状噪声大了再补 diagnosis。
- **设 failure budget。** 按失败类别定义明确 SLO；一旦超预算，就要触发 stop-shipping 讨论。

## 练习

1. 运行 `code/main.py`。确认 circuit breaker 能压住 retry storm。调节 failure threshold，观察权衡关系。
2. 实现一个 **slow-failure proxy**：统计 3 个并行 agent 之间的 agreement rate。当它突然下跌时触发告警。再通过逐步提高输出相关性来模拟 monoculture drift。
3. 阅读 Cemri et al.（arXiv:2503.13657）。从他们的 7 个 MAS 中挑一个，映射出其前 3 大失败类别。它和 MAST 的预测是否一致？
4. 阅读 Groupthink 论文（arXiv:2508.05687）。判断五类模式里哪一种最难在生产环境中检测，并提出一个 proxy metric。
5. 为你熟悉的一个多智能体系统设计一套 STRATUS 风格的 detection-diagnosis-validation 三件套。detection 看哪些症状？diagnosis 推荐哪些缓解动作？validation 如何确认它们生效？

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------------|------------------------|
| MAST | “2026 年的分类法” | Cemri 2025 提出的框架；含 3 大根类别与 14 个子类型。 |
| 规格问题 | “角色含糊” | 任务或角色定义不足；智能体不知道自己该做什么。 |
| 协调失败 | “状态漂移” | 智能体之间通信或同步失效。 |
| 验证缺口 | “没人检查” | 输出未经独立验证便被接受。 |
| 群体思维家族 | “同质化失败” | 单一文化、从众、ToM 缺失、混合动机与级联错误。 |
| 单一文化崩溃 | “同一模型，同类幻觉” | 共享基础模型或训练数据造成的相关错误。 |
| 重试风暴 | “级联式错误放大” | 一次失败触发重试，重试又进一步放大下游负载。 |
| 熔断器 | “错误率过高时快速失败” | 错误率超过阈值时打开，改为短路或返回默认响应。 |
| STRATUS | “事件响应三件套” | 检测、诊断和验证三类智能体；缓解成功率提高 1.5 倍。 |
| 记忆污染 | “幻觉扩散” | 共享记忆被污染，下游智能体在错误事实之上继续推理。 |

## 延伸阅读

- [Cemri et al. — Why Do Multi-Agent LLM Systems Fail?](https://arxiv.org/abs/2503.13657) - MAST 分类法，NeurIPS 2025
- [Groupthink failures in multi-agent LLMs](https://arxiv.org/abs/2508.05687) - 单一文化、从众，以及五大家族失败模式
- [STRATUS — specialized agents for MAS incident response](https://neurips.cc/) - NeurIPS 2025 proceedings 条目（detection、diagnosis、validation）
- [Release It! — stability patterns (Nygard)](https://pragprog.com/titles/mnee2/release-it-second-edition/) - circuit breaker 的经典参考
- [Anthropic — Multi-agent research system](https://www.anthropic.com/engineering/multi-agent-research-system) - 生产环境失败模式相关经验
