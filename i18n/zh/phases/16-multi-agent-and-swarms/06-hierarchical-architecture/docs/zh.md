# 分层架构及其失败模式

> 分层架构就是嵌套的 Supervisor：Manager 管理 Sub-manager，Sub-manager 再管理 Worker。CrewAI 的 `Process.hierarchical` 是教科书式实现：一个 `manager_llm` 动态委派任务并验证输出。LangGraph 中的等价形式是 `create_supervisor(create_supervisor(...))`。当任务本身确实对应一张组织结构图时，这种模式最自然；它也最容易陷入管理循环——Manager 分配工作不当、误解下级输出或无法达成共识。顺序模式往往反而胜出。

**Type:** 学习 + 构建
**Languages:** Python（标准库）
**Prerequisites:** 第 16 阶段 · 05（监督者模式）
**Time:** 约 60 分钟

## 问题

一旦理解 Supervisor 模式，下一步自然会问：“如果 Worker 自己也是 Supervisor 呢？”团队下面有子团队，公司下面有多层部门，分层架构就是对这种结构的映射。

问题在于，LLM Manager 与人类管理者并不相同。人类管理者会稳定掌握下属能力的先验认知；LLM Manager 则每一轮都根据当时上下文重新推理组织结构。上下文只要发生微小漂移，整棵树就可能错误分配工作。

## 核心概念

### 结构

```
                 Manager
                 ┌─────┐
                 └──┬──┘
           ┌────────┴────────┐
           ▼                 ▼
       Sub-Mgr A         Sub-Mgr B
       ┌─────┐           ┌─────┐
       └──┬──┘           └──┬──┘
         ┌┴──┬──┐          ┌┴──┐
         ▼   ▼  ▼          ▼   ▼
       W1  W2  W3         W4  W5
```

每个内部节点都负责规划、委派和综合，只有叶节点真正执行工作。

### 擅长的场景

- **清晰的组织映射。** 如果现实任务确实按部门划分（“法务审查文档，财务审查文档，工程团队审查文档，最后为高管汇总”），分层结构能够把它明确表达出来。
- **局部综合。** 每个 Sub-manager 先综合本团队的输出，再交给顶层 Manager。顶层 Manager 看到的是三个 Sub-manager 摘要，而不是十五个 Worker 输出。

### 容易出错的地方

2026 年的事故复盘反复发现三种失败模式：

1. **任务分配错误。** Manager 读取目标后，对拆解方式产生幻觉，把任务委派给错误的 Sub-manager。Sub-manager 会忠实执行收到的任务，因此错误直到顶层综合时才会显现——与原本可由人发现的位置相隔了一层。
2. **误解输出。** Sub-manager 返回“无法验证陈述 X”，顶层 Manager 却总结为“陈述 X 尚未得到确认”。含义在每一层都会漂移。
3. **协调循环。** 两个 Sub-manager 意见不一致；顶层 Manager 要求它们协调；它们重新向下委派；Worker 重新运行；Sub-manager 返回略有不同的答案；循环继续。CrewAI 的 `Process.hierarchical` 使用步骤上限防止这种情况，但上限本身又成了一个超参数。

### 决定性问题

在顺序模式（线性流水线）与分层模式之间选择时，要问：任务是否真的包含彼此独立的子团队，还是只是一条伪装成树的线性流程？如果是后者，使用顺序模式；如果是前者，可以使用分层模式，但必须明确规划协调预算。

### 基于角色的框架实现

CrewAI 的 `Process.hierarchical` 在专业 Crew 之上连接一个 Manager LLM。该 Manager：

- 接收顶层任务，
- 将子任务分配给 Crew，
- 评估 Crew 输出，
- 决定接受、重新委派还是继续迭代。

文档：https://docs.crewai.com/en/introduction（在 Core Concepts 下查找“Hierarchical Process”）。

### 基于图的框架实现

LangGraph 使用嵌套的 `create_supervisor` 调用。内层 Supervisor 拥有自己的图；外层 Supervisor 则把内层图视为不透明节点。相比 CrewAI，这种方式更容易调试（可以分别逐步执行每张图），但更难表达树结构的动态重塑。

参考资料：https://reference.langchain.com/python/langgraph-supervisor。

```figure
swarm-hierarchy-token
```

## 动手构建

`code/main.py` 运行一个三层分级结构：

- 顶层 Manager：将任务拆成“engineering”和“legal”两个分支，
- Engineering Sub-manager：再拆成“frontend”和“backend”两个 Worker，
- Legal Sub-manager：管理一个 Worker。

演示会对比顺利路径（所有人达成一致）和一条**受扰路径**：顶层 Manager 在拆解时把“legal”误标为“finance”，然后观察错误如何级联——Sub-manager 忠实地执行财务工作，顶层综合器报告财务结果，原始法务问题却无人回答。

运行：

```
python3 code/main.py
```

输出会将“要求交付的内容”与“实际交付的内容”清楚地并排展示，覆盖两条路径。

## 实际使用

`outputs/skill-hierarchy-fitness.md` 用于评估给定任务应采用分层模式、顺序模式还是扁平 Supervisor。输入包括任务描述、组织结构和协调预算；输出为模式建议，以及需要防范的具体失败模式。

## 交付成果

如果要交付分层架构，请做到：

- **将树深限制为 2。** 三层已经会让大部分错误逃离可观测范围。
- **明确协调预算。** 设置顶层 Manager 必须作出决定前允许的最大轮数，通常为 2。
- **每次综合都保留来源。** 每个节点的摘要都必须引用生成它的叶节点输出。
- **对拆解漂移告警。** 记录 Manager 每一步的任务拆解，并与用户查询比较。拆解不再覆盖原查询时，应触发告警。

## 练习

1. 运行 `code/main.py`，比较顺利路径与受扰路径。经过多少层 Manager Handoff 后，顶层输出会完全偏离用户的问题？
2. 添加第四层（top → sub → sub-sub → worker）。测量随着深度增加，受扰路径自行纠正和彻底偏离各自出现的频率。
3. 在每个 Sub-manager 下实现一个“Canary”Worker，让它始终原封不动地接收用户原始问题。使用 Canary 的答案检测拆解漂移。Canary 与综合答案不一致时，Manager 应如何响应？
4. 阅读 CrewAI 的 `Process.hierarchical` 文档。找出 CrewAI 采用的一项具体 Guardrail（步骤上限、manager_llm 约束），并说明它针对哪种失败模式。
5. 比较嵌套 LangGraph Supervisor 与 CrewAI Hierarchical。哪一种更容易发现协调循环？

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------------|------------------------|
| 分层架构 | “组织结构图模式” | Supervisor 管理 Supervisor；只有叶节点执行工作。 |
| Manager LLM | “老板” | 在内部节点进行拆解、分配和验证的 LLM。 |
| 拆解漂移 | “老板偏离了目标” | 顶层 Manager 的拆分不再覆盖原始问题。 |
| 协调循环 | “无休止的会议” | Sub-manager 意见不一，顶层重新委派，Worker 重跑，直到预算耗尽。 |
| 两层上限 | “不要超过两层” | 经验性 Guardrail：三层以上会破坏可观测性。 |
| Canary 问题 | “每层的真实基准” | 始终原封不动接收原始查询、用于检测漂移的 Worker。 |
| 来源链 | “谁说了什么” | 从每次综合追溯到生成它的叶节点输出。 |

## 延伸阅读

- [CrewAI 简介——Process.hierarchical](https://docs.crewai.com/en/introduction)——带 Manager LLM 的经典分层实现
- [LangGraph Supervisor 参考](https://reference.langchain.com/python/langgraph-supervisor)——通过 `create_supervisor` 嵌套 Supervisor
- [Anthropic 工程文章——Research 系统](https://www.anthropic.com/engineering/multi-agent-research-system)——Anthropic 为何有意选择扁平 Supervisor 而非分层模式
- [Cemri 等——多智能体 LLM 系统为何失败？](https://arxiv.org/abs/2503.13657)——MAST 分类法；协调失败章节记录了拆解漂移
