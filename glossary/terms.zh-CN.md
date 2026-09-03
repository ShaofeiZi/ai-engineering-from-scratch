# AI 工程术语表

当课程、论文、模型卡或代码审查引入术语的速度快于解释速度时，请查阅这份术语表。先按确切术语或别名搜索并阅读直接定义，再通过实践说明把概念关联到你能够构建的系统。

每个条目都属于一个学习类别。`相关术语` 会给出接下来值得了解的概念，但不会强制规定固定路径。定义描述的是通用工程含义，不同提供商的具体行为可能有所差异。当 API 契约或模型卡与通用定义不一致时，以当前官方文档为准。

十二个类别分别是：数学与训练、模型与推理、数据与表征、检索与生成、提示与上下文、智能体与工具、评估与安全、AI 原生开发、基础设施与服务、可靠性与运维、安全与治理、多模态系统。

## A

### Activation Checkpointing
- **类别：** 数学与训练
- **实际含义：** 一种训练时的内存优化技术，只保存部分前向传播激活值，并在反向传播时重新计算被省略的部分。
- **为什么重要：** 它通过用额外计算换取更少的激活值存储，让你在固定内存预算下训练更大的模型或更长的序列。
- **实践建议：** 对内存占用高的 transformer block 做检查点，测量额外的 step 时间，并把恢复用检查点与激活重计算设置分开管理。
- **常见误区：** 激活检查点不是可恢复的训练检查点。它只能帮助单次前向和反向传播适配内存，不能在任务崩溃后恢复运行。
- **相关术语：** Autograd, Backpropagation, Checkpoint, Mixed Precision
- **来源：** [以次线性内存成本训练深度网络](https://arxiv.org/abs/1604.06174)

### Activation Function
- **类别：** 数学与训练
- **常见说法：** 层与层之间的非线性操作。
- **实际含义：** 在线性层或仿射层之后施加的函数，用来引入非线性。没有它，带权重和偏置的多层组合最终仍会退化成一个仿射变换。ReLU、GELU 和 SiLU 都是常见选择。它的选型会直接影响训练时梯度能否顺畅传播。
- **延伸学习：** [激活函数](../phases/03-deep-learning-core/04-activation-functions/)
- **相关术语：** ReLU, Gradient, Backpropagation

### Adam (Optimizer)
- **类别：** 数学与训练
- **常见说法：** 你不假思索就会用的优化器。
- **实际含义：** Adaptive Moment Estimation。它把梯度的指数滑动平均与梯度平方的指数滑动平均结合起来，进行偏差校正，并为每个参数自适应调整更新尺度。它是一个很有用的基线，但仍然需要合适的学习率和调度策略。
- **常见误区：** Adam 是很强的基线，但不是放之四海而皆准的最佳优化器。
- **来源：** [Adam 论文](https://arxiv.org/abs/1412.6980)
- **相关术语：** AdamW, Optimizer, Learning Rate

### AdamW
- **类别：** 数学与训练
- **常见说法：** 修正了权重衰减处理方式的 Adam。
- **实际含义：** Adam 的一种变体，把权重衰减从基于梯度的参数更新中解耦出来。相比把 L2 惩罚直接加进 Adam 自适应缩放后的梯度里，这样更容易理解参数收缩的实际行为。
- **常见误区：** 权重衰减解耦并不意味着 AdamW 在所有场景下都是最优。最佳优化器和调度仍取决于模型、数据以及训练规模。
- **来源：** [解耦权重衰减正则化](https://arxiv.org/abs/1711.05101)
- **相关术语：** Adam (Optimizer), Weight Decay, Optimizer

### Admission Control
- **类别：** 可靠性与运维
- **实际含义：** 一种接收前关口，根据系统当前容量、优先级和策略，决定请求是否可以进入有界队列或服务。
- **为什么重要：** 在可控边界拒绝超量工作，可以保护已接纳的请求，避免队列膨胀、超时级联和资源耗尽。
- **实践建议：** 估算请求成本，检查租户和系统容量，原子性地预留所需预算，并在拒绝时明确指出过载范围。只有当问题是暂时性的且调用方的重试预算允许时，才提供重试建议。
- **常见误区：** 准入控制发生在接收之前。负载削减则可以在入口、队列、依赖服务或其他过载边界拒绝或移除工作。
- **相关术语：** Load Shedding, Backpressure, Rate Limit, Saturation
- **来源：** [Google SRE：处理过载](https://sre.google/sre-book/handling-overload/)

### Agent
- **类别：** 智能体与工具
- **常见说法：** 一个会独立思考并行动的自主模型。
- **实际含义：** 一种软件系统，让模型能够围绕目标选择行动、观察工具或环境返回的结果，并在编排策略下继续推进。Agent 可以通过循环、状态机、工作流引擎或人工审批来运作。模型只是其中一个组件，不是整个系统本身。
- **为什么重要：** 可靠性来自模型外围的 harness、工具契约、状态、权限和验证机制。
- **实践建议：** 一个 coding agent 会读取仓库上下文、提出补丁、在沙箱里运行测试，并在部署前停下来等待审批。
- **常见误区：** 自主性是一种被授予权限的程度，而不是每个 agent 都必须具备的属性。
- **延伸学习：** [代理循环](../phases/14-agent-engineering/01-the-agent-loop/)
- **相关术语：** Agent Harness, Agent State, Tool Contract, Human-in-the-Loop (HITL)

### Agent Harness
- **类别：** 智能体与工具
- **实际含义：** 围绕模型运行的运行时层，负责组装上下文、暴露工具、管理状态、执行限制、记录 trace，并决定 agent 何时继续、重试、提问或停止。
- **为什么重要：** 两个使用同一模型的系统，可能因为 harness 提供的上下文、工具、反馈和安全边界不同，而表现出很大差异。
- **实践建议：** 你的 harness 可以限制 agent 最多调用五次工具，在每次补丁被接受后持久化一个检查点，并要求在完成前通过指定测试命令。
- **常见误区：** Harness 的范围比 prompt 模板更广，但又比完整产品更窄。
- **延伸学习：** [最小代理工作台](../phases/14-agent-engineering/32-minimal-agent-workbench/)
- **相关术语：** Agent, Tool Contract, Agent State, Verification Gate, Sandbox

### Agent Memory
- **类别：** 智能体与工具
- **实际含义：** 存储在模型之外、并会在后续 agent 步骤中按需取回的信息，例如过往决策、用户偏好、任务经历或已验证事实。
- **为什么重要：** 它让 agent 能在单个上下文窗口之外保持连续性，而不必把所有历史事件都塞进每次 prompt。
- **实践建议：** 把任务结果以带 provenance 的紧凑形式存储起来，只在相关时取回，并允许用户查看或纠正持久化的个人信息。
- **常见误区：** Agent memory 不等同于 agent state。State 跟踪当前这次运行；memory 则保留可能供未来运行使用的选定信息。
- **相关术语：** Agent State, 上下文工程, Checkpoint, Semantic Cache
- **来源：** [生成式代理](https://arxiv.org/abs/2304.03442)

### Agent State
- **类别：** 智能体与工具
- **实际含义：** agent 在步骤之间显式携带的数据，例如当前目标、已完成动作、工具结果、未决问题、预算、审批状态和产物引用。
- **为什么重要：** 显式 state 让长任务更易恢复、可检查，也更不依赖模型从对话记录里重建进度。
- **实践建议：** 把已选 issue、修改文件、最新测试结果和剩余检查项存入一个强类型对象，并在每次动作后更新它。
- **常见误区：** State 不等同于对话历史。对话记录是证据；state 是用来决定下一步怎么做的紧凑操作记录。
- **延伸学习：** [仓库记忆与状态](../phases/14-agent-engineering/34-repo-memory-and-state/)
- **相关术语：** Checkpoint, Durable Execution, 上下文工程, Handoff

### Agent Skill
- **类别：** 智能体与工具
- **实际含义：** 一种可被发现的过程化指令目录，其入口文件是 `SKILL.md`，并可附带 references、scripts 和 assets，供兼容运行时按阶段加载。
- **为什么重要：** 它把可复用的任务知识从单次对话中独立打包出来，同时让更深层上下文和确定性辅助工具按需可用。
- **实践建议：** 发布一个精炼的名称和路由说明，只在激活后加载工作流，并在任务推进到相应阶段时读取分支专用参考资料。
- **常见误区：** 激活 skill 只是提供上下文。它不会自动暴露工具、授予权限、创建沙箱，也不能证明产出一定正确。
- **延伸学习：** [Agent Skills：可移植契约与运行时边界](../phases/13-tools-and-protocols/22-skills-and-agent-sdks/)
- **相关术语：** Skill Bundle, Skill Catalog, Skill Invocation, Progressive Disclosure, MCP (Model Context Protocol)
- **来源：** [Agent Skills 规范](https://agentskills.io/specification)

### AI Risk Assessment
- **类别：** 安全与治理
- **实际含义：** 一种成文的分析，用于说明 AI 系统如何影响个人、组织和环境，内容包括上下文、风险源、发生可能性、影响、控制措施、残余风险以及监测责任。
- **为什么重要：** 风险不只由模型能力决定。部署场景、受影响群体、人工权限、数据和系统集成方式，都会改变可能的危害和所需控制措施。
- **实践建议：** 定义预期用途和受影响方，识别可信的失败与滥用场景，为控制措施指定负责人，记录残余风险，并为重大变化设定复审触发条件。
- **常见误区：** 风险评估是在既定假设下为决策提供支持。它不是一次性的安全证书，也不能证明所有风险都已经被发现。
- **相关术语：** Threat Model, Guardrails, Human-in-the-Loop (HITL), Data Classification
- **来源：** [NIST AI 风险管理框架](https://www.nist.gov/itl/ai-risk-management-framework)

### Alignment
- **类别：** 评估与安全
- **常见说法：** 让 AI 更安全。
- **实际含义：** 为了让模型或 AI 系统在常规场景和对抗场景中，都尽量符合预期目标、约束和人类偏好而进行的工作。
- **为什么重要：** 一个系统可能在优化表面指标的同时违背用户的真实意图，因此 alignment 不仅需要模型训练，还需要评测、监督和系统级控制。
- **相关术语：** Guardrails, Evaluation (Eval), Human-in-the-Loop (HITL)

### Approval Gate
- **类别：** 智能体与工具
- **实际含义：** 一种控制点，在获得授权人员或策略许可之前，会阻止具有后果性的动作继续执行。
- **为什么重要：** 它在保留可逆工作自动化能力的同时，限制不确定模型决策的影响范围。
- **实践建议：** 可以让 agent 起草数据库迁移并在一次性数据库上运行，但任何生产执行都必须由负责人审批。
- **常见误区：** Approval gate 关心的是动作是否被授权；verification gate 关心的是证据是否表明动作正确。
- **延伸学习：** [验证门](../phases/14-agent-engineering/38-verification-gates/)
- **相关术语：** Human-in-the-Loop (HITL), Verification Gate, Least Privilege

### Approximate Nearest Neighbor (ANN)
- **类别：** 检索与生成
- **实际含义：** 一种搜索方法，无需将查询向量与所有已存向量逐一穷举比较，就能返回大概率属于最近邻的向量。
- **为什么重要：** 近似方法让大规模向量索引具备实用性，但也引入了搜索速度、内存占用和检索召回之间可量化的权衡。
- **实践建议：** 用留出查询集调优索引和搜索参数，然后把延迟和 Recall@K 一起报告，而不是默认所有真实近邻都能被找到。
- **常见误区：** ANN 描述的是一种搜索目标和权衡；HNSW 则是实现它的一种具体索引算法。
- **相关术语：** Vector Database, HNSW, Cosine Similarity, Recall@K
- **来源：** [使用 HNSW 的高效且鲁棒的近似最近邻搜索](https://dl.acm.org/doi/10.1109/TPAMI.2018.2889473)

### Attention
- **类别：** 模型与推理
- **常见说法：** 模型如何关注重要 token。
- **实际含义：** 一种通过比较 query 向量和 key 向量、对结果分数进行归一化，并用这些分数加权组合 value 向量来形成上下文化表示的机制。mask、位置规则或稀疏模式都可以限制哪些位置参与计算。
- **为什么重要：** Attention 让模型能够在序列位置之间路由信息，但它本身并不能解释或证明模型真正理解了什么。
- **常见误区：** Attention 权重是计算系数，不是对模型推理过程的可靠解释。
- **延伸学习：** [从零实现自注意力](../phases/07-transformers-deep-dive/02-self-attention-from-scratch/)
- **来源：** [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- **相关术语：** Self-Attention, Transformer, KV Cache

### Audio Token
- **类别：** 多模态系统
- **实际含义：** 由音频编解码器或 tokenizer 为音频信号的短片段或某种特征产生的离散标识符，有时会跨多个 codebook 编码。
- **为什么重要：** 离散音频表征让序列模型可以用面向 token 的架构来处理、预测、存储或生成声音。
- **实践建议：** 让 codec 与模型一起版本化，保留采样率和 codebook 元数据，测量重建质量，并区分语义型音频 token 与波形压缩型 token。
- **常见误区：** 音频 token 不是固定时长、音素或单词。它的含义和时间跨度取决于 tokenizer 与 codebook 的设计。
- **延伸学习：** [神经音频编解码器](../phases/06-speech-and-audio/13-neural-audio-codecs/)
- **相关术语：** Token, Embedding, Automatic Speech Recognition (ASR), Multimodal Model
- **来源：** [SoundStream](https://arxiv.org/abs/2107.03312)

### Audit Log
- **类别：** 安全与治理
- **实际含义：** 一种持久、受访问控制保护的记录，用来保存与安全或问责相关的事件，包括谁或什么执行了动作、发生了什么变更、发生时间以及最终状态。
- **为什么重要：** 带来实际后果的 agent 行为需要留下证据，以支持调查、策略审查和责任追踪，而不只是性能调试。
- **实践建议：** 记录工具授权、审批决策、外部写入、策略版本和产物标识符，同时对敏感信息做脱敏，并限制日志访问权限。
- **常见误区：** Trace 有助于诊断某一次执行路径；audit log 则保存跨执行、跨时间维度所需的问责事件。
- **相关术语：** Trace, Observability, Approval Gate, Provenance Attestation
- **来源：** [NIST SP 800-92](https://csrc.nist.gov/pubs/sp/800/92/final)

### Autograd
- **类别：** 数学与训练
- **常见说法：** 自动求梯度。
- **实际含义：** 一种记录或变换 tensor 运算的系统，从而能够计算导数，通常基于反向模式自动微分。你只需要写前向计算，框架会推导出反向传播所需的梯度。
- **延伸学习：** [链式法则与自动微分](../phases/01-math-foundations/05-chain-rule-and-autodiff/)
- **相关术语：** Backpropagation, Gradient, Tensor

### Automatic Speech Recognition (ASR)
- **类别：** 多模态系统
- **实际含义：** 把语音信号映射为转写文本的任务与系统流水线，通常还会附带 token 或分段级的时间信息和置信度信息。
- **为什么重要：** 语音接口依赖的不只是语言建模。声学变化、分段、解码、词汇和领域条件都会影响最终转写结果。
- **实践建议：** 从语言、说话人、噪声和领域等维度评估词错误率或字错误率；当下游 grounding 需要时保留时间戳；并测试生产环境中使用的精确音频预处理流程。
- **常见误区：** ASR 负责转写说了什么。判断是谁在说话需要 diarization 或 speaker recognition，而翻译和意图理解又是另外的任务。
- **延伸学习：** [语音识别与 ASR](../phases/06-speech-and-audio/04-speech-recognition-asr/)
- **相关术语：** Audio Token, Encoder, Tokenization, Multimodal Model
- **来源：** [连接时序分类](https://www.cs.toronto.edu/~graves/icml_2006.pdf)

### Autoregressive
- **类别：** 模型与推理
- **常见说法：** 模型一次生成一个词。
- **实际含义：** 一种分解方式，其中每个输出 token 都由它之前的 token 来预测。在生成过程中，被选中的 token 会追加到序列末尾，并成为下一次预测上下文的一部分。
- **常见误区：** 这里的单位是 token，不一定是单词；而且生成时也不一定总是选择概率最高的 token，还可以使用其他 decoding 方法。
- **相关术语：** Token, Temperature, KV Cache

### Autoscaling
- **类别：** 基础设施与服务
- **实际含义：** 一种控制回路，在预设边界内根据观测到的需求、资源使用情况或应用指标，调整服务工作节点的数量或容量。
- **为什么重要：** AI 工作负载变化往往快于人工扩容速度，因此扩缩容决策必须同时考虑模型加载时间、加速器可用性、排队情况和请求成本。
- **实践建议：** 依据与有效工作量相关的需求信号进行扩容，设置最小预热容量，限制缩容抖动，并在新副本接收流量前确认其通过 readiness 检查。
- **常见误区：** 自动扩缩容只能增减容量。它不会让一个已过载的依赖凭空变快，也不能保证能及时拿到足够硬件。
- **延伸学习：** [Kubernetes 上的 GPU 自动伸缩](../phases/17-infrastructure-and-production/03-gpu-autoscaling-kubernetes/)
- **相关术语：** Model Serving, Saturation, Readiness Probe, Backpressure
- **来源：** [Kubernetes 水平 Pod 自动伸缩](https://kubernetes.io/docs/tasks/run-application/horizontal-pod-autoscale/)

### Availability
- **类别：** 可靠性与运维
- **实际含义：** 在既定测量边界下，用户能够获得定义中“可接受服务”的合格交互或时间窗口所占比例。
- **为什么重要：** 服务即使在运行，用户也未必能完成有价值的请求，因此 availability 必须和用户可见的成功绑定，而不能只看进程是否存活。
- **实践建议：** 定义合格事件和可接受结果，只排除有文档记录的特殊情形，在固定窗口内计算指标，并同时调查完全失败和持续性的部分退化。
- **常见误区：** Availability 只是可靠性结果的一部分。它并不能描述延迟、正确性、安全性，或每类用户群体的具体体验。
- **相关术语：** Service Level Indicator (SLI), Service Level Objective (SLO), Error Budget, Incident Response
- **来源：** [Google SRE：服务级目标](https://sre.google/sre-book/service-level-objectives/)

## B

### Backpressure
- **类别：** AI 原生开发
- **实际含义：** 一种流量控制机制，当下游组件无法以当前速率安全处理工作时，它会减慢或拒绝上游流量。
- **为什么重要：** 没有 backpressure，排队中的 agent 运行、工具调用或流式事件就可能耗尽内存、触发限流，并放大重试风暴。
- **实践建议：** 当评估器队列达到上限时，应暂停新的 agent 任务，或返回可重试响应，而不是继续接收无界工作量。
- **常见误区：** Backpressure 在失败发生前保护系统容量；circuit breaker 则是在失败表明依赖已经不健康后，停止继续调用。
- **相关术语：** Rate Limit, Retry with Backoff, Circuit Breaker

### Backpropagation
- **类别：** 数学与训练
- **常见说法：** 神经网络如何学习。
- **实际含义：** 链式法则的一种高效应用，它会把标量 loss 的导数沿计算图向后传播。它负责计算梯度；优化器则利用这些梯度更新参数。
- **常见误区：** 反向传播负责计算梯度，并不决定使用什么更新规则或学习率。
- **名称由来：** 导数信息从 loss 向更早的运算节点反向传播。
- **延伸学习：** [从零实现反向传播](../phases/03-deep-learning-core/03-backpropagation/)
- **相关术语：** Autograd, Gradient, Optimizer

### Batch Size
- **类别：** 数学与训练
- **常见说法：** 一次处理多少个样本。
- **实际含义：** 在一次优化器更新前，有多少个样本的 loss 会共同贡献给一次梯度估计。更大的 batch 可以提升硬件利用率、降低梯度噪声，但也需要更多内存，并且往往需要不同的学习率或调度策略。
- **常见误区：** 不存在放之四海而皆准的 batch size 范围，也不存在“batch 变大多少，学习率就该同比变大多少”的统一规则。
- **相关术语：** Learning Rate, Gradient, Optimizer

### Benchmark Contamination
- **类别：** 评估与安全
- **实际含义：** 评测样本与用于预训练、调优、提示、筛选或以其他方式改进被评估系统的数据之间发生重叠或信息泄漏。
- **为什么重要：** 污染会让 benchmark 分数反映的是系统是否见过相关内容，而不是它对未见任务的泛化能力。
- **实践建议：** 跟踪数据集 provenance，在训练数据源中搜索完全重复和近重复内容，保留私有测试样例，并用新编写的样例持续刷新公开 eval。
- **常见误区：** 污染不只是完全拷贝。改写后的文本、答案键、benchmark 元数据以及反复针对测试集做 prompt 调优，也都可能泄漏评测信息。
- **相关术语：** Data Leakage, Data Deduplication, Eval Set, Exact Match (EM)
- **来源：** [研究现代大语言模型基准中的数据污染](https://arxiv.org/abs/2311.09783)

### BM25
- **类别：** 检索与生成
- **实际含义：** 一种词法排序函数，会根据查询词匹配情况为文档打分，同时考虑词项稀有度、重复出现次数和文档长度。
- **为什么重要：** 它是精确词项检索的强基线，并且在处理标识符、稀有词和领域特定短语时，能与 dense retrieval 形成互补。
- **实践建议：** 先用 BM25 和 dense search 各自召回候选项，再合并它们的排序，最后在引入更昂贵的 reranker 之前评估合并结果。
- **常见误区：** BM25 并不能直接理解语义相似性，而且它的分数在不同查询或索引配置之间也没有统一可比的意义。
- **相关术语：** Hybrid Retrieval, Dense Retrieval, Reranker, RAG (Retrieval-Augmented Generation)
- **来源：** [概率相关性框架：BM25 及其扩展](https://doi.org/10.1561/1500000019)

### Byte Pair Encoding (BPE)
- **类别：** 数据与表征
- **实际含义：** 一种子词切分方法，通过不断合并高频相邻单元，从训练文本中构建固定词表。
- **为什么重要：** 它在词表大小与将稀有词、未见词拆成更小单元进行表示的能力之间取得平衡。
- **实践建议：** 只在获批的语料划分上训练 tokenizer，让其 merge 规则与模型一起做版本管理，并检查它如何切分代码、多语言文本和空白字符。
- **常见误区：** BPE 只是 tokenizer 家族中的一种，不是所有模型生成 token 方式的统一描述。
- **相关术语：** Tokenization, Vocabulary, Token, Embedding
- **来源：** [使用子词单元的稀有词神经机器翻译](https://arxiv.org/abs/1508.07909)

## C

### Calibration
- **类别：** 评估与安全
- **实际含义：** 系统给出的置信度，与该置信度下预测实际正确频率之间的一致程度。
- **为什么重要：** 一个系统平均来看可能很准确，但在用户真正依赖其分数的那些场景中，仍可能表现得危险地过度自信。
- **实践建议：** 按置信度对预测分桶，将置信度与经验准确率进行比较，并在偏差不可接受时进行再校准或选择弃答。
- **常见误区：** Calibration 衡量的是置信度是否可靠，而不是整体准确率、事实性或推理质量。
- **相关术语：** Softmax, Evaluation (Eval), Precision & Recall, Logits
- **来源：** [现代神经网络的校准研究](https://proceedings.mlr.press/v70/guo17a.html)

### Canary Release
- **类别：** 可靠性与运维
- **实际含义：** 一种发布策略，在扩大 rollout 之前，先把新版本暴露给有限的一部分流量或基础设施。
- **为什么重要：** 它能在新模型、prompt、agent 或服务全面上线前，先限制缺陷影响范围，并给你提供生产环境证据。
- **实践建议：** 把一小部分符合条件的流量导向新版本，与对照组比较质量和运维指标，并在出现预定义失败时停止或回滚。
- **常见误区：** 金丝雀发布只能限制暴露范围，不能替代上线前测试、审批流程或回滚准备。
- **相关术语：** Evaluation (Eval), Observability, Rollback, Verification Gate
- **来源：** [Kubernetes Deployments：金丝雀发布](https://kubernetes.io/docs/concepts/workloads/controllers/deployment/#canary-deployment)

### Chain of Thought (CoT)
- **类别：** 提示与上下文
- **常见说法：** 要求模型展示它思考的每一步。
- **实际含义：** 在给出答案之前，用来拆解任务的中间推理过程。prompt 可以要求模型输出可见的理由，而有些系统则会使用不返回给用户的内部推理。
- **为什么重要：** 任务拆解有助于处理多步问题，但一段流畅的推理文本并不能证明答案正确，也不能证明它忠实反映了模型内部的真实计算过程。
- **实践建议：** 可以要求模型先给出简明计划，再独立核查结果，并要求它提供可验证的计算或引用，而不是只依赖一长段推理过程。
- **常见误区：** 思维链不能替代工具、测试或外部验证。
- **延伸学习：** [少样本与思维链](../phases/11-llm-engineering/02-few-shot-cot/)
- **相关术语：** Prompt Engineering, Verification Gate, Evaluation (Eval)

### Checkpoint
- **类别：** 智能体与工具
- **实际含义：** 一种可持久化的快照，用来从某个已知边界恢复。在工作流中，它保存操作状态和产物引用；在模型训练中，它可以保存参数、优化器状态、调度器状态以及训练位置。
- **为什么重要：** 长时间运行的工作流和训练任务可以在中断后恢复，而不必重放已完成工作，也不会丢失代价高昂的进度。
- **实践建议：** 可以在某个已验证步骤之后保存 agent 被接受的补丁和测试证据，也可以在训练任务停机前保存权重、优化器状态、随机状态和数据位置。
- **常见误区：** 工作流检查点和模型训练检查点都服务于恢复这一目标，但保存的状态并不相同。它们都不只是对话记录，也不只是一个没有恢复元数据的权重文件。
- **延伸学习：** [检查点保存与恢复](../phases/19-capstone-projects/47-checkpoint-save-resume/); [仓库记忆与状态](../phases/14-agent-engineering/34-repo-memory-and-state/)
- **相关术语：** Agent State, Durable Execution, Parameter, Optimizer

### Chunked Prefill
- **类别：** 基础设施与服务
- **实际含义：** 一种服务端技术，把长 prompt 的 prefill 工作切成更小、可调度的片段，从而让该 prompt 的处理过程可以与其他请求的 decode 工作交错进行。
- **为什么重要：** 否则，一个超长 prompt 可能独占加速器，拖慢正在进行的生成任务，即便总吞吐看起来正常，也会造成糟糕的尾延迟。
- **实践建议：** 应基于实测工作负载选择 chunk 策略，考虑调度开销，并在混合 prompt 长度场景下比较 prefill 完成时间、decode 延迟和 goodput。
- **常见误区：** Chunked prefill 改变的是 prompt 计算的调度方式。它不会把用户上下文拆成彼此独立的语义块，也不会改变模型的 context window。
- **延伸学习：** [vLLM 服务内部原理](../phases/17-infrastructure-and-production/04-vllm-serving-internals/)
- **相关术语：** Prefill, Decode Phase, Dynamic Batching, Tail Latency
- **来源：** [Sarathi-Serve](https://arxiv.org/abs/2403.02310)

### Chunking
- **类别：** 检索与生成
- **常见说法：** 把文档切成若干片段。
- **实际含义：** 在建立索引之前，把源材料切分成可检索单元。chunk 边界、重叠方式、元数据和文档结构，会共同决定 retrieval 是否能返回足够上下文，同时又不至于把 prompt 塞满。
- **为什么重要：** 合适的 chunking 策略取决于文档形态、查询类型、embedding 模型和评测结果。不存在通用的 token 大小或重叠比例。
- **实践建议：** 先保留标题和代码块的完整性，附加来源元数据，再基于真实问题评估检索质量，之后再调 chunk 大小。
- **相关术语：** RAG (Retrieval-Augmented Generation), Reranker, Grounding

### Circuit Breaker
- **类别：** AI 原生开发
- **实际含义：** 一种可靠性控制措施，在依赖失败次数超过阈值后暂时停止调用，并在之后探测该依赖是否恢复。
- **为什么重要：** 它能防止模型或工具的重复失败继续消耗系统其余部分的延迟预算、成本预算和处理容量。
- **实践建议：** 在提供方连续超时后打开熔断器，进行故障转移或返回受控响应，并在冷却期后允许有限的健康探测。
- **常见误区：** Circuit breaker 响应的是依赖健康状态；rate limit 控制的是允许的请求量。
- **相关术语：** Retry with Backoff, Rate Limit, Model Router, Backpressure

### CNN (Convolutional Neural Network)
- **类别：** 模型与推理
- **常见说法：** 用于图像的神经网络。
- **实际含义：** 一种通过卷积运算（让滤波器在输入上滑动）来检测局部模式的神经网络。层层堆叠卷积后，可以逐步识别更复杂的特征，例如边缘、纹理和物体。
- **常见误区：** 卷积不仅适用于图像，也适用于音频、时间序列和其他网格状数据。
- **相关术语：** Feature, Inductive Bias, Activation Function

### Coding Agent
- **类别：** AI 原生开发
- **实际含义：** 一种面向软件工作的 agent，能够检查仓库、编辑文件、运行开发工具，并利用这些工具输出推进一个有明确范围的工程任务。
- **为什么重要：** 它的价值取决于仓库上下文、工具权限、审查边界和验证流程，而不只是代码生成质量。
- **实践建议：** 给 agent 提供 issue、范围契约、仓库说明和测试命令；在接受结果前，先审查它产出的补丁和证据。
- **常见误区：** 一个只会建议文本的 coding assistant 不一定是 agent。agent 会通过工具执行动作并观察结果。
- **延伸学习：** [技能发现与渐进式披露](../phases/13-tools-and-protocols/24-skill-discovery-and-progressive-disclosure/)
- **相关术语：** Agent Harness, Repository Map, Patch, Scope Contract, Reviewer Agent

### Compensating Action
- **类别：** 智能体与工具
- **实际含义：** 当原始操作无法原子回滚时，通过一个有意设计的新操作，在语义上抵消已经发生的副作用。
- **为什么重要：** 多步 agent 工作流会跨越数据库和外部服务，此时后续失败往往无法通过单个事务撤销前面已经发生的写入。
- **实践建议：** 如果预订工作流已经扣款但预订失败，就应发起可追踪的退款，并保留这两次事件，而不是删除历史记录。
- **常见误区：** 补偿是一种新的业务动作，不是时光倒流。它本身也可能失败，因此同样需要幂等性、监控和升级处理机制。
- **相关术语：** Durable Execution, Idempotency, Checkpoint, Approval Gate
- **来源：** [Sagas](https://dl.acm.org/doi/10.1145/38713.38742)

### Content Provenance
- **类别：** 安全与治理
- **实际含义：** 关于一段媒体内容或其他数字内容的来源与编辑历史的可验证信息，包括相关参与者、工具、变换过程以及附带的声明。
- **为什么重要：** 生成式系统让“内容来源”无法仅凭外观判断，因此消费者和平台都需要可检查的证据来了解内容是如何产生的。
- **实践建议：** 把 provenance 声明绑定到内容本身，用受控身份进行签名，保留变换历史，并在证据缺失或无法验证时明确标示出来。
- **常见误区：** Provenance 可以说明是谁声明了某段历史，以及记录是否被篡改；但它不能证明内容描述的事件一定真实，也不能证明内容一定无害。
- **延伸学习：** [水印、SynthID、Stable Signature 与 C2PA](../phases/18-ethics-safety-alignment/23-watermarking-synthid-stable-signature-c2pa/)
- **相关术语：** Data Provenance, Provenance Attestation, Audit Log, Grounding
- **来源：** [C2PA 技术规范](https://c2pa.org/specifications/specifications/2.2/specs/C2PA_Specification.html)

### Context Compression
- **类别：** 提示与上下文
- **实际含义：** 在尽量保留后续模型决策所需信息的前提下，降低源材料的 token 占用量。
- **为什么重要：** 压缩可以让长任务塞进预算之内，但每一个被省略的细节，都可能让模型丢失证据、约束或尚未解决的状态信息。
- **实践建议：** 应逐字保留权威事实和标识符，对冗余历史做摘要，附上来源指针，并在代表性任务上测试压缩后的上下文。
- **常见误区：** 除非完整保留原文，否则压缩就是有损的。更短的摘要并不自动等价于原始上下文。
- **相关术语：** Token Budget, 上下文工程, Progressive Disclosure, Handoff
- **来源：** [LLMLingua](https://arxiv.org/abs/2310.05736)

### 上下文工程
- **类别：** 提示与上下文
- **实际含义：** 为模型在每一步要接收的完整信息环境做设计，包括指令、选定文件、检索到的证据、工具结果、示例、状态以及输出约束。
- **为什么重要：** 模型表现不佳，往往不是模型本身的问题，而是因为相关证据缺失、过期、顺序不当，或被噪声淹没。
- **实践建议：** 把目标、仓库规则、相关接口、近期工具输出和未决决策打包成一个紧凑的任务包，并随着状态变化持续更新。
- **常见误区：** Prompt engineering 主要关注指令怎么写；context engineering 还要决定哪些证据和状态会进入模型的工作上下文。
- **延伸学习：** [上下文工程](../phases/11-llm-engineering/05-context-engineering/)
- **相关术语：** Context Window, Progressive Disclosure, Agent State, Repository Map

### Context Window
- **类别：** 提示与上下文
- **常见说法：** 模型能记住多少内容。
- **实际含义：** 在特定模型和 API 契约下，单次模型推理可用的最大 token 容量。这个容量可能包括系统指令、消息、检索内容、工具交互以及生成输出，并受到不同服务提供方各自的计量和输出限制。
- **为什么重要：** 只有当应用主动发送或重建对话历史时，这些历史才真正可用。更大的 window 并不保证其中每个细节都会被可靠利用。
- **常见误区：** Context 是一次推理的临时输入；durable memory 则存储在模型之外，并在之后按需重新选回上下文中。
- **延伸学习：** [上下文工程](../phases/11-llm-engineering/05-context-engineering/)
- **相关术语：** Token Budget, 上下文工程, Prompt Cache, Agent State

### Continuous Batching
- **类别：** 基础设施与服务
- **实际含义：** 一种服务端调度器，会在迭代边界动态加入或移除生成请求，而不是等固定 batch 里的所有请求都完成后再统一切换。
- **为什么重要：** 自回归请求会产生不同长度的输出，因此 continuous batching 可以在不让短请求被最长请求拖住的前提下，持续保持加速器利用率。
- **实践建议：** 当容量释放时接纳新请求，跟踪每个请求的延迟，并在活动 batch 或 KV-cache 预算打满时施加 backpressure。
- **常见误区：** Continuous batching 是一种推理调度策略，不是梯度累积，也不是训练中的 batch size 技术。
- **相关术语：** Dynamic Batching, Decode Phase, Backpressure, Rate Limit
- **来源：** [Orca](https://www.usenix.org/conference/osdi22/presentation/yu)

### Contrastive Learning
- **类别：** 数学与训练
- **常见说法：** 通过比较来学习。
- **实际含义：** 一种训练方法：在 embedding 空间里把相似样本拉近，把不相似样本推远。CLIP 就采用了这种思路，对齐匹配的图文对并拉开不匹配的样本。
- **相关术语：** Embedding, Cosine Similarity, Loss Function

### Cosine Similarity
- **类别：** 数据与表征
- **常见说法：** 两个向量有多相似。
- **实际含义：** 两个向量的归一化点积。它比较的是方向而不是大小，对于实值向量，其取值范围为 -1 到 1。
- **常见误区：** 高余弦相似度只有在特定 embedding 模型和数据分布下才有意义，它不能证明两个对象在事实或语义上完全等价。
- **相关术语：** Embedding, Semantic Search, Reranker

### Cost per Successful Task
- **类别：** AI 原生开发
- **实际含义：** 总系统成本除以满足既定成功标准的任务数量，其中包括重试、失败运行、工具调用和评估开销。
- **为什么重要：** 如果一个便宜的模型调用经常失败，或者总要反复人工纠正，那么整个工作流最终仍可能很昂贵。
- **实践建议：** 统计 100 个仓库任务中的模型调用费用和基础设施成本，再除以其中补丁通过测试和评审的任务数量。
- **常见误区：** Cost per token 衡量的是用量；cost per successful task 衡量的则是有效结果。
- **相关术语：** Evaluation (Eval), Retry with Backoff, Model Router, Verification Gate

### Cross-Attention
- **类别：** 多模态系统
- **实际含义：** 一种 attention 形式，其中 query 表示来自一个序列或表示，而 key 和 value 来自另一个序列或表示。
- **为什么重要：** 它让一个信息流能够以可学习的方式从另一个信息流中取回信息，例如语言 token 去关注视觉特征。
- **实践建议：** 明确哪一路输入提供 query、key 和 value，对缺失或无效位置应用 mask，并检查在去掉某个模态后模型是否仍能正常工作。
- **常见误区：** Cross-attention 本身并不天然就是多模态的。它既可以连接两段文本序列，也可以连接其他表示；而 self-attention 则是从同一个序列表征中派生 query、key 和 value。
- **相关术语：** Attention, Self-Attention, Vision-Language Model (VLM), Multimodal Fusion
- **来源：** [Attention Is All You Need](https://arxiv.org/abs/1706.03762)

### Cross-Entropy
- **类别：** 数学与训练
- **常见说法：** 分类损失。
- **实际含义：** 一种基于目标结果负对数概率的 loss。在 next-token 训练中，如果模型给真实下一个 token 分配的概率太低，它就会受到惩罚。
- **常见误区：** 只有在平均方式和对数底数定义一致时，perplexity 才等于平均 cross-entropy 取指数后的结果。
- **相关术语：** Loss Function, Softmax, Perplexity

### CUDA
- **类别：** 模型与推理
- **常见说法：** GPU 编程。
- **实际含义：** NVIDIA 面向兼容 GPU 的通用计算平台与编程模型。深度学习框架会使用 CUDA 库和 kernel，并行执行大量 tensor 运算。
- **常见误区：** GPU 加速不等同于 CUDA；还有其他硬件和软件技术栈可选。
- **相关术语：** Tensor, Mixed Precision, JAX

## D

### Data Augmentation
- **类别：** 数学与训练
- **常见说法：** 制造更多训练数据。
- **实际含义：** 通过构造变体样本来增加训练多样性，例如变换后的图像、加扰后的音频或改写后的文本，而无需重新采集全新数据源。当这些变换保留了任务信号时，它可以帮助减少过拟合。
- **常见误区：** 数据增强必须保留你希望模型学到的目标标签或行为。
- **相关术语：** Overfitting, Epoch, Eval Set

### Data Classification
- **类别：** 安全与治理
- **实际含义：** 把数据划分到有文档定义的敏感性或影响等级中，以便处理、访问、保留、共享和事故响应规则都能与其泄露或丢失后果相匹配。
- **为什么重要：** 如果源文档、prompt、trace 和生成产物都被一视同仁地当作同等敏感，AI 流水线就无法实施比例适当的控制措施。
- **实践建议：** 在数据接入时就完成分类，让标签贯穿衍生产物，按等级限制可用工具和目标位置，并定义数据在变换或聚合后标签应如何变化。
- **常见误区：** 数据分类描述的是保护要求，它不等同于机器学习里的分类任务，也不代表数据一定准确。
- **相关术语：** Data Minimization, Trust Boundary, Least Privilege, Audit Log
- **来源：** [NIST SP 1800-39 初版公开草案：数据分类实践](https://www.nccoe.nist.gov/sites/default/files/2026-02/nist-sp-1800-39-ipd.pdf); [NIST FIPS 199：联邦信息与信息系统分类](https://csrc.nist.gov/pubs/fips/199/final)

### Data Deduplication
- **类别：** 数据与表征
- **实际含义：** 在单个数据集内部或跨数据集之间，检测并移除完全重复或近似重复的样本。
- **为什么重要：** 重复样本会扭曲训练分布、增加记忆化风险、泄漏测试材料，并让评估结果看起来比实际更好。
- **实践建议：** 先规范化内容，再结合精确哈希和相似度方法处理，复审边界样本簇，并记录每个样本是被哪个版本、哪条规则移除的。
- **常见误区：** 去重不是普通的数据清洗。两条不同记录可能合法地共享相同文本，而两段改写文本也可能携带相同的泄漏信息。
- **相关术语：** Data Provenance, Benchmark Contamination, Dataset Split, Overfitting
- **来源：** [训练数据去重让语言模型表现更好](https://arxiv.org/abs/2107.06499)

### Data Exfiltration
- **类别：** 安全与治理
- **实际含义：** 受保护数据被未经授权地从某个系统或信任域转移到无权接收它的人、工具、服务或存储位置。
- **为什么重要：** 即使原始数据存储本身没有被破坏，agent 也可能通过生成文本、工具参数、URL、日志或副作用泄露秘密。
- **实践建议：** 尽量减少可读取的数据，使用目的地 allowlist，检查向外发出的工具调用，对敏感字段做脱敏，并对跨信任边界的异常传输发出告警。
- **常见误区：** Exfiltration 指的是未经授权的数据转移或披露。被授权组件对数据的正常读取不算 exfiltration，但后续使用方式可能会让它演变成 exfiltration。
- **延伸学习：** [EchoLeak 与 AI 的 CVE](../phases/18-ethics-safety-alignment/25-echoleak-cves-for-ai/)
- **相关术语：** Trust Boundary, Least Privilege, Indirect Prompt Injection, Audit Log
- **来源：** [NIST SP 800-53 Rev. 5：AC-4 信息流强制控制](https://csrc.nist.gov/files/pubs/sp/800/53/r5/upd1/final/docs/sp800-53r5-controls.xlsx)

### Data Leakage
- **类别：** 数据与表征
- **实际含义：** 在训练或特征构造过程中，不小心使用了真实预测时本不该可用的信息，或使用了本应属于留出评测边界的信息。
- **为什么重要：** Leakage 会制造看似乐观的指标，一旦系统遇到真正未见过的输入，这些指标就会崩塌。
- **实践建议：** 应在拟合预处理器之前先拆分数据，把未来信息排除在历史特征之外，并让测试标签和 benchmark 答案与 prompt、调优循环隔离开来。
- **常见误区：** Leakage 不只发生在重复样本上。全局归一化统计量、时间戳、从目标变量派生的特征，以及反复基于测试结果修改 prompt，都可能造成信息泄漏。
- **相关术语：** Dataset Split, Benchmark Contamination, Eval Set, Data Provenance
- **来源：** [scikit-learn：数据泄漏](https://scikit-learn.org/stable/common_pitfalls.html#data-leakage)

### Data Lineage
- **类别：** 安全与治理
- **实际含义：** 记录某个数据产物是如何在来源、变换、连接、过滤、版本迭代和下游使用过程中逐步生成出来的。
- **为什么重要：** 当某个来源被更正、撤回或认定为不安全时，lineage 能帮助识别哪些数据集、embedding、评测结果和模型产物可能受到影响。
- **实践建议：** 给输入和输出分配稳定标识符，记录每次变换和版本，保留父子关系，并验证某个受影响来源能否一路追踪到所有衍生产物。
- **常见误区：** Data provenance 更广义地解释来源和流转归属；lineage 则更强调数据产物之间的变换路径和依赖关系。
- **相关术语：** Data Provenance, Datasheet for Datasets, Audit Log, Content Provenance
- **来源：** [W3C PROV-O](https://www.w3.org/TR/prov-o/)

### Data Minimization
- **类别：** 安全与治理
- **实际含义：** 对于个人数据，应把收集、处理、暴露和保留范围限制在特定目的所必需的最小集合内。团队也可以把同样的纪律用于敏感的非个人数据，作为一种工程控制手段。
- **为什么重要：** 每一个不必要地放进 prompt、trace、cache 或工具调用里的字段，都会增加隐私暴露面，并放大被滥用或被攻破后的潜在影响。
- **实践建议：** 在收集前先定义必需字段，在最早边界完成脱敏或聚合，设置保留期限，并在保留可选上下文之前验证它确实能改善可测量的任务结果。
- **常见误区：** 最小化并不意味着什么数据都不留，而是要能够针对既定目的，为每个数据字段、每种用途、每个接收方和每段保留时长给出合理解释。
- **相关术语：** Purpose Limitation, Data Classification, Least Privilege, 上下文工程
- **来源：** [《通用数据保护条例》第 5(1)(c) 条](https://eur-lex.europa.eu/eli/reg/2016/679/oj)

### Data Provenance
- **类别：** 数据与表征
- **实际含义：** 可追踪的信息，用来说明数据来自哪里、由谁或什么进行了变换、用了哪些版本，以及衍生产物如何关联回原始来源。
- **为什么重要：** 为了复现实验结果、遵守使用约束、调查污染问题，以及在来源变化时移除受影响数据，你都需要 provenance。
- **实践建议：** 给数据集分配不可变版本，记录变换任务和来源标识符，并把 lineage 元数据带入 embedding、eval case 和模型产物中。
- **常见误区：** 来源 URL 只是 provenance 的一部分，它并不能说明采集时间、许可协议、过滤过程、变换方式或下游用途。
- **相关术语：** Dataset Split, Data Deduplication, Provenance Attestation, Grounding
- **来源：** [W3C PROV 概览](https://www.w3.org/TR/prov-overview/)

### Dataset Split
- **类别：** 数据与表征
- **实际含义：** 一种有文档记录的样本划分方式，把数据分别拆成用于拟合、开发决策和最终评估的独立子集。
- **为什么重要：** 这种分离能避免“用于选系统的证据”同时又被拿来当作“系统具备泛化能力的独立证明”。
- **实践建议：** 应按真实部署单元来拆分，例如用户、仓库、组织或时间，而不是随机拆分彼此相关的样本行。
- **常见误区：** 随机划分并不自动等于独立划分。近重复样本、未来观测值或来自同一实体的记录，都可能跨越划分边界。
- **相关术语：** Eval Set, Overfitting, Data Leakage, Distribution Shift
- **来源：** [Datasheets for Datasets](https://cacm.acm.org/research/datasheets-for-datasets/)

### Datasheet for Datasets
- **类别：** 安全与治理
- **实际含义：** 一种结构化文档，用来说明数据集的动机、组成、采集过程、预处理方式、用途、分发、维护方式和已知局限。
- **为什么重要：** 一个数据集并不会因为“拿得到”就自动意味着安全或适用。下游建设者需要证据来了解它是如何产生的，以及它的假设会在哪些地方失效。
- **实践建议：** 应让 datasheet 与版本化数据集一起发布，明确谁负责答疑，记录被排除的人群和变换处理，并在数据集发生变化时同步更新文档。
- **常见误区：** Datasheet 记录的是证据和预期用途。它不是许可证、质量保证书，也不能替代面向具体部署场景的评估。
- **延伸学习：** [模型卡、系统卡与数据集卡](../phases/18-ethics-safety-alignment/26-model-system-dataset-cards/)
- **相关术语：** Data Lineage, Data Provenance, Model Card, Dataset Split
- **来源：** [Datasheets for Datasets](https://arxiv.org/abs/1803.09010)

### Deadline Propagation
- **类别：** 可靠性与运维
- **实际含义：** 把端到端剩余时间预算传递给下游调用，让每个依赖都知道原始请求还能有意义地等待多久。
- **为什么重要：** 彼此独立设置的超时，可能会超过用户真正的截止时间，并在结果已经没有价值后，还让被放弃的工作继续消耗系统容量。
- **实践建议：** 在入口处统一设置请求 deadline，对每次下游调用扣除已耗时间，取消已过期工作，并记录究竟是哪个边界耗尽了预算。
- **常见误区：** Deadline 是绝对或剩余的完成边界；retry delay 则控制下一次尝试何时开始，而且必须纳入同一个时间预算内。
- **相关术语：** Retry with Backoff, Retry Budget, Tail Latency, Service Level Objective (SLO)
- **来源：** [gRPC 截止时间](https://grpc.io/docs/guides/deadlines/)

### Decode Phase
- **类别：** 基础设施与服务
- **实际含义：** 自回归推理中的迭代阶段，在输入前缀处理完成后，每次生成一个新 token。
- **为什么重要：** Decode 阶段在计算、内存和调度行为上都不同于 prefill，因此只看一个汇总延迟指标，可能会掩盖真实的服务瓶颈。
- **实践建议：** 应分别测量 token 间延迟和输出吞吐，考虑 KV-cache 占用情况，并测试活动 decode 与新进入 prefill 共享容量时的混合负载表现。
- **常见误区：** Decode phase 不是 encoder-decoder 模型里的 decoder 组件，它指的是运行时的生成阶段。
- **延伸学习：** [解耦式 Prefill 与 Decode](../phases/17-infrastructure-and-production/17-disaggregated-prefill-decode/)
- **相关术语：** Prefill, Autoregressive, KV Cache, Time per Output Token (TPOT)
- **来源：** [DistServe](https://arxiv.org/abs/2401.09670)

### Decoder
- **类别：** 模型与推理
- **常见说法：** 模型的输出侧。
- **实际含义：** 把某种表示映射成输出的组件。在 encoder-decoder transformer 中，decoder 会使用带 mask 的 self-attention 和 cross-attention 来生成输出；而 decoder-only 语言模型则从单一因果堆栈中直接生成。
- **相关术语：** Encoder, Transformer, Autoregressive

### Decoding Strategy
- **类别：** 模型与推理
- **实际含义：** 把模型产生的一串 next-token 分数转换成具体选中 token 和完整输出的算法。
- **为什么重要：** 在同样的 logits 下，greedy、sampling、truncation 和 search 这些策略会带来不同的质量、多样性、延迟和可复现性。
- **实践建议：** 应在 eval 配置中明确任务的 decoding 设置、停止规则和 seed 行为，这样结果才能公平比较。
- **常见误区：** Decoding 改变的是输出如何被选中，并不会改变模型已经训练好的参数，也不会给模型额外注入知识。
- **相关术语：** Autoregressive, Temperature, Top-k Sampling, Nucleus Sampling (Top-p)
- **来源：** [神经文本退化的奇特现象](https://arxiv.org/abs/1904.09751)

### Defense in Depth
- **类别：** 安全与治理
- **实际含义：** 在多个系统边界同时使用相互独立的预防、检测和纠正控制措施，从而避免某一个控制失效就直接决定最终结果。
- **为什么重要：** AI 系统把概率模型、不可信内容、工具和外部服务结合在一起，因此任何单一过滤器或单条 prompt 都不足以构成可靠的安全边界。
- **实践建议：** 应把指令层面的控制，与最小权限、沙箱、schema 校验、重要动作审批、监控以及经过验证的恢复路径组合使用。
- **常见误区：** 控制措施并不是越多越好。各层应覆盖不同的失败模式，并保持可测试，而不是反复依赖同一个假设。
- **相关术语：** Guardrails, Sandbox, Least Privilege, Trust Boundary
- **来源：** [NIST 术语表：纵深防御](https://csrc.nist.gov/glossary/term/defense_in_depth)

### Delegation
- **类别：** 智能体与工具
- **实际含义：** 把一个有明确边界的子任务分配给另一个人或 agent，并同时提供所需上下文、权限、输出契约和回传条件。
- **为什么重要：** 显式 delegation 能支持专业分工和并行工作，同时不丢失责任归属、范围控制和结果整合能力。
- **实践建议：** 给 reviewer agent 明确的文件、评审标准、证据和截止时间，并要求它返回 findings，而不是悄悄修改主产物。
- **常见误区：** 向另一个 agent 发一条模糊消息，并不算可靠 delegation。接收方需要清晰的范围契约和明确的回交方式。
- **相关术语：** Scope Contract, Handoff, Reviewer Agent, Orchestration
- **来源：** [构建高效代理](https://www.anthropic.com/research/building-effective-agents)

### Dense Retrieval
- **类别：** 检索与生成
- **实际含义：** 第一阶段检索方法，会把查询和候选项编码成向量表示，再通过相似度函数对候选项进行排序。
- **为什么重要：** 它能召回那些虽然字面重合很少、但在语义上匹配的改写表达，从而与 BM25 这类词法方法形成互补。
- **实践建议：** 应先为目标领域训练或选择合适的 embedding 模型，对候选向量建立索引，并在接入生成环节前评估检索召回率。
- **常见误区：** Dense retrieval 不是 reranker。前者负责在全集上检索，后者则是在较小候选集合上重新打分。
- **相关术语：** Embedding, Semantic Search, BM25, Hybrid Retrieval
- **来源：** [Dense Passage Retrieval](https://aclanthology.org/2020.emnlp-main.550/)

### Diffusion Model
- **类别：** 模型与推理
- **常见说法：** 一种从噪声生成图像的模型。
- **实际含义：** 一种围绕逐步加噪过程和可学习逆过程训练出的生成模型。采样通常从噪声开始，通过多步去噪生成结果，有时还会在学习到的 latent space 中进行。
- **常见误区：** Diffusion 是一种通用生成框架，不只是图像领域的方法。
- **相关术语：** Latent Space, VAE (Variational Autoencoder), Inference

### Disaggregated Serving
- **类别：** 基础设施与服务
- **实际含义：** 一种服务架构，把 prefill 和 decode 分别放到独立配置的 worker 池中执行，并在两者之间传递所需的 attention state。
- **为什么重要：** Prefill 和 decode 对硬件的压力点不同，因此把它们放进独立 worker 池后，就能分别按各自瓶颈做容量规划和调度，而不是在同一队列里互相争抢。
- **实践建议：** 应测量状态传输成本，让请求在兼容模型版本之间正确路由，根据各自需求信号分别扩缩容，并测试阶段间故障恢复。
- **常见误区：** Disaggregation 分离的是运行时阶段，并不是在单个阶段内部把一个模型切成 tensor parallel 或 pipeline parallel 的分片。
- **延伸学习：** [解耦式 Prefill 与 Decode](../phases/17-infrastructure-and-production/17-disaggregated-prefill-decode/)
- **相关术语：** Prefill, Decode Phase, Model Serving, Goodput
- **来源：** [DistServe](https://arxiv.org/abs/2401.09670)

### Distribution Shift
- **类别：** 评估与安全
- **实际含义：** 用于构建或评估系统的数据分布，与系统上线后实际遇到的数据分布之间存在的差异。
- **为什么重要：** 模型即使通过了留出测试，也可能在用户、任务、语言、工具或运行条件发生变化时失败。
- **实践建议：** 应先定义预期部署切片，按切片监控性能和输入特征，并把新出现的失败样本加入版本化 eval set。
- **常见误区：** Distribution shift 不一定意味着 model drift。模型本身可能没变，变化的只是它所处环境或用户群体。
- **相关术语：** Dataset Split, Eval Set, Overfitting, Model Card
- **来源：** [WILDS](https://proceedings.mlr.press/v139/koh21a.html)

### DPO (Direct Preference Optimization)
- **类别：** 数学与训练
- **常见说法：** 不需要单独 reward model 阶段的偏好训练。
- **实际含义：** 一种偏好优化目标，直接利用相对参考策略的“偏好/拒绝”回答对来训练策略模型，从而避免在这一阶段单独训练显式 reward model 并运行强化学习循环。
- **常见误区：** DPO 仍然依赖偏好数据的质量和覆盖面，也不会消除评测风险或 alignment 风险。
- **延伸学习：** [Direct Preference Optimization](../phases/10-llms-from-scratch/08-dpo/)
- **来源：** [Direct Preference Optimization 论文](https://arxiv.org/abs/2305.18290)
- **相关术语：** RLHF (Reinforcement Learning from Human Feedback), SFT (Supervised Fine-Tuning), Alignment

### Dropout
- **类别：** 数学与训练
- **常见说法：** 随机关闭部分激活值。
- **实际含义：** 在训练时，随机把一部分激活值置零，以促使网络不要过度依赖某一条激活路径。标准推理阶段通常会关闭它，不过 Monte Carlo dropout 会故意保留它，以估计不确定性。
- **相关术语：** Overfitting, Weight Decay, Activation Function

### Durable Execution
- **类别：** 智能体与工具
- **实际含义：** 让工作流在进程崩溃、重启或长时间等待后仍能保留状态和已完成步骤，而不必重复那些已经确认过的副作用。
- **为什么重要：** Agent 任务常常跨越模型调用、工具、审批和外部系统。一次短暂运行的进程不应该成为唯一的进度记录载体。
- **实践建议：** 应持久化每一次工作流状态转换，对外部写入使用幂等键，并在 worker 重启后从最新检查点恢复。
- **常见误区：** Durable execution 并不会自动让所有操作都变得安全。副作用仍然需要幂等性保障和补偿规则。
- **相关术语：** Checkpoint, Agent State, Idempotency, Approval Gate

### Dynamic Batching
- **类别：** 基础设施与服务
- **实际含义：** 一种运行时策略，会根据形状兼容性、最大 batch 大小、优先级和允许的排队时延，从排队请求中组装推理 batch。
- **为什么重要：** 把请求聚成 batch 可以提升硬件利用率，但在流量稀疏或请求差异很大时，等待组 batch 反而可能让延迟更差。
- **实践建议：** 应根据实测延迟目标来设置排队时延和 batch 上限，把不兼容的请求形状分开，并在真实到达率下比较吞吐与尾延迟。
- **常见误区：** Dynamic batching 是从排队工作中组装 batch；continuous batching 则是在自回归生成已经开始运行时，动态改变 batch 成员。
- **延伸学习：** [vLLM 服务内部原理](../phases/17-infrastructure-and-production/04-vllm-serving-internals/)
- **相关术语：** Admission Control, Continuous Batching, Saturation, Tail Latency
- **来源：** [NVIDIA Triton：模型与调度器](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/model_configuration.html#scheduling-and-batching)

## E

### Early Fusion
- **类别：** 多模态系统
- **实际含义：** 在大部分任务特定建模发生之前，就先把多个模态的原始或低层表示融合起来。
- **为什么重要：** 早期融合可以暴露更细粒度的跨模态关系，但也要求各模态表征彼此兼容，并谨慎处理对齐问题和输入缺失问题。
- **实践建议：** 把每个模态转换为明确声明的 token 或特征表示，保留来源与位置信息，在共享 backbone 之前完成融合，并与单模态和晚期融合基线进行比较。
- **常见误区：** Early fusion 描述的是在架构中的哪个位置把各路信息流合并起来，并不保证模型一定能学到有用的跨模态对齐。
- **延伸学习：** [Chameleon 早期融合 Token](../phases/12-multimodal-ai/11-chameleon-early-fusion-tokens/)
- **相关术语：** Late Fusion, Multimodal Fusion, Modality Alignment, Token
- **来源：** [Chameleon：混合模态早期融合基础模型](https://arxiv.org/abs/2405.09818); [多模态机器学习：综述与分类](https://arxiv.org/abs/1705.09406)

### Eigenvalue
- **类别：** 数学与训练
- **常见说法：** PCA 中会用到的一种矩阵性质。
- **实际含义：** 一个标量，用来描述线性变换如何缩放与之对应的非零特征向量，同时不改变它的方向。在协方差矩阵的 PCA 中，更大的特征值对应方差更大的方向。
- **相关术语：** Tensor, Feature, Latent Space

### Embedding
- **类别：** 数据与表征
- **常见说法：** 表示语义的向量。
- **实际含义：** 一种学习得到的映射，把离散对象（词、图像、用户等）映射到连续空间中的稠密向量，并让相似对象彼此靠近
- **常见误区：** 相似性取决于模型、训练目标和距离度量。在一个 embedding 空间里的距离关系，不能直接套用到另一个空间中。
- **名称由来：** 这些对象被放置，也就是被嵌入到了一个几何表示空间中。
- **延伸学习：** [嵌入](../phases/11-llm-engineering/04-embeddings/)
- **相关术语：** Cosine Similarity, Semantic Search, Vector Database

### Encoder
- **类别：** 模型与推理
- **常见说法：** 模型的输入侧。
- **实际含义：** 一个把输入转换为表示的组件。Transformer encoder 通常使用非因果 self-attention，并服从相应 mask，因此每个位置都可以整合来自整个输入的上下文。
- **常见误区：** Encoder-only 模型虽然通常不用于自回归文本生成，但依然可以通过任务头输出结果。
- **相关术语：** Decoder, Transformer, Embedding

### Epoch
- **类别：** 数学与训练
- **常见说法：** 对训练数据完整遍历一遍。
- **实际含义：** 对已定义训练数据集的一次完整遍历。在分布式训练或采样训练中，epoch 的具体实现方式会取决于 data loader 和采样策略。
- **常见误区：** 更多 epoch 并不保证更好的泛化能力；应基于留出数据进行评估。
- **相关术语：** Batch Size, Overfitting, Eval Set

### Error Budget
- **类别：** 可靠性与运维
- **实际含义：** 在一个服务级目标的观测窗口内，在该目标被耗尽之前所允许的失败服务量。
- **为什么重要：** 它为可靠性工作和产品迭代提供了共同的决策边界：当用户可见失败还没耗尽预算时，团队可以继续推动变更；当预算被消耗时，则应放缓风险。
- **实践建议：** 应从 SLO 推导出预算，按原因和用户群体跟踪消耗速率，在预算耗尽前就定义好发布动作，并避免在事故后重置核算。
- **常见误区：** 错误预算不是“允许制造多少事故”的配额，而是从面向用户的可靠性目标推导出的运行策略。
- **相关术语：** Service Level Objective (SLO), Service Level Indicator (SLI), Availability, Incident Response
- **来源：** [Google SRE Workbook：错误预算策略](https://sre.google/workbook/error-budget-policy/)

### Eval Set
- **类别：** 评估与安全
- **别名：** Evaluation set
- **实际含义：** 一组带版本的输入、预期属性、评分规则和元数据，用于围绕某项已定义能力或风险来衡量 AI 系统表现。
- **为什么重要：** 一套可重复的评测集合，能把模糊的质量判断转化成可比较的证据，并在 prompt、模型、工具或检索发生变化后及时发现回归。
- **实践建议：** 应把有代表性的支持类问题、对抗性指令、预期引用和失败标签，放进一份经过审查且独立于开发样例的数据集中。
- **常见误区：** 开发阶段的 eval 用来指导迭代；最终留出测试用于在方案确定后估计性能；标准化 benchmark 则支持在共享协议下进行比较。反复针对任何留出集做调优，都会泄漏测试信息并夸大结果。
- **延伸学习：** [评测驱动的代理开发](../phases/14-agent-engineering/30-eval-driven-agent-development/)
- **相关术语：** Evaluation (Eval), Regression Test, LLM-as-a-Judge, Verification Gate

### Evaluation (Eval)
- **类别：** 评估与安全
- **别名：** Eval
- **实际含义：** 一种定义明确的过程，用显式成功标准、数据、评分器和审查流程来衡量模型或系统在代表性任务上的行为表现。
- **为什么重要：** 如果所谓“成功”只是来自少数 demo 的主观印象，你就无法真正提升可靠性。
- **实践建议：** 在修改 retrieval 前后运行同一批客服场景，对正确性和引用支撑进行打分，并按类别分析失败样本。
- **常见误区：** Benchmark 分数只是某一次评测结果，不是对生产质量的完整说明。
- **延伸学习：** [LLM 评测](../phases/11-llm-engineering/10-evaluation/)
- **相关术语：** Eval Set, LLM-as-a-Judge, Cost per Successful Task, Regression Test

### Exact Match (EM)
- **类别：** 评估与安全
- **实际含义：** 一种指标，只有当输出经过规范化后的表示与某个被接受的参考答案完全一致时，才算正确。
- **为什么重要：** 对于只有一个标准答案的任务，它具有确定性且容易审计，但不会反映部分正确的情况。
- **实践建议：** 在评测前先定义好规范化规则和所有可接受参考答案；当任务可能存在多个正确输出时，再配合使用任务特定检查。
- **常见误区：** 较低的 exact match 分数，可能只是由无害的格式差异造成；反过来，即使字符串完全匹配，也仍可能在上下文中缺乏依据或不安全。
- **相关术语：** ROUGE, Eval Set, Structured Output, Pass@k
- **来源：** [SQuAD](https://aclanthology.org/D16-1264/)

### Expert Parallelism
- **类别：** 基础设施与服务
- **实际含义：** 把 mixture-of-experts 的子网络分布到多个设备上，并把每个 token 的激活路由到承载其所选 expert 的设备。
- **为什么重要：** 稀疏 expert 能在不让每个 token 都执行所有 expert 的前提下提升模型容量，但路由机制也会带来通信、负载均衡和部署位置方面的约束。
- **实践建议：** 应测量各个 expert 的 token 分布，为通信带宽做容量规划，对溢出流量进行有意限制或路由，并在流量导致 expert 需求不均时测试模型质量。
- **常见误区：** Expert parallelism 划分的是由路由器选中的 expert；tensor parallelism 划分的是层内部的 tensor 运算。
- **延伸学习：** [Mixture of Experts](../phases/07-transformers-deep-dive/11-mixture-of-experts/)
- **相关术语：** MoE (Mixture of Experts), Tensor Parallelism, Pipeline Parallelism, Model Serving
- **来源：** [GShard](https://arxiv.org/abs/2006.16668)

## F

### Feature
- **类别：** 数据与表征
- **常见说法：** 数据集中的一列。
- **实际含义：** 数据中一个可测量的单独属性。在传统机器学习里，特征通常由人手工设计；在深度学习里，网络会从原始数据中自动学习特征。
- **常见误区：** 一列存储的数据里可能包含多个有用特征，而学得的表示里也可能包含无法用简单人类标签命名的特征。
- **相关术语：** Embedding, Latent Space, Inductive Bias

### Few-Shot
- **类别：** 提示与上下文
- **常见说法：** 在提示里给模型几个示例。
- **实际含义：** 一种上下文学习方式：在目标输入之前放入少量示例，让模型据此推断期望的任务、格式或决策边界。
- **为什么重要：** 示例的质量与覆盖范围，比一个放之四海而皆准的示例数量更重要。质量差或彼此矛盾的示例会降低可靠性。
- **相关术语：** Zero-Shot, In-Context Learning, Prompt Engineering, Context Window

### Fine-tuning
- **类别：** 数学与训练
- **常见说法：** 用你的数据训练模型。
- **实际含义：** 在预训练参数基础上，围绕更窄的数据集或目标继续训练。具体更新哪些参数取决于方法，可以更新全部参数、部分参数，或新增的适配器参数。
- **为什么重要：** Fine-tuning 可以调整行为、风格、格式或任务表现，但当事实必须保持最新且可追溯时，它并不能可靠替代检索。
- **常见误区：** Fine-tuning 会影响模型内部编码的知识，但它并不是把记录直接追加进模型内部一个可搜索的数据库。
- **延伸学习：** [Fine-Tuning and LoRA](../phases/11-llm-engineering/08-fine-tuning-lora/)
- **相关术语：** SFT (Supervised Fine-Tuning), LoRA (Low-Rank Adaptation), QLoRA, RAG (Retrieval-Augmented Generation)

### Flaky Test
- **类别：** AI 原生开发
- **实际含义：** 在代码和预期测试环境都没有相关变化的情况下，同样的测试在等价运行中却时而通过、时而失败的测试。
- **为什么重要：** 测试波动会削弱验证关卡，也会让人或 agent 习惯于忽略真实失败，或一再重试直到碰到一次假通过。
- **实践建议：** 保留失败时的随机种子和环境；只有在明确责任人和截止时间的前提下才临时隔离；随后修复未受控的时间、并发、网络、执行顺序或共享状态依赖。
- **常见误区：** 如果一个测试能够稳定暴露产品中间歇出现的 bug，那是有价值的证据，并不一定是 flaky test。
- **相关术语：** Regression Test, Test Oracle, Retry with Backoff, Verification Gate
- **来源：** [De-Flake Your Tests](https://conferences.computer.org/icsme/pdfs/ICSME2020-1oOutvkGTwF4GyVvNtr3Mm/561900a736/561900a736.pdf)

### FlashAttention
- **类别：** 基础设施与服务
- **实际含义：** 一种精确的 attention 算法，通过分块计算来减少加速器不同内存层级之间的数据搬运，同时避免在高带宽内存中显式展开完整的 attention 矩阵。
- **为什么重要：** Attention 的瓶颈常常不在算术计算，而在内存搬运，长序列下尤为明显，因此具备 IO 感知能力的 kernel 往往能提升实际速度和内存效率。
- **实践建议：** 选用与模型张量形状、mask、dtype 和硬件兼容的 kernel，验证数值误差是否可接受，并以端到端延迟做基准，不要把论文中的提速倍数当成固定结论照搬。
- **常见误区：** FlashAttention 改变的是 attention 的计算方式，而不是它要得到的数学结果；它和 KV cache、quantization 也是不同层面的技术。
- **延伸学习：** [KV Cache and Flash Attention](../phases/07-transformers-deep-dive/12-kv-cache-flash-attention/)
- **相关术语：** Attention, Self-Attention, KV Cache, Mixed Precision
- **来源：** [FlashAttention](https://arxiv.org/abs/2205.14135)

### Function Calling
- **类别：** 智能体与工具
- **常见说法：** 会使用工具的模型。
- **实际含义：** 一种由提供商或应用暴露的接口，模型通过它发出结构化请求，指定要调用的工具及其参数。应用代码负责校验请求、执行操作，并可将结果返回给模型进入下一步。
- **常见误区：** 模型只是请求进行 function call；是否执行、如何执行，由受信任的应用代码决定。仅有 function calling 并不等于一个完整的 agent。
- **延伸学习：** [Function Calling](../phases/11-llm-engineering/09-function-calling/)
- **相关术语：** Structured Output, Tool Contract, Agent, MCP (Model Context Protocol)

## G

### GAN (Generative Adversarial Network)
- **类别：** 模型与推理
- **常见说法：** 训练中彼此对抗的两个神经网络。
- **实际含义：** 一个生成器网络尝试生成逼真的数据，另一个判别器网络尝试区分真假。二者共同训练：生成器越来越擅长骗过判别器，判别器也越来越擅长识别伪造内容。
- **相关术语：** Loss Function, Latent Space, Diffusion Model

### Goodput
- **类别：** 基础设施与服务
- **实际含义：** 在给定工作负载下，满足既定服务约束的已完成请求速率，例如同时满足 TTFT 和逐 token 延迟目标的请求完成速率。
- **为什么重要：** 原始吞吐量可能上升，但用户感受到的慢请求却更多。Goodput 只统计真正满足服务约定的那部分工作。
- **实践建议：** 先明确请求分布和延迟阈值，只统计达标完成的请求，并在总体速率之外同时报告分位数；不要把目标不同的系统直接拿来比较。
- **常见误区：** Goodput 不是所有已完成请求的总吞吐，也不是模型天然自带的通用属性；它取决于工作负载和成功阈值的定义。
- **延伸学习：** [Inference Metrics and Goodput](../phases/17-infrastructure-and-production/08-inference-metrics-goodput/)
- **相关术语：** Service Level Objective (SLO), Time to First Token (TTFT), Time per Output Token (TPOT), Cost per Successful Task
- **来源：** [DistServe](https://arxiv.org/abs/2401.09670)

### GPT
- **类别：** 模型与推理
- **常见说法：** 任何聊天机器人的统称。
- **实际含义：** Generative Pre-trained Transformer，指一类先在序列预测目标上完成预训练、再适配下游用途的生成式 transformer 模型。产品名称和模型架构名称不能混为一谈。
- **名称由来：** Generative 指能生成输出，pre-trained 指先经历广泛的初始训练阶段，transformer 则指其所属的架构家族。
- **相关术语：** Transformer, Autoregressive, LLM (Large Language Model)

### Graceful Degradation
- **类别：** 可靠性与运维
- **实际含义：** 当容量或依赖受损时，不是让所有请求一起失败，而是通过降低可选质量、功能、新鲜度或负载，保住一个边界清晰的核心服务能力。
- **为什么重要：** AI 系统往往依赖多个较慢或可能出错的组件，因此在部分故障时，预先定义好的降级模式可以保护关键用户结果。
- **实践建议：** 预先定义哪些能力允许被关闭，让运维人员清楚看到当前处于降级模式，确保安全检查仍然生效，在依赖故障下验证回退路径，并在恢复时有意识地切回完整服务。若正确性、安全性、时效性或已承诺的服务契约发生实质变化，也要明确告知用户。
- **常见误区：** Graceful degradation 不是悄悄返回一个更差的结果、假装什么都没发生。运维侧始终需要可见性；当降级模式实质改变结果或服务契约时，用户也需要被明确告知。
- **延伸学习：** [Production LLM Application](../phases/11-llm-engineering/13-production-app/)
- **相关术语：** Circuit Breaker, Load Shedding, Model Router, Availability
- **来源：** [Google SRE: Addressing Cascading Failures](https://sre.google/sre-book/addressing-cascading-failures/)

### Gradient
- **类别：** 数学与训练
- **常见说法：** 损失函数的斜率。
- **实际含义：** 由偏导数组成的向量，指向函数增长最快的方向。在机器学习中，为了最小化损失，我们沿着梯度的反方向前进，也就是做 gradient descent。
- **常见误区：** 优化器不一定只是沿负梯度迈一步，它还可能对梯度做变换、平均、裁剪或自适应调整。
- **相关术语：** Backpropagation, Gradient Descent, Optimizer

### Gradient Accumulation
- **类别：** 数学与训练
- **实际含义：** 在执行一次优化器更新前，先对多个 microbatch 的梯度进行求和或求平均。
- **为什么重要：** 当单个设备无法一次装下全部样本和激活值时，它可以帮助你近似实现更大的有效 batch。
- **实践建议：** 要保证 loss 的缩放方式前后一致，只在累计到设定数量的 microbatch 后再调用优化器，并测量归一化或分布式同步是否改变了训练行为。
- **常见误区：** Gradient accumulation 能降低单步所需的激活内存，但它并不能完全复现一次性处理整个大 batch 的所有性质。
- **相关术语：** Batch Size, Mixed Precision, Optimizer, Backpropagation
- **来源：** [PyTorch AMP examples: Gradient accumulation](https://docs.pytorch.org/docs/stable/notes/amp_examples.html#gradient-accumulation)

### Gradient Clipping
- **类别：** 数学与训练
- **实际含义：** 在优化器更新之前，如果梯度值或其整体范数超过设定阈值，就对其进行限制。
- **为什么重要：** 它可以防止异常巨大的梯度把某一步训练带偏，进而产生非有限数值。
- **实践建议：** 记录裁剪前的范数；若使用 mixed precision，应在反缩放之后再裁剪；如果频繁触发裁剪，要去排查训练不稳定的根因，而不是把裁剪当成替代诊断的办法。
- **常见误区：** 裁剪控制的是更新幅度，它并不能修复无效数据、错误的损失函数，或长期不合适的学习率。
- **相关术语：** Gradient, NaN (Not a Number), Mixed Precision, Learning Rate
- **来源：** [On the difficulty of training recurrent neural networks](https://arxiv.org/abs/1211.5063)

### Gradient Descent
- **类别：** 数学与训练
- **常见说法：** 在损失曲面上往下走。
- **实际含义：** 一类利用目标函数负梯度来更新参数的优化方法，梯度通常由 batch 估计，而不是基于整个数据集精确计算。
- **相关术语：** Gradient, Learning Rate, Optimizer

### Grounding
- **类别：** 检索与生成
- **实际含义：** 把生成的答案或动作与系统能够识别并核查的证据、状态或观测结果连接起来。
- **为什么重要：** Grounding 让系统的输出有了不依赖自由生成的依据，也更容易识别缺乏支撑的说法。
- **实践建议：** 先检索出相关政策条款，要求答案引用它；如果某项主张并未得到所引段落支持，就拒绝该主张。
- **常见误区：** 把文档塞进 prompt 只是创造了 grounding 的机会，并不保证模型一定会正确使用这些材料。
- **延伸学习：** [Retrieval-Augmented Generation](../phases/11-llm-engineering/06-rag/)
- **相关术语：** RAG (Retrieval-Augmented Generation), Hallucination, Verification Gate, Reranker

### Guardrails
- **类别：** 评估与安全
- **常见说法：** 围绕模型的一圈安全过滤器。
- **实际含义：** 用于约束输入、工具使用、输出、权限和升级路径的系统控制措施。它们可以包括 schema、策略检查、分类器、allowlist、sandboxing、审批流程以及动作后的验证。
- **为什么重要：** 没有任何单一过滤器能够覆盖所有失败模式，因此控制措施应依据风险分层叠加。
- **常见误区：** Guardrails 能降低风险，但并不能证明一个 AI 系统就是安全的。
- **延伸学习：** [Guardrails](../phases/11-llm-engineering/12-guardrails/)
- **相关术语：** Least Privilege, Approval Gate, Sandbox, Evaluation (Eval)

## H

### Hallucination
- **类别：** 评估与安全
- **常见说法：** 模型在撒谎。
- **实际含义：** 指生成内容是错误的、没有现有证据支持的，或与任务所依据的真实来源不一致。即便输出很流畅、模型也并非有意欺骗，这种情况依然会发生。
- **为什么重要：** 你通常无法检查某个说法是否曾出现在训练数据里，因此生产环境中的检查应聚焦于是否有依据、是否正确、以及是否可追溯。
- **实践建议：** 对事实性回答要求提供引用证据，并检查每条引用是否真的支持对应主张。
- **常见误区：** Hallucination 是输出质量上的失败，不是对模型意图的判断。
- **相关术语：** Grounding, RAG (Retrieval-Augmented Generation), Verification Gate

### Handoff
- **类别：** AI 原生开发
- **实际含义：** 在人与人、或 agent 与 agent 之间进行的结构化任务移交，要求把目标、当前状态、证据、已做决策、约束条件和剩余工作一并传递清楚。
- **为什么重要：** 高质量的 handoff 能避免接手者从长篇对话里重新拼装整个任务，也能避免重复已经完成的动作。
- **实践建议：** 把已确认的计划、变更文件、测试命令与结果、尚未解决的风险，以及下一步的明确动作，整理成一份紧凑的任务包传递出去。
- **常见误区：** 摘要只说明发生了什么；handoff 还要说明哪个状态是权威状态，以及接下来应该做什么。
- **延伸学习：** [Multi-Session Handoff](../phases/14-agent-engineering/40-multi-session-handoff/)
- **相关术语：** Agent State, Checkpoint, Scope Contract, Progressive Disclosure

### HNSW
- **类别：** 检索与生成
- **别名：** Hierarchical Navigable Small World
- **实际含义：** 一种 approximate nearest neighbor 索引，用分层邻近图来组织向量，并从较粗的上层逐步搜索到更细的下层。
- **为什么重要：** 在穷举比较代价过高的规模下，它是让高召回向量检索变得可行的常见方法。
- **实践建议：** 围绕延迟、内存和 Recall@K 目标调节建图与查询参数，并在 embedding 版本变化时重建索引。
- **常见误区：** HNSW 是一种索引算法，不是相似度度量、embedding 模型，也不是完整的向量数据库。
- **相关术语：** Approximate Nearest Neighbor (ANN), Vector Database, Embedding, Recall@K
- **来源：** [使用 HNSW 的高效且鲁棒的近似最近邻搜索](https://dl.acm.org/doi/10.1109/TPAMI.2018.2889473)

### Human-in-the-Loop (HITL)
- **类别：** 智能体与工具
- **别名：** Human oversight, human review
- **实际含义：** 一种工作流设计：在 AI 驱动流程中的特定节点，由人提供判断、修正、审批或升级处理。
- **为什么重要：** 人的介入最适合放在高影响、高歧义或不可逆的边界上，而不是把它当成每一步之后模糊兜底的默认选项。
- **实践建议：** 让 agent 自动处理常规请求分类；但对不确定或高价值的情况，要连同证据和拟采取的动作一起交给审核者处理。
- **常见误区：** HITL 不会自动让系统变安全。审核者仍然需要时间、上下文、权限，以及清晰的决策标准。
- **相关术语：** Approval Gate, Verification Gate, Agent, Guardrails

### Hybrid Retrieval
- **类别：** 检索与生成
- **实际含义：** 一种把不同检索方法产生的信号结合起来的检索方式，常见组合是词法匹配与稠密向量相似度，再对结果进行合并或 rerank。
- **为什么重要：** 精确标识符、罕见术语和语义改写在检索中的表现差异很大，因此单一检索信号容易漏掉有用证据。
- **实践建议：** 先用 BM25 风格关键词检索和 embedding 检索各自召回候选，再合并排序，最后针对用户查询对合并后的候选集做 rerank。
- **常见误区：** Hybrid retrieval 负责组合候选召回信号；reranker 则是在候选已经被召回之后，再施加第二层相关性模型。
- **延伸学习：** [Advanced RAG](../phases/11-llm-engineering/07-advanced-rag/)
- **相关术语：** Semantic Search, Reranker, RAG (Retrieval-Augmented Generation), Embedding

### Hyperparameter
- **类别：** 数学与训练
- **常见说法：** 一个需要调节的设置。
- **实际含义：** 指那些影响模型结构、优化方式、数据处理或推理过程的配置选择，而不是像普通模型参数那样通过训练学出来。典型例子包括 learning rate、batch size、层数和 decoding 设置。
- **常见误区：** 有些 hyperparameter 在训练前就定下，有些则可以在训练日程中途或推理时调整。
- **相关术语：** Parameter, Learning Rate, Batch Size, Temperature

## I

### Idempotency
- **类别：** AI 原生开发
- **实际含义：** 一种性质：对同一个操作、使用同一个身份重复执行时，不会在第一次成功之后再产生额外副作用。
- **为什么重要：** 在分布式 agent 系统里，重试是常态。没有 idempotency 的话，一次结果不确定的响应就可能导致支付、评论、部署或记录被重复创建。
- **实践建议：** 给工具请求附上 idempotency key，并持久化保存已完成结果，这样重试时就能直接返回已有结果，而不是再次执行写操作。
- **常见误区：** Idempotency 并不意味着每次响应都必须逐字节完全相同；它的含义是预期的状态变化不会被重复执行。
- **来源：** [HTTP Semantics: idempotent methods](https://www.rfc-editor.org/rfc/rfc9110.html#name-idempotent-methods)
- **相关术语：** Retry with Backoff, Durable Execution, Checkpoint

### Image Token
- **类别：** 多模态系统
- **实际含义：** 一种与具体模型相关的视觉单位，可表示为向量或离散编码，通常来自图像 patch、区域，或学习得到的视觉码本条目。
- **为什么重要：** 把视觉输入转换成序列，才能让 transformer 风格的组件把图像与文本或其他已 token 化的模态一起处理。
- **实践建议：** 要说明 token 是连续 patch 还是离散编码，保留空间位置信息，测试分辨率与长宽比变化的影响，并把视觉 token 计入模型输入预算。
- **常见误区：** 一个 image token 不一定对应一个像素、一个物体，或一块固定物理区域；它的范围取决于所使用的视觉编码器或 tokenizer。
- **延伸学习：** [Vision-Language Models](../phases/04-computer-vision/25-vision-language-models/)
- **相关术语：** Patch Embedding, Token, VAE (Variational Autoencoder), Vision Transformer (ViT)
- **来源：** [Vision Transformer](https://arxiv.org/abs/2010.11929); [VQ-VAE](https://arxiv.org/abs/1711.00937)

### In-Context Learning
- **类别：** 提示与上下文
- **实际含义：** 模型不通过常规参数更新，而是根据当前输入中提供的指令、示例或模式临时调整自身行为。
- **为什么重要：** 它解释了为什么同一个预训练模型在权重不变的情况下，仅凭上下文就能完成新任务。
- **实践建议：** 把有代表性的示例放在目标输入前面，测试不同顺序和格式的变体，并让评测样例与示例本身保持分离。
- **常见误区：** In-context learning 是一种临时条件化，不是 fine-tuning、不是持久记忆，也不代表模型就一定真正推断出了你想要的规则。
- **相关术语：** Few-Shot, Zero-Shot, Context Window, Prompt Engineering
- **来源：** [Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165)

### Incident Response
- **类别：** 可靠性与运维
- **实际含义：** 针对威胁服务、数据、安全性或安全防护的事件，所进行的协调流程，包括检测、分析、遏制、恢复、沟通，以及事后学习。
- **为什么重要：** 在事故处理中，清晰的角色分工和证据比临场逞英雄更重要，尤其是在模型行为和分布式依赖让故障边界变得模糊时。
- **实践建议：** 要定义清楚事件等级和指挥角色，保留 trace 与审计记录，停止有害动作，沟通影响范围，验证恢复情况，并持续跟踪纠正工作直到完成。
- **常见误区：** Incident response 处理的是事件本身及其后果；根因分析和长期预防工作会在服务初步恢复后继续进行。
- **延伸学习：** [SRE for AI](../phases/17-infrastructure-and-production/23-sre-for-ai/)
- **相关术语：** Observability, Audit Log, Postmortem, Availability
- **来源：** [Google SRE: Managing Incidents](https://sre.google/sre-book/managing-incidents/); [NIST SP 800-61 Rev. 3](https://csrc.nist.gov/pubs/sp/800/61/r3/final)

### Indirect Prompt Injection
- **类别：** 安全与治理
- **实际含义：** 一种 prompt injection 攻击，它不是直接通过用户指令输入，而是通过系统检索到或观察到的内容投递进来，例如网页、文档、邮件、图像文字或工具结果。
- **为什么重要：** agent 在执行授权任务时，可能会接触到由攻击者控制的指令内容，并误把这些内容当成有权威性的指导。
- **实践建议：** 把外部内容标记为不可信数据，并与指令层分离；尽量缩小工具权限；对重要动作要求审批；同时把恶意检索内容纳入回归测试。
- **常见误区：** Indirect 说的是投递路径，而不是攻击更弱。隐藏在检索内容里的指令，后果可能和直接来自用户的恶意 prompt 一样严重。
- **延伸学习：** [Indirect Prompt Injection](../phases/18-ethics-safety-alignment/15-indirect-prompt-injection/)
- **相关术语：** Prompt Injection, Instruction Hierarchy, Trust Boundary, Data Exfiltration
- **来源：** [Compromising Real-World LLM-Integrated Applications with Indirect Prompt Injection](https://arxiv.org/abs/2302.12173)

### Inductive Bias
- **类别：** 模型与推理
- **常见说法：** 内置在学习系统里的假设。
- **实际含义：** 指那些让某些函数形式或表示方式比其他形式更容易被学到的结构性或统计性假设。比如 convolution 偏好局部性与滤波器共享，causal masking 偏好根据前文位置进行预测。
- **常见误区：** Transformer 依然具有 inductive bias，这些偏置来自 tokenization、位置处理、masking、架构设计、训练数据和训练目标。
- **相关术语：** CNN (Convolutional Neural Network), Transformer, Feature

### Inference
- **类别：** 模型与推理
- **常见说法：** 运行一个已经训练好的模型。
- **实际含义：** 执行一个已训练模型，用来产生预测、分数、embedding 或生成 token，而不会对其参数执行常规训练更新。
- **常见误区：** 即便模型权重保持不变，应用在 inference 过程中仍然可能更新 cache、会话状态或外部记忆。
- **相关术语：** Autoregressive, Streaming, KV Cache

### Instruction Following
- **类别：** 提示与上下文
- **实际含义：** 模型把自然语言指令与给定上下文映射为符合任务要求和约束条件的行为的能力。
- **为什么重要：** 语言生成可以很流畅，但依然可能不遵守用户要求的操作方式、输出格式、边界条件或优先级。
- **实践建议：** 应当把 instruction adherence 与回答质量分开评估，测试材料要覆盖冲突约束、格式要求、无关上下文以及拒绝场景。
- **常见误区：** Instruction following 不等于事实正确、不等于 alignment，也不等于对所有看起来像指令的字符串都照单全收。
- **相关术语：** SFT (Supervised Fine-Tuning), Prompt Engineering, Instruction Hierarchy, Alignment
- **来源：** [Finetuned Language Models Are Zero-Shot Learners](https://arxiv.org/abs/2109.01652)

### Instruction Hierarchy
- **类别：** 提示与上下文
- **实际含义：** 一套用于解决不同权威来源之间指令冲突的规则，比如应用策略、用户输入和不可信检索内容之间的冲突。
- **为什么重要：** agent 系统会把受信目标与外部文本混在一起，因此当低权威内容与高权威约束冲突时，模型和 harness 都必须有预先定义好的处理方式。
- **实践建议：** 把不可信工具输出明确标记为数据，把更高优先级的约束放在这些内容之外单独保留，并同时测试直接冲突与间接冲突场景。
- **常见误区：** Instruction hierarchy 可以改善行为表现，但它不是安全边界；真正限制后果的，仍然是 least privilege 和审批控制。
- **相关术语：** System Prompt, Prompt Injection, Least Privilege, Tool Contract
- **来源：** [The Instruction Hierarchy](https://arxiv.org/abs/2404.13208)

### Inter-Token Latency (ITL)
- **类别：** 基础设施与服务
- **实际含义：** 对单个请求而言，两个连续输出 token 到达事件之间经过的时间。对于首个 token 之后的任一输出 token，它计算为 `t_i - t_(i-1)`。
- **为什么重要：** 单个间隔能够暴露 decode 停顿和 streaming 抖动，而这些问题往往会被按请求求平均的指标掩盖，尤其是在 batching、抢占或混合负载下。
- **实践建议：** 记录每一个首 token 之后的时间间隔，并关联对应请求和 token 位置；随后按工作负载、输出长度和并发度报告分布，不要因为汇总而抹掉请求边界。
- **常见误区：** ITL 指的是相邻 token 之间某一个具体时间间隔。Time per output token 是对这些间隔在单个请求内求出的平均值，而 time to first token 统计的是开始 streaming 之前的等待时间。
- **延伸学习：** [Inference Metrics and Goodput](../phases/17-infrastructure-and-production/08-inference-metrics-goodput/)
- **相关术语：** Time per Output Token (TPOT), Time to First Token (TTFT), Decode Phase, Tail Latency
- **来源：** [DistServe](https://arxiv.org/abs/2401.09670)

## J

### Jailbreak
- **类别：** 安全与治理
- **实际含义：** 一种对抗性输入或交互策略，目的是让模型做出其训练机制或应用控制本来试图阻止的行为。
- **为什么重要：** 成功的 jailbreak 会暴露出声明策略与实际行为之间的落差；如果模型还能控制工具或受保护数据，后果会更严重。
- **实践建议：** 围绕被禁止的行为构造测试家族，变化输入格式和交互长度，同时衡量拒绝效果和有害完成情况，并把已确认的失败沉淀为带版本的对抗性 eval。
- **常见误区：** Jailbreak 针对的是模型或系统的行为限制；prompt injection 则是把 instruction following 重定向到攻击者的目标上。一段交互里，二者可能同时出现。
- **延伸学习：** [Jailbreak Taxonomy](../phases/19-capstone-projects/82-jailbreak-taxonomy/)
- **相关术语：** Prompt Injection, Red Teaming, Guardrails, Eval Set
- **来源：** [Universal and Transferable Adversarial Attacks on Aligned Language Models](https://arxiv.org/abs/2307.15043)

### JAX
- **类别：** 数学与训练
- **常见说法：** 一个类似 NumPy 的加速机器学习系统。
- **实际含义：** 一个 Python 库，用于通过自动微分、编译、向量化以及跨加速器并行执行来变换数值函数。它的这些变换机制在显式状态管理和函数式风格代码下效果最好。
- **常见误区：** JAX 并不禁止所有带状态编程，但如果在被变换的函数内部隐藏可变状态，就可能产生错误结果或不受支持的行为。
- **延伸学习：** [Introduction to JAX](../phases/03-deep-learning-core/12-intro-to-jax/)
- **来源：** [JAX documentation](https://docs.jax.dev/en/latest/)
- **相关术语：** Autograd, Tensor, CUDA

## K

### Knowledge Distillation
- **类别：** 数学与训练
- **实际含义：** 训练一个 student model 去复现更强 teacher 的特定行为或输出分布，通常还会同时使用常规目标标签。
- **为什么重要：** 当直接提供 teacher 的服务不切实际时，它可以把有用的行为迁移到更小或更便宜的模型里。
- **实践建议：** 先定义 teacher outputs、temperature、student loss 和 held-out eval set，再把 student 同 teacher 以及仅用标签训练的 baseline 做对比。
- **常见误区：** Distillation 转移的是训练分布上的行为，并不会复制 teacher 的每一项能力、事实知识或安全属性。
- **相关术语：** Fine-tuning, Loss Function, Logits, Quantization
- **来源：** [Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531)

### KV Cache
- **类别：** 模型与推理
- **常见说法：** 一种让 token 生成更快的缓存。
- **实际含义：** 在 autoregressive generation 中保存较早位置的 key 和 value tensor。复用它们可以避免在每个 decoding step 上都为未变化的 prefix 重新计算 attention projection。
- **为什么重要：** 它能减少重复计算，但会占用随 sequence length、layers、batch 和 model configuration 增长的内存。
- **常见误区：** KV cache 是单个序列在运行时的 attention state。Prefix caching 会在请求之间复用符合条件的 KV state，而 prompt caching 是更宽泛的 provider 或应用级复用契约。
- **延伸学习：** [KV Cache and Flash Attention](../phases/07-transformers-deep-dive/12-kv-cache-flash-attention/)
- **相关术语：** Attention, Autoregressive, Prefix Caching, Prompt Cache

## L

### Late Fusion
- **类别：** 多模态系统
- **实际含义：** 先用独立的 encoder 或 predictor 处理各个 modality，再在接近任务输出的位置合并它们的高层表征、分数或决策。
- **为什么重要：** 分支彼此独立时，可以使用各自适配 modality 的架构，也更能容忍输入缺失；但它们可能错过只有更早融合阶段才能捕捉到的细粒度交互。
- **实践建议：** 分别校准每个 branch，定义缺失 modality 时如何合并，比较 score-level 与 feature-level 的组合方式，并把各 branch 单独运行作为 ablation。
- **常见误区：** Late fusion 描述的是合并发生的位置。它不等于简单平均，也不保证各个 modality 的贡献相同。
- **延伸学习：** [Cross-Attention Fusion](../phases/19-capstone-projects/61-cross-attention-fusion/)
- **相关术语：** Early Fusion, Multimodal Fusion, Modality, Evaluation (Eval)
- **来源：** [Multimodal Deep Learning](https://ai.stanford.edu/~ang/papers/icml11-MultimodalDeepLearning.pdf); [多模态机器学习：综述与分类](https://arxiv.org/abs/1705.09406)

### Latent Space
- **类别：** 数据与表征
- **常见说法：** 模型的隐藏表征空间。
- **实际含义：** 一种学习得到的表征空间，其中的坐标编码了对模型有用的因素。它可能比输入维度更低，但并不是所有 latent representation 都必须承担压缩作用。
- **常见误区：** 相邻点是否真的相似，只能依据模型和训练目标实际学到的内容来判断。
- **相关术语：** Embedding, VAE (Variational Autoencoder), Feature

### Learning Rate
- **类别：** 数学与训练
- **常见说法：** 每一步优化走多大。
- **实际含义：** optimizer 用来控制 parameter update 幅度的缩放因子。数值过大可能让训练失稳；数值过小则会让有效进展慢到不切实际。
- **常见误区：** 实际的 update 还取决于 optimizer、schedule、gradient scale、batch 以及参数历史。
- **相关术语：** Optimizer, Gradient Descent, Batch Size

### Learning Rate Schedule
- **类别：** 数学与训练
- **实际含义：** 一种会随着训练推进而改变 optimizer learning rate 的策略，变化依据可以是 steps、epochs、metrics，或预定义曲线。
- **为什么重要：** 训练的不同阶段往往适合不同的 update scale，因此固定不变的 rate 可能在前期不稳定、在后期又浪费。
- **实践建议：** 把 schedule 与 optimizer configuration 一起做版本管理，记录每一步的实际 rate，并在相同 token 或 update budget 下比较不同 schedule。
- **常见误区：** scheduler 控制的是 learning rate 随时间如何变化；它并不决定 optimizer step 何时发生，也不保证一定收敛。
- **相关术语：** Learning Rate, Warmup, Optimizer, Epoch
- **来源：** [SGDR](https://arxiv.org/abs/1608.03983); [Attention Is All You Need](https://arxiv.org/abs/1706.03762)

### Least Privilege
- **类别：** 评估与安全
- **实际含义：** 只给 model、agent、tool 或用户当前任务所需的最小权限，而且只在需要的那段时间内授予。
- **为什么重要：** 模型可能出错，也可能执行恶意指令。更窄的权限可以降低单次失败造成的损害。
- **实践建议：** 可以给文档 agent 读取源码和写入单个分支的权限，但不要给 production credentials 或 merge permission。
- **常见误区：** Authentication 证明的是身份；least privilege 限制的是这个身份能做什么。
- **相关术语：** Sandbox, Approval Gate, Prompt Injection, Tool Contract

### LLM (Large Language Model)
- **类别：** 模型与推理
- **常见说法：** AI 应用的大脑。
- **实际含义：** 一种具备足够容量并接受过广泛训练的 language model，能够通过 prompting 或 adaptation 完成多种语言任务。当前多数 LLM 使用 transformer architecture 和 sequence-prediction objective，但规模门槛、数据来源和训练配方并不统一。
- **常见误区：** LLM 是系统里的一个模型组件。tools、retrieval、state、policies 和 product logic 都存在于它外围的系统中。
- **相关术语：** Transformer, Autoregressive, Agent Harness

### LLM-as-a-Judge
- **类别：** 评估与安全
- **实际含义：** 用 language model 按照 rubric 对另一个系统的输出进行打分、比较、分类或点评。
- **为什么重要：** 它能把一些难以写成 exact-match test 的质量维度扩展到可规模化评估，比如清晰度或指令遵循度。
- **实践建议：** 把任务、候选答案、参考证据和结构化 rubric 提供给独立的 evaluator model，再用人工复核过的样本来校准它的评分。
- **常见误区：** judge model 不是 ground truth。它可能受到顺序、篇幅、文风、prompt 措辞或共享模型缺陷的偏置影响。
- **延伸学习：** [评测驱动的代理开发](../phases/14-agent-engineering/30-eval-driven-agent-development/)
- **相关术语：** Evaluation (Eval), Eval Set, Verification Gate, Precision & Recall

### Load Shedding
- **类别：** 可靠性与运维
- **实际含义：** 当需求超过系统能够产出有用结果的容量时，在一个或多个过载边界上有意拒绝、丢弃或取消部分工作。
- **为什么重要：** 在过载时继续接受所有请求，会让排队不断加剧，直到几乎所有请求都错过截止时间，恢复也会更困难。
- **实践建议：** 应尽量在最早且信息充分的边界做 shedding，在可能时保住高优先级和已接纳的工作，明确过载范围，并且只有当问题是暂时性的且请求仍在 retry budget 内时，才把响应标记为可重试。
- **常见误区：** Load shedding 并不只针对已经接受的工作。Admission control 特指接纳前的闸门，而 rate limiting 即使在容量尚有余量时也可以执行使用策略。
- **相关术语：** Admission Control, Backpressure, Rate Limit, Graceful Degradation
- **来源：** [Google SRE：处理过载](https://sre.google/sre-book/handling-overload/)

### Logits
- **类别：** 模型与推理
- **实际含义：** 模型在 normalization function 或 decoding rule 把候选结果变成最终选择之前，给各候选结果打出的未归一化数值分数。
- **为什么重要：** Temperature、softmax、top-k 和 top-p 都直接作用于 logits 或由其推导而来，因此 logits 把模型计算和最终生成的 tokens 连接起来。
- **实践建议：** 当 API 暴露 logits 或 log probabilities 时应查看它们，在 sampling 之前先施加 masks，并避免把原始数值大小当成已校准的置信度。
- **常见误区：** logits 不是 probability；如果没有明确定义的变换，也不能跨不相关的位置、模型或任务直接比较。
- **相关术语：** Softmax, Temperature, Token, Cross-Entropy
- **来源：** [Attention Is All You Need](https://arxiv.org/abs/1706.03762)

### LoRA (Low-Rank Adaptation)
- **类别：** 数学与训练
- **常见说法：** 参数高效的 fine-tuning。
- **实际含义：** 一种让 base weights 保持冻结、只为选定 layers 学习 low-rank update matrices 的方法。与 full-parameter fine-tuning 相比，它能减少可训练参数数量，也可能降低训练内存占用。
- **常见误区：** 实际的内存和速度收益取决于 rank、target modules、optimizer state、activation memory、quantization 以及具体实现。
- **延伸学习：** [Fine-Tuning and LoRA](../phases/11-llm-engineering/08-fine-tuning-lora/)
- **来源：** [LoRA paper](https://arxiv.org/abs/2106.09685)
- **相关术语：** Fine-tuning, QLoRA, Parameter

### Loss Function
- **类别：** 数学与训练
- **常见说法：** 衡量训练误差的一个数。
- **实际含义：** 一种把 predictions 和 targets（有时还包括 regularization terms）映射为单个数值目标的函数，而优化过程会尝试减小这个值。loss 决定了训练会直接奖励或惩罚哪些错误。
- **常见误区：** 训练 loss 低，并不保证模型在生产任务上就一定有用、安全，或具备泛化能力。
- **相关术语：** Cross-Entropy, Gradient, Evaluation (Eval)

### Lost in the Middle
- **类别：** 提示与上下文
- **实际含义：** 一种长上下文失败模式：模型表现会随证据位置变化，当关键信息位于上下文中段时，效果可能下降。
- **为什么重要：** 把证据塞进 context window，并不意味着模型会以同等可靠性使用每个位置上的信息。
- **实践建议：** 测试多种证据摆放位置，减少干扰项，把决策关键约束放在更容易保持显著的位置，并用 source 去核对答案。
- **常见误区：** 这是一种被观察到的行为模式，不是对所有模型、任务和位置都同样成立的固定规律。
- **相关术语：** Context Window, 上下文工程, Eval Set, Grounding
- **来源：** [Lost in the Middle](https://aclanthology.org/2024.tacl-1.9/)

## M

### Maximum Marginal Relevance (MMR)
- **类别：** 检索与生成
- **实际含义：** 一种选择规则，在与 query 的相关性和相对已选条目的新颖性之间取得平衡。
- **为什么重要：** 它可以减少冗余 chunks，让有限的 context budget 覆盖更多彼此不同的证据。
- **实践建议：** 先检索一个 candidate pool，再用文档化的 relevance-diversity 权重选择下一个条目，并同时评估答案质量和 source coverage。
- **常见误区：** MMR 只是让现有 candidate set 更加多样化；它不会找回缺失证据，也不能证明选中的 passages 就是正确的。
- **相关术语：** Reranker, Chunking, RAG (Retrieval-Augmented Generation), Grounding
- **来源：** [The Use of MMR, Diversity-Based Reranking for Reordering Documents and Producing Summaries](https://www.cs.cmu.edu/~jgc/publication/MMR_DiversityBased_Reranking_SIGIR_1998.pdf)

### MCP (Model Context Protocol)
- **类别：** 智能体与工具
- **常见说法：** AI 应用连接 tools 和 context 的一种标准方式。
- **实际含义：** 一种开放的 JSON-RPC 协议，让 host 可以连接那些通过明确的 request、result、discovery 和 transport contract 暴露 tools、resources、prompts 与 extensions 的 server。在 2026-07-28 版本中，每个 request 都会携带自己的 protocol version 和 client capabilities，而不是依赖 initialization handshake 或 protocol session。
- **常见误区：** MCP 标准化的是发现和交换过程。它并不决定哪个 tool 调用是安全的，不负责授予权限，也不会禁止应用使用显式 state handle。
- **延伸学习：** [Model Context Protocol](../phases/11-llm-engineering/14-model-context-protocol/)
- **来源：** [MCP 2026-07-28 key changes](https://modelcontextprotocol.io/specification/2026-07-28/changelog)
- **相关术语：** Stateless MCP, Multi Round-Trip Request (MRTR), Function Calling, Tool Contract, Least Privilege

### Membership Inference
- **类别：** 安全与治理
- **实际含义：** 一种攻击，通过观察模型输出或其他可获取信号，推断某条特定记录或样本是否出现在模型训练数据中。
- **为什么重要：** 即使模型不会逐字复现某条记录，可区分的行为仍可能泄露它是否参与过敏感数据集训练的信息。
- **实践建议：** 应在真实 query interface 下测试有代表性的 members 和 non-members，减少不必要的 confidence signals，降低数据暴露，并结合 utility requirements 评估隐私防护手段。
- **常见误区：** Membership inference 关注的是某条记录是否参与过训练。Model extraction 试图复现模型行为，而 direct memorization 测试的是内容能否被恢复出来。
- **延伸学习：** [Differential Privacy for LLMs](../phases/18-ethics-safety-alignment/22-differential-privacy-for-llms/)
- **相关术语：** Data Leakage, Data Minimization, Eval Set, Data Classification
- **来源：** [Membership Inference Attacks Against Machine Learning Models](https://doi.org/10.1109/SP.2017.41)

### Mixed Precision
- **类别：** 数学与训练
- **常见说法：** 使用更低精度的计算来换取速度和内存收益。
- **实际含义：** 一种数值策略：对不同运算使用不同数据类型，通常让大量矩阵运算使用较低精度，而把需要更大范围或更高稳定性的数值保留在较高精度。
- **常见误区：** 速度、内存和精度效果取决于硬件、数据类型、scaling method、kernels 和模型本身，并不是固定倍数关系。
- **相关术语：** Tensor, CUDA, NaN (Not a Number), Quantization

### Modality
- **类别：** 多模态系统
- **实际含义：** 一种具有自身结构和采集过程的信息形态，比如 text、image、audio、video、depth 或 sensor measurements。
- **为什么重要：** 不同 modality 在采样率、噪声、空间或时间结构以及缺失数据行为上都不同，因此单一的 preprocessing 假设很少适用于全部情况。
- **实践建议：** 在设计 alignment 或 fusion 之前，应先记录每个 modality 的 source、units、resolution、timing、preprocessing 和 missing-value policy。
- **常见误区：** modality 不只是文件扩展名或某一列特征。一个 modality 可以有多种编码方式，而一个样本也可以同时包含多个 modalities。
- **延伸学习：** [MIO Any-to-Any Streaming](../phases/12-multimodal-ai/16-mio-any-to-any-streaming/)
- **相关术语：** Multimodal Model, Token, Tensor, Embedding
- **来源：** [ImageBind: One Embedding Space To Bind Them All](https://arxiv.org/abs/2305.05665); [多模态机器学习：综述与分类](https://arxiv.org/abs/1705.09406)

### Modality Alignment
- **类别：** 多模态系统
- **实际含义：** 学习或建立不同 modality 表征之间的对应关系，使语义上或时间上相关的对象能够被匹配起来。
- **为什么重要：** 如果系统无法在结构不同的输入之间对应到同一个事件、对象或概念，fusion 和 cross-modal retrieval 就会失败。
- **实践建议：** 应定义 positive 和 negative pairs，保留时间或空间 metadata，评估不匹配样本，并把 alignment 与下游任务准确率分开衡量。
- **常见误区：** Alignment 的作用是让表征之间可以比较或建立对应关系，并不要求它们变得完全相同，也不会抹掉 modality-specific information。
- **延伸学习：** [Projection Layer Modality Alignment](../phases/19-capstone-projects/60-projection-layer-modality-align/)
- **相关术语：** Shared Embedding Space, Contrastive Learning, Grounding, Multimodal Fusion
- **来源：** [Learning Transferable Visual Models From Natural Language Supervision](https://proceedings.mlr.press/v139/radford21a.html)

### Model Card
- **类别：** 评估与安全
- **实际含义：** 一种结构化报告，用来说明模型的预期用途、评估条件、性能特征、局限性，以及相关的伦理或安全考量。
- **为什么重要：** 它为下游构建者提供判断依据，帮助他们评估已有证据是否适用于自己的用户和部署条件。
- **实践建议：** 应记录 model version、训练与评估范围、subgroup results、已知失败模式、禁止用途，以及每条结论对应的日期。
- **常见误区：** model card 传达的是证据和局限性；它不是认证、担保、系统级 threat model，也不能替代面向具体部署场景的评估。
- **相关术语：** Eval Set, Dataset Split, Distribution Shift, Alignment
- **来源：** [Model Cards for Model Reporting](https://dl.acm.org/doi/10.1145/3287560.3287596)

### Model Router
- **类别：** AI 原生开发
- **实际含义：** 一种组件，会根据 capability、latency、cost、context size、policy 和当前可用性等要求，为请求选择 model 或 provider。
- **为什么重要：** 不同任务和失败条件适合不同模型，而 routing 可以在不把所有请求都送到最大模型的前提下提升结果质量。
- **实践建议：** 把低风险 extraction 交给快模型，把复杂 code review 交给更强模型，fail over 时也只切到满足同一数据策略的 provider。
- **常见误区：** Routing 是策略决策；随机负载均衡只是在分发流量。
- **相关术语：** Evaluation (Eval), Circuit Breaker, Rate Limit, Cost per Successful Task

### Model Serving
- **类别：** 基础设施与服务
- **实际含义：** 负责加载带版本的 model artifacts、接收 inference requests、调度执行、管理资源，并在 operational contract 下返回结果的 runtime 与 API 层。
- **为什么重要：** 即使模型本身能力很强，如果 queueing、batching、placement、versioning、cancellation 和 response boundary 没有被明确设计，最终产品依然可能不可靠。
- **实践建议：** 应固定 model 和 tokenizer 版本，校验 request limits，暴露 readiness 与 latency signals，控制 concurrency，并在导入生产流量前验证 rollback。
- **常见误区：** Model serving 的范围大于单次 inference 调用，但又小于完整应用；完整应用还可能包含 retrieval、tools、policy 和 user state。
- **延伸学习：** [Self-Hosted Serving Selection](../phases/17-infrastructure-and-production/28-self-hosted-serving-selection/)
- **相关术语：** Inference, Model Router, Autoscaling, Observability
- **来源：** [Clipper](https://arxiv.org/abs/1612.03079)

### MoE (Mixture of Experts)
- **类别：** 模型与推理
- **常见说法：** 一种对每个 token 只激活部分参数的大模型。
- **实际含义：** 一种包含多个 expert subnetwork 和学习型 router 的架构，router 会为每个输入单元（通常是每个 token）选择其中一部分。通过 sparse activation，可以在不让每次 forward pass 都使用全部 expert 的情况下提升总参数容量。
- **为什么重要：** compute、memory、communication、routing balance 和质量表现都取决于具体架构与 serving system。
- **常见误区：** 除非模型开发者明确披露，否则产品名称本身不能证明它就是 MoE 架构。
- **延伸学习：** [Mixture of Experts](../phases/07-transformers-deep-dive/11-mixture-of-experts/)
- **相关术语：** Transformer, Model Router, Parameter

### Multimodal Fusion
- **类别：** 多模态系统
- **实际含义：** 把来自多个 modality 的证据或学习到的表征结合起来，生成联合表征、预测结果或生成输出。
- **为什么重要：** 不同 modality 可以提供互补证据，但简单粗暴的组合也可能放大噪声、时序误差，或让某一条主导流过度压制其他信息。
- **实践建议：** 应先建立 single-modality baseline，明确 fusion point 和 masks，测试缺失与互相矛盾的输入，并报告每个评估切片主要由哪些 modalities 驱动。
- **常见误区：** Fusion 指的是组合操作本身。Alignment 负责建立对应关系，而把两个 modality 放进同一个 request 并不能证明这两件事真的成功发生了。
- **延伸学习：** [Cross-Attention Fusion](../phases/19-capstone-projects/61-cross-attention-fusion/)
- **相关术语：** Early Fusion, Late Fusion, Cross-Attention, Modality Alignment
- **来源：** [Multimodal Deep Learning](https://ai.stanford.edu/~ang/papers/icml11-MultimodalDeepLearning.pdf); [多模态机器学习：综述与分类](https://arxiv.org/abs/1705.09406)

### Multimodal Model
- **类别：** 多模态系统
- **实际含义：** 一种能够通过 representation、alignment、fusion、translation 或 coordinated prediction 来学习、关联或生成多种 modality 的模型。
- **为什么重要：** 多模态能力取决于各 modality 如何交互，而不只是能接收几种输入类型；并且每一道 representation boundary 都可能出错。
- **实践建议：** 应记录支持的输入输出组合，分别评估各 modality 单独使用和联合使用的表现，测试缺失或冲突输入，并把 preprocessing version 与模型一起追踪。
- **常见误区：** 由独立 image model 和 text model 组成的 pipeline 在系统层面可以算多模态，但它未必是一个联合训练出的 multimodal model。
- **延伸学习：** [MIO Any-to-Any Streaming](../phases/12-multimodal-ai/16-mio-any-to-any-streaming/)
- **相关术语：** Modality, Vision-Language Model (VLM), Multimodal Fusion, Transformer
- **来源：** [Flamingo: a Visual Language Model for Few-Shot Learning](https://arxiv.org/abs/2204.14198); [多模态机器学习：综述与分类](https://arxiv.org/abs/1705.09406)

### Multi Round-Trip Request (MRTR)
- **类别：** 智能体与工具
- **别名：** MRTR
- **实际含义：** 一种 MCP request pattern：某次操作先返回 `resultType: input_required`，并附带一个或多个 `inputRequests`，随后 client 使用 `inputResponses` 和原样返回的 `requestState` 重试最初的方法调用。
- **为什么重要：** 它允许 stateless server 在不发起由 server 主导的 JSON-RPC 交互、也不保存 protocol session state 的情况下请求 user、model 或 root input。
- **实践建议：** 先从 `tools/call` 返回 input request，在 host 中收集经过授权的响应，再用新的 JSON-RPC id 重试同一个 tool call。
- **常见误区：** `requestState` 是不可信的 round-trip data。在把它用于授权或业务决策前，应先做完整性保护，也不要把它当成 server-side session identifier。
- **延伸学习：** [MCP Roots and Elicitation](../phases/13-tools-and-protocols/12-mcp-roots-and-elicitation/)
- **相关术语：** Stateless MCP, MCP (Model Context Protocol), Human-in-the-Loop (HITL), Tool Contract
- **来源：** [MCP Multi Round-Trip Requests](https://modelcontextprotocol.io/specification/2026-07-28/basic/patterns/mrtr)

## N

### NaN (Not a Number)
- **类别：** 数学与训练
- **常见说法：** 数值计算出问题的信号。
- **实际含义：** 一种表示未定义或不可表示数值结果的 floating-point value。在训练中，NaN 可能来自非法运算、overflow、不稳定的 normalization、过大的 update，或更早就已损坏的数值。
- **实践建议：** 先找到第一个 non-finite tensor，检查它的输入，并在对应操作附近加入 assertions 或 anomaly detection。
- **相关术语：** Mixed Precision, Learning Rate, Gradient

### Normalization
- **类别：** 数学与训练
- **常见说法：** 把数据缩放到标准范围。
- **实际含义：** 一类利用给定统计量对 inputs、activations 或 features 做重新缩放或重中心化的变换。Batch normalization 和 layer normalization 使用的轴不同，在 training 和 inference 阶段的行为也不同。
- **常见误区：** Normalization 可以提升优化稳定性，但并不总能支持更大的 learning rate，也不会自动改善所有架构。
- **相关术语：** Tensor, Activation Function, Mixed Precision

### Nucleus Sampling (Top-p)
- **类别：** 模型与推理
- **别名：** Top-p sampling
- **实际含义：** 一种 decoding method：从累计概率刚好达到设定阈值的最小 next-token 候选集合中进行采样。
- **为什么重要：** candidate set 的大小会随分布自适应变化，不确定性更分散时保留更多选项，概率更集中时保留更少。
- **实践建议：** 评估阈值时，应保持 temperature 和 stop settings 不变，并为每个结果记录完整的 decoding configuration。
- **常见误区：** Top-p 是概率质量阈值，而 top-k 始终保留固定上限数量的候选项。
- **相关术语：** Top-k Sampling, Temperature, Decoding Strategy, Softmax
- **来源：** [神经文本退化的奇特现象](https://arxiv.org/abs/1904.09751)

## O

### Observability
- **类别：** AI 原生开发
- **实际含义：** 一种能力：能够根据记录下来的 inputs、outputs、state transitions、tool calls、timings、costs、errors 和 evaluation signals 来理解 AI 系统的行为。
- **为什么重要：** AI 系统的失败往往横跨 model、retrieval、tools 和 orchestration，多路关联证据是定位故障边界所必需的。
- **实践建议：** 应在 retrieval、model calls、tool execution、approvals 和最终 scoring 全链路记录 trace ID，同时落实脱敏和访问控制。
- **常见误区：** Logging 只是收集事件；Observability 则让这些事件足够结构化、彼此关联，从而能够回答运维层面的问题。
- **延伸学习：** [Agent Observability Platforms](../phases/14-agent-engineering/24-agent-observability-platforms/)
- **相关术语：** Trace, Evaluation (Eval), Agent State, Time to First Token (TTFT)

### Optimizer
- **类别：** 数学与训练
- **常见说法：** 更新权重的算法。
- **实际含义：** 一种把 gradients 转换为 parameter updates 的算法。普通的 stochastic gradient descent 是简单 baseline；momentum、Adam 等 optimizer 会利用历史信息或自适应缩放来改变 update。不同选择在内存占用、稳定性和调参行为上各不相同。
- **常见误区：** optimizer 负责消费 gradients；backpropagation 负责计算 gradients。
- **相关术语：** Adam (Optimizer), AdamW, Gradient, Learning Rate

### Orchestration
- **类别：** 智能体与工具
- **实际含义：** 一种控制逻辑，用来在 model 和 tool 步骤之间编排顺序、分支、委派、重试、暂停、恢复和终止工作。
- **为什么重要：** 可靠的 agent 行为依赖于模型之外的显式 workflow 决策，尤其是在任务存在依赖关系或重要副作用时。
- **实践建议：** 应把稳定步骤编码成 workflow 或 state machine，只把有边界的决策暴露给模型，并在对外写入之前持久化状态转换。
- **常见误区：** Orchestration 不等于 autonomy，也不等于 multi-agent system；单个 agent 也可以通过确定性的 workflow 被编排。
- **相关术语：** Agent Harness, Planning, Delegation, Durable Execution
- **来源：** [构建高效代理](https://www.anthropic.com/research/building-effective-agents)

### Overfitting
- **类别：** 数学与训练
- **常见说法：** 模型把训练数据背下来了。
- **实际含义：** 一种泛化差距现象：模型在训练数据上的表现明显好于在有代表性的未见数据上的表现。memorization 可能是成因之一，但在工程上看到的症状是泛化能力差。
- **实践建议：** 应比较 training metrics 与 held-out metrics，检查 subgroup failure，并测试数据质量、regularization、early stopping 或 model capacity 等方面的调整。
- **相关术语：** Underfitting, Dropout, Weight Decay, Eval Set

## P

### Paged KV Cache
- **类别：** 基础设施与服务
- **实际含义：** 一种 KV-cache 内存管理器，将 attention state 存储在固定大小的块中，并把逻辑序列位置映射到物理块，而不是要求每条序列都对应一段连续分配的内存。
- **为什么重要：** 可变序列长度会造成碎片化和不可预测的增长，因此基于块的分配能够提升可用内存，并支持更灵活的共享。
- **实践建议：** 根据工作负载测量结果选择块大小，跟踪分配与驱逐，在请求之间隔离状态，并在内存压力下测试取消和前缀共享。
- **常见误区：** Paged KV cache 管理的是运行时的 attention-state 内存。它不会把模型参数移到磁盘，也不会扩展模型训练时的上下文上限。
- **延伸学习：** [vLLM 服务内部原理](../phases/17-infrastructure-and-production/04-vllm-serving-internals/)
- **相关术语：** KV Cache, Prefix Caching, Context Window, Model Serving
- **来源：** [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180)

### Parameter
- **类别：** 模型与推理
- **常见说法：** 一个用来描述模型规模的数字。
- **实际含义：** 训练过程中学到的一个数值，常见形式包括 weight、bias、embedding 元素或 normalization parameter。参数量是衡量模型容量的一种指标，但它并不能直接决定质量、内存占用或服务成本。
- **常见误区：** 每个参数对应的内存开销取决于数值格式、量化元数据、分片方式、optimizer state、activations 以及运行时额外开销。
- **相关术语：** Weight, MoE (Mixture of Experts), Quantization

### Pass@k
- **类别：** 评估与安全
- **实际含义：** 在一组任务上，至少有 1 个来自 k 个采样候选的结果通过既定正确性测试的任务占比。
- **为什么重要：** 它衡量的是在代码生成这类可由自动验证器检查每个候选结果的任务中，多次采样尝试所带来的价值。
- **实践建议：** 在固定配置下独立生成候选结果，对每个候选运行相同且隔离的测试，并在报告中同时给出 k 以及采样和估计器的细节。
- **常见误区：** Pass@k 不是单次尝试的准确率，更高的分数也可能只是反映了更大的尝试预算，而不是更好的首答质量。
- **相关术语：** Coding Agent, Regression Test, Eval Set, Test Oracle
- **来源：** [Evaluating Large Language Models Trained on Code](https://arxiv.org/abs/2107.03374)

### Patch
- **类别：** AI 原生开发
- **实际含义：** 一种可供审阅的一个或多个文件变更表示形式，通常体现为相对于某个已知基线版本的新增与删除。
- **为什么重要：** patch 为人和智能体提供了一个边界明确的工件，可在不接收整个工作目录的前提下进行审查、测试、应用或拒绝。
- **实践建议：** 要求 coding agent 返回 unified diff，然后验证它只修改了允许触碰的文件，并且能够干净地应用到预期的 commit 上。
- **常见误区：** patch 记录的是文件变更，而不是发布这些变更所需的推理过程、测试证据或审批。
- **延伸学习：** [Workbench for Real Repositories](../phases/14-agent-engineering/41-workbench-for-real-repos/)
- **相关术语：** Coding Agent, Worktree, Scope Contract, Regression Test

### Patch Embedding
- **类别：** 多模态系统
- **实际含义：** 一种学习得到的投影，把图像 patch 转换为固定宽度的向量，并作为 transformer 输入序列中的一个元素。
- **为什么重要：** 它建立了空间图像网格与序列模型之间的接口，其中 patch 大小决定 token 数量以及保留的局部细节。
- **实践建议：** 记录 patch 和图像尺寸，显式处理 padding 或 resizing，加入位置信息，并测量分辨率变化对准确率与 token 成本的影响。
- **常见误区：** patch embedding 是 patch 的向量表示，不是语义目标检测器，也不保证 patch 边界会与视觉实体边界对齐。
- **延伸学习：** [Vision Transformer Patch Tokens](../phases/12-multimodal-ai/01-vision-transformer-patch-tokens/)
- **相关术语：** Vision Transformer (ViT), Image Token, Embedding, Token
- **来源：** [An Image is Worth 16x16 Words](https://arxiv.org/abs/2010.11929)

### Perplexity
- **类别：** 模型与推理
- **常见说法：** 语言模型对某个数据集有多“惊讶”。
- **实际含义：** 在既定 tokenization 与对数约定下，平均负对数似然取指数后的结果。数值越低，表示模型给被评估序列分配的概率越高。
- **常见误区：** Perplexity 不能在不同 tokenizer 或不同评测设置之间直接比较，也不能直接衡量事实正确性或实用性。
- **相关术语：** Cross-Entropy, Token, Evaluation (Eval)

### Pipeline Parallelism
- **类别：** 基础设施与服务
- **实际含义：** 将模型中按顺序排列的一组组层切分到不同设备上，并让 microbatch 或请求像流水线一样通过这些阶段。
- **为什么重要：** 它让模型可以突破单个设备的内存限制，但阶段不均衡、pipeline bubbles、activation 传输以及故障协同都会影响可用性能。
- **实践建议：** 平衡各阶段成本，选择合适的 microbatch 调度方案，测量空闲时间和互连流量，并对模型与 checkpoint 的分片元数据进行版本管理。
- **常见误区：** Pipeline parallelism 按深度切分层；Tensor parallelism 则切分单层内部的 tensor 运算。
- **延伸学习：** [Scaling and Distributed Training](../phases/10-llms-from-scratch/05-scaling-distributed/)
- **相关术语：** Tensor Parallelism, Expert Parallelism, Batch Size, Model Serving
- **来源：** [GPipe](https://arxiv.org/abs/1811.06965)

### Planning
- **类别：** 智能体与工具
- **实际含义：** 构建、选择或修订一组行动及其依赖关系，以便从当前状态推进到目标状态。
- **为什么重要：** 显式计划会在智能体执行昂贵或不可逆操作之前，把前提假设和步骤顺序暴露出来。
- **实践建议：** 先要求给出一个简短且考虑依赖关系的计划，再用可用工具和权限对其进行校验；当观测结果推翻某个假设时，再重新规划。
- **常见误区：** 生成出的计划只是一个提案，并不能证明这些步骤一定可行、充分或安全。
- **相关术语：** Agent State, ReAct, Orchestration, Verification Gate
- **来源：** [LLM+P](https://arxiv.org/abs/2304.11477)

### Postmortem
- **类别：** 可靠性与运维
- **实际含义：** 一种可长期保留的事故记录，说明影响范围、发现方式、响应过程、促成条件、恢复过程以及后续责任行动，而不是用追责替代分析。
- **为什么重要：** 一次已经恢复的故障仍然具有证据价值。记录系统状态和决策过程，能把单次事件转化为减少复发和缩短响应时间的改进。
- **实践建议：** 基于 traces 和 logs 构建时间线，区分触发事件与促成条件，分配带日期的后续行动，并复盘每项行动是否真正改变了相关控制点。
- **常见误区：** postmortem 不是会议纪要，也不是追查某个人错误的过程。它应当产出可验证的系统改进。
- **相关术语：** Incident Response, Regression Test, Audit Log, Observability
- **来源：** [Google SRE: Postmortem Culture](https://sre.google/sre-book/postmortem-culture/)

### Precision & Recall
- **类别：** 评估与安全
- **常见说法：** 用于衡量分类或检索质量的两个指标。
- **实际含义：** Precision 关注被标记出来的项中有多少是正确的；recall 关注所有相关项中有多少被找到了。对于同一个固定评分模型，当你调整决策阈值时，提高 recall 往往会降低 precision，反之亦然。更好的模型则可能同时提升两者。F1 是它们的调和平均数。
- **常见误区：** 合适的阈值和评价指标取决于各类错误的代价，以及目标类别本身的基数。
- **相关术语：** Eval Set, Semantic Search, Guardrails

### Prefill
- **类别：** 基础设施与服务
- **别名：** Prefill Phase
- **实际含义：** 推理开始阶段，负责处理所有输入 token，生成它们的表示以及后续 autoregressive generation 所需的 attention state。
- **为什么重要：** prompt 形状、排队情况与缓存复用都会影响 prefill 成本；而且 prefill 与 decode 对算力的竞争方式不同，因此它会显著影响启动延迟和服务调度。
- **实践建议：** 记录 prompt token 数和 prefill 延迟，区分排队时间与执行时间，比较有缓存和无缓存前缀的差异，并在有活跃 decode 流量时测试长 prompt。
- **常见误区：** Prefill 是运行时处理 prompt 的阶段，不是生成出的第一个 token 本身。只有在 prefill 和所有排队完成之后，第一个 token 才会出现。
- **延伸学习：** [解耦式 Prefill 与 Decode](../phases/17-infrastructure-and-production/17-disaggregated-prefill-decode/)
- **相关术语：** Decode Phase, KV Cache, Time to First Token (TTFT), Chunked Prefill
- **来源：** [Sarathi-Serve](https://www.usenix.org/system/files/osdi24-agrawal.pdf); [DistServe](https://arxiv.org/abs/2401.09670)

### Prefix Caching
- **类别：** 基础设施与服务
- **实际含义：** 在不同请求之间复用由相同且符合条件的 token 前缀生成的 KV-cache 块，从而让服务运行时跳过重复的前缀计算。
- **为什么重要：** 共享的 system instructions、模板或文档会消耗大量 prefill 计算，但只有当 token 序列和缓存资格都匹配时，复用才会真正带来收益。
- **实践建议：** 把稳定 token 放在请求特定内容之前，将模型和 tokenizer 版本纳入缓存标识，隔离租户敏感状态，监控命中率，并把缓存驱逐视为正常现象。
- **常见误区：** Prefix caching 复用的是精确 token 前缀对应的运行时 attention state。Prompt caching 是更宽泛的 provider 或应用层契约，而 semantic caching 复用的是相似请求的历史结果。
- **延伸学习：** [Inference Optimization](../phases/10-llms-from-scratch/12-inference-optimization/)
- **相关术语：** Prompt Cache, Semantic Cache, KV Cache, Paged KV Cache
- **来源：** [SGLang](https://arxiv.org/abs/2312.07104)

### Progressive Disclosure
- **类别：** AI 原生开发
- **实际含义：** 先向人或模型提供最小但足够有用的上下文，再在任务或证据确实需要时逐步揭示更深层细节。
- **为什么重要：** 它能在保留权威细节可按需获取的同时，控制上下文噪声与成本。
- **实践建议：** 先给 coding agent 仓库规则和整体地图；只有在它识别出相关模块之后，再加载完整实现文件。
- **常见误区：** Progressive disclosure 是分阶段开放细节，而不是故意隐瞒做出决策所必需的信息。
- **延伸学习：** [Workbench for Real Repositories](../phases/14-agent-engineering/41-workbench-for-real-repos/)
- **相关术语：** 上下文工程, Repository Map, Token Budget, Handoff

### Prompt Cache
- **类别：** 提示与上下文
- **实际含义：** 对相同或符合条件的 prompt 前缀复用 provider 侧或应用侧的计算结果，从而让重复推理跳过部分预处理工作。
- **为什么重要：** 当满足 provider 的缓存契约时，稳定的指令和大型共享文档在重复调用中会变得更便宜或更快。
- **实践建议：** 把稳定的策略文本放在请求特定内容之前，监控缓存命中元数据，并把未命中视为正常现象，因为不同 provider 的资格条件和生存期并不相同。
- **常见误区：** prompt cache 是 provider 或应用层的复用契约，内部可能会用到 prefix caching。Prefix caching 专指复用符合条件的精确 token KV state，而 semantic caching 复用的是对足够相似请求的历史结果。
- **延伸学习：** [Prompt Caching](../phases/11-llm-engineering/15-prompt-caching/)
- **相关术语：** Semantic Cache, Prefix Caching, KV Cache, Time to First Token (TTFT)

### Prompt Engineering
- **类别：** 提示与上下文
- **常见说法：** 通过措辞让模型按任务要求行事。
- **实际含义：** 为模型设计面向任务的指令、示例、约束和输出要求，以提升其在特定任务上的表现。
- **常见误区：** prompt 的措辞无法弥补证据缺失、不安全的权限设置、糟糕的工具契约或缺位的评估。
- **延伸学习：** [Prompt Engineering](../phases/11-llm-engineering/01-prompt-engineering/)
- **相关术语：** 上下文工程, Few-Shot, System Prompt, Structured Output

### Prompt Injection
- **类别：** 评估与安全
- **常见说法：** 一种会把模型带偏的对抗性指令。
- **实际含义：** 一种攻击或失效模式，其中不受信内容会影响模型，使其无视原定指令、泄露数据、滥用工具，或采取超出用户目标的行动。这些内容可能直接来自用户，也可能间接来自检索页面、文件、消息或工具输出。
- **为什么重要：** 模型通过同一语言通道同时处理指令和数据，因此仅靠输入过滤，无法可靠地区分所有恶意指令与合法内容。
- **实践建议：** 把外部内容视为不可信，将其与带有授权性质的指令隔离，尽量收紧工具权限，对重要写操作要求审批，并验证输出与动作。
- **常见误区：** Prompt injection 在技术机制上并不等同于 SQL injection，而更强的 system prompt 也不是完整防线。
- **延伸学习：** [Prompt Injection Defense](../phases/14-agent-engineering/27-prompt-injection-defense/)
- **来源：** [OWASP prompt injection guidance](https://genai.owasp.org/llmrisk/llm01-prompt-injection/)
- **相关术语：** Least Privilege, Sandbox, Approval Gate, Tool Contract

### Prompt Sensitivity
- **类别：** 提示与上下文
- **实际含义：** 在任务意图不变的前提下，prompt 的措辞、顺序、格式或示例发生变化所引起的模型输出或测量表现差异。
- **为什么重要：** 一个只在某种顺手表述下表现良好的系统，对真实用户来说可能并不可靠，也可能在评估中造成误导。
- **实践建议：** 构造语义等价的 prompt 变体，按案例测量波动，并把这些变体保留在 regression tests 中，而不是只针对单个 eval set 优化一条 prompt。
- **常见误区：** Sensitivity 并不总是 prompt 本身的缺陷；它也可能暴露出歧义、模型鲁棒性不足、解码不稳定，或评分规则不充分。
- **相关术语：** Prompt Engineering, Eval Set, Regression Test, Few-Shot
- **来源：** [ProSA](https://aclanthology.org/2024.findings-emnlp.108/)

### Provenance Attestation
- **类别：** 安全与治理
- **实际含义：** 一种经过认证、机器可读的元数据，用来把某个工件与其生成方式、生成地点、生成时间以及所用输入等声明绑定起来。
- **为什么重要：** 它让自动化策略和审阅者能够验证供应链声明，而不是仅仅相信一条未签名的构建说明。
- **实践建议：** 在构建系统中生成 attestation，将其绑定到 artifact digest，使用受控身份进行签名，并在发布前完成验证。
- **常见误区：** 签名能够标识 attester 并保护完整性，但它并不能证明 attestation 中的每一项声明都为真。
- **相关术语：** Data Provenance, Reproducible Build, Audit Log, Verification Gate
- **来源：** [SLSA Software Attestations](https://slsa.dev/spec/v1.2/attestation-model)

### Purpose Limitation
- **类别：** 安全与治理
- **实际含义：** 对于个人数据，只能为已明确说明且具体的目的进行收集和使用；若要用于新目的，则必须具备适当的兼容性或授权依据。
- **为什么重要：** 某些数据在一个工作流中是可接受的，但若被悄悄复用于模型训练、评估、个性化或无关分析，就可能引发隐私与治理风险。
- **实践建议：** 为每个数据集记录使用目的，在开放访问前用它校验新 pipeline，将不兼容的用途隔离开来，并在用途发生变化时要求有书面决策记录。
- **常见误区：** Purpose limitation 约束的是为什么使用数据；Data minimization 约束的是这个目的实际需要多少数据。
- **相关术语：** Data Minimization, Data Classification, AI Risk Assessment, Audit Log
- **来源：** [General Data Protection Regulation, Article 5(1)(b)](https://eur-lex.europa.eu/eli/reg/2016/679/oj)

## Q

### QLoRA
- **类别：** 数学与训练
- **常见说法：** 带量化基座模型的 LoRA。
- **实际含义：** 一种参数高效的 fine-tuning 方法：将预训练基座模型冻结在低比特量化表示中，同时在需要的地方以更高精度计算来训练 LoRA adapters。
- **为什么重要：** 它可以减少适配大模型所需的内存，但节省幅度和效果质量取决于模型、本征 rank、optimizer、序列长度、硬件和具体实现。
- **常见误区：** QLoRA 并不保证特定的内存占用，也不保证与 full fine-tuning 之间存在固定的质量差距。
- **延伸学习：** [Fine-Tuning and LoRA](../phases/11-llm-engineering/08-fine-tuning-lora/)
- **来源：** [QLoRA paper](https://arxiv.org/abs/2305.14314)
- **相关术语：** LoRA (Low-Rank Adaptation), Quantization, Fine-tuning

### Quantization
- **类别：** 模型与推理
- **常见说法：** 用更少的 bit 来存储或计算模型数值。
- **实际含义：** 用更低精度的格式表示 weights、activations 或 caches，以降低内存、带宽或计算成本。不同方法在校准方式、粒度、数据类型，以及转换发生在训练前、训练中还是训练后等方面各不相同。
- **常见误区：** 从一种标称 bit 宽度切换到另一种，并不保证端到端内存或速度会按同样比例变化，因为元数据、kernel、cache 和硬件支持也会产生影响。
- **相关术语：** QLoRA, Mixed Precision, Parameter

## R

### RAG (Retrieval-Augmented Generation)
- **类别：** 检索与生成
- **常见说法：** 一种借助检索到的知识来回答问题的模型。
- **实际含义：** 一种系统模式：先检索与请求相关的证据，再在模型回答或行动之前，把选出的内容提供给生成模型。检索可以采用 lexical、vector、structured 或 hybrid 方法。
- **为什么重要：** RAG 能在不把信息写进模型权重的前提下，让当前或私有证据可被利用，但 retrieval 和 grounding 必须分别评估。
- **名称由来：** Retrieval 负责找到证据，augmentation 负责把选中的证据加入上下文，generation 负责产出响应。
- **延伸学习：** [Retrieval-Augmented Generation](../phases/11-llm-engineering/06-rag/)
- **来源：** [Retrieval-Augmented Generation paper](https://arxiv.org/abs/2005.11401)
- **相关术语：** Grounding, Hybrid Retrieval, Reranker, Hallucination

### Rate Limit
- **类别：** AI 原生开发
- **实际含义：** 一种策略，用于在既定时间窗口或容量窗口内限制请求数、token 数、并发工作量或其他资源使用。
- **为什么重要：** 它能保护 provider 和你自己的系统，避免过载、失控的成本以及不公平的资源占用。
- **实践建议：** 按租户执行 token 和并发限制，读取 provider 的重试元数据，并以可预期的方式对超量工作进行排队或拒绝。
- **常见误区：** rate limit 控制的是允许使用多少；backpressure 传播的是系统中下游容量约束。
- **相关术语：** Backpressure, Retry with Backoff, Circuit Breaker

### ReAct
- **类别：** 智能体与工具
- **实际含义：** 一种智能体模式：在决定下一步之前，交替进行任务推理、执行具体动作，并接收环境返回的 observation。
- **为什么重要：** 环境反馈能够修正假设，并为后续决策提供 grounding，而不是迫使模型只依赖内部生成把整个任务一次性做完。
- **实践建议：** 提供一小组带类型的工具，返回简洁 observation，限制循环次数，并验证最终工件，而不是保存私有推理轨迹。
- **常见误区：** ReAct 是一种 prompting 与控制模式，并不保证自治性、正确性或工具使用的安全性。
- **相关术语：** Agent, Function Calling, Planning, Grounding
- **来源：** [ReAct](https://arxiv.org/abs/2210.03629)

### Readiness Probe
- **类别：** 可靠性与运维
- **实际含义：** 一种诊断机制，用来告诉流量路由层某个服务实例当前是否能够接收请求。
- **为什么重要：** 一个进程即便还活着，也可能尚未加载模型、依赖不可用，或 warmup 尚未完成，因此过早送入流量会带来本可避免的失败。
- **实践建议：** 检查提供服务所需的最低依赖，在启动和摘流期间让 readiness 失败，保持 probe 足够轻量，并且不要仅仅因为 readiness 为 false 就重启进程。
- **常见误区：** Readiness 决定流量是否可以进入；Liveness 决定进程是否应被重启，而两者都不能证明每次模型响应都是正确的。
- **延伸学习：** [Production LLM Application](../phases/11-llm-engineering/13-production-app/)
- **相关术语：** Autoscaling, Model Serving, Availability, Graceful Degradation
- **来源：** [Kubernetes Liveness, Readiness, and Startup Probes](https://kubernetes.io/docs/concepts/configuration/liveness-readiness-startup-probes/)

### Recall@K
- **类别：** 检索与生成
- **实际含义：** 对于单个 query，Recall@K 定义为 `|relevant items intersecting the top k| / |relevant items|`。数据集得分则是在既定规则下对各 query 得分进行汇总。
- **为什么重要：** 它告诉你某个 retrieval 阶段是否为下游 generation 或 reranking 提供了足够多的相关候选项。
- **实践建议：** 定义相关性判定、k、聚合方式，以及对没有任何已判定相关项的 query 的处理策略，然后重点检查那些一条相关证据都没召回的 query。
- **常见误区：** 高 Recall@K 并不意味着首条结果足够好、排序足够合理，或最终答案已经 grounded。对于没有相关项的 query，由于分母为零，必须显式规定是排除还是赋值。
- **相关术语：** Precision & Recall, Eval Set, Reranker, Approximate Nearest Neighbor (ANN)
- **来源：** [BEIR](https://openreview.net/forum?id=wCu6T5xFjeJ)

### Reciprocal Rank Fusion (RRF)
- **类别：** 检索与生成
- **实际含义：** 一种 rank 融合方法，通过累加多个结果列表中各条目的贡献值来进行合并，而该贡献值会随着条目在各列表中的排名下降而递减。
- **为什么重要：** 它能融合 lexical、dense 或多 query 排名，而不要求它们的原始分数处在同一量纲上。
- **实践建议：** 先检索出彼此独立的候选列表，再基于稳定的文档标识去重，应用一个带版本控制的融合常数，并将结果与每个单独的 retriever 分别对比评估。
- **常见误区：** RRF 融合的是 rank，而不是 embeddings 或 relevance scores；对于所有输入列表里都不存在的条目，它也无能为力。
- **相关术语：** Hybrid Retrieval, BM25, Dense Retrieval, Reranker
- **来源：** [Reciprocal Rank Fusion Outperforms Condorcet and Individual Rank Learning Methods](https://dl.acm.org/doi/10.1145/1571941.1572114)

### Red Teaming
- **类别：** 安全与治理
- **实际含义：** 一种结构化的对抗测试流程，由经授权的测试者依据已记录的目标、威胁假设、测试案例和证据来寻找失败点。
- **为什么重要：** 常规质量测试很少会探索系统在遭遇操纵、误用、目标冲突或有意绕过控制时会如何表现。
- **实践建议：** 从 threat model 推导攻击路径，在隔离环境中执行，记录可复现案例，按层修复问题，并把已确认的失败转化为 regression evals。
- **常见误区：** 一组 jailbreak prompts 并不能构成完整的 red-team 方案，red teaming 也无法证明不存在未知失败。
- **相关术语：** Threat Model, Guardrails, Prompt Injection, Eval Set
- **来源：** [Red Teaming Language Models with Language Models](https://arxiv.org/abs/2202.03286)

### Regression Test
- **类别：** AI 原生开发
- **实际含义：** 一种可重复执行的检查，用来保护那些已经被证明可正常工作的行为，尤其是在代码、prompt、模型、retrieval 或工具发生变化之后。
- **为什么重要：** AI 系统的改动可能会提升平均质量，但同时悄悄把之前修复过的问题重新带回来。
- **实践建议：** 把已经修复的 prompt-injection 事故转化为一个永久 eval case，并要求它在下一次部署前必须通过。
- **常见误区：** regression test 保护的是某个具体的预期行为；广义 benchmark 则是在更广的任务分布上估计性能。
- **延伸学习：** [评测驱动的代理开发](../phases/14-agent-engineering/30-eval-driven-agent-development/)
- **相关术语：** Eval Set, Verification Gate, Patch, Evaluation (Eval)

### ReLU
- **类别：** 数学与训练
- **常见说法：** 一种简单的 activation function。
- **实际含义：** Rectified Linear Unit，定义为 `f(x) = max(0, x)`。它计算代价低，并且正半轴不易饱和；但在负输入上梯度为零，可能导致神经元失活。
- **相关术语：** Activation Function, Gradient, CNN (Convolutional Neural Network)

### Repository Instructions
- **类别：** AI 原生开发
- **实际含义：** 一种受版本控制的说明，告诉 coding agents 仓库如何组织、适用哪些命令与约定、需要遵守哪些边界，以及如何验证工作结果。
- **为什么重要：** 它把反复口耳相传的经验知识转化为伴随代码存在的本地上下文，并且可以按子项目分别定义。
- **实践建议：** 在仓库根目录放置 `AGENTS.md`，为子目录补充更窄范围的说明文件，并写清楚构建、测试、生成文件、安全和贡献规则。
- **常见误区：** Repository instructions 是对源码和人工文档的补充；它们不会覆盖用户当前请求，也不能保证智能体一定会正确遵守。
- **相关术语：** Repository Map, Scope Contract, Coding Agent, Progressive Disclosure
- **来源：** [AGENTS.md specification](https://agents.md/)

### Repository Map
- **类别：** AI 原生开发
- **实际含义：** 一种紧凑且持续维护的仓库说明，涵盖重要目录、职责边界、入口点、构建命令、测试、生成文件以及本地说明。
- **为什么重要：** 它能帮助 coding agent 在加载大文件或误改错误子系统之前，先找到正确的证据和入口。
- **实践建议：** 先根据目录树和清单生成索引，再补充关于模块边界和验证命令的权威说明。
- **常见误区：** 原始文件树只能展示名字；repository map 会解释哪些路径真正重要，以及它们与当前任务的关系。
- **延伸学习：** [仓库记忆与状态](../phases/14-agent-engineering/34-repo-memory-and-state/)
- **相关术语：** Coding Agent, Progressive Disclosure, Scope Contract, 上下文工程

### Reproducible Build
- **类别：** AI 原生开发
- **实际含义：** 一种构建过程：其声明的源码、环境和指令都可以被独立重跑，并产出逐位一致的指定工件。
- **为什么重要：** 它让工件不再只能依赖最初产出它的机器或智能体来验证，同时还能暴露隐藏的构建输入。
- **实践建议：** 固定 toolchain 和依赖，去除时间戳与不稳定排序，记录环境信息，然后比较独立重建后的 artifact digest。
- **常见误区：** 一个构建连续成功两次，只能说明它具备可重复的证据；而 reproducibility 还要求满足所声明的独立条件，并得到完全相同的输出。
- **相关术语：** Repository Instructions, Verification Gate, Provenance Attestation, Software Bill of Materials (SBOM)
- **来源：** [Reproducible Builds definition](https://reproducible-builds.org/docs/definition/)

### Reranker
- **类别：** 检索与生成
- **实际含义：** 一种第二阶段模型或评分函数，通过更丰富的 query 与候选项比较方式，对一个较小的候选集重新排序。
- **为什么重要：** 快速的一阶段 retrieval 用来尽量覆盖候选项，而 reranking 则能改善最终进入有限 context window 的证据质量。
- **实践建议：** 先用 hybrid search 检索出 50 个候选，再用 cross-encoder 为每个 query-document 对打分，最后把前 5 个有支持性的 chunk 送入 generation。
- **常见误区：** reranker 并不会搜索整个语料库；它只会重排 retrieval 已经找出来的候选项。
- **相关术语：** Hybrid Retrieval, Semantic Search, RAG (Retrieval-Augmented Generation)

### Retry Budget
- **类别：** 可靠性与运维
- **实际含义：** 对重试流量施加的上限，通常以相对于原始请求的比例或某个时间窗口内的额度来表示，用于防止重试无限吞噬容量。
- **为什么重要：** 当某个依赖变慢或失败时，不受限制的重试会在系统最缺冗余容量的时候成倍放大负载。
- **实践建议：** 把重试与首次尝试分开计数，按服务和租户设置上限，遵守截止时间，采用带 jitter 的 backoff，并且对非瞬时性或非幂等失败停止重试。
- **常见误区：** retry budget 限制的是额外尝试次数；error budget 衡量的是在 SLO 下允许用户可见的不可靠性额度。
- **相关术语：** Retry with Backoff, Error Budget, Rate Limit, Admission Control
- **来源：** [Google SRE: Addressing Cascading Failures](https://sre.google/sre-book/addressing-cascading-failures/)

### Retry with Backoff
- **类别：** AI 原生开发
- **实际含义：** 对失败的瞬时性操作进行重试，并在每次之间使用越来越长的延迟，通常还会加入随机 jitter 和严格的重试上限。
- **为什么重要：** 立即且同步的重试会加剧故障、耗尽 rate limits，并重复产生副作用。
- **实践建议：** 对 provider timeout 采用有上限的指数退避重试，遵循服务器给出的重试建议，并为任何写操作复用同一个 idempotency key。
- **常见误区：** 不要重试永久性的校验错误或权限错误；对于非幂等操作，如果没有去重策略，也不要重试。
- **相关术语：** Idempotency, Rate Limit, Circuit Breaker, Backpressure

### Reviewer Agent
- **类别：** AI 原生开发
- **实际含义：** 一种被指派去审查另一个智能体产出物或决策的智能体，它依据明确标准返回问题发现或结论。
- **为什么重要：** 角色分离有助于发现遗漏，但前提是 reviewer 拿到的是独立证据和具体 rubric，否则效果有限。
- **实践建议：** 当一个 agent 产出 patch 后，把 diff、scope contract、repository rules 和测试输出交给独立 reviewer，并要求给出精确到行的发现。
- **常见误区：** 第二次模型调用并不会自动变得独立或正确。共享上下文、模型偏差和模糊标准，都可能把同样的错误再复现一遍。
- **延伸学习：** [Reviewer Agent](../phases/14-agent-engineering/39-reviewer-agent/)
- **相关术语：** Coding Agent, Verification Gate, Scope Contract, LLM-as-a-Judge

### RLHF (Reinforcement Learning from Human Feedback)
- **类别：** 数学与训练
- **常见说法：** 用人类偏好来训练模型。
- **实际含义：** 一类 pipeline：先利用人类反馈学习 reward 或 preference signal，再据此优化模型策略。具体实现各不相同，也不一定都采用同一种 reinforcement-learning algorithm。
- **常见误区：** RLHF 优化的是从收集到的反馈中学得的代理目标，它并不能保证对所有用户或所有情境都实现广义 alignment。
- **延伸学习：** [Reinforcement Learning from Human Feedback](../phases/10-llms-from-scratch/07-rlhf/)
- **来源：** [InstructGPT paper](https://arxiv.org/abs/2203.02155)
- **相关术语：** DPO (Direct Preference Optimization), SFT (Supervised Fine-Tuning), Alignment

### Rollback
- **类别：** 可靠性与运维
- **实际含义：** 当当前发布违反运维、质量或安全标准时，将系统恢复到先前已知状态的部署版本或配置。
- **为什么重要：** 即便经过上线前评估，agent 和模型的改动仍可能在生产环境中失败，因此恢复路径必须在 rollout 之前就设计好。
- **实践建议：** 保留带版本的工件和配置，定义 rollback 触发条件，演练相关命令及其对数据的影响，并在恢复后验证服务健康状况。
- **常见误区：** 代码回滚并不会自动撤销数据库迁移、外部副作用、缓存输出，或错误版本已经写入的数据。
- **相关术语：** Canary Release, Checkpoint, Regression Test, Durable Execution
- **来源：** [Kubernetes Deployments: Rolling Back](https://kubernetes.io/docs/concepts/workloads/controllers/deployment/#rolling-back-a-deployment)

### ROUGE
- **类别：** 评估与安全
- **常见说法：** 一种常用于摘要任务的参考文本重叠指标。
- **实际含义：** 一类指标族，用 n-gram overlap、longest common subsequence 等单位来比较生成文本与参考文本。
- **常见误区：** 表层重叠可能错过语义等价的情况，也可能奖励照抄措辞，却无法证明事实质量。
- **相关术语：** Evaluation (Eval), Precision & Recall, LLM-as-a-Judge

## S

### Sandbox
- **类别：** 智能体与工具
- **实际含义：** 一种隔离执行环境，用来限制智能体访问文件、进程、网络目标、凭证和宿主资源的能力。
- **为什么重要：** 生成出的代码和工具调用可能出错，也可能带有恶意。隔离能限制其影响范围，并让一次性验证变得可行。
- **实践建议：** 在一个临时容器里运行测试，底层环境只读、工作区可写但范围受限、不含生产密钥，并设置明确的网络 allowlist。
- **常见误区：** sandbox 能降低影响范围，但并不能证明其中运行的代码就是正确或无害的。
- **延伸学习：** [Production Agent Runtimes](../phases/14-agent-engineering/29-production-runtimes/)
- **相关术语：** Least Privilege, Approval Gate, Coding Agent, Guardrails

### Saturation
- **类别：** 可靠性与运维
- **实际含义：** 受限资源或服务耗尽其容量的程度，其中也包括那些无法及时开始处理的排队工作。
- **为什么重要：** 单看 utilization 也许还算正常，但内存、加速卡槽位、队列深度或下游 quota 可能已经开始限制有效吞吐。
- **实践建议：** 识别每个关键资源，测量活跃与等待中的工作量，把 saturation 与 tail latency 和错误关联起来，并在队列进入不稳定增长区间之前发出告警。
- **常见误区：** Saturation 不是一个通用百分比。真正的瓶颈资源及其排队行为取决于工作负载和系统架构。
- **相关术语：** Observability, Autoscaling, Backpressure, Tail Latency
- **来源：** [Google SRE: Monitoring Distributed Systems](https://sre.google/sre-book/monitoring-distributed-systems/)

### Scope Contract
- **类别：** AI 原生开发
- **实际含义：** 一种具体约定，用来定义任务目标、允许和禁止触碰的范围、预期产出物、验证要求以及停止条件。
- **为什么重要：** 它能防止智能体把一个小修小补扩展成无法审查的重构，也能防止其在没有证据的情况下声称任务完成。
- **实践建议：** 明确说明只有 parser 模块及其测试可以修改，公共 API 必须保持兼容，并且指定的测试套件必须通过。
- **常见误区：** 任务描述只说明你想要什么；scope contract 还会定义边界与证明标准。
- **延伸学习：** [Scope Contracts](../phases/14-agent-engineering/36-scope-contracts/)
- **相关术语：** Coding Agent, Patch, Verification Gate, Handoff

### Self-Attention
- **类别：** 模型与推理
- **常见说法：** token 决定哪些其他 token 更重要。
- **实际含义：** 一种 attention，其中 queries、keys 和 values 都来自同一序列表示。缩放后的相似度分数会被归一化，并在 causal、padding、local 或其他 mask 的约束下用于加权组合 values。
- **为什么重要：** 它能构建对上下文敏感的 token 表示，但允许的 attention 模式取决于具体架构。
- **常见误区：** 并不是每个 token 在任何时候都能关注所有其他 token。Causal 和 sparse 模型会刻意限制这些连接。
- **延伸学习：** [从零实现自注意力](../phases/07-transformers-deep-dive/02-self-attention-from-scratch/)
- **相关术语：** Attention, Transformer, Context Window

### Semantic Cache
- **类别：** AI 原生开发
- **实际含义：** 一种缓存：当新的请求在选定表示和阈值下被判断为足够相似时，就复用之前的结果。
- **为什么重要：** 它能为重复意图降低延迟和成本，但一旦匹配错误，也可能返回过时或不适合当前用户的输出。
- **实践建议：** 按归一化后的意图缓存低风险 FAQ 答案，在 key 中加入租户和策略版本，并对个性化或时效性请求绕过缓存。
- **常见误区：** 语义相似并不保证两个请求拥有相同的正确答案。semantic cache 复用的是历史结果；prefix caching 复用的是精确 token 的 KV state；prompt caching 则遵循 provider 或应用层的资格规则。
- **相关术语：** Prompt Cache, Embedding, Cost per Successful Task, Grounding

### Semantic Search
- **类别：** 检索与生成
- **常见说法：** 按语义而不是精确词面进行搜索。
- **实际含义：** 一种 retrieval 方法：把 query 与候选项表示到 embedding space 中，再用向量相似度函数对候选项排序。
- **为什么重要：** 它能检索到改写表达和概念相关文本，但精确标识符和稀有字符串仍然可能需要 lexical search。
- **相关术语：** Embedding, Hybrid Retrieval, Vector Database, Reranker

### Separation of Duties
- **类别：** 安全与治理
- **实际含义：** 把彼此冲突的职责或权限拆分到独立角色上，使单一主体无法在没有另一方授权决策的情况下独自完成高风险操作。
- **为什么重要：** 一个被攻陷的账号或出错的 agent，不应同时具备提出、审批、执行并掩盖同一项重要变更的能力。
- **实践建议：** 将工件创建与发布审批分离，使用不同身份，保留两边决策到 audit log 中，并为紧急访问定义事后复审机制。
- **常见误区：** Separation of duties 关注的是相互制衡的权限，而不只是把工作分配给几个共享同一套凭证的人或 agents。
- **相关术语：** Approval Gate, Reviewer Agent, Audit Log, Least Privilege
- **来源：** [NIST SP 800-53 Rev. 5](https://csrc.nist.gov/pubs/sp/800/53/r5/upd1/final)

### Service Level Indicator (SLI)
- **类别：** 可靠性与运维
- **实际含义：** 在某个与用户相关的明确边界上，对服务行为进行量化测量的指标，例如成功请求比例或低于某阈值的延迟比例。
- **为什么重要：** 只有当被观测的行为、纳入统计的事件以及测量点都明确时，关于可靠性的讨论才真正具备可操作性。
- **实践建议：** 定义分子、分母、排除项、数据来源和聚合窗口，然后验证这个指标是否真的追踪到了用户实际感受到的结果。
- **常见误区：** SLI 是测量值；SLO 是在既定时间范围内施加在该测量值上的目标。
- **相关术语：** Service Level Objective (SLO), Availability, Tail Latency, Observability
- **来源：** [Google SRE：服务级目标](https://sre.google/sre-book/service-level-objectives/)

### Service Level Objective (SLO)
- **类别：** 可靠性与运维
- **实际含义：** 针对某个已定义人群和测量窗口，为 service-level indicator 设定的目标区间或阈值。
- **为什么重要：** 它把期望的用户结果转化为监控、容量、发布风险和事故决策的运行边界。
- **实践建议：** 选择用户真正关心的指标，根据产品需求而非当前表现设定目标，定义统计窗口和排除项，并配套 error-budget 策略。
- **常见误区：** SLO 是内部的可靠性目标；合同层面的 service-level agreement 可能包含补偿条款，并且采用不同定义。
- **延伸学习：** [Inference Metrics and Goodput](../phases/17-infrastructure-and-production/08-inference-metrics-goodput/)
- **相关术语：** Service Level Indicator (SLI), Error Budget, Availability, Goodput
- **来源：** [Google SRE：服务级目标](https://sre.google/sre-book/service-level-objectives/)

### SFT (Supervised Fine-Tuning)
- **类别：** 数学与训练
- **常见说法：** 基于示例输入和期望输出进行训练。
- **实际含义：** 在成对的输入与期望响应上对预训练模型进行 fine-tuning，使其学会训练分布中展示出来的行为。
- **常见误区：** SFT 能适配的行为远不止聊天场景，而示例质量决定了什么样的行为会被强化。
- **相关术语：** Fine-tuning, DPO (Direct Preference Optimization), RLHF (Reinforcement Learning from Human Feedback)

### Shadow Traffic
- **类别：** 可靠性与运维
- **实际含义：** 把线上真实请求流量复制一份发送到候选系统进行观察，而候选系统的响应不进入主用户响应路径。由于复制过去的请求仍会执行，因此它的副作用必须被隔离。
- **为什么重要：** 它让候选系统在尽量不影响用户的前提下承受真实输入形态和真实负载，从而暴露合成测试中看不到的问题。
- **实践建议：** 移除或 token 化敏感字段，把工具和依赖路由到 sandboxed 或 no-op 目标，在能力边界上阻断写操作，保留请求关联性，并避免 shadow 流量与用户流量争抢资源。
- **常见误区：** 把候选响应隔离在主路径之外，并不意味着执行过程就没有副作用。canary release 的不同之处在于，它会让候选系统为真实用户承接一部分受控流量。
- **延伸学习：** [Shadow, Canary, and Progressive Delivery](../phases/17-infrastructure-and-production/20-shadow-canary-progressive/)
- **相关术语：** Canary Release, Evaluation (Eval), Trace, Model Serving
- **来源：** [Istio Traffic Mirroring](https://istio.io/latest/docs/tasks/traffic-management/mirroring/)

### Shared Embedding Space
- **类别：** 多模态系统
- **实际含义：** 一个共享向量空间，在其中，不同模态的表示可以使用同一个相似度函数进行比较。
- **为什么重要：** 它使跨模态检索与匹配成为可能，例如用文本找图片，而不要求两者拥有相同的原始表示。
- **实践建议：** 有意识地构造配对和非配对负样本，在目标需要时对向量做归一化，同时评估两个检索方向，并检查不同子群体和语言上的表现。
- **常见误区：** 共享同样的向量维度，并不会自动形成共享语义空间。跨模态可比性必须由训练目标和训练数据来建立。
- **延伸学习：** [CLIP Contrastive Pretraining](../phases/12-multimodal-ai/02-clip-contrastive-pretraining/)
- **相关术语：** Embedding, Cosine Similarity, Modality Alignment, Semantic Search
- **来源：** [Learning Transferable Visual Models From Natural Language Supervision](https://proceedings.mlr.press/v139/radford21a.html)

### Skill Bundle
- **类别：** 智能体与工具
- **实际含义：** 一个完整、可安装的 skill 目录，包含 `SKILL.md` 以及工作流所需的所有 reference、script、asset、fixture 和配套文件。
- **为什么重要：** 如果只复制入口文件，可能会留下表面看似有效、实际上指向缺失资源的说明，或者丢失工作流依赖的确定性代码。
- **实践建议：** 把整棵目录作为一个整体安装，记录 hash 和源码版本，校验已安装副本，并在替换已有 bundle 前展示冲突情况。
- **常见误区：** `SKILL.md` 是入口点，但不一定就是完整工件。
- **延伸学习：** [Skill Evals, Packaging, and Portability](../phases/13-tools-and-protocols/27-skill-evals-packaging-and-portability/)
- **相关术语：** Agent Skill, Skill Catalog, Reproducible Build, Provenance Attestation
- **来源：** [Agent Skills 规范](https://agentskills.io/specification)

### Skill Catalog
- **类别：** 智能体与工具
- **实际含义：** 一种对模型可见的紧凑型可用技能清单，通常包含名称、描述、内部来源标识等路由元数据，而不是每个 skill 的完整内容。
- **为什么重要：** catalog 让智能体在无需把所有已安装包都载入工作上下文的情况下，发现相关流程。
- **实践建议：** 先校验包，再应用明确的重名策略，衡量序列化后的 catalog 预算，并为那些被截短、被省略或被遮蔽的条目保留诊断信息。
- **常见误区：** catalog 中存在某条目，只表示这个 skill 可以被发现；并不意味着其正文已激活，或其工具已被授权。
- **延伸学习：** [技能发现与渐进式披露](../phases/13-tools-and-protocols/24-skill-discovery-and-progressive-disclosure/)
- **相关术语：** Skill Discovery, Skill Invocation, Progressive Disclosure, Token Budget
- **来源：** [Agent Skills 规范](https://agentskills.io/specification)

### Skill Discovery
- **类别：** 智能体与工具
- **实际含义：** 一种运行时 pipeline：搜索已配置根路径，识别候选 skill 目录，校验其 package contract，附加 scope 与 provenance，解决冲突，并发布符合条件的 catalog entries。
- **为什么重要：** 确定性的发现流程能在模型开始路由之前，就把缺失、格式错误、被遮蔽或不安全的 package 诊断出来。
- **实践建议：** 明确搜索范围和重名处理策略，决定如何处理 symlink，拒绝资源逃逸，并记录每个候选项为何被接受或拒绝。
- **常见误区：** Skill discovery 不是对所有名为 `SKILL.md` 的文件做无限制递归搜索；安装位置和优先级属于运行时策略的一部分。
- **延伸学习：** [技能发现与渐进式披露](../phases/13-tools-and-protocols/24-skill-discovery-and-progressive-disclosure/)
- **相关术语：** Skill Catalog, Skill Bundle, Progressive Disclosure, Trust Boundary
- **来源：** [Agent Skills client implementation guide](https://agentskills.io/client-implementation/adding-skills-support)

### Skill Invocation
- **类别：** 智能体与工具
- **实际含义：** 一种由运行时介导的过程：合格的人、模型、应用或其他 skill 选择某个 skill，并使其指令进入当前工作上下文。
- **为什么重要：** 显式用户访问、隐式模型路由、激活、参数绑定、工具权限和实际执行，都是彼此独立且具有不同失败模式的决策。
- **实践建议：** 定义 actor policy，用正例和近似误触请求评估描述文案，记录被选中的 package 身份，并把宿主相关的 invocation 字段保存在经过测试的 adapter 中。
- **常见误区：** Invocation 会激活指令，但不会自动执行命令，也不会绕过审批与 sandbox 策略。
- **延伸学习：** [Skill Invocation and Routing](../phases/13-tools-and-protocols/25-skill-invocation-and-routing/)
- **相关术语：** Agent Skill, Skill Catalog, Approval Gate, Sandbox
- **来源：** [Evaluating Agent Skills](https://agentskills.io/skill-creation/evaluating-skills)

### Softmax
- **类别：** 数学与训练
- **常见说法：** 一种把 logits 转成归一化正值的函数。
- **实际含义：** 一个定义为 `softmax(x_i) = exp(x_i) / sum(exp(x_j))` 的函数，实际实现时会加入数值稳定化处理。它的输出都为正且总和为一，因此可以参数化 categorical distribution。
- **常见误区：** Softmax 的输出值并不会自动成为关于现实世界正确性的校准概率。
- **相关术语：** Temperature, Cross-Entropy, Attention

### Software Bill of Materials (SBOM)
- **类别：** 安全与治理
- **别名：** SBOM
- **实际含义：** 一种结构化清单，用来记录与某个产品或工件相关的软件组件及其关系，通常包含版本、供应方、许可证和标识符等信息。
- **为什么重要：** 当软件发生变化或暴露漏洞时，你需要一份组件清单，才能评估受影响依赖、许可证义务以及供应链暴露面。
- **实践建议：** 在可信构建过程中生成 SBOM，将其绑定到发布工件，在策略检查中验证它，并在依赖或打包方式变化时及时更新。
- **常见误区：** SBOM 是一份清单，而不是安全性、许可证合规性或组件真实存在性的证明；除非其生成过程和 provenance 本身可信。
- **相关术语：** Provenance Attestation, Reproducible Build, Data Provenance, Audit Log
- **来源：** [SPDX 3.0.1 specification](https://spdx.github.io/spdx-spec/v3.0/)

### Speculative Decoding
- **类别：** 模型与推理
- **实际含义：** 一种推理方法：由更便宜的 draft 过程先提出若干 token，再由目标模型并行地为这些 draft 位置打分。在 exact sampling 变体中，接受与修正规则会保持目标模型的输出分布不变。
- **为什么重要：** 当 draft token 被接受时，它能减少目标模型串行 decoding 的工作量，而且无需改动目标模型已经训练好的权重。
- **实践建议：** 在真实 prompt 上测量接受率和端到端延迟，把 draft model 的开销也算进去，并验证实现确实保持了目标的 decoding distribution。
- **常见误区：** Speculative decoding 不是普通的模型路由，也不是未经验证的自动补全。exact 变体通过接受与修正规则保持目标分布；approximate 变体则可能用放弃这种保证来换取速度。
- **相关术语：** Autoregressive, KV Cache, Decoding Strategy, Tokens per Second (TPS)
- **来源：** [Fast Inference from Transformers via Speculative Decoding](https://proceedings.mlr.press/v202/leviathan23a.html)

### Stateless MCP
- **类别：** 智能体与工具
- **实际含义：** MCP 2026-07-28 的一种请求模型：每个请求都在 `params._meta` 中携带协议版本和客户端能力，结果则带有显式 `resultType`；协议状态不再依赖 initialization handshake、连接或 `Mcp-Session-Id` 来建立。
- **为什么重要：** 任何 worker 都可以仅根据请求内容和授权上下文来验证并处理请求，这避免了隐式连接亲和，也让水平路由更容易被推理和维护。
- **实践建议：** 实现 `server/discover`，在每次调用时重建请求元数据，对照 JSON-RPC body 验证传输头，并在需要连续性时，把 server 签发的应用句柄作为普通工具参数传递下去。
- **常见误区：** Stateless MCP 去掉的是协议层 session，而不是应用状态、传输连接、streaming responses、tasks 或显式 handles。
- **延伸学习：** [MCP Fundamentals](../phases/13-tools-and-protocols/06-mcp-fundamentals/)
- **相关术语：** MCP (Model Context Protocol), Multi Round-Trip Request (MRTR), Tool Contract, Idempotency
- **来源：** [MCP 2026-07-28 key changes](https://modelcontextprotocol.io/specification/2026-07-28/changelog); [MCP Streamable HTTP](https://modelcontextprotocol.io/specification/2026-07-28/basic/transports/streamable-http)

### Stochastic Gradient Descent (SGD)
- **类别：** 数学与训练
- **别名：** SGD
- **实际含义：** 一类 optimizer：用从采样样本或 minibatch 估计出的梯度来更新参数，而不是依赖完整训练数据集上的梯度。
- **为什么重要：** 它是理解 gradient noise、momentum、batch scaling 以及现代训练中各类自适应 optimizer 的基础。
- **实践建议：** 记录 batch 采样方式、learning rate、是否使用 momentum 以及调度策略，再在相同更新预算或 token 预算下比较验证表现。
- **常见误区：** 在当前实践里，SGD 通常指的是 minibatch SGD，而且它有效的 learning rate 并不遵循某一条通用的 batch scaling 规律。
- **相关术语：** Gradient Descent, Batch Size, Learning Rate, Optimizer
- **来源：** [Optimization Methods for Large-Scale Machine Learning](https://arxiv.org/abs/1606.04838); [Accurate, Large Minibatch SGD](https://arxiv.org/abs/1706.02677)

### Stop Sequence
- **类别：** 模型与推理
- **实际含义：** 一种由应用指定的 token 或文本模式，当解码系统遇到它时就会停止生成。
- **为什么重要：** Stop sequence 可以为输出协议和多段式生成设定边界，而不必等待模型在语义上自己判断“已经结束”。
- **实践建议：** 选择无歧义的分隔符，测试 tokenization 与部分 streaming 匹配情况，并且仍然要执行输出长度和 schema validation。
- **常见误区：** stop sequence 是一种机械性的解码停止条件，而不是答案完整或智能体目标已完成的证明。
- **相关术语：** Decoding Strategy, Structured Output, Token, Termination Condition
- **来源：** [Transformers text-generation documentation](https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)

### Streaming
- **类别：** 模型与推理
- **常见说法：** 边生成边展示输出。
- **实际含义：** 在完整结果尚未准备好之前，就持续发送增量响应事件。具体 stream 内容可能包括 token 文本、structured deltas、tool-call arguments、usage metadata 或状态事件，这取决于 API 设计。
- **为什么重要：** 它能提升用户感知到的响应速度，但并不会减少模型产出完整答案的真实耗时。
- **常见误区：** 网络传输方式、事件形态和 chunk 边界都具有 provider 特性，不能保证恰好与单词或 token 对齐。
- **延伸学习：** [Production LLM Application](../phases/11-llm-engineering/13-production-app/)
- **相关术语：** Time to First Token (TTFT), Autoregressive, Observability

### Structured Output
- **类别：** 智能体与工具
- **实际含义：** 对模型输出施加机器可读 schema 的约束或验证，使应用代码无需解析自由文本，就能直接消费结构化字段。
- **为什么重要：** 它能减少模型到软件边界处的格式歧义，并支持字段级验证与重试。
- **实践建议：** 要求 incident triage 结果必须包含允许的 severity enum、evidence array 和可空的 escalation reason；凡是不符合 schema 的响应，一律拒绝。
- **常见误区：** 即便输出通过了 schema 校验，里面的值仍可能是错的。结构化不等于事实验证。
- **延伸学习：** [Structured Outputs](../phases/11-llm-engineering/03-structured-outputs/)
- **相关术语：** Function Calling, Tool Contract, Verification Gate

### Swarm
- **类别：** 智能体与工具
- **常见说法：** 多个智能体协作，但没有固定的单一控制者。
- **实际含义：** 一种松耦合的 multi-agent 模式：局部 agent 的决策和消息交换共同产生系统级行为。这个术语使用并不统一，因此必须明确说明具体拓扑、状态归属以及终止规则。
- **常见误区：** 光有多个具名 agents，并不保证真的形成有效分工或涌现式协作。
- **相关术语：** Agent, Reviewer Agent, Handoff, Agent State

### System Prompt
- **类别：** 提示与上下文
- **常见说法：** 由开发者控制、用于约束模型交互的指令。
- **实际含义：** 一种由 provider 定义的 instruction message 或配置项，由应用提供，用于在该 provider 的指令层级中建立行为规范与约束。
- **为什么重要：** system instructions 可以引导模型行为，但并不能保证始终保密，也不应被视为安全边界。
- **常见误区：** 不同 API 在优先级规则、消息角色、持久性和可见性上都可能不同，应以当前 provider contract 为准。
- **延伸学习：** [Instructions as Executable Constraints](../phases/14-agent-engineering/33-instructions-as-executable-constraints/)
- **相关术语：** Prompt Engineering, Prompt Injection, 上下文工程, Guardrails

## T

### Tail Latency
- **类别：** 可靠性与运维
- **实际含义：** 请求中最慢那一部分所经历的延迟，通常在给定工作负载和时间窗口下用高百分位数来概括。
- **为什么重要：** 平均值看起来可能很健康，但由于排队、竞争、重试或请求成本差异，一部分用户实际等待时间会明显更长。
- **实践建议：** 按路由和工作负载报告多个百分位数，根据书面规则把 timeout 记为删失观测或失败观测，并沿依赖链追踪慢请求。
- **常见误区：** Tail latency 不是单个最慢请求；如果没有百分位、统计人群和测量边界，它本身没有意义。
- **延伸学习：** [Inference Metrics and Goodput](../phases/17-infrastructure-and-production/08-inference-metrics-goodput/)
- **相关术语：** Time to First Token (TTFT), Time per Output Token (TPOT), Saturation, Goodput
- **来源：** [The Tail at Scale](https://research.google/pubs/the-tail-at-scale/)

### Temperature
- **类别：** 模型与推理
- **常见说法：** 一个“创造力”设置。
- **实际含义：** 一种 decoding 参数，会在形成概率分布之前对 logits 做缩放。更高的正值通常会让分布更平坦；更低的正值则会让分布更尖锐。
- **为什么重要：** Temperature 改变的是采样行为，而不是模型的知识储备或事实正确性。
- **常见误区：** temperature 设为零时，通常会实现为 greedy decoding，但具体行为和是否确定性，仍取决于 provider、sampler、seed 支持以及 serving system。
- **相关术语：** Softmax, Autoregressive, Token

### Tensor
- **类别：** 数据与表征
- **常见说法：** 一种用于数值计算的多维数组。
- **实际含义：** 一种带有 shape、data type 和 device placement 的 typed array，框架用它来表示输入、参数、activations 和 gradients。Automatic-differentiation metadata 取决于框架和具体操作，并不是每个 tensor 天然自带的属性。
- **相关术语：** Autograd, Parameter, Mixed Precision

### Tensor Parallelism
- **类别：** 基础设施与服务
- **实际含义：** 把模型单层内部的 tensor 运算切分到多个设备上执行，并在该层计算过程中通过 collective communication 合并部分结果。
- **为什么重要：** 它让单层可以同时利用多个设备的内存和算力，但如果互连或切分方式不合适，频繁通信就会反过来主导总开销。
- **实践建议：** 让切分维度匹配模型形状，基准测试 collective 流量，把各 rank 放在高速互连上，并把 sharding 布局与 checkpoints 和 serving 配置一起记录下来。
- **常见误区：** Tensor parallelism 切分的是层内计算；Pipeline parallelism 则是把不同的层组放到不同设备上。
- **延伸学习：** [Scaling and Distributed Training](../phases/10-llms-from-scratch/05-scaling-distributed/)
- **相关术语：** Tensor, Pipeline Parallelism, Expert Parallelism, Parameter
- **来源：** [Megatron-LM](https://arxiv.org/abs/1909.08053)

### Termination Condition
- **类别：** 智能体与工具
- **实际含义：** 一种显式规则：当智能体运行成功、失败、耗尽预算、触及安全边界或需要升级处理时，用来结束或暂停运行。
- **为什么重要：** 没有 termination condition，智能体就可能陷入循环、重复产生副作用、浪费预算，或在目标尚未满足时误报完成。
- **实践建议：** 在启动循环之前，先定义成功证据、最大步数与成本、不可重试错误以及升级状态。
- **常见误区：** stop sequence 结束的是文本生成；termination condition 决定的是任务或工作流是否该停止。
- **相关术语：** Agent Harness, Token Budget, Verification Gate, Stop Sequence
- **来源：** [AutoGen](https://arxiv.org/abs/2308.08155)

### Test Oracle
- **类别：** AI 原生开发
- **实际含义：** 一种机制、规范、参考实现、不变量或人工判断，用来决定观察到的程序行为是否正确。
- **为什么重要：** 仅仅生成测试输入还不够；自动化验证还需要一个独立依据来判定每个结果是否正确。
- **实践建议：** 优先使用可执行不变量、reference implementation、schema 和确定性的期望输出；对于仍需人工判断的部分，要明确记录。
- **常见误区：** 写出代码的那个模型，不应仅仅因为被问了一句“你自己写得对吗”，就被当作独立 oracle。
- **相关术语：** Regression Test, Verification Gate, Eval Set, Human-in-the-Loop (HITL)
- **来源：** [The Oracle Problem in Software Testing](https://www.computer.org/csdl/journal/ts/2015/05/06963470/13rRUx0geBw)

### Threat Model
- **类别：** 安全与治理
- **实际含义：** 一份成文记录，用来说明受保护资产、trust boundaries、潜在对手、假定能力、攻击路径、影响范围以及计划中的控制措施。
- **为什么重要：** 如果不先说清楚这些控制要保护什么、对抗谁，以及基于哪些假设，就无法评价它们是否合理。
- **实践建议：** 梳理数据和权限在模型、retrieval、工具、用户与外部服务之间的流动，然后把可信的滥用路径转化为 red-team 案例与缓解措施。
- **常见误区：** threat model 用于给可信风险排优先级；它不是一份能证明系统安全或预测所有未来攻击的检查表。
- **相关术语：** Least Privilege, Prompt Injection, Sandbox, Red Teaming
- **来源：** [NIST SP 800-154](https://csrc.nist.gov/pubs/sp/800/154/ipd); [NIST Generative AI Profile](https://nvlpubs.nist.gov/nistpubs/ai/nist.ai.600-1.pdf)

### Time per Output Token (TPOT)
- **类别：** 基础设施与服务
- **实际含义：** 对于一个输出 token 数为 `N > 1` 的请求，其首 token 之后的平均间隔定义为 `(t_N - t_1) / (N - 1)`。系统级分布则是在此基础上对各请求平均值进行聚合。
- **为什么重要：** 用户可能很快就收到第一个 token，但之后的内容却慢慢流出来，因此仅看启动延迟并不能完整描述生成响应性。
- **实践建议：** 为每个请求单独计算 TPOT，按输出长度和并发度报告跨请求的百分位数，并避免把所有 token 间隔混在一起统计，也不要直接比较 tokenizer 或测量边界不同的系统。
- **常见误区：** TPOT 是按请求计算的平均值；单个 inter-token latency 指的是相邻两个 token 之间的一次间隔，而 time to first token 还包括输出开始前的等待时间。
- **延伸学习：** [Inference Metrics and Goodput](../phases/17-infrastructure-and-production/08-inference-metrics-goodput/)
- **相关术语：** Decode Phase, Time to First Token (TTFT), Streaming, Goodput
- **来源：** [DistServe](https://arxiv.org/abs/2401.09670)

### Time to First Token (TTFT)
- **类别：** 模型与推理
- **别名：** TTFT
- **实际含义：** 在既定测量边界下，从提交生成请求到客户端收到第一个输出 token 或内容事件所经历的时间。
- **为什么重要：** TTFT 会显著影响用户感知到的响应速度，也能暴露排队、prompt 处理、缓存或网络延迟问题。
- **实践建议：** 按模型、prompt 长度、地域和缓存状态记录客户端侧 TTFT，并把它与总完成时间区分开来。
- **常见误区：** TTFT 不是 tokens per second。前者衡量启动延迟，后者衡量输出开始后的生成吞吐。
- **相关术语：** Streaming, Prompt Cache, Observability, Token Budget

### Token
- **类别：** 数据与表征
- **常见说法：** 模型输入或输出中一个接近“词大小”的片段。
- **实际含义：** 一种由特定模型 tokenizer 从文本、字节、图像、音频或其他输入表示生成的整数标识。一个 token 可以是整个单词、词的一部分、标点、空白、字节序列，或特殊控制符号。
- **常见误区：** 字符与 token 的换算比例会因语言、内容和 tokenizer 而异，因此应使用目标模型的 tokenizer 或 provider 工具来统计。
- **延伸学习：** [Tokenizers](../phases/10-llms-from-scratch/01-tokenizers/)
- **相关术语：** Token Budget, Context Window, Autoregressive

### Token Budget
- **类别：** 提示与上下文
- **实际含义：** 一种对 token 容量的显式分配方案，把容量划分给 instructions、evidence、history、tool results、reasoning 或工作空间以及输出。
- **为什么重要：** 每一个被纳入的 token 都会争抢上下文容量、延迟预算和成本预算。做预算规划会迫使你优先保留高价值证据。
- **实践建议：** 预留输出容量，限制检索 chunk 数量，把旧的工具结果总结进状态里，并在达到模型上限之前停止或压缩上下文。
- **常见误区：** token budget 是一种规划约束，不等同于模型的最大 context window。
- **延伸学习：** [上下文工程](../phases/11-llm-engineering/05-context-engineering/)
- **相关术语：** Context Window, 上下文工程, Progressive Disclosure, Cost per Successful Task

### Tokenization
- **类别：** 数据与表征
- **实际含义：** 把输入表示转换为某个特定模型或 tokenizer 能接受的有序 token 标识序列。
- **为什么重要：** Tokenization 决定了序列长度、词表边界、成本计量、截断行为，以及文本或代码在 embedding 之前如何被表示。
- **实践建议：** 使用目标模型对应的精确 tokenizer，对 tokenizer 与相关工件一起做版本管理，并测试多语言文本、代码、空白字符和特殊 tokens。
- **常见误区：** Tokenization 并不总是“按词切分”；两个模型也可能对同一输入给出不同的 token 数量和 ID。
- **相关术语：** Token, Vocabulary, Byte Pair Encoding (BPE), Embedding
- **来源：** [使用子词单元的稀有词神经机器翻译](https://arxiv.org/abs/1508.07909)

### Tokens per Second (TPS)
- **类别：** 基础设施与服务
- **别名：** TPS, output token throughput
- **实际含义：** 一种吞吐指标，用来报告某个 serving system 在给定范围与工作负载下，每单位时间产出多少 output tokens。
- **为什么重要：** 它补充了启动延迟指标，展示输出开始后生成推进得有多快，以及服务在负载下的表现。
- **实践建议：** 说明 TPS 是按请求统计还是聚合统计，排除或单独标识 prefill，并报告 batch、大并发、序列长度、硬件和百分位延迟。
- **常见误区：** TPS 不能在不同 tokenizer、工作负载、质量设置或测量边界之间直接比较。
- **相关术语：** Time to First Token (TTFT), Streaming, Prefill, Observability
- **来源：** [Sarathi-Serve](https://www.usenix.org/system/files/osdi24-agrawal.pdf)

### Tool Contract
- **类别：** 智能体与工具
- **实际含义：** 围绕某个工具边界形成的完整约定，涵盖目的、typed inputs、outputs、校验、权限、副作用、错误、超时、幂等性，以及返回给调用方的证据。
- **为什么重要：** schema 告诉模型有哪些字段；contract 告诉整个系统这个工具在什么条件下才算安全，以及失败必须如何处理。
- **实践建议：** 定义一个文件写入工具时，应明确允许的根目录、预期基线版本、最大尺寸、dry-run 模式、显式冲突错误，以及返回的 patch hash。
- **常见误区：** JSON Schema 只是 tool contract 的一部分，而不是全部。
- **延伸学习：** [Tool Use and Function Calling](../phases/14-agent-engineering/06-tool-use-and-function-calling/)
- **相关术语：** Function Calling, Structured Output, Least Privilege, Idempotency

### Top-k Sampling
- **类别：** 模型与推理
- **实际含义：** 一种 decoding 方法：把下一 token 的分布限制在得分最高的 k 个候选上，对它们的概率重新归一化，再从中采样。
- **为什么重要：** 它在保留固定候选数上限的同时，去掉了采样分布中那条很长的低概率尾部。
- **实践建议：** 把 k 与 temperature、top-p 和 stop 设置一起评估，并把完整采样器配置与生成结果一起记录下来。
- **常见误区：** Top-k 使用固定候选数；top-p 使用概率质量阈值，因此候选数会随着每一步而变化。
- **相关术语：** Nucleus Sampling (Top-p), Temperature, Decoding Strategy, Logits
- **来源：** [神经文本退化的奇特现象](https://arxiv.org/abs/1904.09751)

### Trace
- **类别：** AI 原生开发
- **实际含义：** 围绕单个请求或任务形成的一条关联记录，覆盖 model calls、retrieval、tools、状态迁移、重试、审批与评估。
- **为什么重要：** 它能帮助你重建在多步骤工作流中，时间、成本和失败究竟是从哪里进入系统的。
- **实践建议：** 在 agent harness 中传递同一个 trace identifier，并为每次模型调用和工具操作附加经过脱敏的 spans。
- **常见误区：** trace 应记录运维证据，而不是暴露隐藏的模型推理、密钥或未经脱敏的敏感内容。
- **延伸学习：** [OpenTelemetry GenAI Conventions](../phases/14-agent-engineering/23-otel-genai-conventions/)
- **来源：** [OpenTelemetry traces](https://opentelemetry.io/docs/concepts/signals/traces/)
- **相关术语：** Observability, Agent State, Time to First Token (TTFT), Evaluation (Eval)

### Transfer Learning
- **类别：** 数学与训练
- **常见说法：** 把预训练模型复用于新任务。
- **实际含义：** 从一个数据分布或训练目标上学到的表示或参数出发，再将其适配到另一个任务。哪些部分可迁移、如何更新，取决于架构和任务本身。
- **常见误区：** Transfer 并不只发生在后层；当源任务与目标任务差异很大时，也不能保证迁移一定成功。
- **相关术语：** Fine-tuning, Feature, SFT (Supervised Fine-Tuning)

### Transformer
- **类别：** 模型与推理
- **常见说法：** 许多现代语言模型背后的基础架构。
- **实际含义：** 一种由 attention、位置信息、feed-forward sublayers、residual connections 和 normalization 组成的神经网络架构。Encoder、decoder 和 encoder-decoder 变体会使用不同的 mask 和信息流方式。
- **为什么重要：** 训练阶段可以并行处理许多序列位置，而 autoregressive generation 仍然需要逐步产出输出。
- **常见误区：** Self-attention 并不意味着每个 transformer 都支持毫无限制的全连接 attention。
- **延伸学习：** [Build a Full Transformer](../phases/07-transformers-deep-dive/05-full-transformer/)
- **来源：** [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- **相关术语：** Attention, Self-Attention, Encoder, Decoder

### Trust Boundary
- **类别：** 安全与治理
- **实际含义：** 一种接口：数据、指令、身份或权限在不同信任假设下运行的组件或主体之间跨越时，就发生在这里。
- **为什么重要：** 边界跨越之处，正是系统必须完成身份认证、数据验证、权限约束，并决定哪些声明可以影响行为的地方。
- **实践建议：** 围绕用户、model context、retrieval sources、tools、networks 和 data stores 画出边界，并为每一次跨越指定验证与授权规则。
- **常见误区：** 网络边界只是 trust boundary 的一种。不受信的文档文本进入高权限 agent 上下文时，同样是在跨越 trust boundary。
- **延伸学习：** [Jailbreak Taxonomy](../phases/19-capstone-projects/82-jailbreak-taxonomy/)
- **相关术语：** Threat Model, Least Privilege, Sandbox, Indirect Prompt Injection
- **来源：** [Microsoft Learn: Trust Boundary, the Trust Zone Change Element](https://learn.microsoft.com/en-us/training/modules/tm-create-a-threat-model-using-foundational-data-flow-diagram-elements/6-trust-boundary-the-trust-zone-change-element); [OWASP Threat Modeling](https://owasp.org/www-community/Threat_Modeling)

## U

### Underfitting
- **类别：** 数学与训练
- **常见说法：** 模型无法很好地拟合训练任务。
- **实际含义：** 模型或训练设置缺乏足够的有效容量、优化能力、特征或训练信号，因此无法捕捉训练数据中的有用模式。
- **实践建议：** 先排查数据和优化问题，再考虑延长训练时间、调整特征、减弱过度正则化，或增加合适的容量。
- **相关术语：** Overfitting, Loss Function, Hyperparameter

## V

### VAE (Variational Autoencoder)
- **类别：** 模型与推理
- **常见说法：** 一种概率生成式自编码器。
- **实际含义：** 一种带潜变量的模型，通过重建目标和正则项进行训练，使近似后验保持接近选定先验。重参数化估计器允许梯度穿过随机的潜变量采样过程。
- **常见误区：** VAE 并不会强制所有潜变量分布都变成某个固定的高斯分布；具体的先验与近似后验都是建模选择。
- **来源：** [Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114)
- **相关术语：** Latent Space, Encoder, Decoder, Diffusion Model

### Vector Database
- **类别：** 检索与生成
- **常见说法：** 针对向量相似度搜索优化的数据库。
- **实际含义：** 一种存储与索引系统，支持在向量表征上执行最近邻查询，通常还具备元数据过滤、持久化和近似索引能力。
- **常见误区：** 向量数据库负责存储和搜索向量。它并不会生成高质量 embedding，也不保证检索结果一定相关。
- **相关术语：** Embedding, Semantic Search, Hybrid Retrieval

### Verification Gate
- **类别：** 评估与安全
- **实际含义：** 一种控制点，在既定证据满足正确性或质量标准之前阻止流程继续推进。
- **为什么重要：** 它会把模型“已完成”的声明转化为基于证据的判断。
- **实践建议：** 在 patch 成功应用、范围内测试通过、禁止修改的文件保持不变且所需产物存在之前，不允许编码任务结束。
- **常见误区：** Verification 检查的是证据是否满足标准。Approval 授予的是继续执行的权限，即使相关证据已经明确。
- **延伸学习：** [验证门](../phases/14-agent-engineering/38-verification-gates/)
- **相关术语：** Approval Gate, Regression Test, Scope Contract, Structured Output

### Vision-Language Model (VLM)
- **类别：** 多模态系统
- **实际含义：** 一种学习视觉表征与语言表征之间关系，或对二者进行联合处理的模型，可用于检索、描述、问答或 grounded generation 等任务。
- **为什么重要：** VLM 的表现取决于视觉编码器、语言组件、连接机制、训练数据和分辨率策略，而不是某个笼统的能力标签。
- **实践建议：** 要评估纯文本对照和纯视觉对照，改变图像分辨率与版式，在可能时要求给出证据定位，并按视觉能力与语言能力分别报告失败情况。
- **常见误区：** 能够接收图像并不代表模型能正确使用图像，而且 VLM 也不一定具备生成图像的能力。
- **延伸学习：** [Vision-Language Models](../phases/04-computer-vision/25-vision-language-models/)
- **相关术语：** Multimodal Model, Vision Transformer (ViT), Cross-Attention, Visual Grounding
- **来源：** [CLIP](https://arxiv.org/abs/2103.00020); [Flamingo](https://arxiv.org/abs/2204.14198)

### Vision Transformer (ViT)
- **类别：** 多模态系统
- **实际含义：** 一种视觉架构，将图像表示为带位置信息的 patch embedding 序列，并用 transformer encoder blocks 处理该序列。
- **为什么重要：** 它为视觉数据提供了序列模型接口，但性能与计算开销取决于 patch 大小、分辨率、预训练方式和归纳偏置。
- **实践建议：** 要让 patch 划分和归一化方式与训练阶段保持一致，考虑 position embedding 在新分辨率下的表现，并在目标数据集上与合适的视觉基线进行比较。
- **常见误区：** ViT 是一类架构，不是所有能接收图像的 transformer；它的 patches 也并不天然对应语义对象。
- **延伸学习：** [Vision Transformers](../phases/04-computer-vision/14-vision-transformers/)
- **相关术语：** Transformer, Patch Embedding, Self-Attention, Encoder
- **来源：** [An Image is Worth 16x16 Words](https://arxiv.org/abs/2010.11929)

### Visual Grounding
- **类别：** 多模态系统
- **实际含义：** 把语言表达与图像或视频中的空间证据对应起来，例如某个区域、目标、mask 或被跟踪的实体。
- **为什么重要：** 流畅的视觉回答也可能缺乏依据，而 grounding 会让所指对象可检查，并支持区域级评估。
- **实践建议：** 要求回答同时给出 box、mask 或时间片段，测试指代模糊和指代不存在的情况，并将定位准确性与语言正确性分开评分。
- **常见误区：** Visual grounding 解决的是被提及证据“在哪里”。通用图像描述可以描述场景，但不会为每个说法都做定位。
- **延伸学习：** [Cross-Attention Fusion](../phases/19-capstone-projects/61-cross-attention-fusion/)
- **相关术语：** Grounding, Vision-Language Model (VLM), Attention, Evaluation (Eval)
- **来源：** [MDETR](https://arxiv.org/abs/2104.12763)

### Vocabulary
- **类别：** 数据与表征
- **实际含义：** token 标识符与 tokenizer 可输出单位之间的有限映射，其中包括普通 token、字节级 token 和特殊控制 token。
- **为什么重要：** Vocabulary 的设计会影响序列长度、多语言覆盖、代码表示、embedding 大小，以及 tokenizer 与模型权重之间的兼容性。
- **实践建议：** 要让 vocabulary 及特殊 token 分配与模型一起版本化，测试 encode-decode 往返一致性，并且绝不要仅凭 token 名称看起来相似就替换 tokenizer。
- **常见误区：** 模型的 vocabulary 不是人类词语字典；其中很多条目其实是片段、字节、空白模式或控制符号。
- **相关术语：** Tokenization, Byte Pair Encoding (BPE), Token, Embedding
- **来源：** [使用子词单元的稀有词神经机器翻译](https://arxiv.org/abs/1508.07909)

## W

### Warmup
- **类别：** 数学与训练
- **实际含义：** 训练初期的一个阶段，在此期间学习率会从较小的值逐步升到主调度目标值。
- **为什么重要：** 训练早期的梯度和优化器统计量可能并不稳定，尤其是在大 batch 或 transformer 训练中，因此一开始就使用完整幅度的更新可能会破坏优化过程。
- **实践建议：** 应以 step 数或已处理 token 数定义 warmup，记录实际曲线，并结合 batch、优化器和总训练预算一起调参。
- **常见误区：** 并不是所有模型都需要 warmup，它也不会让原本不合适的学习率自动变得安全可用。
- **相关术语：** Learning Rate Schedule, Learning Rate, Batch Size, AdamW
- **来源：** [Accurate, Large Minibatch SGD](https://arxiv.org/abs/1706.02677)

### Weight
- **类别：** 数学与训练
- **常见说法：** 模型内部学到的一个数值。
- **实际含义：** 模型变换中的一个可训练系数。Weights 通常以 tensors 的形式组织，优化过程会调整它们以降低训练目标。
- **常见误区：** 并非每个参数都叫 weight；bias、embedding 和 normalization scale 也都是参数。
- **相关术语：** Parameter, Tensor, Optimizer

### Weight Decay
- **类别：** 数学与训练
- **常见说法：** 在优化过程中收缩 weights 的正则化方法。
- **实际含义：** 一种在训练过程中减小特定参数幅度的更新规则，常见做法是让 weights 乘上一个独立于梯度更新之外的收缩因子。
- **为什么重要：** 它可以提升泛化能力，但合适的系数以及应排除的参数组取决于模型、优化器、调度方式和数据。
- **常见误区：** 对于某些简单优化器，decoupled weight decay 等价于 L2 loss penalty；但对 Adam 这类自适应优化器而言，通常并不等价。
- **相关术语：** AdamW, Overfitting, Optimizer

### Worktree
- **类别：** AI 原生开发
- **实际含义：** 在 Git 中，worktree 是附着在某个仓库及其分支或提交上的工作目录；它共享对象存储，但拥有自己检出的文件和索引。
- **为什么重要：** 独立的 worktree 让人和 agent 可以并行工作，而不必反复切换或覆盖同一个 checkout。
- **实践建议：** 为每个 coding agent 分配具名 feature branch 和明确的 worktree 路径，然后通过正常的 Git 历史审查并集成 patch。
- **常见误区：** worktree 隔离的是检出的文件，而不是机器上的所有进程、端口、缓存、数据库或密钥。
- **延伸学习：** [Workbench for Real Repositories](../phases/14-agent-engineering/41-workbench-for-real-repos/)
- **来源：** [git-worktree documentation](https://git-scm.com/docs/git-worktree)
- **相关术语：** Coding Agent, Patch, Scope Contract, Handoff

## Z

### Zero-Shot
- **类别：** 提示与上下文
- **常见说法：** 在当前 prompt 中不给示例，直接要求模型完成任务。
- **实际含义：** 在即时输入中不提供该任务的示范样例，仅依靠指令或任务设定来完成任务。
- **常见误区：** Zero-shot 并不意味着模型没有接受过相关预训练、指令微调，也不意味着它没有工具或检索到的上下文。
- **相关术语：** Few-Shot, Prompt Engineering, Transfer Learning

### Zero Trust
- **类别：** 安全与治理
- **实际含义：** 一种安全模型，不会因为网络位置或资产归属就默认信任，而是根据身份、设备、资源、策略和当前上下文来评估每一次访问请求。
- **为什么重要：** AI 工具和 agent 会跨越本地文件、云服务、模型与外部内容，因此仅凭“可信内网”来授予权限，范围过大。
- **实践建议：** 要认证每一个行为主体和工作负载，对每一次资源操作进行授权，签发短时凭证，做好访问分段，并持续记录和重新评估与策略相关的信号。
- **常见误区：** Zero Trust 不意味着什么都不信任，也不意味着阻断所有自动化。它意味着把信任决策做成显式的、有边界的，并且能够持续验证。
- **延伸学习：** [Security, Secrets, and Audit](../phases/17-infrastructure-and-production/25-security-secrets-audit/)
- **相关术语：** Least Privilege, Trust Boundary, Approval Gate, Audit Log
- **来源：** [NIST SP 800-207](https://csrc.nist.gov/pubs/sp/800/207/final)
