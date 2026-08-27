# 综合项目 05——自主研究智能体（AI-Scientist 级）

> Sakana 的 AI-Scientist-v2 已经发表过完整论文，Agent Laboratory 能够运行实验，Allen AI 则公开了执行轨迹。到 2026 年，这类系统通常在实验树上进行“规划—执行—验证”搜索，并配有成本预算、沙箱代码执行、接受视觉反馈的 LaTeX 写作器，以及自动化的 NeurIPS 风格评审模型组合。本综合项目要求你搭建这样一套系统，将每篇论文的端到端成本控制在 30 美元以内，并通过 Sakana 文档所记录的沙箱逃逸红队测试。

**Type:** 综合项目
**Languages:** Python（智能体与沙箱）、LaTeX（输出）
**Prerequisites:** 第 2 阶段（机器学习）、第 3 阶段（深度学习）、第 7 阶段（Transformer）、第 10 阶段（从零构建 LLM）、第 14 阶段（智能体）、第 15 阶段（自主系统）、第 16 阶段（多智能体）、第 18 阶段（安全）
**Phases exercised:** P0 · P2 · P3 · P7 · P10 · P14 · P15 · P16 · P18
**Time:** 40 小时

## 问题

自主研究智能体在 2026 年迈过了一道门槛。Sakana AI 的 AI-Scientist-v2 登上了《Nature》，其中生成的论文通过了研讨会同行评审。ShinkaEvolve（ICLR 2026）又把这条路线扩展到假设演化，AMD 的 Agent Laboratory 则公开了可复现的执行轨迹。这些智能体并不神奇：它们是在候选实验树上反复执行“规划—执行—验证”，同时设置成本上限、使用固定随机种子的沙箱，并自动评审结果。系统是否可靠，取决于循环设计、预算控制和安全保障。

你将在一个范围明确的领域中实现这套循环，例如对一亿参数 Transformer 的注意力稀疏性做消融实验。第一次运行能否发现新成果并不重要；真正有价值的是背后的基础设施，包括树搜索、实验沙箱、写作与评审循环，以及红队测试报告。Sakana 团队记录过沙箱逃逸故障，你的智能体必须通过同类红队测试。

## 核心概念

这个智能体采用最佳优先树搜索。每个节点都是一份实验规格，包含假设、配置、代码和预期结果。扩展节点时，只对原方案做小幅修改，例如更换优化器、调整批大小或移除某个组件。每个子节点都在全新的沙箱中运行，并受到严格的资源限制。实验结果会反馈给评分函数，再按“新颖性 × 质量 × 剩余预算”对节点排序。搜索持续到预算耗尽，最后将最佳分支写成论文。

写作器支持多模态输入。它先生成 LaTeX 初稿，完成编译并渲染图表，再把生成的 PDF 交给 Claude Opus 4.7 的视觉模式，检查版式、图表可读性以及论断与证据是否一致。随后，五个 LLM 评审模型按照 NeurIPS 风格分别评估新颖性、严谨性、清晰度、可复现性和影响力。平均分低于阈值时，论文连同评审意见一起退回写作器修改。

安全机制是整套系统的基础。每项实验都在 E2B 或 Daytona 沙箱中运行，禁止访问外部网络，并限制运行时长和资源用量。智能体生成的代码还要经过策略层检查，拦截可能逃逸沙箱的系统调用。红队报告需要复现 Sakana 记录的攻击面，包括进程分叉炸弹（fork bomb）、文件系统逃逸，以及由 LLM 生成的网络请求代码。

## 架构

```
seed idea + domain
      |
      v
  literature search (Semantic Scholar + OpenAlex + FAISS cache)
      |
      v
  LangGraph plan-execute-verify tree
      |
      v
  +--- expand node ----+      per-node sandbox
  |                    |      (E2B / Daytona)
  v                    v      resource caps
  child_1           child_k   no network egress
  |                    |      deterministic seeds
  v                    v
  run experiment       run experiment
  |                    |
  v                    v
  score nodes by (novelty, quality, budget)
      |
      v
  best branch -> LaTeX writer
      |
      v
  compile + vision critique (Opus 4.7 vision)
      |
      v
  reviewer ensemble (5 LLM judges, NeurIPS rubric)
      |
      v
  paper.pdf + review.md + trace.json
```

## 技术栈

- 编排：LangGraph，支持检查点和人工审批关卡
- 树搜索：在实验节点上运行自定义最佳优先搜索（采用 Sakana v2 的 AB-MCTS 风格）
- 沙箱：每个实验使用一个 E2B 沙箱，Docker-in-Docker 作为回退；通过 cgroups 施加资源上限
- 文献层：Semantic Scholar Graph API + OpenAlex + 本地 FAISS 摘要缓存
- 写作器：LaTeX 模板 + Claude Opus 4.7（视觉模式），用于评议图表和检查版式
- 评审器：由 5 个模型组成的评审组合（Opus 4.7、GPT-5.4、Gemini 3 Pro、DeepSeek R1、Qwen3-Max），采用加权汇总
- 实验框架：PyTorch 2.5 负责实际实验，W&B 负责记录
- 可观测性：Langfuse 记录智能体执行轨迹，并为每篇论文设置 30 美元的硬性预算

```figure
ce-experiment-tree
```

## 动手构建

1. **确定初始想法和领域范围。** 选择一个初始想法，例如“研究十亿参数以下 Transformer 的注意力图稀疏模式”。定义模型、数据集和计算预算所构成的搜索空间。

2. **文献扫描。** 查询 Semantic Scholar 和 OpenAlex，找出被引次数最高的 50 篇相关论文，将摘要缓存在本地，再生成一页领域综述。

3. **搭建树结构。** 用初始假设创建根节点。实现 `expand(node) -> children`，要求每个子节点只修改一项配置；再实现 `score(node)`，对新颖性、质量和预算三项乘积加权。

4. **沙箱封装。** 每个实验都通过 `docker run --network=none --memory=8g --cpus=2 --pids-limit=256 --read-only` 运行，或采用等效的 E2B 策略。将随机种子写入沙箱，并以只读方式挂载实验输出。

5. **规划—执行—验证循环。** `plan` 提出子节点；`execute` 在沙箱中运行实验并收集日志和指标；`verify` 对指标执行单元检查，例如损失是否下降、消融实验是否隔离了目标效应。失败节点的原因要存回搜索树。

6. **写作器。** 预算耗尽后选出最佳分支，用 matplotlib 渲染图表，再让 Claude Opus 4.7 根据该分支的执行轨迹生成 LaTeX 初稿。编译后，将 PDF 交给 Opus 4.7 的视觉模式评议，并据此反复修改。

7. **评审模型组合。** 五个模型按照 NeurIPS 风格的量表，从新颖性、严谨性、清晰度、可复现性和影响力五方面为初稿评分。平均分低于 4.0/5 时，将评审意见连同初稿退回写作器。最多重写 3 轮，随后强制停止。

8. **红队测试。** 构建或接入一组针对沙箱的对抗任务，包括进程分叉炸弹、网络数据外传、文件系统逃逸，以及 LLM 生成的 shell 元字符攻击。确认所有攻击均被拦截，并记录结果。

9. **可复现性。** 每篇论文都必须附带树搜索轨迹 JSON、随机种子、W&B 运行链接、沙箱配置，以及说明如何端到端复现实验的 README。

## 实际使用

```
$ ai-scientist run --seed "attention sparsity in sub-1B transformers" --budget 30
[lit]    50 papers, digest in 12s
[tree]   expanded 8 nodes, budget 12/30
[exec]   node #3 sparsity=top-8, loss=2.83 (best so far)
[exec]   node #6 sparsity=top-4, loss=3.12 (worse)
[exec]   ...
[tree]   chose branch rooted at node #3 (novelty 0.62, quality 0.81)
[write]  LaTeX draft v1 complete
[vision] critique: figure 2 legend too small, claim-evidence ok
[write]  draft v2 after 3 edits
[review] mean 4.2/5 (novelty 3.9, rigor 4.3, clarity 4.1, repro 4.5, impact 4.2)
[done]   paper.pdf + review.md + trace.json     $28.40 spent
```

## 交付成果

`outputs/skill-ai-scientist.md` 是最终交付物。输入初始想法、研究领域和 30 美元预算后，它会运行完整流水线，输出一篇可供评审的论文及其可复现性材料包。

| 权重 | 标准 | 测量方式 |
|:-:|---|---|
| 25 | 论文质量 | 参照已发表的研讨会论文进行盲审式量表评估 |
| 20 | 实验严谨性 | 基线、随机种子和消融实验齐全；每项论断都由结果表中的数据支持 |
| 20 | 成本与算力控制 | 严格执行每篇论文 30 美元的上限，并由 Langfuse 跟踪 |
| 20 | 安全性 | 通过沙箱红队测试；网络策略和紧急停止开关均验证有效 |
| 15 | 可复现性 | 使用相同随机种子时，一条命令即可重新运行并复现论文结果 |
| **100** | | |

## 练习

1. 在同一领域中，分别以三个不同的初始想法运行整条流水线。比较树搜索中重叠的部分，找出重复消耗的算力。

2. 在执行实验之前，为预计成本超过 5 美元的节点增加人工审批关卡。测量总成本因此降低了多少。

3. 将评审模型组合换成单个模型。在一组留出的已知低质量论文上测试，测量误接受率。

4. 增加网络数据外传红队测试：让智能体编写代码，尝试用 `curl` 访问外部地址。确认 `--network=none` 策略能够拦截，并记录这次尝试。

5. 将树搜索与平坦随机基线比较：两者预算相同，但后者不使用扩展策略。报告“新颖性 × 质量”的增益。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|-----------------|------------------------|
| 树搜索（Tree search） | “AB-MCTS 式扩展” | 按“新颖性 × 质量 × 预算”得分，对实验节点进行最佳优先探索 |
| 沙箱（Sandbox） | “实验隔离” | 禁止联网、限制 CPU 和内存、固定随机种子且输入只读的容器环境 |
| 视觉评议（Vision critique） | “先渲染、再阅读” | 将论文编译为 PDF，再交给 VLM 检查版式以及论断与证据是否一致 |
| 评审模型组合（Reviewer ensemble） | “自动化同行评审” | 多个 LLM 按 NeurIPS 量表评分，并用加权结果决定流水线是否放行 |
| 新颖性分数（Novelty score） | “这是新成果吗？” | 一项启发式评分，会惩罚与 50 篇论文缓存过于相似的结果 |
| 成本上限（Cost ceiling） | “美元预算” | 每篇论文总开销的硬性上限，由 Langfuse 计数器和运行前估算共同控制 |
| 红队测试（Red team） | “沙箱逃逸审计” | 一组对抗任务；如果策略有误，这些任务就可能逃逸沙箱 |

## 延伸阅读

- [Sakana AI-Scientist-v2 代码仓库](https://github.com/SakanaAI/AI-Scientist-v2) — 可供参考的生产级研究智能体
- [Sakana AI-Scientist-v1 论文（arXiv:2408.06292）](https://arxiv.org/abs/2408.06292) — 最初介绍该方法的论文
- [ShinkaEvolve (Sakana ICLR 2026)](https://sakana.ai) — 进化式扩展方向
- [Agent Laboratory (AMD)](https://github.com/SamuelSchmidgall/AgentLaboratory) — 多角色研究实验室框架
- [LangGraph 文档](https://langchain-ai.github.io/langgraph/) — 编排层参考
- [Semantic Scholar Graph API](https://api.semanticscholar.org/) — 文献搜索入口
- [E2B 沙箱](https://e2b.dev) — 实验隔离参考
- [NeurIPS 评审指南](https://neurips.cc/Conferences/2026/Reviewer-Guidelines) — 评审模型组合所采用的评分准则
