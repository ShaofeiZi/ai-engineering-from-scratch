# LLM 功能的 A/B 测试 —— GrowthBook、Statsig 与 “凭感觉上线” 的问题

> 传统 A/B 测试并不是为非确定性的 LLM 设计的。这里最关键的区分是：eval 回答“模型能不能把这件事做对”，A/B 测试回答“用户是否真的在意这次改动”。两者缺一不可，2026 年再靠“看起来更好了”上线，已经不够格了。真正值得测的有三类：prompt engineering（措辞与提示结构）、model selection（GPT-4 vs GPT-3.5 vs OSS，权衡准确率、成本和延迟）、generation parameters（temperature、top-p）。真实案例里，聊天机器人某个 reward-model 变体把会话时长拉高了 +70%，留存拉高了 +30%；Nextdoor 的 AI 主题行实验在 reward function 调优后把 CTR 再推高了 +1%；Khan Academy 的 Khanmigo 则持续在“延迟 vs 数学正确率”这条轴上做迭代。平台层面分野也很清楚：**Statsig**（2025 年 9 月被 OpenAI 以 11 亿美元收购）主打 sequential testing、CUPED 和一体化产品；**GrowthBook** 则是开源、warehouse-native，支持 Bayesian、Frequentist、Sequential、CUPED、SRM 检查，以及 Benjamini-Hochberg 和 Bonferroni 修正。你最终选谁，通常取决于团队是否偏好直接写 warehouse SQL，以及“被 OpenAI 收购”这件事对组织是否敏感。

**Type:** 学习
**Languages:** Python（标准库，玩具级序贯检验模拟器）
**Prerequisites:** 阶段 17 · 13（可观测性）、阶段 17 · 20（渐进式部署）
**Time:** 约 60 分钟

## 学习目标

- 区分 eval（“模型能否完成任务”）与 A/B 测试（“用户是否在意这次变化”）。
- 列出三条可测试轴线：prompt、model、parameters，并为每一类选择合适指标。
- 解释 CUPED、sequential testing 和 Benjamini-Hochberg 多重比较修正。
- 根据 warehouse-SQL 偏好和企业对并购归属的态度，在 Statsig 与 GrowthBook 之间做选择。

## 问题

你手工调了一轮 system prompt。主观感觉更好了，于是直接上线。结果转化率的波动其实只是噪声，你却开始怀疑指标体系。或者你切了一个新模型，转化完全没动，到底是模型退化了，还是改动太小、根本测不出来？你不知道，因为你根本没有做 A/B。

eval 只能回答：在一个带标注的数据集上，模型能不能完成指定任务。它并不能告诉你：真实用户是否更喜欢这个输出。后者只能靠在线、随机、受控实验来判断，而且这个实验还必须有足够统计功效、能控制非确定性，并且对多次比较做修正。

## 概念

### Evals 与 A/B tests 的区别

**Evals**：离线、基于标注集、带 judge（rubric、LLM-as-judge 或人工）。它回答的是：“在这组固定分布上，输出是否正确、有帮助、足够安全？”

**A/B test**：在线、面对真实用户、随机分流。它回答的是：“新变体是否推动了真正重要的用户级指标？”

两者都需要。eval 用来在曝光前拦截回归，A/B 测试则用来在上线后确认产品层面的真实收益。

### 到底该测什么

1. **Prompt engineering**：措辞、system prompt 结构、示例组织方式。指标可以是任务成功率、用户留存、单次请求成本。
2. **Model selection**：GPT-4、GPT-3.5-Turbo、Llama-OSS 等模型之间的选择。指标往往是准确率（任务结果）+ 单次请求成本 + P99 延迟，是典型的多目标优化。
3. **Generation parameters**：temperature、top-p、max_tokens 等采样参数。指标则要贴合具体任务，例如输出多样性与确定性之间的权衡。

### CUPED：用前置数据降方差

Controlled-experiments Using Pre-Experiment Data。它的思路是在比较实验后期结果前，先用实验前数据把一部分方差“回归掉”。常见效果是把方差降低 30-70%，等价于免费提升有效样本量。

Statsig 和 GrowthBook 都内置了这个能力。

### 序贯检验

经典 A/B 测试假设样本量在实验开始前就固定。sequential test 则允许你“一边看结果一边决定是否停”，同时仍然控制反复查看带来的假阳性率。像 mSPRT 或 Howard 的 confidence sequences 这样的 always-valid 方法，允许你在胜负非常明显时提前停实验。

### 多重比较修正

如果你同时跑 20 个 95% 置信度的 A/B 测试，按概率总会平白撞出一个假阳性。Bonferroni correction 会把每个测试可用的 α 再收紧；Benjamini-Hochberg 则控制 false discovery rate，保守程度更低。GrowthBook 两种都支持。

### SRM：样本比例失配

用户通常通过 assignment hash 被随机分到不同变体。如果你设计的是 50/50 分流，实际却跑成了 47/53，那通常不是“运气不好”，而是分流链路哪里坏了。SRM 检查就是用来抓这个问题的。两个平台都内置支持。

### Statsig 与 GrowthBook 怎么选

**Statsig**：
- 2025 年 9 月被 OpenAI 以 11 亿美元收购，是托管式 SaaS。
- 提供 sequential testing、CUPED、held-out populations。
- 一体化能力比较强：feature flags、experimentation、observability 都打包。
- 最适合已经想买整套产品、并且不介意 OpenAI 所有权的团队。

**GrowthBook**：
- 开源（MIT），warehouse-native，可以直接从 Snowflake、BigQuery、Redshift 读取数据。
- 支持 Bayesian、Frequentist、Sequential 多种引擎。
- 提供 CUPED、SRM、Bonferroni、BH 修正。
- 可自托管，也可用托管云。
- 最适合数据团队掌控指标层、习惯直接做 warehouse SQL、并且偏好 OSS 的组织。

### 非确定性会让统计功效更难算

同一个 prompt，LLM 每次生成的结果并不完全相同。传统功效分析默认观测是 IID 的；一旦引入 LLM 非确定性，名义样本量通常会高估真实有效样本量。经验上，所需样本量最好再乘上大约 1.3-1.5x 作为安全边际。

### 真实案例的结果

- 聊天机器人 reward model 变体：会话时长 +70%，留存 +30%。
- Nextdoor 主题行实验：reward function 调优后 CTR 再增 +1%。
- Khan Academy Khanmigo：持续在延迟与数学正确率之间做迭代权衡。

### 反模式：凭感觉上线

几乎每个资深工程师都能说出一两个“因为感觉更好就上线”的功能。多数情况下，它们都悄悄拖累了团队几个月都没发现的产品指标。A/B 测试的价值就在这里：它逼你拿真实用户结果说话。

### 你应该记住的数字

- Statsig 被 OpenAI 收购：11 亿美元，时间是 2025 年 9 月。
- GrowthBook：开源 MIT，支持 Bayesian + Frequentist + Sequential。
- CUPED 常见降方差幅度：30-70%。
- LLM 非确定性：样本量通常需要额外预留 +30-50% 缓冲。

```figure
mx-sequential-test
```

## 用起来

`code/main.py` 会模拟一个同时带固定边界和 sequential 边界的 A/B 测试，展示为什么 sequential 方法能更早停出明显赢家。

## 交付物

这一课会产出 `outputs/skill-ab-plan.md`。输入某个功能改动、工作负载和基线指标后，它会帮你选平台、定义闸门，并估算样本量。

## 练习

1. 运行 `code/main.py`。如果预期 lift 是 5%，基线转化率是 3%，要做到 80% power，大约需要多大样本量？
2. 面向受监管、要求 on-prem 的医疗客户时，你会选 Statsig 还是 GrowthBook？
3. 设计一个测试 GPT-4 vs GPT-3.5 在单个 resolved ticket 成本上的 A/B。主指标、guardrail 指标和次指标分别是什么？
4. 你的 canary 通过了，但 A/B 测试显示转化率下降了 -1.2%。你还会发版吗？写出升级处理标准。
5. 对一个 pre-period 方差占 post-period 60% 的场景应用 CUPED。估算有效样本量能提升多少。

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|----------------|------------------------|
| Eval | “离线测试” | 对模型能力做带标注集评估 |
| A/B test | “实验” | 面向真实用户的在线随机对照比较 |
| CUPED | “降方差” | 用实验前数据做回归，降低方差 |
| Sequential test | “可以边看边停的测试” | 允许提前停止、且统计上始终有效的程序 |
| Multiple comparison | “家族错误问题” | 同时跑太多测试会抬高假阳性 |
| Bonferroni | “很紧的修正” | 把 α 按测试数量拆分 |
| Benjamini-Hochberg | “BH FDR” | 控制假发现率，比 Bonferroni 更不保守 |
| SRM | “分流坏了” | Sample ratio mismatch，说明分配链路有 bug |
| Statsig | “OpenAI 那家” | 商业一体化平台，2025 年被收购 |
| GrowthBook | “那个 OSS” | MIT 协议、warehouse-native 的实验平台 |
| mSPRT | “序列概率比检验” | 经典 sequential procedure |

## 延伸阅读

- [GrowthBook — How to A/B Test AI](https://blog.growthbook.io/how-to-a-b-test-ai-a-practical-guide/)
- [Statsig — 超越提示词：数据驱动的 LLM 优化](https://www.statsig.com/blog/llm-optimization-online-experimentation)
- [Statsig vs GrowthBook comparison](https://www.statsig.com/perspectives/ab-testing-feature-flags-comparison-tools)
- [Deng et al. — CUPED](https://www.exp-platform.com/Documents/2013-02-CUPED-ImprovingSensitivityOfControlledExperiments.pdf)
- [Howard — 置信序列](https://arxiv.org/abs/1810.08240)
