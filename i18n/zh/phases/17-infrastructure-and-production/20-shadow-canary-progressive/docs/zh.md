# LLM 的影子流量、金丝雀发布与渐进式部署

> LLM 的上线流程，叠加了软件部署里最难处理的几类问题：没有可靠的单元测试、失败模式分散、反馈信号还会延迟出现。标准顺序通常是：(1) shadow mode，把线上请求复制给候选模型，只记录和比对，不影响真实用户；它能抓住明显的分布问题，但绝不是质量保证；(2) canary rollout，按 10% → 25% → 50% → 75% → 100% 逐步放量，并在每一步设置 gate；关注 latency percentile、cost/request、error/refusal rate、output length distribution、user-feedback rate；(3) 稳定性确认之后，再对差异明显的候选方案做 A/B testing。LLM 的非确定性无法彻底消除：即便输入完全一样，GPU 浮点非结合性再叠加 batch-size 变化，也可能让准确率在多次运行之间波动到 15%。成本也不是常数，一个“效果提高 20%”的模型，单次调用成本可能是原来的 3 倍。真正决定成败的是 rollback 速度：如果你必须重新部署才能回滚，那就说明你已经太慢。策略应放在 config/flags 中；模型应登记在 registry 并钉住 digest；rollback 应该等于“切换 policy + 恢复阈值 + 重新 pin 旧模型”，耗时以秒计，而不是以小时计。

**Type:** 学习
**Languages:** Python（标准库，玩具级金丝雀发布进度模拟器）
**Prerequisites:** 阶段 17 · 13（可观测性）、阶段 17 · 21（A/B 测试）
**Time:** 约 60 分钟

## 学习目标

- 区分 shadow mode、canary rollout 和 A/B testing 各自的作用与边界。
- 列举五个 LLM 特有的 canary 指标：latency、cost/request、error/refusal、output-length distribution、user feedback。
- 解释为什么 LLM 的非确定性波动会改变 rollout 中“稳定”的定义。
- 设计一个以秒为单位完成的 rollback 路径，而不是依赖重新部署。

## 问题

你上线了一个新模型。离线评估显示准确率提升了 3%。于是你直接切到生产。24 小时内，成本上涨 40%，用户点踩上涨 8%，还有 3 张客户工单反馈“回答变奇怪了”。你决定回滚，但重新部署需要 3 个小时。整个周末都被拖垮了。

这几乎每一步都本可以避免。shadow mode 本来可以在用户看到之前就发现那 40% 的成本飙升；canary 本来可以在 10% 流量时就因为点踩上升而停下；基于 policy flag 的 rollback 本来只需要 30 秒。真正的难点，不在于“离线 eval 看起来不错”，而在于如何把它安全地过渡成“真实用户也满意”。

## 核心概念

### 影子模式

候选模型接收与生产模型完全相同的请求，但它的输出只会被记录，不会返回给用户，因此对用户是零影响。通常要记录：

- 输出内容，以及它与生产结果之间的差异。
- token 数量，也就是成本变化。
- 延迟。
- refusal 和 error。

它能抓住的主要问题包括：成本异常飙升、输出长度回归、明显的拒答变化、硬错误。它抓不住的是：用户真正会感知到的质量差异。所以 shadow 更像 smoke test，而不是质量测试。

### 金丝雀发布

canary 的核心是带 gate 的渐进式流量切换。常见放量路径是：1% → 10% → 25% → 50% → 75% → 100%。每一步都要看 5 组指标：

1. **Latency percentiles**：P50、P95、P99。触发 breach 的例子是 canary 的 P99 超过 baseline 的 1.5x。
2. **Cost per request**：混合后的单请求成本。若高于 baseline 20% 以上，可视为 breach。
3. **错误 / 拒答率**：包括 5xx 和显式 refusal。若达到 baseline 的 2x，可视为 breach。
4. **输出长度分布**：均值加 P99。如果输出长度分布整体漂移，就该停。
5. **User-feedback rate**：点踩率、工单率等。若达到 baseline 的 1.5x，就需要告警。

### 非确定性就是新的方差来源

同样的输入，不一定会产出同样的输出。原因包括：

- GPU FP non-associativity，也就是浮点归约顺序会随着 batch 而变化。
- batch-size variance，同一条 prompt 落在 batch 128 和 batch 16 里，行为可能不同。
- sampling，只要 temperature > 0，就天然会引入波动。

实际测量里，同一份 eval set 在多次运行间，准确率波动可以高达 15%。因此 rollout 里的“稳定”，不再意味着“与 baseline 完全相同”，而是“指标仍落在预期波动范围内”。你的 gate 必须设在噪声地板之上，否则只能制造误报。

### 成本本身就是变量

一个效果提高 20% 的模型，完全可能每次调用贵 3 倍。cost/request 因此必须是五个 gate 之一。只提升质量、却破坏 unit economics 的模型，同样应该被回滚。

### 回滚才是真正的武器

- **Policy flag**：通过 feature flag system 在配置层调整流量百分比，通常只需几秒。
- **Model pinning**：在 registry 层钉住 digest，避免模型被自动升级。
- **Rollback**：就是把 flag 切回去，再把 pinned digest 指回旧模型，整个过程应该以秒为单位完成。

如果你的系统想回滚就必须重新部署，那说明在真正 rollout 之前，系统设计就还没准备好。

### 常用工具

**Argo Rollouts** / **Flagger**：用于 Kubernetes 渐进式发布的控制器，可以与 Istio/Linkerd 的按权重路由能力配合使用。

**Istio 加权路由**：在 service mesh 层做流量切分。

**KServe / Seldon Core**：模型服务框架，内建 canary 能力。

**Feature flags**：例如 LaunchDarkly、Flagsmith、Unleash。优点是 policy flip 不需要 redeploy。

### 指标采样节奏

canary gate 通常每 5–15 分钟检查一次，具体取决于流量规模。若只有 1% 流量、总请求量又只有 10 req/min，那一个窗口里只有 50–150 个样本，足够看 latency，但对 user feedback 来说仍然偏噪。10% 流量会让样本量提升约 10 倍。因此每一级 rollout 都应该停得足够久，先积累足够样本，再决定是否继续。

### A/B 这一步是可选的

如果新模型与旧模型差异明显，例如行为不同、成本曲线不同、语气不同，那么 canary 通过后，可以在 50% 流量阶段做 A/B test。如果它只是旧模型的改进版，那么 canary gate 通过后直接推到 100% 往往就够了。

### 你需要记住的数字

- 常见 canary progression：1% → 10% → 25% → 50% → 75% → 100%。
- 非确定性波动上限：相同输入的多次运行间，可能出现高达 15% 的准确率差异。
- 五个 canary 指标：latency、cost、error/refusal、output length、user feedback。
- 成本 gate：高于 baseline 20% 就算 breach。
- rollback 的目标速度：秒级，而不是小时级。

```figure
i4-canary-ramp
```

## 用起来

`code/main.py` 会模拟一个带有回归注入的 canary rollout，并报告 rollout 会停在哪个阶段，以及具体是哪一个 gate 被触发。

## 交付物

本课产出 `outputs/skill-rollout-runbook.md`。它会根据 candidate model、baseline 和风险容忍度，生成一份 shadow→canary→100% 的 rollout 计划。

## 练习

1. 运行 `code/main.py`。注入一个 25% 的成本回归。canary 会在哪个阶段停住？
2. 你的新模型离线准确率提升 3%，但 cost/request 增加了 18%。这算不算该上线？取决于 policy，请分别写出两条决策路径。
3. 设计一个端到端不超过 60 秒的 rollback 方案，并列出所需基础设施。
4. 你的 eval 里，非确定性波动大约是 ±7%。那 canary gate 应该怎么设，才能避免误报？你会用什么 multiplier？
5. shadow mode 在 canary 之前就发现成本上涨 40%。请写出对应的 shadow alert rule。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------|----------|
| Shadow mode | “复制到新模型” | 把请求零影响地复制给候选模型，仅用于记录 |
| Canary | “渐进式放量” | 对真实用户逐步放量、并设置 gate 的 rollout |
| Gates | “放量检查项” | 用于阻止继续放量的指标阈值 |
| Non-determinism | “LLM 方差” | 无法完全消除的多次运行差异 |
| Policy flag | “切 flag 回滚” | 在配置层完成的秒级回滚 |
| Model pin | “registry digest” | 指向特定模型版本的不可变引用 |
| Argo Rollouts | “K8s 渐进发布” | Kubernetes 原生的 canary/rollback 控制器 |
| KServe | “inference K8s” | 带有 canary 原语的模型服务框架 |
| Istio weighted | “mesh 流量切分” | service mesh 层的流量切分 |

## 延伸阅读

- [TianPan — 在不破坏生产的前提下发布 AI 功能](https://tianpan.co/blog/2026-04-09-llm-gradual-rollout-shadow-canary-ab-testing)
- [MarkTechPost — 安全部署 ML 模型](https://www.marktechpost.com/2026/03/21/safely-deploying-ml-models-to-production-four-controlled-strategies-a-b-canary-interleaved-shadow-testing/)
- [APXML — 高级 LLM 部署模式](https://apxml.com/courses/mlops-for-large-models-llmops/chapter-4-llm-deployment-serving-optimization/advanced-llm-deployment-patterns)
- [Argo Rollouts 文档](https://argo-rollouts.readthedocs.io/)
- [Flagger docs](https://docs.flagger.app/)
