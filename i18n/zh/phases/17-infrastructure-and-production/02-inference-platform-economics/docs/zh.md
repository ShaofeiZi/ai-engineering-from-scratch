# 推理平台经济学：Fireworks、Together、Baseten、Modal、Replicate、Anyscale

> 到了 2026 年，推理市场早就不只是“出租 GPU 时间”了。它已经分裂成三类：定制芯片路线（Groq、Cerebras、SambaNova）、GPU 平台路线（Baseten、Together、Fireworks、Modal），以及 API-first 市场路线（Replicate、DeepInfra）。Fireworks 在 2026 年 5 月 1 日把 GPU 租金上调了 $1/hr，而它在每天 10T+ tokens 的量级上拿到 $4B 估值，说明“靠规模驱动”的模型是跑得通的。Baseten 在 2026 年 1 月以 $5B 估值完成了 $300M Series E。竞争定位规则其实很简单：Fireworks 优化延迟，Together 优化目录宽度，Baseten 优化企业级打磨，Modal 优化 Python-native DX，Replicate 优化多模态覆盖，Anyscale 优化分布式 Python。学完这课，你应该能给一家初创公司直接递出一张可执行的选型矩阵。

**Type:** 学习
**Languages:** Python（标准库，玩具级单次调用经济性比较器）
**Prerequisites:** 第 17 阶段 · 01（托管 LLM 平台），第 17 阶段 · 04（服务引擎内部原理）
**Time:** 约 60 分钟

## 学习目标

- 说出三类市场分层（custom silicon、GPU platforms、API-first），并把每家厂商放到正确分层中。
- 解释为什么“按 token 收费”的 API 定价会逐渐贴近 serving engine 的成本曲线，而不是硬件本身的成本曲线。
- 至少在三家厂商之间算出 effective cost per request，并解释什么时候按分钟计费（Baseten、Modal）会优于按 token 计费。
- 判断在不同工作负载下，哪一个平台应该成为默认选项：serverless bursty、steady high-throughput、fine-tuned variants、多模态。

## 问题

你已经评估过 hyperscaler 的托管平台了。现在你决定自己需要一个更窄、更快的选择：也许用 Fireworks 换延迟，用 Together 换目录广度，用 Baseten 托一个定制微调模型。问题来了，你现在有六个现实可选项，但它们的价格页根本不能直接横向对齐。Fireworks 写的是 $/M tokens；Baseten 写的是 $/minute；Modal 写的是 $/second；Replicate 写的是 $/prediction。不先建工作负载模型，你没法真正把它们放在一张表里比。

更麻烦的是，每张价格页背后的商业模式都不同。Fireworks 在共享 GPU 上运行自己的自研引擎（FireAttention），所以它的按 token 价格背后其实反映的是它的利用率曲线。Baseten 提供 Truss + dedicated GPUs，因此按分钟收费本质上卖的是独占性。Modal 则是真正的 Python serverless，按秒计费，冷启动还可以压到亚秒或几秒级。表面上大家都在卖“LLM 推理结果”，实际上底层对应着完全不同的成本函数。

这一课就是把这六家放进同一个经济模型中，并告诉你它们分别在什么场景下获胜。

## 概念

### 三个分层

**Custom silicon**：Groq（LPU）、Cerebras（WSE）、SambaNova（RDU）。在同一个模型上，通常能比基于 GPU 的集群实现 5-10x 更快的 decode。代价是 per-token 价格更高，例如 Groq 在 2025 年底的 Llama-70B 上大约是 ~$0.99/M。它并非低成本路线，但对于 voice agents 和实时翻译这种极致延迟场景，它经常是最优生产选择。

**GPU platforms**：Baseten、Together、Fireworks、Modal、Anyscale。它们运行在 NVIDIA（2026 年常见是 H100、H200、B200）或少数 AMD 上，处在“原始 GPU 租赁”（RunPod、Lambda）和“超大云托管服务”（Bedrock）之间的经济中间层。

**API-first marketplaces**：Replicate、DeepInfra、OpenRouter、Fal。特点是目录广、按 prediction 或按秒计费、强调 time-to-first-call 极短。

### Fireworks：延迟优化型 GPU 平台

- 自研 FireAttention 引擎，对外宣称在等价配置下延迟可比 vLLM 低 4x。
- 提供 batch tier，对于非交互式工作负载，大约是 serverless 价格的 50%。
- 微调模型按 base model 价格提供服务，这是它相对很多 LoRA 额外加价平台的真实差异点。
- 到 2026 年中，按需 GPU 租金上调了 $1/hour，自 2026 年 5 月 1 日生效。
- 财务信号也足够强：$4B 估值，每天处理 10T+ tokens。

### Together：目录广度优化型

- 提供 200+ 模型，开源上游新模型发布后几天内往往就会上架。
- 在同等 LLM 模型上，价格通常比 Replicate 低 50-70%，“AI Native Cloud” 的核心卖点就是目录与吞吐量。
- 推理、微调和训练都可以通过同一套 API 完成。

### Baseten：企业打磨优化型

- Truss 框架把模型打包、依赖、secrets、serving config 全部收敛到一个 manifest 里。
- GPU 覆盖从 T4 到 B200，按分钟计费，并且在冷启动缓解上做得不错。
- SOC 2 Type II、HIPAA-ready，常见于金融和医疗场景。
- 2026 年 1 月完成 $300M Series E，估值来到 $5B，由 CapitalG、IVP、NVIDIA 等参与。

### Modal：Python-native 优化型

- 纯 Python 形式的 infrastructure-as-code。你可以用 `@modal.function(gpu="A100")` 装饰一个函数，然后一条命令就部署。
- 按秒计费。冷启动通常是 2-4 秒，小模型可以做到 <1 秒。
- 2025 年以 $1.1B 估值完成 $87M Series B。在独立开发者调查里，它的 DX 评分通常是最强一档。

### Replicate：多模态覆盖型

- 按 prediction 计费，是很多 image、video、audio 模型的默认入口平台。
- 集成生态好，Zapier、Vercel、CMS 插件这些外围能力很丰富。
- LLM 的 per-token 价格竞争力一般，但在多模态种类上经常赢。

### Anyscale：Ray-native 路线

- 基于 Ray；RayTurbo 是 Anyscale 的专有推理引擎，用来和 vLLM 竞争。
- 最适合分布式 Python 工作负载，其中推理只是更大计算图中的一个节点。
- 提供托管 Ray clusters，并和 Ray AIR、Ray Serve 高度集成。

### 按 token 收费与按分钟收费：什么时候各自胜出

按 token 计费适合延迟不敏感、且流量爆发性很强的工作负载，因为你只为真实消耗付费。按分钟计费则适合高利用率、负载更可预测的场景，因为一旦 GPU 利用率打上去，它会比按 token 更便宜。

一个很粗但很有用的经验法则是：当 dedicated GPU 的 sustained utilization 超过约 30% 时，按分钟计费（Baseten、Modal）就开始打赢按 token 计费（Fireworks、Together）。低于这个区间时，按 token 往往更划算，因为你避免了为空闲时间买单。

### 真正的护城河是自研引擎

几乎每一家建立在 vLLM 或 SGLang 之上的平台，都声称自己有“自研引擎”：FireAttention、RayTurbo、Baseten 自家的 serving stack。这里营销成分很多。更诚实的表述是：vLLM + SGLang 代表了 2026 年大约 80% 的生产级开源推理基础，而平台层真正的差异，更多落在 DX、归因能力和 SLA 上，而不是“某个 attention kernel 名字听起来更新”。

### 你应该记住的数字

- Fireworks GPU rental：自 2026 年 5 月 1 日起，上涨 $1/hr。
- Fireworks 的公开 claim：在等价配置上，延迟可比 vLLM 低 4x。
- Together：在 LLM 上通常比 Replicate 便宜 50-70%。
- Baseten valuation：$5B（Series E，2026 年 1 月，$300M 轮）。
- Modal valuation：$1.1B（Series B，2025 年）。
- 当 sustained utilization 超过约 30% 时，per-minute 往往会打赢 per-token。

```figure
cost-per-token
```

## 学以致用

`code/main.py` 会在一个合成工作负载上比较这六家厂商的不同定价模型，并输出 $/day 与 effective $/M tokens。运行它，你就能看到 per-token 和 per-minute 之间的 break-even 到底在哪。

## 交付成果

这一课会产出 `outputs/skill-inference-platform-picker.md`。给它一个 workload profile、SLA 和预算，它会选出主推理平台，并给出次选方案。

## 练习

1. 运行 `code/main.py`。对于一张 H100 上跑 70B 模型，Baseten（per-minute）会在什么 sustained utilization 下开始优于 Fireworks（per-token）？自己推导 crossover，再和经验法则对照。
2. 你的产品同时提供 image generation、chat 和 speech-to-text。分别给这些模态选平台，并说出一个把它们统一起来的 gateway pattern。
3. Fireworks 把你的主模型价格又上调了 $1/hr。如果你有 40% 的流量可以转到 batch tier（50% off），模型总成本会怎么变化？
4. 某个强监管客户要求 SOC 2 Type II + HIPAA + dedicated GPUs。哪三家平台是 viable 的？其中谁在 FinOps 层面最好用？
5. 比较 Llama 3.1 70B 在 Fireworks serverless、Together on-demand、Baseten dedicated、Replicate API 上每 1,000 次 prediction 的成本。10 predictions/day 时谁最便宜？10,000 predictions/day 时谁最便宜？

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------|----------|
| Custom silicon | “非 GPU 芯片” | Groq LPU、Cerebras WSE、SambaNova RDU，都是为 decode 优化的专用芯片 |
| FireAttention | “Fireworks 引擎” | 自定义 attention kernel；对外宣称比 vLLM 低 4x 延迟 |
| Truss | “Baseten 的格式” | 把模型打包、依赖、secrets 和 serving config 收进同一个 manifest |
| Per-token | “API 定价” | 按消耗的 token 计费；不会为空闲时间付费 |
| Per-minute | “专属实例定价” | 按 GPU 墙钟时间计费；高利用率时更划算 |
| Per-prediction | “Replicate 定价” | 按单次模型调用计费；常见于图像和视频 |
| RayTurbo | “Anyscale 引擎” | 构建在 Ray 上的专有推理引擎；在 Ray 集群里和 vLLM 竞争 |
| Batch tier | “五折队列” | 非交互式任务的折扣队列；Fireworks、OpenAI 等都常见 |
| 基础模型同价微调 | “Fireworks LoRA” | LoRA 请求按基础模型费率收费，是它的差异点之一 |

## 延伸阅读

- [Fireworks Pricing](https://fireworks.ai/pricing) — 按 token 费率、batch tier 与 GPU 租赁价格。
- [Baseten Pricing](https://www.baseten.co/pricing/) — 按分钟费率、承诺容量与企业分层。
- [Modal Pricing](https://modal.com/pricing) — 按秒 GPU 费率与免费层。
- [Together AI Pricing](https://www.together.ai/pricing) — 模型目录与按 token 费率。
- [Anyscale Pricing](https://www.anyscale.com/pricing) — RayTurbo 与托管 Ray 定价。
- [Northflank — Fireworks AI Alternatives](https://northflank.com/blog/7-best-fireworks-ai-alternatives-for-inference) — 竞品对比评估。
- [Infrabase — AI Inference API Providers 2026](https://infrabase.ai/blog/ai-inference-api-providers-compared) — 供应商版图综览。
