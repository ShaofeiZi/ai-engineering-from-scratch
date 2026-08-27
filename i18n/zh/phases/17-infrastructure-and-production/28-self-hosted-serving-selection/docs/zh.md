# 自托管服务选型：让引擎匹配硬件与规模

> 引擎选型取决于硬件、规模和生态，而不是看一眼排行榜。到了 2026 年，自托管推理的主流引擎基本集中在四个：llama.cpp、Ollama、vLLM、SGLang，而 TGI 已经落到 maintenance mode。**llama.cpp** 在 CPU 场景下最快，模型支持面最广，对量化和线程也拥有最完整的控制。**Ollama** 是开发笔记本上的一键安装方案，但通常会比 llama.cpp 慢约 15-30%（Go + CGo + HTTP serialization），在接近生产的负载下吞吐差距可达到 3x。**TGI 于 2025 年 12 月 11 日进入 maintenance mode**，后续只收 bug fix；它的原始吞吐通常比 vLLM 慢约 10%，但历史上在 observability 和 Hugging Face 生态整合上很强。由于进入 maintenance mode，它不再适合作为新项目的长期默认，SGLang 或 vLLM 更稳。**vLLM** 是通用生产场景的默认选择，v0.15.1（2026 年 2 月）新增了 PyTorch 2.10、RTX Blackwell SM120 与 H200 优化。**SGLang** 则是 agentic 多轮、prefix-heavy 工作负载的专长选手，已经在生产中运行于 400,000+ GPUs（xAI、LinkedIn、Cursor、Oracle、GCP、Azure、AWS）。硬件约束上：CPU-first → llama.cpp；AMD / 非 NVIDIA → vLLM 是支持最完整的路径（TRT-LLM 被锁定在 NVIDIA 上）。2026 年常见的流水线模式是：dev = Ollama，staging = llama.cpp，prod = vLLM 或 SGLang。不同引擎接受的权重格式也不同，llama.cpp 家族偏 GGUF，GPU 引擎偏 Hugging Face safetensors，因此阶段之间往往还要做一次格式转换。

**Type:** 学习
**Languages:** Python（标准库，引擎决策树遍历器）
**Prerequisites:** 阶段 17 中所有覆盖引擎的课程（04、06、07、09、18）
**Time:** 约 45 分钟

## 学习目标

- 根据硬件（CPU / AMD / NVIDIA Hopper / Blackwell）、规模（1 用户 / 100 / 10,000）和工作负载（通用聊天 / agent / 长上下文）选择合适引擎。
- 说出 TGI 在 2026 年的 maintenance-mode 状态（2025 年 12 月 11 日）以及这为什么会让新项目更偏向 vLLM 或 SGLang。
- 描述 dev / staging / prod 的流水线模式，以及 GGUF 到 safetensors 的格式转换通常处于哪个环节。
- 解释为什么 “CPU-first” 会指向 llama.cpp，而 “AMD” 会排除 TRT-LLM。

## 问题

你的团队刚启动一个新的自托管 LLM 项目。一个工程师说用 Ollama，另一个说用 vLLM，第三个说“不是 TGI 开箱即用吗？”三个人都没完全错，但他们各自只对一部分场景成立。

到了 2026 年，这个选择树必须按顺序走：先看硬件，再看规模，最后看工作负载。还有一个发生在 2025 年 12 月 11 日的关键事件，也就是 TGI 进入 maintenance mode，它直接改变了新项目的默认选项。

## 概念

### 五个引擎

| 引擎 | 最适合 | 说明 |
|--------|----------|-------|
| **llama.cpp** | CPU / 边缘设备 / 最少依赖 / 最广模型支持 | CPU 上最快，控制力最强 |
| **Ollama** | 开发笔记本、单用户、一键安装 | 比 llama.cpp 慢 15-30%；生产吞吐可能差 3x |
| **TGI** | HF 生态、受监管行业 | **自 2025 年 12 月 11 日起进入维护模式** |
| **vLLM** | 通用生产场景、100+ 用户 | 2026 年最广泛的生产默认；v0.15.1 发布于 2026 年 2 月 |
| **SGLang** | 智能体多轮、前缀复用密集型工作负载 | 已在生产中的 400,000+ 张 GPU 上运行 |

### 硬件优先的决策方式

**CPU-first** → llama.cpp。Ollama 也能跑，但更慢。其他引擎在 CPU 上都没有竞争力。

**AMD GPU** → vLLM 是支持最完善的路径（支持 AMD ROCm）。SGLang 也可行。TRT-LLM 只支持 NVIDIA，所以直接排除。

**NVIDIA Hopper（H100 / H200）** → vLLM、SGLang、TRT-LLM 都属于顶级选择。

**NVIDIA Blackwell（B200 / GB200）** → TRT-LLM 是吞吐冠军（见 Phase 17 · 07），vLLM 和 SGLang 紧随其后。

**Apple Silicon（M 系列）** → llama.cpp（Metal）。Ollama 本质上是对这一路径的封装。

### 规模第二位

**1 用户 / 本地开发** → Ollama。一条命令就能装好，几秒内出 first token。

**10-100 用户 / 小团队** → 单 GPU 的 vLLM。

**100-10k 用户 / 生产** → vLLM production stack（Phase 17 · 18）或 SGLang。

**10k+ 用户 / 企业** → vLLM production stack + disaggregated（Phase 17 · 17）+ LMCache（Phase 17 · 18）。

### 工作负载第三位

**通用聊天 / Q&A** → vLLM 是更稳的通用默认。

**Agentic 多轮（tools、planning、memory）** → SGLang 的 RadixAttention（Phase 17 · 06）通常更有优势。

**RAG 且 prefix 重复使用很多** → SGLang 更适合。

**代码生成** → vLLM 没问题；SGLang 在 cache 利用上略占优。

**长上下文（128K+）** → vLLM + chunked prefill；或 SGLang + tiered KV。

### TGI 的维护模式陷阱

Hugging Face TGI 已于 2025 年 12 月 11 日进入 maintenance mode，后续只收 bug fix。从历史上看，它的 observability 很强，对 Hugging Face 生态（model cards、safety tools）的整合也很好，但在原始吞吐上通常略落后于 vLLM。

到了 2026 年，对新项目来说应该默认避开 TGI。已有的 TGI 部署可以继续用，但应当规划迁移。vLLM 和 SGLang 是更安全的默认选择。

### 流水线模式

典型流程是：dev（Ollama）→ staging（llama.cpp）→ prod（vLLM）。不同引擎接受的权重格式不同：llama.cpp 家族通常使用 GGUF，GPU 引擎通常使用 Hugging Face safetensors，所以中间经常需要加一层格式转换。这样工程师可以在笔记本上快速迭代，staging 更贴近量化形态，而 prod 则使用真正的 serving target。

### Ollama 的边界

Ollama 很适合开发，但不适合作为共享生产环境。Go HTTP serialization 会增加额外开销，并发管理也比 vLLM 简单，OpenTelemetry 支持也更弱。它适合“一人一机一条命令”的场景，真正进入共享服务后应切到 vLLM。

### 自托管和托管是另一道决策

Phase 17 · 01（managed hyperscalers）和 · 02（inference platforms）讨论的是托管方案。本课假设你已经决定走自托管。通常这样做的原因包括：数据驻留、定制 fine-tune、规模上的总体拥有成本，以及托管平台上没有你要的 domain model。

### 你应该记住的数字

- TGI maintenance mode：2025 年 12 月 11 日。
- vLLM v0.15.1：2026 年 2 月；支持 PyTorch 2.10 与 Blackwell SM120。
- SGLang 生产足迹：400,000+ GPUs。
- Ollama 相比 llama.cpp 的吞吐差距：大约慢 15-30%，在生产负载下可能差到 3x。

```figure
data-parallel
```

## 用起来

`code/main.py` 是一个决策树 walker：输入硬件 + 规模 + 工作负载，它会给出引擎选择并解释原因。

## 交付物

这一课会产出 `outputs/skill-engine-picker.md`。它会根据约束条件选出引擎，并写出迁移计划。

## 练习

1. 用你的硬件 / 规模 / 工作负载运行 `code/main.py`。输出和你的直觉一致吗？
2. 你的基础设施有 12 张 H100 和 8 张 AMD MI300X。该选什么引擎？为什么 TRT-LLM 不在候选里？
3. 团队想在 2026 年继续使用 TGI，因为“我们最熟悉”。请论证迁移的必要性。
4. 从 Ollama 开发切到 vLLM 生产，在量化、配置和 observability 上会发生哪些变化？
5. 一个 RAG 产品的 P99 prefix length 是 8K，而且跨租户存在大量复用。该选什么引擎？又该如何和 Phase 17 · 11 + 18 组合？

## 关键术语

| 术语 | 人们怎么说 | 它实际意味着什么 |
|------|----------------|------------------------|
| llama.cpp | “那个 CPU 版” | 模型支持最广，CPU 上速度最快 |
| Ollama | “笔记本那个” | 一键安装，但只有开发级吞吐 |
| TGI | “HF 的 serving” | 自 2025 年 12 月起处于 maintenance mode |
| vLLM | “默认选项” | 2026 年最广泛的生产基线 |
| SGLang | “agentic 那个” | 擅长 prefix-heavy 与 RadixAttention 场景 |
| TRT-LLM | “NVIDIA 专属” | Blackwell 吞吐冠军，但只能跑在 NVIDIA 上 |
| GGUF | “llama.cpp 格式” | 打包好的 K-quant 权重格式 |
| Production-stack | “vLLM K8s” | Phase 17 · 18 的参考部署模式 |
| 流水线模式 | “dev→stage→prod” | Ollama → llama.cpp → vLLM，不同阶段权重格式不同 |

## 延伸阅读

- [AI Made Tools — 2026：vLLM、Ollama、llama.cpp 与 TGI 对比](https://www.aimadetools.com/blog/vllm-vs-ollama-vs-llamacpp-vs-tgi/)
- [Morph — 2026 年的 llama.cpp 与 Ollama 对比](https://www.morphllm.com/comparisons/llama-cpp-vs-ollama)
- [n1n.ai — LLM 推理引擎综合对比](https://explore.n1n.ai/blog/llm-inference-engine-comparison-vllm-tgi-tensorrt-sglang-2026-03-13)
- [PremAI — 10 Best vLLM Alternatives 2026](https://blog.premai.io/10-best-vllm-alternatives-for-llm-inference-in-production-2026/)
- [TGI maintenance announcement](https://github.com/huggingface/text-generation-inference) — 维护模式公告发布说明。
- [vLLM v0.15.1 release notes](https://github.com/vllm-project/vllm/releases)
