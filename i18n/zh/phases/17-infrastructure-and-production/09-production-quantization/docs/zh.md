# 生产量化：AWQ、GPTQ、GGUF K-quants、FP8、MXFP4/NVFP4

> 量化格式不是一个放之四海而皆准的选择，它取决于硬件、serving engine 与 workload。GGUF Q4_K_M 或 Q5_K_M 主导 CPU 与边缘场景，通过 llama.cpp 与 Ollama 交付。若你在 vLLM 中需要同一底座模型挂多组 LoRA，GPTQ 会胜出。AWQ 配合 Marlin-AWQ kernels 在 7B 级模型上可做到约 741 tok/s，同时在 INT4 中给出最好的 Pass@1，是 2026 年数据中心生产环境的默认选项。FP8 则是 Hopper、Ada 与 Blackwell 上的中间地带：几乎无损，且支持广泛。NVFP4 与 MXFP4（Blackwell microscaling）更激进，必须逐块验证。团队最常踩的两个坑是：校准数据集必须匹配部署域；KV cache 与权重量化是两回事，AWQ 教程里那句“my model is 4 GB now”往往忘了生产批量下还要再吃掉 10-30 GB 的 KV cache。

**Type:** 学习
**Languages:** Python（标准库，玩具级跨格式内存与吞吐比较器）
**Prerequisites:** 第 10 阶段 · 13（量化基础），第 17 阶段 · 04（服务引擎内部原理）
**Time:** 约 75 分钟

## 学习目标

- 说出 2026 年六种生产量化格式及其最适合的使用场景。
- 在给定硬件（CPU vs GPU，Hopper vs Blackwell）、引擎（vLLM、TRT-LLM、llama.cpp）与工作负载（常规聊天、推理、多 LoRA）的前提下选出合适格式。
- 计算某个格式节省了多少权重内存，以及仍然原封不动保留下来的 KV cache 有多大。
- 指出会让量化模型在真实业务流量上退化的 calibration-dataset 坑点。

## 问题

量化会减少内存占用与 HBM 带宽需求，而这正是 decode 所依赖的资源。一个 FP16 70B 模型有 140 GB 权重。把权重量化到 INT4（AWQ 或 GPTQ）后，模型会缩到 35 GB，可以装进一张 H100，同时还能给 KV cache 留出空间。这一点非常关键，因为在 128 路并发、2k context 的条件下，仅 KV cache 就会吃掉 20-30 GB。

但量化不是免费的。激进量化会损伤质量，尤其是在重推理任务上。不同格式只适配某些引擎。不同硬件也原生支持不同精度。2026 年的格式动物园是真实存在的，你不能直接照抄别人的选择，必须根据自己的栈来选。

## 概念

### 六种格式

| 格式 | 位数 | 最适合的场景 | 引擎 |
|------|------|--------------|------|
| GGUF Q4_K_M / Q5_K_M | 4-5 | CPU、边缘设备、笔记本 | llama.cpp, Ollama |
| GPTQ | 4-8 | vLLM 上的多 LoRA | vLLM, TGI |
| AWQ | 4 | 数据中心 GPU 生产环境 | vLLM (Marlin-AWQ), TGI |
| FP8 | 8 | Hopper/Ada/Blackwell 数据中心 | vLLM, TRT-LLM, SGLang |
| MXFP4 | 4 | Blackwell 多用户场景 | TRT-LLM |
| NVFP4 | 4 | Blackwell 多用户场景 | TRT-LLM |

### GGUF：CPU/edge 默认选择

GGUF 是文件格式，而不完全是一种量化方案。它把 K-quant 变体（Q2_K、Q3_K_M、Q4_K_M、Q5_K_M、Q6_K、Q8_0）打包进同一个容器。Q4_K_M 与 Q5_K_M 是生产默认选项，在 4-5 bit 下接近 BF16 质量。它之所以是 CPU 或 edge serving 的最佳选择，是因为 llama.cpp 远远是最快的 CPU inference engine。

在 vLLM 中的吞吐惩罚约为：7B 模型只有 ~93 tok/s。这个格式并没有为 GPU kernels 做优化。只在部署目标是 CPU/edge 时使用 GGUF，其他情况不要用。

### GPTQ：vLLM 中的 multi-LoRA

GPTQ 是一种带 calibration pass 的 post-training quantization 算法。Marlin kernels 让它在 GPU 上足够快（相较非 Marlin GPTQ 约有 2.6x 提升）。7B 上大约是 ~712 tok/s。

它的独特优势在于：GPTQ-Int4 在 vLLM 中支持 LoRA adapters。如果你要服务一个 base model，再挂上 10-50 个 fine-tuned variants（每个都以 LoRA 形式提供），GPTQ 就是你的路。至少到 2026 年初，NVFP4 还不支持 LoRA。

### AWQ：数据中心 GPU 默认选择

Activation-aware Weight Quantization。在量化过程中，它会保护约 1% 最显著的权重。Marlin-AWQ kernels 可带来相较 naive 实现约 10.9x 的加速。在 7B 上约为 ~741 tok/s，并在 INT4 格式里取得最好的 Pass@1。

除非你需要 multi-LoRA（GPTQ）或更激进的 Blackwell FP4（NVFP4），否则为新的 GPU serving 直接选 AWQ。

### FP8：可靠的中间地带

8-bit floating point。几乎无损，支持广泛。Hopper Tensor Cores 可原生加速 FP8，Blackwell 也延续这一能力。当质量不可妥协时（推理、医疗、代码生成），FP8 是 2026 年的稳妥默认值。它的内存节省只有 INT4 的一半，但质量风险明显更低。

### MXFP4 / NVFP4：Blackwell 上的激进选项

Microscaling FP4。每个权重块都有独立 scale factor。它在 Blackwell Tensor Cores 上既激进又有硬件加速。相较 FP8，它把每 token 字节数再砍半，这正是 Phase 17 · 07 里提到的经济性收益。

注意事项：
- 尚不支持 LoRA（2026 年初）。
- 在重推理 workload 上会出现可见质量下降。
- 必须基于你自己的 eval set 逐模型验证。

### 校准陷阱

AWQ 与 GPTQ 都需要一个 calibration dataset，常见选择是 C4 或 WikiText。对于代码、医疗、法律这类 domain model，如果你用通用网页文本做校准，算法就可能错误判断哪些权重最该保护。HumanEval 上的 Pass@1 可能会掉好几个点。

解决办法是：用领域内数据做 calibration。通常几百条领域样本就够了。真正 ship 之前，先在 eval set 上测。

### KV cache 陷阱

AWQ 会把权重缩到 4 bit。KV cache 是独立的，仍然保留在 FP16/FP8。以一个采用 AWQ 的 70B 模型为例：

- Weights: ~35 GB（从 140 GB 压到 INT4 后的结果）
- 在 128 并发 × 2k context 下的 KV cache: ~20 GB
- Activations: ~5 GB
- Total: ~60 GB，可装进 H100 80GB。

天真地说“我把模型量化到 4 GB 了”，其实是把另外 30-50 GB 完全忘了。规划容量时，必须从整体上核算 HBM。

另外，KV cache quantization（FP8 KV 或 INT8 KV）是另一套独立选择，也有自己的一组 tradeoffs。它直接影响 attention accuracy，并不是白捡的收益。

### AWQ INT4 对推理任务有风险

Chain-of-thought、数学、长上下文代码生成，这些任务都会明显受到激进量化影响。AWQ INT4 在 MATH 上会掉约 3-5 分。对于推理密集型 workload，应当交付 FP8 或 BF16，并接受相应的内存成本。

### 2026 选型指南

- CPU/edge 场景：GGUF Q4_K_M。直接选它。
- GPU 服务、常规聊天、无 LoRA：AWQ。
- GPU 服务、多 LoRA：带 Marlin 的 GPTQ。
- 推理密集型 workload：FP8。
- Blackwell 数据中心，且质量已验证：NVFP4 + FP8 KV。
- 如果拿不准：对每个候选格式都跑一次 1,000 样本评估。

```figure
gpu-memory-breakdown
```

## 动手用

`code/main.py` 会计算一组模型尺寸在六种格式下的内存占用（weights + KV + activations）与相对吞吐。它会展示 KV cache 何时成为主导、权重压缩何时真正划算，以及 FP8 为什么是安全选择。

## 交付物

本课产出 `outputs/skill-quantization-picker.md`。给定硬件、模型尺寸、workload 类型与质量容忍度，它会选出一种格式，并生成对应的 calibration/validation 计划。

## 练习

1. 运行 `code/main.py`。对一个 70B、128 并发、2k context 的模型，计算每种格式的总 HBM。哪一种格式能让你装进一张 H100 80GB？
2. 你有一个 7B coding model。选一种格式并说明理由。如果你对质量容忍度判断错了，恢复路径是什么？
3. 计算一个 medical domain model 做 AWQ calibration 所需的数据集规模。为什么数据更多并不总是更好？
4. 阅读 Marlin-AWQ kernel 的论文或 release notes。用三句话解释为什么 AWQ 在 7B 上可以做到 741 tok/s，而原始 GPTQ 只有 ~712。
5. 在什么情况下，把 AWQ weights 与 FP8 KV cache 组合起来是合理的？又什么时候应该把 KV 继续保留在 BF16？

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------|----------|
| GGUF | “llama.cpp 格式” | 打包 K-quant 变体的文件格式；CPU/edge 默认选择 |
| Q4_K_M | “Q4 K M” | 4-bit K-quant medium；GGUF 的生产默认 |
| GPTQ | “gee pee tee q” | 带 calibration 的 post-train INT4；在 vLLM 中支持 LoRA |
| AWQ | “a w q” | Activation-aware INT4；配合 Marlin kernels；INT4 中 Pass@1 最佳 |
| Marlin kernels | “快速 INT4 kernels” | 为 Hopper 上 INT4 定制的 CUDA kernels；可带来约 10x 加速 |
| FP8 | “8 位浮点” | Hopper/Ada/Blackwell 上安全的默认精度 |
| MXFP4 / NVFP4 | “微缩 4 位浮点” | Blackwell 上带逐块 scale factor 的 4-bit FP |
| Calibration dataset | “校准数据” | 用于选择量化参数的输入文本；必须与业务域匹配 |
| KV cache quantization | “KV INT8” | 与权重量化独立的选择；会影响 attention accuracy |

## 延伸阅读

- [VRLA Tech — LLM Quantization 2026](https://vrlatech.com/llm-quantization-explained-int4-int8-fp8-awq-and-gptq-in-2026/) — 对比式 benchmark 汇总。
- [Jarvis Labs — vLLM 量化完整指南](https://jarvislabs.ai/blog/vllm-quantization-complete-guide-benchmarks) — 按格式整理的吞吐数字。
- [PremAI — GGUF vs AWQ vs GPTQ vs bitsandbytes 2026](https://blog.premai.io/llm-quantization-guide-gguf-vs-awq-vs-gptq-vs-bitsandbytes-compared-2026/) — 格式选型对比。
- [vLLM docs — Quantization](https://docs.vllm.ai/en/latest/features/quantization/index.html) — 支持的格式与参数开关。
- [AWQ paper (arXiv:2306.00978)](https://arxiv.org/abs/2306.00978) — AWQ 的原始论文。
- [GPTQ paper (arXiv:2210.17323)](https://arxiv.org/abs/2210.17323) — GPTQ 的原始论文。
