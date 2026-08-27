# 水印技术：SynthID、Stable Signature、C2PA

> 到了 2026 年，AI 生成内容的 provenance 基本由三类技术共同构成。SynthID（Google DeepMind）最早在 2023 年 8 月用于图像水印，2024 年 5 月扩展到文本与视频（Gemini + Veo），2024 年 10 月又通过 Responsible GenAI Toolkit 开源了文本版本，并在 2025 年 11 月随着 Gemini 3 Pro 推出统一的多媒体检测器。文本水印的做法是以人眼几乎察觉不到的方式微调 next-token sampling probabilities；图像和视频水印则要尽量在 compression、cropping、filters、frame-rate changes 之后仍能保留下来。Stable Signature（Fernandez et al., ICCV 2023, arXiv:2303.15435）则是另一条路线，它通过微调 latent diffusion decoder，让每张输出图像都包含固定消息；在只保留 10% 内容的裁剪图像上，检测率仍可超过 90%，同时将 FPR 控制在 1e-6 以下。随后 2024 年 5 月的 “Stable Signature is Unstable”（arXiv:2405.07145）又表明，只要进一步 fine-tuning，就能在基本不损伤图像质量的前提下移除该水印。C2PA 则是加密签名、可篡改留痕的元数据标准（C2PA 2.2 Explainer 2025）。水印与 C2PA 不是替代关系，而是互补关系：metadata 可能被剥离，但能承载更丰富的 provenance；水印则更容易穿越转码流程，但携带的信息量有限。

**Type:** 构建
**Languages:** Python (stdlib, token-watermark embed + detect)
**Prerequisites:** 阶段 10 · 04（采样）、阶段 01 · 09（信息论）
**Time:** 约 75 分钟

## 学习目标

- 描述 token-level watermarking，也就是 SynthID-text 风格文本水印，以及它为什么可以被检测出来。
- 描述 Stable Signature，以及 2024 年将其破坏掉的 removal attack。
- 说明 C2PA 的角色，以及为什么它与 watermarking 是互补关系。
- 描述几个关键限制：model-specific signal、paraphrase 下的脆弱性，以及 meaning-preserving attacks（arXiv:2508.20228）。

## 问题

从 2023 到 2024 年，deepfakes 与 AI-generated content 已经大规模进入政治传播和消费场景。水印因此被提出为一种技术性的 provenance signal：在内容生成时就打上标记，之后再由检测器识别。到 2025 年的证据表明，没有任何水印方案能做到无条件稳健，但如果把水印和 C2PA metadata 叠加使用，已经足以构成一套可用的 provenance 方案。

## 概念

### 文本水印（SynthID-text style）

Kirchenbauer et al. 2023 提出的机制，后来被 Google 工程化落地：

1. 在每一个 decoding step，对前 K 个 tokens 做哈希，从而把词表伪随机地划分成 “green” 和 “red” 两个集合。
2. 通过给 green logits 加上 δ，让采样过程更偏向 green set。
3. 于是最终生成文本里，green tokens 的数量会多于随机情形下的期望值。

检测时，需要重新对每个前缀做哈希，统计生成结果中 green tokens 的数量，再计算 z-score。对带水印文本来说，z-score 会大于 0；对人类自然文本来说，它通常接近 0。

这种方案的性质是：
- 对读者几乎不可感知，因为 δ 足够小，质量损失有限。
- 只要能访问词表划分函数，就能做检测。
- 不抗 paraphrase，一旦内容被改写，信号就会被破坏。

SynthID-text 已在 2024 年 10 月通过 Google 的 Responsible GenAI Toolkit 开源。

### Stable Signature（图像）

Fernandez et al. 在 ICCV 2023 提出的方案。其思路是微调 latent diffusion decoder，让每一张生成图像的 latent representation 中都嵌入固定的二进制消息。检测时再用一个 neural decoder 从 latent 中把这段消息读出来。即便只保留原图 10% 的内容，检测率仍能超过 90%，且 FPR<1e-6。

但 2024 年 5 月的 “Stable Signature is Unstable”（arXiv:2405.07145）表明，只要再对 decoder 进行 fine-tuning，就可以把水印移除，同时基本保持图像质量不变。也就是说，这类水印在对抗性场景下的稳健性其实是有限的，因为 post-generation fine-tuning 的成本并不高。

### SynthID unified detector（2025 年 11 月）

随着 Gemini 3 Pro 一起发布的，是一个统一的多媒体检测器。它可以在同一套 API 中读取文本、图像、音频、视频里的 SynthID 信号，从而把 Google 的 provenance stack 串成一体。

### C2PA

C2PA 的全称是 Coalition for Content Provenance and Authenticity。它是一套加密签名、可篡改留痕的 metadata 标准。根据 C2PA 2.2 Explainer（2025），一份 C2PA manifest 会记录 provenance claims，例如谁创建了内容、什么时候创建、经历了哪些变换，并由创建者的密钥签名。

它与 watermarking 的互补关系体现在：
- Metadata 可能被剥离；水印则不那么容易被剥离。
- Metadata 信息量丰富，可以携带完整 provenance chain；水印通常只能携带少量 bits。
- C2PA 依赖平台采用；水印则在生成时自动嵌入。

Google 已经在 Search、Ads 和 “About this image” 里同时使用这两层能力。

### 局限性

- **Model-specific。** SynthID 只能给启用了 SynthID 的模型输出打标。一个没有 SynthID 的模型生成出来的内容天然就没有该信号，因此 “no SynthID signal” 不能被当作真实性证明。
- **Paraphrase。** 文本水印不具备在 meaning-preserving paraphrase 下持续存在的能力。
- **Transformation attacks。** arXiv:2508.20228（2025）表明，一些保持语义不变的攻击可以同时摧毁文本水印以及许多图像水印。
- **Fine-tune removal。** 正如 “Stable Signature is Unstable” 所展示的，后续 fine-tuning 可以移除嵌入式图像水印。

### EU AI Act Article 50

欧盟关于 AI-generated content labeling 的透明度守则仍在起草中。根据 [European Commission status page](https://digital-strategy.ec.europa.eu/en/policies/code-practice-ai-generated-content)，第一版草案发布于 2025 年 12 月，第二版草案发布于 2026 年 3 月，最终稿预计在 2026 年 6 月完成。截至 2026 年 4 月，这套 Code 仍然处于草案阶段，时间表也可能继续变化。这一层是要求技术层落地的监管层要求。Deepfakes 必须被标注。

### 它在 Phase 18 里的位置

Lessons 22-23 讨论的是模型发出来的内容本身，也就是 private data 与 provenance signal。Lesson 27 会继续进入 training-data governance。Lesson 24 则是要求这些技术措施存在的监管框架。

```figure
an-watermark-greenlist
```

## 用它

`code/main.py` 会构建一个玩具文本水印系统。token 被表示成 0..N-1 的整数；带水印的采样过程会偏向哈希定义出的 green set；检测器则计算 green-token z-score。你可以观察 1000-token 生成结果上的检测效果，观看 paraphrase 如何破坏信号，并测量人类文本上的 false-positive rate。

## 交付它

这一课会产出 `outputs/skill-provenance-audit.md`。给定一个声称具有 provenance 的内容部署，它会审计：是否存在水印机制、是否存在 C2PA signing chain、两者各自的 adversarial robustness，以及每种模态上的覆盖范围。

## 练习

1. 运行 `code/main.py`。分别报告带水印的 1000-token 生成结果与人类撰写文本的 z-score，并找出在 95% confidence threshold 下的 false-positive rate。

2. 实现一个 paraphrase attack，用同义词替换 30% 的 tokens。然后重新测量 z-score。

3. 阅读 Kirchenbauer et al. 2023 的第 6 节关于 robustness 的讨论。为什么文本水印会在 paraphrase 下失效，而图像水印却能在 cropping 下继续存活？

4. 设计一个同时使用 SynthID-text 和 C2PA metadata 的部署。描述消费者最终能看到的 provenance chain，并指出每个组成部分各自的一个 failure mode。

5. 2024 年的 “Stable Signature is Unstable” 结果表明，fine-tuning 可以移除图像水印。请设计一个限制此类攻击的部署控制，例如要求 fine-tuned checkpoints 必须是签名发布物。

## 关键术语

| 术语 | 人们通常怎么说 | 实际含义 |
|------|-----------------|------------------------|
| SynthID | “Google 的水印” | 跨模态的来源标记信号，覆盖文本、图像、音频和视频 |
| 词元水印（Token watermark） | “Kirchenbauer 风格” | 通过绿色词元的 z 分数检测、采用偏置采样生成的文本水印 |
| Stable Signature | “图像水印” | 通过微调解码器实现的水印；发表于 ICCV 2023 |
| C2PA | “元数据标准” | 经过密码学签名、能够显现篡改痕迹的来源元数据 |
| 改写鲁棒性（Paraphrase robustness） | “改写会破坏水印吗” | 文本水印抵抗改写的能力；目前仍然有限 |
| 微调移除（Fine-tune removal） | “对抗性去水印” | 通过微调解码器移除图像水印的攻击 |
| 跨模态检测器（Cross-modal detector） | “统一 SynthID” | 2025 年 11 月推出、覆盖多种模态的统一 API |

## 延伸阅读

- [Kirchenbauer et al. — A Watermark for Large Language Models (ICML 2023, arXiv:2301.10226)](https://arxiv.org/abs/2301.10226) — 词元水印机制
- [Fernandez et al. — Stable Signature (ICCV 2023, arXiv:2303.15435)](https://arxiv.org/abs/2303.15435) — 图像水印论文
- ["Stable Signature is Unstable" (arXiv:2405.07145)](https://arxiv.org/abs/2405.07145) — 水印移除攻击
- [Google DeepMind — SynthID](https://deepmind.google/models/synthid/) — 跨模态水印
- [C2PA 2.2 Explainer (2025)](https://c2pa.org/specifications/specifications/2.2/explainer/Explainer.html) — 元数据标准
