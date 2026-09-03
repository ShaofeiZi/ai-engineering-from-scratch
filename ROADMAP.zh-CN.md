# 路线图

用于跟踪每个阶段和课程进度的状态记录器。本文件中的状态符号会反馈到
网站（`site/build.js` 会将它们解析进 `site/data.js`）；请勿更改其形式。

预计总耗时：约 323 小时，按你自己的节奏推进。

**图例：** ✅ 已完成 &nbsp;·&nbsp; 🚧 进行中 &nbsp;·&nbsp; ⬚ 计划中

## 阶段 0：环境与工具 — ✅（约 14 小时）

| # | 课程 | 状态 | 预计 |
|---|--------|--------|------|
| 01 | 开发环境 | ✅ | ~75 min |
| 02 | Git 与协作 | ✅ | ~45 min |
| 03 | GPU 配置与云端 | ✅ | ~75 min |
| 04 | API 与密钥 | ✅ | ~75 min |
| 05 | Jupyter Notebook | ✅ | ~75 min |
| 06 | Python 环境 | ✅ | ~75 min |
| 07 | 面向 AI 的 Docker | ✅ | ~75 min |
| 08 | 编辑器配置 | ✅ | ~75 min |
| 09 | 数据管理 | ✅ | ~75 min |
| 10 | 终端与 Shell | ✅ | ~45 min |
| 11 | 面向 AI 的 Linux | ✅ | ~45 min |
| 12 | 调试与性能分析 | ✅ | ~75 min |

## 阶段 1：数学基础 — ✅（约 23 小时）

| # | 课程 | 状态 | 预计 |
|---|--------|--------|------|
| 01 | 线性代数直觉 | ✅ | ~45 min |
| 02 | 向量、矩阵与运算 | ✅ | ~75 min |
| 03 | 矩阵变换与特征值 | ✅ | ~75 min |
| 04 | 面向机器学习的微积分——导数与梯度 | ✅ | ~45 min |
| 05 | 链式法则与自动微分 | ✅ | ~75 min |
| 06 | 概率与分布 | ✅ | ~45 min |
| 07 | 贝叶斯定理与统计思维 | ✅ | ~75 min |
| 08 | 优化——梯度下降家族 | ✅ | ~75 min |
| 09 | 信息论——熵、KL 散度 | ✅ | ~45 min |
| 10 | 降维——PCA、t-SNE、UMAP | ✅ | ~75 min |
| 11 | 奇异值分解 | ✅ | ~75 min |
| 12 | 张量运算 | ✅ | ~75 min |
| 13 | 数值稳定性 | ✅ | ~45 min |
| 14 | 范数与距离 | ✅ | ~45 min |
| 15 | 面向机器学习的统计学 | ✅ | ~45 min |
| 16 | 采样方法 | ✅ | ~75 min |
| 17 | 线性方程组 | ✅ | ~75 min |
| 18 | 凸优化 | ✅ | ~75 min |
| 19 | 面向 AI 的复数 | ✅ | ~45 min |
| 20 | 傅里叶变换 | ✅ | ~75 min |
| 21 | 面向机器学习的图论 | ✅ | ~45 min |
| 22 | 随机过程 | ✅ | ~45 min |

## 阶段 2：机器学习基础 — ✅（约 21 小时）

| # | 课程 | 状态 | 预计 |
|---|--------|--------|------|
| 01 | 什么是机器学习——类型与分类 | ✅ | ~45 min |
| 02 | 从零实现线性回归 | ✅ | ~75 min |
| 03 | 逻辑回归与分类 | ✅ | ~75 min |
| 04 | 决策树与随机森林 | ✅ | ~75 min |
| 05 | 支持向量机 | ✅ | ~75 min |
| 06 | K 近邻与距离度量 | ✅ | ~75 min |
| 07 | 无监督学习——K-Means、DBSCAN | ✅ | ~75 min |
| 08 | 特征工程与特征选择 | ✅ | ~75 min |
| 09 | 模型评估——指标、交叉验证 | ✅ | ~75 min |
| 10 | 偏差、方差与学习曲线 | ✅ | ~45 min |
| 11 | 集成方法——Boosting、Bagging、Stacking | ✅ | ~75 min |
| 12 | 超参数调优与 AutoML | ✅ | ~75 min |
| 13 | 机器学习流水线与实验跟踪 | ✅ | ~75 min |
| 14 | 朴素贝叶斯——多项式、高斯、伯努利 | ✅ | ~75 min |
| 15 | 时间序列基础 | ✅ | ~45 min |
| 16 | 异常检测 | ✅ | ~75 min |
| 17 | 处理不平衡数据 | ✅ | ~75 min |
| 18 | 特征选择 | ✅ | ~75 min |

## 阶段 3：深度学习核心 — ✅（约 15 小时）

| # | 课程 | 状态 | 预计 |
|---|--------|--------|------|
| 01 | 感知机——一切的起点 | ✅ | ~45 min |
| 02 | 多层网络与前向传播 | ✅ | ~75 min |
| 03 | 从零实现反向传播 | ✅ | ~75 min |
| 04 | 激活函数——ReLU、Sigmoid、GELU 及原因 | ✅ | ~45 min |
| 05 | 损失函数——MSE、交叉熵、对比损失 | ✅ | ~45 min |
| 06 | 优化器——SGD、动量、Adam、AdamW | ✅ | ~75 min |
| 07 | 正则化——Dropout、权重衰减、BatchNorm | ✅ | ~75 min |
| 08 | 权重初始化与训练稳定性 | ✅ | ~45 min |
| 09 | 学习率调度与预热 | ✅ | ~45 min |
| 10 | 构建你自己的迷你框架 | ✅ | ~120 min |
| 11 | PyTorch 入门 | ✅ | ~75 min |
| 12 | JAX 入门 | ✅ | ~75 min |
| 13 | 神经网络调试 | ✅ | ~75 min |

## 阶段 4：计算机视觉 — ✅（约 27 小时）

| # | 课程 | 状态 | 预计 |
|---|--------|--------|------|
| 01 | 图像基础——像素、通道、色彩空间 | ✅ | ~45 min |
| 02 | 从零实现卷积 | ✅ | ~75 min |
| 03 | CNN——从 LeNet 到 ResNet | ✅ | ~75 min |
| 04 | 图像分类 | ✅ | ~75 min |
| 05 | 迁移学习与微调 | ✅ | ~75 min |
| 06 | 目标检测——从零实现 YOLO | ✅ | ~75 min |
| 07 | 语义分割——U-Net | ✅ | ~75 min |
| 08 | 实例分割——Mask R-CNN | ✅ | ~75 min |
| 09 | 图像生成——GAN | ✅ | ~75 min |
| 10 | 图像生成——扩散模型 | ✅ | ~75 min |
| 11 | Stable Diffusion——架构与微调 | ✅ | ~75 min |
| 12 | 视频理解——时序建模 | ✅ | ~45 min |
| 13 | 3D 视觉——点云、NeRF | ✅ | ~45 min |
| 14 | 视觉 Transformer（ViT） | ✅ | ~45 min |
| 15 | 实时视觉——端侧部署 | ✅ | ~75 min |
| 16 | 构建完整的视觉流水线 | ✅ | ~120 min |
| 17 | 自监督视觉——SimCLR、DINO、MAE | ✅ | ~75 min |
| 18 | 开放词汇视觉——CLIP | ✅ | ~45 min |
| 19 | OCR 与文档理解 | ✅ | ~45 min |
| 20 | 图像检索与度量学习 | ✅ | ~45 min |
| 21 | 关键点检测与姿态估计 | ✅ | ~45 min |
| 22 | 从零实现 3D 高斯泼溅 | ✅ | ~90 min |
| 23 | 扩散 Transformer 与整流流 | ✅ | ~75 min |
| 24 | SAM 3 与开放词汇分割 | ✅ | ~60 min |
| 25 | 视觉-语言模型（ViT-MLP-LLM） | ✅ | ~75 min |
| 26 | 单目深度与几何估计 | ✅ | ~60 min |
| 27 | 多目标跟踪与视频记忆 | ✅ | ~60 min |
| 28 | 世界模型与视频扩散 | ✅ | ~75 min |

## 阶段 5：NLP——从基础到进阶 — ✅（约 30 小时）

| # | 课程 | 状态 | 预计 |
|---|--------|--------|------|
| 01 | [文本处理——分词、词干提取、词形还原](phases/05-nlp-foundations-to-advanced/01-text-processing) | ✅ | ~45 min |
| 02 | [词袋模型、TF-IDF 与文本表示](phases/05-nlp-foundations-to-advanced/02-bag-of-words-tfidf) | ✅ | ~75 min |
| 03 | [词嵌入——从零实现 Word2Vec](phases/05-nlp-foundations-to-advanced/03-word-embeddings-word2vec) | ✅ | ~75 min |
| 04 | [GloVe、FastText 与子词嵌入](phases/05-nlp-foundations-to-advanced/04-glove-fasttext-subword) | ✅ | ~45 min |
| 05 | [情感分析](phases/05-nlp-foundations-to-advanced/05-sentiment-analysis) | ✅ | ~75 min |
| 06 | [命名实体识别（NER）](phases/05-nlp-foundations-to-advanced/06-named-entity-recognition) | ✅ | ~75 min |
| 07 | [词性标注与句法解析](phases/05-nlp-foundations-to-advanced/07-pos-tagging-parsing) | ✅ | ~45 min |
| 08 | [文本分类——用于文本的 CNN 与 RNN](phases/05-nlp-foundations-to-advanced/08-cnns-rnns-for-text) | ✅ | ~75 min |
| 09 | [序列到序列模型](phases/05-nlp-foundations-to-advanced/09-sequence-to-sequence) | ✅ | ~75 min |
| 10 | [注意力机制——突破性进展](phases/05-nlp-foundations-to-advanced/10-attention-mechanism) | ✅ | ~45 min |
| 11 | [机器翻译](phases/05-nlp-foundations-to-advanced/11-machine-translation) | ✅ | ~75 min |
| 12 | [文本摘要](phases/05-nlp-foundations-to-advanced/12-text-summarization) | ✅ | ~75 min |
| 13 | [问答系统](phases/05-nlp-foundations-to-advanced/13-question-answering) | ✅ | ~75 min |
| 14 | [信息检索与搜索](phases/05-nlp-foundations-to-advanced/14-information-retrieval-search) | ✅ | ~75 min |
| 15 | [主题建模——LDA、BERTopic](phases/05-nlp-foundations-to-advanced/15-topic-modeling) | ✅ | ~45 min |
| 16 | [文本生成——Transformer 之前的语言模型](phases/05-nlp-foundations-to-advanced/16-text-generation-pre-transformer) | ✅ | ~45 min |
| 17 | [聊天机器人——从规则到神经](phases/05-nlp-foundations-to-advanced/17-chatbots-rule-to-neural) | ✅ | ~75 min |
| 18 | [多语言 NLP](phases/05-nlp-foundations-to-advanced/18-multilingual-nlp) | ✅ | ~45 min |
| 19 | [子词分词——BPE、WordPiece、Unigram、SentencePiece](phases/05-nlp-foundations-to-advanced/19-subword-tokenization) | ✅ | ~60 min |
| 20 | [结构化输出与受限解码](phases/05-nlp-foundations-to-advanced/20-structured-outputs-constrained-decoding) | ✅ | ~60 min |
| 21 | [NLI 与文本蕴含](phases/05-nlp-foundations-to-advanced/21-nli-textual-entailment) | ✅ | ~60 min |
| 22 | [嵌入模型深入剖析](phases/05-nlp-foundations-to-advanced/22-embedding-models-deep-dive) | ✅ | ~60 min |
| 23 | [面向 RAG 的分块策略](phases/05-nlp-foundations-to-advanced/23-chunking-strategies-rag) | ✅ | ~60 min |
| 24 | [共指消解](phases/05-nlp-foundations-to-advanced/24-coreference-resolution) | ✅ | ~60 min |
| 25 | [实体链接与消歧](phases/05-nlp-foundations-to-advanced/25-entity-linking) | ✅ | ~60 min |
| 26 | [关系抽取与知识图谱构建](phases/05-nlp-foundations-to-advanced/26-relation-extraction-kg) | ✅ | ~60 min |
| 27 | [LLM 评估——RAGAS、DeepEval、G-Eval](phases/05-nlp-foundations-to-advanced/27-llm-evaluation-frameworks) | ✅ | ~75 min |
| 28 | [长上下文评估——NIAH、RULER、LongBench、MRCR](phases/05-nlp-foundations-to-advanced/28-long-context-evaluation) | ✅ | ~60 min |
| 29 | [对话状态跟踪](phases/05-nlp-foundations-to-advanced/29-dialogue-state-tracking) | ✅ | ~75 min |

## 阶段 6：语音与音频 — ✅（约 18 小时）

| # | 课程 | 状态 | 预计 |
|---|--------|--------|------|
| 01 | [音频基础——波形、采样、傅里叶变换](phases/06-speech-and-audio/01-audio-fundamentals) | ✅ | ~45 min |
| 02 | [频谱图、Mel 尺度与音频特征](phases/06-speech-and-audio/02-spectrograms-mel-features) | ✅ | ~45 min |
| 03 | [音频分类](phases/06-speech-and-audio/03-audio-classification) | ✅ | ~75 min |
| 04 | [语音识别（ASR）](phases/06-speech-and-audio/04-speech-recognition-asr) | ✅ | ~45 min |
| 05 | [Whisper——架构与微调](phases/06-speech-and-audio/05-whisper-architecture-finetuning) | ✅ | ~75 min |
| 06 | [说话人识别与验证](phases/06-speech-and-audio/06-speaker-recognition-verification) | ✅ | ~45 min |
| 07 | [文本转语音（TTS）](phases/06-speech-and-audio/07-text-to-speech) | ✅ | ~75 min |
| 08 | [声音克隆与语音转换](phases/06-speech-and-audio/08-voice-cloning-conversion) | ✅ | ~75 min |
| 09 | [音乐生成](phases/06-speech-and-audio/09-music-generation) | ✅ | ~75 min |
| 10 | [音频语言模型](phases/06-speech-and-audio/10-audio-language-models) | ✅ | ~45 min |
| 11 | [实时音频处理](phases/06-speech-and-audio/11-real-time-audio-processing) | ✅ | ~75 min |
| 12 | [构建语音助手流水线](phases/06-speech-and-audio/12-voice-assistant-pipeline) | ✅ | ~120 min |
| 13 | [神经音频编解码器——EnCodec、SNAC、Mimi、DAC](phases/06-speech-and-audio/13-neural-audio-codecs) | ✅ | ~60 min |
| 14 | [语音活动检测与轮次切换](phases/06-speech-and-audio/14-voice-activity-detection-turn-taking) | ✅ | ~45 min |
| 15 | [流式语音到语音——Moshi、Hibiki](phases/06-speech-and-audio/15-streaming-speech-to-speech-moshi-hibiki) | ✅ | ~75 min |
| 16 | [语音反欺骗与音频水印](phases/06-speech-and-audio/16-anti-spoofing-audio-watermarking) | ✅ | ~75 min |
| 17 | [音频评估——WER、MOS、MMAU、排行榜](phases/06-speech-and-audio/17-audio-evaluation-metrics) | ✅ | ~60 min |

## 阶段 7：Transformer 深入剖析 — ✅（约 14 小时）

| # | 课程 | 状态 | 预计 |
|---|--------|--------|------|
| 01 | [为什么需要 Transformer——RNN 的问题](phases/07-transformers-deep-dive/01-why-transformers) | ✅ | ~45 min |
| 02 | [从零实现自注意力](phases/07-transformers-deep-dive/02-self-attention-from-scratch) | ✅ | ~75 min |
| 03 | [多头注意力](phases/07-transformers-deep-dive/03-multi-head-attention) | ✅ | ~75 min |
| 04 | [位置编码——正弦、RoPE、ALiBi](phases/07-transformers-deep-dive/04-positional-encoding) | ✅ | ~45 min |
| 05 | [完整的 Transformer——编码器 + 解码器](phases/07-transformers-deep-dive/05-full-transformer) | ✅ | ~75 min |
| 06 | [BERT——掩码语言建模](phases/07-transformers-deep-dive/06-bert-masked-language-modeling) | ✅ | ~45 min |
| 07 | [GPT——因果语言建模](phases/07-transformers-deep-dive/07-gpt-causal-language-modeling) | ✅ | ~75 min |
| 08 | [T5、BART——编码器-解码器模型](phases/07-transformers-deep-dive/08-t5-bart-encoder-decoder) | ✅ | ~45 min |
| 09 | [视觉 Transformer（ViT）](phases/07-transformers-deep-dive/09-vision-transformers) | ✅ | ~45 min |
| 10 | [音频 Transformer——Whisper 架构](phases/07-transformers-deep-dive/10-audio-transformers-whisper) | ✅ | ~45 min |
| 11 | [混合专家（MoE）](phases/07-transformers-deep-dive/11-mixture-of-experts) | ✅ | ~45 min |
| 12 | [KV 缓存、Flash Attention 与推理优化](phases/07-transformers-deep-dive/12-kv-cache-flash-attention) | ✅ | ~75 min |
| 13 | [缩放定律](phases/07-transformers-deep-dive/13-scaling-laws) | ✅ | ~45 min |
| 14 | [从零构建 Transformer——毕业项目](phases/07-transformers-deep-dive/14-build-a-transformer-capstone) | ✅ | ~120 min |
| 15 | [注意力变体——滑动窗口、稀疏、差分](phases/07-transformers-deep-dive/15-attention-variants) | ✅ | ~60 min |
| 16 | [投机解码——草拟、验证、重复](phases/07-transformers-deep-dive/16-speculative-decoding) | ✅ | ~60 min |

## 阶段 8：生成式 AI — ✅（约 14 小时）

| # | 课程 | 状态 | 预计 |
|---|--------|--------|------|
| 01 | [生成式模型——分类与历史](phases/08-generative-ai/01-generative-models-taxonomy-history/) | ✅ | ~45 min |
| 02 | [自编码器与 VAE](phases/08-generative-ai/02-autoencoders-vae/) | ✅ | ~75 min |
| 03 | [GAN——生成器对抗判别器](phases/08-generative-ai/03-gans-generator-discriminator/) | ✅ | ~75 min |
| 04 | [条件 GAN 与 Pix2Pix](phases/08-generative-ai/04-conditional-gans-pix2pix/) | ✅ | ~75 min |
| 05 | [StyleGAN](phases/08-generative-ai/05-stylegan/) | ✅ | ~45 min |
| 06 | [扩散模型——从零实现 DDPM](phases/08-generative-ai/06-diffusion-ddpm-from-scratch/) | ✅ | ~75 min |
| 07 | [潜在扩散与 Stable Diffusion](phases/08-generative-ai/07-latent-diffusion-stable-diffusion/) | ✅ | ~75 min |
| 08 | [ControlNet、LoRA 与图像条件控制](phases/08-generative-ai/08-controlnet-lora-conditioning/) | ✅ | ~75 min |
| 09 | [图像修复、外绘与编辑](phases/08-generative-ai/09-inpainting-outpainting-editing/) | ✅ | ~75 min |
| 10 | [视频生成](phases/08-generative-ai/10-video-generation/) | ✅ | ~45 min |
| 11 | [音频生成](phases/08-generative-ai/11-audio-generation/) | ✅ | ~45 min |
| 12 | [3D 生成](phases/08-generative-ai/12-3d-generation/) | ✅ | ~45 min |
| 13 | [流匹配与整流流](phases/08-generative-ai/13-flow-matching-rectified-flows/) | ✅ | ~45 min |
| 14 | [评估——FID、CLIP Score、人类偏好](phases/08-generative-ai/14-evaluation-fid-clip-score/) | ✅ | ~45 min |
| 19 | [视觉自回归建模（VAR）：下一尺度预测](phases/08-generative-ai/19-visual-autoregressive-var) | ✅ | ~90 min |

## 阶段 9：强化学习 — ✅（约 13 小时）

| # | 课程 | 状态 | 预计 |
|---|--------|--------|------|
| 01 | MDP、状态、动作与奖励 | ✅ | ~45 min |
| 02 | 动态规划 | ✅ | ~75 min |
| 03 | 蒙特卡洛方法 | ✅ | ~75 min |
| 04 | 时间差分——Q-Learning、SARSA | ✅ | ~75 min |
| 05 | 深度 Q 网络（DQN） | ✅ | ~75 min |
| 06 | 策略梯度方法——REINFORCE | ✅ | ~75 min |
| 07 | Actor-Critic——A2C、A3C | ✅ | ~75 min |
| 08 | 近端策略优化（PPO） | ✅ | ~75 min |
| 09 | 奖励建模与 RLHF | ✅ | ~45 min |
| 10 | 多智能体强化学习 | ✅ | ~45 min |
| 11 | 仿真到真实迁移 | ✅ | ~45 min |
| 12 | 面向游戏的强化学习 | ✅ | ~75 min |

## 阶段 10：从零构建 LLM — ✅（约 26 小时）

| # | 课程 | 状态 | 预计 |
|---|--------|--------|------|
| 01 | [分词器——BPE、WordPiece、SentencePiece](phases/10-llms-from-scratch/01-tokenizers) | ✅ | ~45 min |
| 02 | [从零构建分词器](phases/10-llms-from-scratch/02-building-a-tokenizer) | ✅ | ~75 min |
| 03 | [预训练数据流水线](phases/10-llms-from-scratch/03-data-pipelines) | ✅ | ~75 min |
| 04 | [预训练迷你 GPT（124M）](phases/10-llms-from-scratch/04-pre-training-mini-gpt) | ✅ | ~120 min |
| 05 | [扩展——分布式训练、FSDP、DeepSpeed](phases/10-llms-from-scratch/05-scaling-distributed) | ✅ | ~75 min |
| 06 | [指令微调——SFT](phases/10-llms-from-scratch/06-instruction-tuning-sft) | ✅ | ~75 min |
| 07 | [RLHF——奖励模型 + PPO 训练](phases/10-llms-from-scratch/07-rlhf) | ✅ | ~75 min |
| 08 | [DPO——直接偏好优化](phases/10-llms-from-scratch/08-dpo) | ✅ | ~75 min |
| 09 | [宪法 AI 与自我改进](phases/10-llms-from-scratch/09-constitutional-ai-self-improvement) | ✅ | ~45 min |
| 10 | [评估——基准、评测、LM Harness](phases/10-llms-from-scratch/10-evaluation) | ✅ | ~75 min |
| 11 | [量化——INT8、GPTQ、AWQ、GGUF](phases/10-llms-from-scratch/11-quantization) | ✅ | ~75 min |
| 12 | [推理优化](phases/10-llms-from-scratch/12-inference-optimization) | ✅ | ~75 min |
| 13 | [构建完整的 LLM 流水线](phases/10-llms-from-scratch/13-building-complete-llm-pipeline) | ✅ | ~120 min |
| 14 | [开源模型——架构详解](phases/10-llms-from-scratch/14-open-models-architecture-walkthroughs) | ✅ | ~45 min |
| 15 | [投机解码与 EAGLE-3](phases/10-llms-from-scratch/15-speculative-decoding-eagle3) | ✅ | ~75 min |
| 16 | [差分注意力（V2）](phases/10-llms-from-scratch/16-differential-attention-v2) | ✅ | ~60 min |
| 17 | [原生稀疏注意力（DeepSeek NSA）](phases/10-llms-from-scratch/17-native-sparse-attention) | ✅ | ~60 min |
| 18 | [多 Token 预测（MTP）](phases/10-llms-from-scratch/18-multi-token-prediction) | ✅ | ~60 min |
| 19 | [DualPipe 并行](phases/10-llms-from-scratch/19-dualpipe-parallelism) | ✅ | ~60 min |
| 20 | [DeepSeek-V3 架构详解](phases/10-llms-from-scratch/20-deepseek-v3-walkthrough) | ✅ | ~75 min |
| 21 | [Jamba——混合 SSM-Transformer](phases/10-llms-from-scratch/21-jamba-hybrid-ssm-transformer) | ✅ | ~60 min |
| 22 | [异步与 Hogwild! 推理](phases/10-llms-from-scratch/22-async-hogwild-inference) | ✅ | ~60 min |
| 25 | [投机解码与 EAGLE](phases/10-llms-from-scratch/25-speculative-decoding) | ✅ | ~75 min |
| 34 | [梯度检查点与激活重计算](phases/10-llms-from-scratch/34-gradient-checkpointing) | ✅ | ~70 min |

## 阶段 11：LLM 工程 — ✅（约 17 小时）

| # | 课程 | 状态 | 预计 |
|---|--------|--------|------|
| 01 | [提示工程——技巧与模式](phases/11-llm-engineering/01-prompt-engineering) | ✅ | ~45 min |
| 02 | [少样本、思维链、思维树](phases/11-llm-engineering/02-few-shot-cot) | ✅ | ~45 min |
| 03 | [结构化输出](phases/11-llm-engineering/03-structured-outputs) | ✅ | ~75 min |
| 04 | [嵌入与向量表示](phases/11-llm-engineering/04-embeddings) | ✅ | ~75 min |
| 05 | [上下文工程](phases/11-llm-engineering/05-context-engineering) | ✅ | ~75 min |
| 06 | [RAG——检索增强生成](phases/11-llm-engineering/06-rag) | ✅ | ~75 min |
| 07 | [高级 RAG](phases/11-llm-engineering/07-advanced-rag) | ✅ | ~75 min |
| 08 | [使用 LoRA 与 QLoRA 微调](phases/11-llm-engineering/08-fine-tuning-lora) | ✅ | ~75 min |
| 09 | [函数调用与工具使用](phases/11-llm-engineering/09-function-calling) | ✅ | ~75 min |
| 10 | [LLM 应用的评估与测试](phases/11-llm-engineering/10-evaluation) | ✅ | ~45 min |
| 11 | [缓存、限流与成本优化](phases/11-llm-engineering/11-caching-cost) | ✅ | ~45 min |
| 12 | [护栏、安全与内容过滤](phases/11-llm-engineering/12-guardrails) | ✅ | ~45 min |
| 13 | [构建生产级 LLM 应用](phases/11-llm-engineering/13-production-app) | ✅ | ~120 min |
| 14 | [模型上下文协议（MCP）](phases/11-llm-engineering/14-model-context-protocol) | ✅ | ~75 min |
| 15 | [提示缓存与上下文缓存](phases/11-llm-engineering/15-prompt-caching) | ✅ | ~60 min |

## 阶段 12：多模态 AI — ✅（约 65 小时）

| # | 课程 | 状态 | 预计 |
|---|--------|--------|------|
| 01 | [视觉 Transformer 与 Patch-Token 原语](phases/12-multimodal-ai/01-vision-transformer-patch-tokens) | ✅ | ~120 min |
| 02 | [CLIP 与对比式视觉-语言预训练](phases/12-multimodal-ai/02-clip-contrastive-pretraining) | ✅ | ~180 min |
| 03 | [BLIP-2 与作为模态桥梁的 Q-Former](phases/12-multimodal-ai/03-blip2-qformer-bridge) | ✅ | ~180 min |
| 04 | [Flamingo 与门控交叉注意力](phases/12-multimodal-ai/04-flamingo-gated-cross-attention) | ✅ | ~120 min |
| 05 | [LLaVA 与视觉指令微调](phases/12-multimodal-ai/05-llava-visual-instruction-tuning) | ✅ | ~180 min |
| 06 | [任意分辨率视觉：Patch-n'-Pack 与 NaFlex](phases/12-multimodal-ai/06-any-resolution-patch-n-pack) | ✅ | ~120 min |
| 07 | [开源权重 VLM 配方：真正重要的因素](phases/12-multimodal-ai/07-open-weight-vlm-recipes) | ✅ | ~180 min |
| 08 | [LLaVA-OneVision：单图、多图、视频](phases/12-multimodal-ai/08-llava-onevision-single-multi-video) | ✅ | ~180 min |
| 09 | [Qwen-VL 家族与动态帧率视频](phases/12-multimodal-ai/09-qwen-vl-family-dynamic-fps) | ✅ | ~120 min |
| 10 | [InternVL3 原生多模态预训练](phases/12-multimodal-ai/10-internvl3-native-multimodal) | ✅ | ~120 min |
| 11 | [Chameleon 与早期融合的纯 Token 方案](phases/12-multimodal-ai/11-chameleon-early-fusion-tokens) | ✅ | ~180 min |
| 12 | [Emu3 用于生成的下一 Token 预测](phases/12-multimodal-ai/12-emu3-next-token-for-generation) | ✅ | ~120 min |
| 13 | [Transfusion 自回归 + 扩散](phases/12-multimodal-ai/13-transfusion-autoregressive-diffusion) | ✅ | ~180 min |
| 14 | [Show-o 与离散扩散统一](phases/12-multimodal-ai/14-show-o-discrete-diffusion-unified) | ✅ | ~120 min |
| 15 | [Janus-Pro 解耦编码器](phases/12-multimodal-ai/15-janus-pro-decoupled-encoders) | ✅ | ~120 min |
| 16 | [MIO 任意到任意流式](phases/12-multimodal-ai/16-mio-any-to-any-streaming) | ✅ | ~120 min |
| 17 | [视频-语言时序定位](phases/12-multimodal-ai/17-video-language-temporal-grounding) | ✅ | ~180 min |
| 18 | [百万 Token 上下文下的长视频理解](phases/12-multimodal-ai/18-long-video-million-token) | ✅ | ~180 min |
| 19 | [音频语言模型：从 Whisper 到 AF3](phases/12-multimodal-ai/19-audio-language-whisper-to-af3) | ✅ | ~180 min |
| 20 | [全能模型：Thinker-Talker](phases/12-multimodal-ai/20-omni-models-thinker-talker) | ✅ | ~180 min |
| 21 | [具身 VLA：RT-2、OpenVLA、π0、GR00T](phases/12-multimodal-ai/21-embodied-vlas-openvla-pi0-groot) | ✅ | ~180 min |
| 22 | [文档与图表理解](phases/12-multimodal-ai/22-document-diagram-understanding) | ✅ | ~180 min |
| 23 | [ColPali 视觉原生文档 RAG](phases/12-multimodal-ai/23-colpali-vision-native-rag) | ✅ | ~180 min |
| 24 | [多模态 RAG 与跨模态检索](phases/12-multimodal-ai/24-multimodal-rag-cross-modal) | ✅ | ~180 min |
| 25 | [多模态智能体与计算机使用（毕业项目）](phases/12-multimodal-ai/25-multimodal-agents-computer-use) | ✅ | ~240 min |

## 阶段 13：工具与协议 — ✅（约 43 小时）

| # | 课程 | 状态 | 预计 |
|---|--------|--------|------|
| 01 | [工具接口](phases/13-tools-and-protocols/01-the-tool-interface/) | ✅ | ~45 min |
| 02 | [函数调用深入剖析](phases/13-tools-and-protocols/02-function-calling-deep-dive/) | ✅ | ~75 min |
| 03 | [并行与流式工具调用](phases/13-tools-and-protocols/03-parallel-and-streaming-tool-calls/) | ✅ | ~75 min |
| 04 | [结构化输出](phases/13-tools-and-protocols/04-structured-output/) | ✅ | ~75 min |
| 05 | [工具模式设计](phases/13-tools-and-protocols/05-tool-schema-design/) | ✅ | ~45 min |
| 06 | [MCP 基础：无状态请求与 JSON-RPC](phases/13-tools-and-protocols/06-mcp-fundamentals/) | ✅ | ~55 min |
| 07 | [构建 MCP 服务器：无状态 Python 与 TypeScript](phases/13-tools-and-protocols/07-building-an-mcp-server/) | ✅ | ~85 min |
| 08 | [构建 MCP 客户端：发现、路由与双时代回退](phases/13-tools-and-protocols/08-building-an-mcp-client/) | ✅ | ~85 min |
| 09 | [MCP 传输：stdio 与无状态流式 HTTP](phases/13-tools-and-protocols/09-mcp-transports/) | ✅ | ~65 min |
| 10 | [MCP 资源与提示：面向无状态服务器的可寻址上下文](phases/13-tools-and-protocols/10-mcp-resources-and-prompts/) | ✅ | ~60 min |
| 11 | [MCP 模型输入：采样迁移与无状态 MRTR](phases/13-tools-and-protocols/11-mcp-sampling/) | ✅ | ~75 min |
| 12 | [显式作用域与无状态 elicitation](phases/13-tools-and-protocols/12-mcp-roots-and-elicitation/) | ✅ | ~60 min |
| 13 | [MCP 任务扩展：在无状态核心上承载持久工作](phases/13-tools-and-protocols/13-mcp-async-tasks/) | ✅ | ~90 min |
| 14 | [无状态协议上的 MCP 应用](phases/13-tools-and-protocols/14-mcp-apps/) | ✅ | ~75 min |
| 15 | [MCP 安全：投毒元数据、路由与 MRTR 状态](phases/13-tools-and-protocols/15-mcp-security-tool-poisoning/) | ✅ | ~60 min |
| 16 | [MCP 授权：CIMD、签发方绑定、PKCE 与逐步升级](phases/13-tools-and-protocols/16-mcp-security-oauth-2-1/) | ✅ | ~90 min |
| 17 | [无状态 MCP 网关与注册中心准入](phases/13-tools-and-protocols/17-mcp-gateways-and-registries/) | ✅ | ~75 min |
| 18 | [生产中的 MCP 鉴权：签发方绑定的注册与令牌](phases/13-tools-and-protocols/18-mcp-auth-production/) | ✅ | ~90 min |
| 19 | [A2A 协议](phases/13-tools-and-protocols/19-a2a-protocol/) | ✅ | ~75 min |
| 20 | [OpenTelemetry GenAI](phases/13-tools-and-protocols/20-opentelemetry-genai/) | ✅ | ~75 min |
| 21 | [LLM 路由层](phases/13-tools-and-protocols/21-llm-routing-layer/) | ✅ | ~45 min |
| 22 | [智能体技能：可移植契约与运行时边界](phases/13-tools-and-protocols/22-skills-and-agent-sdks/) | ✅ | ~90 min |
| 23 | [毕业项目：无状态工具生态](phases/13-tools-and-protocols/23-capstone-tool-ecosystem/) | ✅ | ~120 min |
| 24 | [技能发现与渐进式披露](phases/13-tools-and-protocols/24-skill-discovery-and-progressive-disclosure/) | ✅ | ~105 min |
| 25 | [技能调用与路由](phases/13-tools-and-protocols/25-skill-invocation-and-routing/) | ✅ | ~105 min |
| 26 | [技能权限、沙箱与信任](phases/13-tools-and-protocols/26-skill-permissions-sandboxes-and-trust/) | ✅ | ~120 min |
| 27 | [技能评估、打包与可移植性](phases/13-tools-and-protocols/27-skill-evals-packaging-and-portability/) | ✅ | ~150 min |
| 28 | [MCP 工具契约与内容](phases/13-tools-and-protocols/28-mcp-tool-contracts-and-content/) | ✅ | ~120 min |
| 29 | [MCP 可靠性、取消与流控](phases/13-tools-and-protocols/29-mcp-reliability-cancellation-and-flow-control/) | ✅ | ~120 min |
| 30 | [MCP 注册中心供应链：准入、漂移与回滚](phases/13-tools-and-protocols/30-mcp-registry-supply-chain-and-drift/) | ✅ | ~90 min |
| 31 | [MCP 一致性工程：版本、证据与运维](phases/13-tools-and-protocols/31-mcp-conformance-versioning-and-operations/) | ✅ | ~100 min |

## 阶段 14：智能体工程 — ✅（约 55 小时）

| # | 课程 | 状态 | 预计 |
|---|--------|--------|------|
| 01 | 智能体循环 | ✅ | ~60 min |
| 02 | ReWOO 与规划-执行 | ✅ | ~60 min |
| 03 | Reflexion 与言语强化学习 | ✅ | ~60 min |
| 04 | 思维树与 LATS | ✅ | ~75 min |
| 05 | Self-Refine 与 CRITIC | ✅ | ~60 min |
| 06 | 工具使用与函数调用 | ✅ | ~60 min |
| 07 | 智能体记忆——虚拟上下文与内存分页 | ✅ | ~75 min |
| 08 | 记忆块与睡眠时计算 | ✅ | ~75 min |
| 09 | 混合记忆——向量 + 图 + KV | ✅ | ~75 min |
| 10 | 技能库与终身学习（Voyager） | ✅ | ~75 min |
| 11 | 基于 HTN 与进化搜索的规划 | ✅ | ~75 min |
| 12 | Anthropic 的工作流模式 | ✅ | ~60 min |
| 13 | 有状态图编排——持久执行与检查点 | ✅ | ~75 min |
| 14 | 面向智能体的 Actor 模型 | ✅ | ~75 min |
| 15 | 基于角色的智能体团队——角色、任务、流程 | ✅ | ~60 min |
| 16 | OpenAI Agents SDK——交接、护栏、追踪 | ✅ | ~75 min |
| 17 | 作为库的 Harness——子智能体与会话存储 | ✅ | ~75 min |
| 18 | 生产级智能体运行时 | ✅ | ~45 min |
| 19 | 基准——SWE-bench、GAIA、AgentBench | ✅ | ~60 min |
| 20 | 基准——WebArena 与 OSWorld | ✅ | ~60 min |
| 21 | 计算机使用——Claude、OpenAI CUA、Gemini | ✅ | ~60 min |
| 22 | 语音智能体——Pipecat 与 LiveKit | ✅ | ~60 min |
| 23 | OpenTelemetry GenAI 语义约定 | ✅ | ~60 min |
| 24 | 智能体可观测性——Langfuse、Phoenix、Opik | ✅ | ~45 min |
| 25 | 多智能体辩论与协作 | ✅ | ~60 min |
| 26 | 失效模式——智能体为何崩溃 | ✅ | ~60 min |
| 27 | 提示注入与 PVE 防御 | ✅ | ~75 min |
| 28 | 编排模式——监督者、群体、分层 | ✅ | ~60 min |
| 29 | 生产运行时——队列、事件、定时 | ✅ | ~60 min |
| 30 | 评估驱动的智能体开发 | ✅ | ~60 min |
| 31 | 智能体工作台：能力强的模型为何仍会失败 | ✅ | ~45 min |
| 32 | 最小化智能体工作台 | ✅ | ~45 min |
| 33 | 作为可执行约束的智能体指令 | ✅ | ~50 min |
| 34 | 仓库记忆与持久状态 | ✅ | ~60 min |
| 35 | 面向智能体的初始化脚本 | ✅ | ~45 min |
| 36 | 作用域契约与任务边界 | ✅ | ~50 min |
| 37 | 运行时反馈循环 | ✅ | ~50 min |
| 38 | 验证门 | ✅ | ~55 min |
| 39 | 评审智能体：将构建者与判分者分离 | ✅ | ~55 min |
| 40 | 多会话交接 | ✅ | ~50 min |
| 41 | 真实仓库上的工作台 | ✅ | ~60 min |
| 42 | 毕业项目：发布可复用的智能体工作台包 | ✅ | ~75 min |
| 43 | 在智能体写代码之前先界定任务 | ✅ | ~60 min |
| 44 | 构建有证据支撑的执行计划 | ✅ | ~65 min |
| 45 | 以隔离与合并契约委派智能体工作 | ✅ | ~70 min |
| 46 | 把每次智能体纠正转化为系统改进 | ✅ | ~65 min |
| 47 | 在选择输出之前先定义结果 | ✅ | ~60 min |
| 48 | 发现人们实际执行的工作流 | ✅ | ~70 min |
| 49 | 梳理假设并先解决风险最高的一个 | ✅ | ~65 min |
| 50 | 选择能改变决策的最小切片 | ✅ | ~65 min |
| 51 | 编写保留判断力的规格说明 | ✅ | ~75 min |
| 52 | 在结果出现之前设计成功指标 | ✅ | ~70 min |
| 53 | 有意识地选择原型、试点或生产 | ✅ | ~70 min |
| 54 | 构建带所有权与退役机制的反馈棘轮 | ✅ | ~75 min |

## 阶段 15：自主系统 — ✅（约 20 小时）

| # | 课程 | 状态 | 预计 |
|---|--------|--------|------|
| 01 | 从聊天机器人到长程智能体（METR） | ✅ | ~45 min |
| 02 | STaR、V-STaR、Quiet-STaR——自学习推理 | ✅ | ~60 min |
| 03 | AlphaEvolve——进化式编码智能体 | ✅ | ~60 min |
| 04 | Darwin Gödel Machine——自我修改的智能体 | ✅ | ~60 min |
| 05 | AI Scientist v2——工作坊级研究 | ✅ | ~60 min |
| 06 | 自动化对齐研究（Anthropic AAR） | ✅ | ~60 min |
| 07 | 递归自我改进——能力 vs 对齐 | ✅ | ~60 min |
| 08 | 有界自我改进设计 | ✅ | ~60 min |
| 09 | 自主编码智能体全景（SWE-bench、CodeAct） | ✅ | ~45 min |
| 10 | 面向自主智能体的权限模式 | ✅ | ~45 min |
| 11 | 浏览器智能体与间接提示注入 | ✅ | ~45 min |
| 12 | 面向长时间运行智能体的持久执行 | ✅ | ~60 min |
| 13 | 动作预算、迭代上限与成本调控 | ✅ | ~60 min |
| 14 | 终止开关、熔断器、金丝雀令牌 | ✅ | ~60 min |
| 15 | HITL——提议后提交 | ✅ | ~60 min |
| 16 | 检查点与回滚 | ✅ | ~60 min |
| 17 | 宪法 AI 与规则覆盖 | ✅ | ~60 min |
| 18 | Llama Guard 与输入/输出分类 | ✅ | ~45 min |
| 19 | Anthropic 负责任扩展政策 v3.0 | ✅ | ~45 min |
| 20 | OpenAI 防备框架与 DeepMind FSF | ✅ | ~45 min |
| 21 | METR 时间跨度与外部评估 | ✅ | ~60 min |
| 22 | CAIS、CAISI 与社会级风险 | ✅ | ~45 min |

## 阶段 16：多智能体与群体 — ✅（约 28 小时）

| # | 课程 | 状态 | 预计 |
|---|--------|--------|------|
| 01 | [为什么需要多智能体](phases/16-multi-agent-and-swarms/01-why-multi-agent/) | ✅ | ~45 min |
| 02 | [FIPA-ACL 渊源与言语行为](phases/16-multi-agent-and-swarms/02-fipa-acl-heritage/) | ✅ | ~60 min |
| 03 | [通信协议](phases/16-multi-agent-and-swarms/03-communication-protocols/) | ✅ | ~45 min |
| 04 | [多智能体原语模型](phases/16-multi-agent-and-swarms/04-primitive-model/) | ✅ | ~60 min |
| 05 | [监督者 / 编排者-工作者模式](phases/16-multi-agent-and-swarms/05-supervisor-orchestrator-pattern/) | ✅ | ~75 min |
| 06 | [分层架构与分解漂移](phases/16-multi-agent-and-swarms/06-hierarchical-architecture/) | ✅ | ~60 min |
| 07 | [心智社会与多智能体辩论](phases/16-multi-agent-and-swarms/07-society-of-mind-debate/) | ✅ | ~75 min |
| 08 | [角色专精——规划者 / 评审者 / 执行者 / 验证者](phases/16-multi-agent-and-swarms/08-role-specialization/) | ✅ | ~75 min |
| 09 | [并行群体与网络化架构](phases/16-multi-agent-and-swarms/09-parallel-swarm-networks/) | ✅ | ~60 min |
| 10 | [群聊与发言者选择](phases/16-multi-agent-and-swarms/10-group-chat-speaker-selection/) | ✅ | ~60 min |
| 11 | [交接与例程（无状态编排）](phases/16-multi-agent-and-swarms/11-handoffs-and-routines/) | ✅ | ~60 min |
| 12 | [A2A——智能体到智能体协议](phases/16-multi-agent-and-swarms/12-a2a-protocol/) | ✅ | ~75 min |
| 13 | [共享记忆与黑板模式](phases/16-multi-agent-and-swarms/13-shared-memory-blackboard/) | ✅ | ~75 min |
| 14 | [面向智能体的共识与拜占庭容错](phases/16-multi-agent-and-swarms/14-consensus-and-bft/) | ✅ | ~75 min |
| 15 | [投票、自一致与辩论拓扑](phases/16-multi-agent-and-swarms/15-voting-debate-topology/) | ✅ | ~75 min |
| 16 | [谈判与讨价还价](phases/16-multi-agent-and-swarms/16-negotiation-bargaining/) | ✅ | ~75 min |
| 17 | [生成式智能体与涌现式模拟](phases/16-multi-agent-and-swarms/17-generative-agents-simulation/) | ✅ | ~75 min |
| 18 | [心智理论与涌现式协调](phases/16-multi-agent-and-swarms/18-theory-of-mind-coordination/) | ✅ | ~75 min |
| 19 | [面向 LLM 的群体优化（PSO、ACO）](phases/16-multi-agent-and-swarms/19-swarm-optimization-pso-aco/) | ✅ | ~75 min |
| 20 | [MARL——MADDPG、QMIX、MAPPO](phases/16-multi-agent-and-swarms/20-marl-maddpg-qmix-mappo/) | ✅ | ~90 min |
| 21 | [智能体经济、Token 激励、声誉](phases/16-multi-agent-and-swarms/21-agent-economies/) | ✅ | ~75 min |
| 22 | [生产扩展——队列、检查点、持久化](phases/16-multi-agent-and-swarms/22-production-scaling-queues-checkpoints/) | ✅ | ~75 min |
| 23 | [失效模式——MAST、群体思维、单一文化、级联](phases/16-multi-agent-and-swarms/23-failure-modes-mast-groupthink/) | ✅ | ~75 min |
| 24 | [评估与协调基准](phases/16-multi-agent-and-swarms/24-evaluation-coordination-benchmarks/) | ✅ | ~75 min |
| 25 | [案例研究与 2026 年前沿水平](phases/16-multi-agent-and-swarms/25-case-studies-2026-sota/) | ✅ | ~90 min |

## 阶段 17：基础设施与生产 — ✅（约 32 小时）

| # | 课程 | 状态 | 预计 |
|---|--------|--------|------|
| 01 | 托管式 LLM 平台——Bedrock、Azure OpenAI、Vertex AI | ✅ | ~60 min |
| 02 | 推理平台经济学——Fireworks、Together、Baseten、Modal | ✅ | ~60 min |
| 03 | Kubernetes 上的 GPU 自动伸缩——Karpenter、KAI Scheduler | ✅ | ~75 min |
| 04 | 推理引擎内部——PagedAttention、连续批处理、分块预填充 | ✅ | ~75 min |
| 05 | 生产中的 EAGLE-3 投机解码 | ✅ | ~60 min |
| 06 | 前缀缓存服务——RadixAttention 与 KV 复用 | ✅ | ~60 min |
| 07 | 硬件专用推理编译——Blackwell 上的 FP8 与 NVFP4 | ✅ | ~75 min |
| 08 | 推理指标——TTFT、TPOT、ITL、Goodput、P99 | ✅ | ~60 min |
| 09 | 生产级量化——AWQ、GPTQ、GGUF、FP8、NVFP4 | ✅ | ~75 min |
| 10 | Serverless LLM 的冷启动缓解 | ✅ | ~60 min |
| 11 | 多区域 LLM 服务与 KV 缓存局部性 | ✅ | ~60 min |
| 12 | 边缘推理——ANE、Hexagon、WebGPU、Jetson | ✅ | ~60 min |
| 13 | LLM 可观测性栈选型 | ✅ | ~60 min |
| 14 | 提示缓存与语义缓存的经济学 | ✅ | ~60 min |
| 15 | 批处理 API——50% 折扣成为行业标准 | ✅ | ~45 min |
| 16 | 作为降本原语的模型路由 | ✅ | ~60 min |
| 17 | 预填充/解码分离——NVIDIA Dynamo 与 llm-d | ✅ | ~75 min |
| 18 | 生产服务栈——KV 卸载与缓存感知路由 | ✅ | ~60 min |
| 19 | AI 网关——LiteLLM、Portkey、Kong、Bifrost | ✅ | ~60 min |
| 20 | 影子、金丝雀与渐进式部署 | ✅ | ~60 min |
| 21 | A/B 测试 LLM 功能——GrowthBook 与 Statsig | ✅ | ~60 min |
| 22 | LLM API 负载测试——k6、LLMPerf、GenAI-Perf | ✅ | ~75 min |
| 23 | 面向 AI 的 SRE——多智能体事件响应 | ✅ | ~60 min |
| 24 | LLM 生产的混沌工程 | ✅ | ~60 min |
| 25 | 安全——密钥、PII 清洗、审计日志 | ✅ | ~60 min |
| 26 | 合规——SOC 2、HIPAA、GDPR、欧盟 AI 法案、ISO 42001 | ✅ | ~60 min |
| 27 | 面向 LLM 的 FinOps——单位经济与多租户归因 | ✅ | ~60 min |
| 28 | 自托管服务选型——让引擎匹配硬件与规模 | ✅ | ~45 min |

## 阶段 18：伦理、安全与对齐 — ✅（约 31 小时）

| # | 课程 | 状态 | 预计 |
|---|--------|--------|------|
| 01 | [指令遵循作为对齐信号](phases/18-ethics-safety-alignment/01-instruction-following-alignment-signal) | ✅ | ~45 min |
| 02 | [奖励黑客与古德哈特定律](phases/18-ethics-safety-alignment/02-reward-hacking-goodhart) | ✅ | ~60 min |
| 03 | [直接偏好优化家族](phases/18-ethics-safety-alignment/03-direct-preference-optimization-family) | ✅ | ~60 min |
| 04 | [谄媚作为 RLHF 的放大](phases/18-ethics-safety-alignment/04-sycophancy-rlhf-amplification) | ✅ | ~45 min |
| 05 | [宪法 AI 与 RLAIF](phases/18-ethics-safety-alignment/05-constitutional-ai-rlaif) | ✅ | ~60 min |
| 06 | [元优化与欺骗性对齐](phases/18-ethics-safety-alignment/06-mesa-optimization-deceptive-alignment) | ✅ | ~75 min |
| 07 | [潜伏智能体——持久欺骗](phases/18-ethics-safety-alignment/07-sleeper-agents-persistent-deception) | ✅ | ~60 min |
| 08 | [前沿模型中的上下文内密谋](phases/18-ethics-safety-alignment/08-in-context-scheming-frontier-models) | ✅ | ~60 min |
| 09 | [对齐伪装](phases/18-ethics-safety-alignment/09-alignment-faking) | ✅ | ~60 min |
| 10 | [AI 控制——即便被颠覆也保持安全](phases/18-ethics-safety-alignment/10-ai-control-subversion) | ✅ | ~75 min |
| 11 | [可扩展监督与弱到强泛化](phases/18-ethics-safety-alignment/11-scalable-oversight-weak-to-strong) | ✅ | ~60 min |
| 12 | [红队——PAIR 与自动化攻击](phases/18-ethics-safety-alignment/12-red-teaming-pair-automated-attacks) | ✅ | ~75 min |
| 13 | [多样本越狱](phases/18-ethics-safety-alignment/13-many-shot-jailbreaking) | ✅ | ~45 min |
| 14 | [ASCII 艺术与视觉越狱](phases/18-ethics-safety-alignment/14-ascii-art-visual-jailbreaks) | ✅ | ~60 min |
| 15 | [间接提示注入](phases/18-ethics-safety-alignment/15-indirect-prompt-injection) | ✅ | ~75 min |
| 16 | [红队工具——Garak、Llama Guard、PyRIT](phases/18-ethics-safety-alignment/16-red-team-tooling-garak-llamaguard-pyrit) | ✅ | ~75 min |
| 17 | [WMDP 与双用途能力评估](phases/18-ethics-safety-alignment/17-wmdp-dual-use-evaluation) | ✅ | ~60 min |
| 18 | [前沿安全框架——RSP、PF、FSF](phases/18-ethics-safety-alignment/18-frontier-safety-frameworks-rsp-pf-fsf) | ✅ | ~75 min |
| 19 | [模型福祉研究](phases/18-ethics-safety-alignment/19-model-welfare-research) | ✅ | ~45 min |
| 20 | [偏见与代表性危害](phases/18-ethics-safety-alignment/20-bias-representational-harm) | ✅ | ~60 min |
| 21 | [公平性准则——群体、个体、反事实](phases/18-ethics-safety-alignment/21-fairness-criteria-group-individual-counterfactual) | ✅ | ~60 min |
| 22 | [面向 LLM 的差分隐私](phases/18-ethics-safety-alignment/22-differential-privacy-for-llms) | ✅ | ~60 min |
| 23 | [水印——SynthID、Stable Signature、C2PA](phases/18-ethics-safety-alignment/23-watermarking-synthid-stable-signature-c2pa) | ✅ | ~75 min |
| 24 | [监管框架——欧盟、美国、英国、韩国](phases/18-ethics-safety-alignment/24-regulatory-frameworks-eu-us-uk-korea) | ✅ | ~75 min |
| 25 | [EchoLeak 与面向 AI 的 CVE](phases/18-ethics-safety-alignment/25-echoleak-cves-for-ai) | ✅ | ~45 min |
| 26 | [模型、系统与数据集卡片](phases/18-ethics-safety-alignment/26-model-system-dataset-cards) | ✅ | ~60 min |
| 27 | [数据来源与训练数据治理](phases/18-ethics-safety-alignment/27-data-provenance-training-governance) | ✅ | ~60 min |
| 28 | [对齐研究生态——MATS、Redwood、Apollo、METR](phases/18-ethics-safety-alignment/28-alignment-research-ecosystem) | ✅ | ~45 min |
| 29 | [审核系统——OpenAI、Perspective、Llama Guard](phases/18-ethics-safety-alignment/29-moderation-systems-openai-perspective-llamaguard) | ✅ | ~60 min |
| 30 | [双用途风险——网络、生物、化学、核](phases/18-ethics-safety-alignment/30-dual-use-risk-cyber-bio-chem-nuclear) | ✅ | ~75 min |

## 阶段 19：毕业项目 — ✅（约 620 小时）

| # | 项目 | 状态 | 预计 |
|---|---------|--------|------|
| 01 | [终端原生编码智能体](phases/19-capstone-projects/01-terminal-native-coding-agent) | ✅ | ~35 hr |
| 02 | [代码库上的 RAG（跨仓库语义搜索）](phases/19-capstone-projects/02-rag-over-codebase) | ✅ | ~30 hr |
| 03 | [实时语音助手（ASR 到 LLM 到 TTS）](phases/19-capstone-projects/03-realtime-voice-assistant) | ✅ | ~30 hr |
| 04 | [多模态文档问答（视觉优先）](phases/19-capstone-projects/04-multimodal-document-qa) | ✅ | ~30 hr |
| 05 | [自主研究智能体（AI-Scientist 类）](phases/19-capstone-projects/05-autonomous-research-agent) | ✅ | ~40 hr |
| 06 | [面向 Kubernetes 的 DevOps 故障排查智能体](phases/19-capstone-projects/06-devops-troubleshooting-agent) | ✅ | ~30 hr |
| 07 | [端到端微调流水线](phases/19-capstone-projects/07-end-to-end-fine-tuning-pipeline) | ✅ | ~35 hr |
| 08 | [生产级 RAG 聊天机器人（受监管行业）](phases/19-capstone-projects/08-production-rag-chatbot) | ✅ | ~30 hr |
| 09 | [代码迁移智能体（仓库级升级）](phases/19-capstone-projects/09-code-migration-agent) | ✅ | ~30 hr |
| 10 | [多智能体软件工程团队](phases/19-capstone-projects/10-multi-agent-software-team) | ✅ | ~40 hr |
| 11 | [LLM 可观测性与评估仪表盘](phases/19-capstone-projects/11-llm-observability-dashboard) | ✅ | ~25 hr |
| 12 | [视频理解流水线（场景到问答）](phases/19-capstone-projects/12-video-understanding-pipeline) | ✅ | ~30 hr |
| 13 | [带注册中心与治理的无状态 MCP 服务器](phases/19-capstone-projects/13-mcp-server-with-registry) | ✅ | ~25 hr |
| 14 | [投机解码推理服务器](phases/19-capstone-projects/14-speculative-decoding-server) | ✅ | ~30 hr |
| 15 | [宪法安全 Harness + 红队靶场](phases/19-capstone-projects/15-constitutional-safety-harness) | ✅ | ~25 hr |
| 16 | [GitHub Issue 到 PR 的自主智能体](phases/19-capstone-projects/16-github-issue-to-pr-agent) | ✅ | ~30 hr |
| 17 | [个人 AI 导师（自适应、多模态）](phases/19-capstone-projects/17-personal-ai-tutor) | ✅ | ~30 hr |
| 20 | [智能体 Harness 循环契约](phases/19-capstone-projects/20-agent-harness-loop-contract) | ✅ | ~90 min |
| 21 | [带模式校验的工具注册中心](phases/19-capstone-projects/21-tool-registry-schema-validation) | ✅ | ~90 min |
| 22 | [基于换行分隔 stdio 的 JSON-RPC 2.0](phases/19-capstone-projects/22-jsonrpc-stdio-transport) | ✅ | ~90 min |
| 23 | [函数调用分发器](phases/19-capstone-projects/23-function-call-dispatcher) | ✅ | ~90 min |
| 24 | [规划-执行控制流](phases/19-capstone-projects/24-plan-execute-control-flow) | ✅ | ~90 min |
| 25 | [验证门与观测预算](phases/19-capstone-projects/25-verification-gates-observation-budget) | ✅ | ~90 min |
| 26 | [带黑名单与路径牢笼的沙箱运行器](phases/19-capstone-projects/26-sandbox-runner-denylist) | ✅ | ~90 min |
| 27 | [带固定任务的评估 Harness](phases/19-capstone-projects/27-eval-harness-fixture-tasks) | ✅ | ~90 min |
| 28 | [基于 OTel GenAI Span 与 Prometheus 指标的可观测性](phases/19-capstone-projects/28-observability-otel-traces) | ✅ | ~90 min |
| 29 | [Harness 上的端到端编码智能体](phases/19-capstone-projects/29-end-to-end-coding-task-demo) | ✅ | ~90 min |
| 30 | [从零实现 BPE 分词器](phases/19-capstone-projects/30-bpe-tokenizer-from-scratch) | ✅ | ~90 min |
| 31 | [带滑动窗口的分词数据集](phases/19-capstone-projects/31-tokenized-dataset-sliding-window) | ✅ | ~90 min |
| 32 | [Token 与位置嵌入](phases/19-capstone-projects/32-token-positional-embeddings) | ✅ | ~90 min |
| 33 | [多头自注意力](phases/19-capstone-projects/33-multihead-self-attention) | ✅ | ~90 min |
| 34 | [从零实现 Transformer 块](phases/19-capstone-projects/34-transformer-block) | ✅ | ~90 min |
| 35 | [GPT 模型组装](phases/19-capstone-projects/35-gpt-model-assembly) | ✅ | ~90 min |
| 36 | [训练循环与评估](phases/19-capstone-projects/36-training-loop-eval) | ✅ | ~90 min |
| 37 | [加载预训练权重](phases/19-capstone-projects/37-loading-pretrained-weights) | ✅ | ~90 min |
| 38 | [通过头部替换进行分类器微调](phases/19-capstone-projects/38-classifier-finetuning) | ✅ | ~90 min |
| 39 | [通过监督微调进行指令微调](phases/19-capstone-projects/39-instruction-tuning-sft) | ✅ | ~90 min |
| 40 | [从零实现直接偏好优化](phases/19-capstone-projects/40-dpo-from-scratch) | ✅ | ~90 min |
| 41 | [完整评估流水线](phases/19-capstone-projects/41-eval-pipeline) | ✅ | ~90 min |
| 42 | [大型语料下载器](phases/19-capstone-projects/42-large-corpus-downloader) | ✅ | ~90 min |
| 43 | [HDF5 分词语料](phases/19-capstone-projects/43-hdf5-tokenized-corpus) | ✅ | ~90 min |
| 44 | [带线性预热的余弦学习率](phases/19-capstone-projects/44-cosine-lr-warmup) | ✅ | ~90 min |
| 45 | [梯度裁剪与混合精度](phases/19-capstone-projects/45-gradient-clipping-amp) | ✅ | ~90 min |
| 46 | [梯度累积](phases/19-capstone-projects/46-gradient-accumulation) | ✅ | ~90 min |
| 47 | [检查点保存与恢复](phases/19-capstone-projects/47-checkpoint-save-resume) | ✅ | ~90 min |
| 48 | [从零实现分布式数据并行与 FSDP](phases/19-capstone-projects/48-distributed-fsdp-ddp) | ✅ | ~90 min |
| 49 | [语言模型评估 Harness](phases/19-capstone-projects/49-lm-eval-harness) | ✅ | ~90 min |
| 50 | [假设生成器](phases/19-capstone-projects/50-hypothesis-generator) | ✅ | ~90 min |
| 51 | [文献检索](phases/19-capstone-projects/51-literature-retrieval) | ✅ | ~90 min |
| 52 | [实验运行器](phases/19-capstone-projects/52-experiment-runner) | ✅ | ~90 min |
| 53 | [结果评估器](phases/19-capstone-projects/53-result-evaluator) | ✅ | ~90 min |
| 54 | [论文撰写器](phases/19-capstone-projects/54-paper-writer) | ✅ | ~90 min |
| 55 | [评审循环](phases/19-capstone-projects/55-critic-loop) | ✅ | ~90 min |
| 56 | [迭代调度器](phases/19-capstone-projects/56-iteration-scheduler) | ✅ | ~90 min |
| 57 | [端到端研究演示](phases/19-capstone-projects/57-end-to-end-research-demo) | ✅ | ~90 min |
| 58 | [视觉编码器分块](phases/19-capstone-projects/58-vision-encoder-patches) | ✅ | ~90 min |
| 59 | [视觉 Transformer 编码器](phases/19-capstone-projects/59-vit-transformer) | ✅ | ~90 min |
| 60 | [用于模态对齐的投影层](phases/19-capstone-projects/60-projection-layer-modality-align) | ✅ | ~90 min |
| 61 | [交叉注意力融合](phases/19-capstone-projects/61-cross-attention-fusion) | ✅ | ~90 min |
| 62 | [视觉-语言预训练](phases/19-capstone-projects/62-vision-language-pretraining) | ✅ | ~90 min |
| 63 | [多模态评估](phases/19-capstone-projects/63-multimodal-eval) | ✅ | ~90 min |
| 64 | [分块策略对比](phases/19-capstone-projects/64-chunking-strategies-advanced) | ✅ | ~90 min |
| 65 | [结合 BM25 与稠密嵌入的混合检索](phases/19-capstone-projects/65-hybrid-retrieval-bm25-dense) | ✅ | ~90 min |
| 66 | [交叉编码器重排器](phases/19-capstone-projects/66-reranker-cross-encoder) | ✅ | ~90 min |
| 67 | [查询重写：HyDE、多查询与分解](phases/19-capstone-projects/67-query-rewriting-hyde) | ✅ | ~90 min |
| 68 | [RAG 评估：精确率、召回率、MRR、nDCG、忠实度、答案相关性](phases/19-capstone-projects/68-rag-eval-precision-recall) | ✅ | ~90 min |
| 69 | [端到端 RAG 系统](phases/19-capstone-projects/69-end-to-end-rag-system) | ✅ | ~90 min |
| 70 | [任务规格格式](phases/19-capstone-projects/70-task-spec-format) | ✅ | ~90 min |
| 71 | [经典指标](phases/19-capstone-projects/71-classical-metrics) | ✅ | ~90 min |
| 72 | [代码执行指标](phases/19-capstone-projects/72-code-exec-metric) | ✅ | ~90 min |
| 73 | [困惑度与校准](phases/19-capstone-projects/73-perplexity-calibration) | ✅ | ~90 min |
| 74 | [排行榜聚合](phases/19-capstone-projects/74-leaderboard-aggregation) | ✅ | ~90 min |
| 75 | [端到端评估运行器](phases/19-capstone-projects/75-end-to-end-eval-runner) | ✅ | ~90 min |
| 76 | [从零实现集合通信算子](phases/19-capstone-projects/76-collective-ops-from-scratch) | ✅ | ~90 min |
| 77 | [从零实现数据并行 DDP](phases/19-capstone-projects/77-data-parallel-ddp) | ✅ | ~90 min |
| 78 | [ZeRO 优化器状态分片](phases/19-capstone-projects/78-zero-parameter-sharding) | ✅ | ~90 min |
| 79 | [流水线并行与气泡分析](phases/19-capstone-projects/79-pipeline-parallel) | ✅ | ~90 min |
| 80 | [分片检查点与原子恢复](phases/19-capstone-projects/80-checkpoint-sharded-resume) | ✅ | ~90 min |
| 81 | [端到端分布式训练](phases/19-capstone-projects/81-end-to-end-distributed-train) | ✅ | ~90 min |
| 82 | [越狱分类学](phases/19-capstone-projects/82-jailbreak-taxonomy) | ✅ | ~90 min |
| 83 | [提示注入检测器](phases/19-capstone-projects/83-prompt-injection-detector) | ✅ | ~90 min |
| 84 | [拒绝评估](phases/19-capstone-projects/84-refusal-evaluation) | ✅ | ~90 min |
| 85 | [内容分类器集成](phases/19-capstone-projects/85-content-classifier-integration) | ✅ | ~90 min |
| 86 | [宪法规则引擎](phases/19-capstone-projects/86-constitutional-rules-engine) | ✅ | ~90 min |
| 87 | [端到端安全门](phases/19-capstone-projects/87-end-to-end-safety-gate) | ✅ | ~90 min |

---

**合计：20 个阶段，523 节课程 | 523 节已完成 | 预计约 1,079 小时**

想要帮忙吗？任选一节 ⬚ 课程并提交 PR。参见 [CONTRIBUTING.md](CONTRIBUTING.md)。
