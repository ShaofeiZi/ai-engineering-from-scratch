# 机器翻译

> 三十年来，机器翻译一直是支撑自然语言处理研究投入的任务，至今仍在持续创造价值。

**Type:** 构建
**Languages:** Python
**Prerequisites:** 阶段 5 · 10（注意力机制）、阶段 5 · 04（GloVe、FastText、子词）
**Time:** 约 75 分钟

## 问题

模型读取一种语言的句子，再输出另一种语言的句子。长度会变化，词序会变化，有些源词对应多个目标词，有些则反过来。习语不接受一对一映射。英语里的“我想你”对应法语 “tu me manques”，直译更接近“你让我感到缺失”。任何词级对齐都无法经受这种变化。

机器翻译这项任务推动自然语言处理发明了编码器—解码器、注意力机制、Transformer，最终催生了整个大语言模型范式。每一次进步都源于两个事实：翻译质量可以测量，而人与机器之间的差距又顽固地存在。

本课跳过历史梳理，直接讲解 2026 年可实际使用的流水线：预训练多语言编码器—解码器（NLLB-200 或 mBART）、子词分词、束搜索、BLEU 与 chrF 评估，以及那些仍会在未被发现的情况下进入生产环境的少数失败模式。

## 概念

![机器翻译流水线：分词 → 编码 → 使用注意力解码 → 还原词元](../../../../../../phases/05-nlp-foundations-to-advanced/11-machine-translation/assets/mt-pipeline.svg)

现代机器翻译使用在平行文本上训练的 Transformer 编码器—解码器。编码器按照源语言的分词方式读取输入，解码器通过交叉注意力（第 10 课）使用编码器输出，逐个子词生成目标文本。解码时使用束搜索，以避开贪心解码陷阱。最终输出会经过词元还原和大小写还原，再与参考译文评分。

有三项操作选择会决定真实环境中的机器翻译质量。

- **分词器。** 在混合语言语料库上训练的 SentencePiece BPE。跨语言共享词表是 NLLB 能实现零样本语言对的原因。
- **模型大小。** NLLB-200 distilled 600M 可以在笔记本电脑上运行。NLLB-200 3.3B 是公开的生产默认版本，54.5B 则是研究规模的上限。
- **解码。** 通用内容使用 4～5 的束宽；使用长度惩罚避免输出过短；需要术语一致性时采用约束解码。

```figure
seq2seq-alignment
```

## 动手构建

### 第 1 步：调用预训练机器翻译模型

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_id = "facebook/nllb-200-distilled-600M"
tok = AutoTokenizer.from_pretrained(model_id, src_lang="eng_Latn")
model = AutoModelForSeq2SeqLM.from_pretrained(model_id)

src = "The cats are running."
inputs = tok(src, return_tensors="pt")

out = model.generate(
    **inputs,
    forced_bos_token_id=tok.convert_tokens_to_ids("fra_Latn"),
    num_beams=5,
    length_penalty=1.0,
    max_new_tokens=64,
)
print(tok.batch_decode(out, skip_special_tokens=True)[0])
```

```text
Les chats courent.
```

这里有三点至关重要。`src_lang` 告诉分词器应采用哪种文字体系和切分方式；`forced_bos_token_id` 告诉解码器应生成哪种语言。二者都是 NLLB 特有的技巧；mBART 和 M2M-100 各有自己的约定，不能互换。

### 第 2 步：BLEU 与 chrF

BLEU 衡量输出与参考译文之间的 n 元语法重叠度。它使用四种参考 n 元语法大小（1～4）、精确率的几何平均值，以及针对过短输出的简短惩罚。分数范围是 [0, 100]。它很常用，却很难解释：30 BLEU 表示“可用”，40 表示“良好”，50 表示“极佳”；低于 1 BLEU 的差异只是噪声。

chrF 衡量字符级 F 分数。对于形态丰富、容易被 BLEU 低估匹配程度的语言，它更敏感，因此常与 BLEU 一起报告。

```python
import sacrebleu

hypotheses = ["Les chats courent."]
references = [["Les chats courent."]]

bleu = sacrebleu.corpus_bleu(hypotheses, references)
chrf = sacrebleu.corpus_chrf(hypotheses, references)
print(f"BLEU: {bleu.score:.1f}  chrF: {chrf.score:.1f}")
```

始终使用 `sacrebleu`。它会规范分词，使不同论文中的分数可以比较。自行编写 BLEU 计算，正是产生误导性基准的常见方式。

### 三层评估体系（2026）

现代机器翻译评估使用三个互补的指标家族。交付时至少使用其中两种。

- **启发式指标**（BLEU、chrF）。速度快、依赖参考译文、可解释，但对释义不敏感。用于与旧结果比较和发现回归。
- **学习式指标**（COMET、BLEURT、BERTScore）。在人工判断上训练的神经模型，用于比较译文与源文和参考译文的语义相似度。自 2023 年起，COMET 与机器翻译研究的相关性最高；到 2026 年，在重视质量的场景中，它已成为生产默认指标。
- **大语言模型裁判**（无参考译文）。提示大型模型从流畅度、充分性、语气和文化适切性等方面为译文评分。当评分标准设计良好时，以 GPT-4 为裁判与人类判断的一致率约为 80%。适用于没有参考译文的开放式内容。

实用的 2026 年技术栈是：用 `sacrebleu` 计算 BLEU 与 chrF，用 `unbabel-comet` 计算 COMET，再使用带提示的大语言模型获取最终的用户侧信号。在将任何指标用于生产数据之前，都要先用 50～100 个经人工标注的样本进行校准。

无参考指标（COMET-QE、BLEURT-QE、大语言模型裁判）让你可以在没有参考译文时评估翻译，这对不存在参考译文的长尾语言对尤其重要。

### 第 3 步：生产环境中会出什么问题

上面的流水线有 80% 的时间能生成流畅译文，却会在剩余 20% 的情况下悄然失败。下面是这些失败模式的名称：

- **幻觉。** 模型编造源文中不存在的内容，在不熟悉的领域词汇中很常见。症状是输出很流畅，却声称了源文未曾陈述的事实。缓解方法：对领域术语使用约束解码；受监管内容交由人工审阅；监控长度远超输入的输出。
- **目标语言错误。** 模型翻译成了错误语言。NLLB 在罕见语言对上出人意料地容易发生这种问题。缓解方法：检查 `forced_bos_token_id`，并始终用语言识别模型检查输出。
- **术语漂移。** “Sign up”在文档 1 中译成“s'inscrire”，在文档 2 中却变成“créer un compte”。对于 UI 文本和面向用户的字符串，一致性比原始质量更重要。缓解方法：使用词汇表约束解码或译后词典替换。
- **正式程度不匹配。** 法语的“tu”与“vous”、日语的敬语等级。模型会选择训练数据中更常见的形式，但对于面向客户的内容，这通常是错误的。缓解方法：如果模型支持，在提示前缀中加入正式程度词元；或者在仅含正式语体的语料库上微调小模型。
- **短输入导致长度爆炸。** 非常短的输入句子常会产生过长译文，因为源词元少于约 5 个时，长度惩罚会突然失效。缓解方法：按源文本长度成比例设置硬性最大长度。

### 第 4 步：面向领域微调

预训练模型是通才。使用领域平行数据微调，可以显著改善法律、医学或游戏对话翻译。方法并不复杂：

```python
from transformers import Trainer, TrainingArguments
from datasets import Dataset

pairs = [
    {"src": "The defendant pleaded guilty.", "tgt": "L'accusé a plaidé coupable."},
]

ds = Dataset.from_list(pairs)


def preprocess(ex):
    return tok(
        ex["src"],
        text_target=ex["tgt"],
        truncation=True,
        max_length=128,
        padding="max_length",
    )


ds = ds.map(preprocess, remove_columns=["src", "tgt"])

args = TrainingArguments(output_dir="out", per_device_train_batch_size=4, num_train_epochs=3, learning_rate=3e-5)
Trainer(model=model, args=args, train_dataset=ds).train()
```

几千个高质量平行样本，胜过几十万个从网页抓取的噪声样本。训练数据质量是影响生产效果最大的单一杠杆。

## 学以致用

2026 年的生产级机器翻译技术栈：

| 用例 | 推荐起点 |
|---------|---------------------------|
| 任意语言互译，覆盖 200 种语言 | `facebook/nllb-200-distilled-600M`（笔记本电脑）或 `nllb-200-3.3B`（生产） |
| 以英语为中心，覆盖 50 种语言且追求高质量 | `facebook/mbart-large-50-many-to-many-mmt` |
| 短时运行、低成本推理、英语与法语/德语/西班牙语互译 | Helsinki-NLP / Marian 模型 |
| 延迟敏感的浏览器端场景 | ONNX 量化 Marian（约 50 MB） |
| 追求最高质量且愿意付费 | 使用翻译提示的 GPT-4 / Claude / Gemini |

截至 2026 年，大语言模型在若干语言对上已经胜过专用机器翻译模型，尤其擅长习语内容和长上下文。代价是逐词元费用与延迟。如果上下文长度、风格一致性或通过提示适配领域比吞吐量更重要，就选择大语言模型。

## 交付成果

保存为 `outputs/skill-mt-evaluator.md`：

```markdown
---
name: mt-evaluator
description: Evaluate a machine translation output for shipping.
version: 1.0.0
phase: 5
lesson: 11
tags: [nlp, translation, evaluation]
---

Given a source text and a candidate translation, output:

1. Automatic score estimate. BLEU and chrF ranges you would expect. State whether a reference is available.
2. Five-point human-verifiable check list: (a) content preservation (no hallucinations), (b) correct language, (c) register / formality match, (d) terminology consistency with glossary if provided, (e) no truncation or length explosion.
3. One domain-specific issue to probe. E.g., for legal: named entities and statute citations. For medical: drug names and dosages. For UI: placeholder variables `{name}`.
4. Confidence flag. "Ship" / "Ship with review" / "Do not ship". Tie to the severity of issues found in step 2.

Refuse to ship a translation without a language-ID check on output. Refuse to evaluate without a reference unless the user explicitly opts in to reference-free scoring (COMET-QE, BLEURT-QE). Flag any content over 1000 tokens as likely needing chunked translation.
```

## 练习

1. **简单。** 使用 `nllb-200-distilled-600M` 把一段包含 5 个句子的英语段落译成法语，再译回英语。衡量回译文本与原文的接近程度。你应当看到语义得到保留，但用词有所漂移。
2. **中等。** 使用 `fasttext lid.176` 或 `langdetect` 对翻译输出实现语言识别检查。将它集成进机器翻译调用，使目标语言错误的生成结果在返回前就被拦截。
3. **困难。** 在你任选的 5000 对领域语料上微调 `nllb-200-distilled-600M`。在留出集上分别测量微调前后的 BLEU，并报告哪些句型有所改善、哪些出现退步。

## 关键术语

| 术语 | 人们通常怎么说 | 实际含义 |
|------|-----------------|-----------------------|
| BLEU | 翻译分数 | 带简短惩罚的 n 元语法精确率，范围为 [0, 100]。 |
| chrF | 字符 F 分数 | 字符级 F 分数，对形态丰富的语言更敏感。 |
| NMT | 神经机器翻译 | 在平行文本上训练的 Transformer 编码器—解码器，是 2017 年后的默认方案。 |
| NLLB | 不让任何语言掉队 | Meta 的 200 语言机器翻译模型家族。 |
| 约束解码 | 受控输出 | 强制特定词元或 n 元语法在输出中出现或不出现。 |
| 幻觉 | 编造内容 | 模型输出了源文不支持的内容。 |

## 延伸阅读

- [Costa-jussà 等（2022），不让任何语言掉队：扩展以人为本的机器翻译](https://arxiv.org/abs/2207.04672)——NLLB 论文。
- [Post（2018），呼吁清晰报告 BLEU 分数](https://aclanthology.org/W18-6319/)——为什么 `sacrebleu` 是唯一正确的 BLEU 报告方式。
- [Popović（2015），chrF：用于自动机器翻译评估的字符 n 元语法 F 分数](https://aclanthology.org/W15-3049/)——chrF 论文。
- [Hugging Face 机器翻译指南](https://huggingface.co/docs/transformers/tasks/translation)——实用的微调教程。
