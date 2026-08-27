# 从零构建词元化器

> 第 01 课给了你一个玩具，这一课则会给你一件利器。

**Type:** 构建
**Languages:** Python
**Prerequisites:** 阶段 10 第 01 课（词元化器：BPE、WordPiece、SentencePiece）
**Time:** 约 90 分钟

## 学习目标

- 构建生产级 BPE 词元化器，正确处理 Unicode、空白规范化和特殊词元
- 实现字节级回退，使词元化器可以编码任何输入（包括表情符号、中日韩文字与代码），且不会产生未知词元
- 添加预词元化正则表达式，在应用 BPE 合并前按词边界切分文本
- 在语料库上训练自定义词元化器，并在多语言文本上对照 tiktoken 评估其压缩率

## 问题

第 01 课的 BPE 词元化器可以处理英语文本。现在试着给它输入日语、表情符号，或混用制表符与空格的 Python 代码。

它会坏掉。

问题不在 BPE，而在于实现并不完整。生产级词元化器要能处理任意编码的原始字节，在切分前规范化 Unicode，管理永远不会参与合并的特殊词元，将预词元化与子词切分串联起来，而且必须足够快，不能成为处理 15 万亿词元的训练流水线中的瓶颈。

GPT-2 的词表有 50,257 个词元，Llama 3 有 128,256 个，GPT-4 约有 100,000 个。这些并不是玩具规模。支撑这些词表的合并表在数百 GB 文本上训练，而词元化器外围的机制——规范化、预词元化、注入特殊词元、聊天模板格式化——决定了它只能处理“hello world”，还是能够处理整个互联网。

你将亲手构建这些机制。

## 概念

### 完整流水线

生产级词元化器不是单一算法，而是由五个阶段组成的流水线，每个阶段解决不同问题。

```mermaid
graph LR
    A[Raw Text] --> B[Normalize]
    B --> C[Pre-Tokenize]
    C --> D[BPE Merge]
    D --> E[Special Tokens]
    E --> F[Token IDs]

    style A fill:#1a1a2e,stroke:#e94560,color:#fff
    style B fill:#1a1a2e,stroke:#e94560,color:#fff
    style C fill:#1a1a2e,stroke:#e94560,color:#fff
    style D fill:#1a1a2e,stroke:#e94560,color:#fff
    style E fill:#1a1a2e,stroke:#e94560,color:#fff
    style F fill:#1a1a2e,stroke:#e94560,color:#fff
```

每个阶段都有明确职责：

| 阶段 | 作用 | 重要性 |
|-------|-------------|----------------|
| 规范化 | NFKC Unicode；可选转小写；可选移除重音符号 | “fi”连字（U+FB01）会变成“fi”（两个字符）。若不处理，同一个词会得到不同词元。 |
| 预词元化 | 在 BPE 前把文本切成片段 | 防止 BPE 跨越词边界合并。“the cat”绝不应产生“e c”这个词元。 |
| BPE 合并 | 对字节序列应用学习得到的合并规则 | 核心压缩步骤，把原始字节转换为子词词元。 |
| 特殊词元 | 注入 [BOS]、[EOS]、[PAD] 与聊天模板标记 | 这些词元拥有固定 ID，永远不参与 BPE 合并。模型依靠它们表达结构。 |
| ID 映射 | 把词元字符串转换为整数 ID | 模型读取整数，而不是字符串。 |

### 字节级 BPE

第 01 课的词元化器处理 UTF-8 字节，这个选择是正确的。但我们跳过了一个重要问题：这些字节无法构成有效 UTF-8 时怎么办？

字节级 BPE 把每一种可能的字节值（0～255）都视为有效词元。基础词表恰好有 256 项。任何文件——文本、二进制文件、损坏的数据——都可以被词元化，而且不会产生未知词元。

GPT-2 还加入了一项技巧：把每个字节映射为可打印的 Unicode 字符，让词表便于阅读。在它的映射中，字节 0x20（空格）会变成字符“G”。这纯粹是显示层面的处理，算法并不关心。

真正的威力在于，字节级 BPE 能处理世界上的每种语言。一个汉字占 3 个 UTF-8 字节，日文字符可能占 3～4 个字节，阿拉伯文、天城文和表情符号也都只是字节序列。BPE 查找这些字节序列中的模式，与查找英语 ASCII 字节中的模式完全相同。

### 预词元化

在 BPE 接触文本之前，需要先把文本切成片段，以防合并算法生成跨越词边界的词元。

GPT-2 使用一条正则表达式切分文本：

```
'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
```

这个表达式会切分缩写（“don't”变为“don”+“'t”）、可带前导空格的单词、数字、标点与空白。前导空格会保留在单词上——因此“the cat”变成 [“ the”, “ cat”]，而不是 [“the”, “ ”, “cat”]。

Llama 使用 SentencePiece，完全跳过正则表达式。它把原始字节流视为一个长序列，让 BPE 算法自行找出边界。这种做法更简单，却给了 BPE 更大的自由，可以生成跨词词元。

这个选择很重要。GPT-2 的正则表达式会阻止词元化器把一个单词末尾的“the”和下一个单词开头的“the”合并。SentencePiece 允许这种情况，因此有时压缩效率更高，但词元的可解释性更差。

### 特殊词元

每个生产级词元化器都会为结构标记保留词元 ID：

| 词元 | 用途 | 使用者 |
|-------|---------|---------|
| `[BOS]` / `<s>` | 序列开始 | Llama 3、GPT |
| `[EOS]` / `</s>` | 序列结束 | 所有模型 |
| `[PAD]` | 批次对齐所需的填充 | BERT、T5 |
| `[UNK]` | 未知词元（字节级 BPE 不需要它） | BERT、WordPiece |
| `<\|im_start\|>` | 聊天消息边界起点 | ChatGPT、Qwen |
| `<\|im_end\|>` | 聊天消息边界终点 | ChatGPT、Qwen |
| `<\|user\|>` | 用户轮次标记 | Llama 3 |
| `<\|assistant\|>` | 助手轮次标记 | Llama 3 |

特殊词元永远不会被 BPE 拆分。在运行合并算法之前，先进行精确匹配，用固定 ID 替换它们，再正常对周围文本进行词元化。

### 聊天模板

这是最容易让人困惑、也是多数实现最容易出错的地方。

向聊天模型发送消息时，API 接收的是消息列表：

```
[
  {"role": "system", "content": "You are helpful."},
  {"role": "user", "content": "Hello"},
  {"role": "assistant", "content": "Hi there!"}
]
```

模型看不到 JSON，只会看到展平后的词元序列。聊天模板使用特殊词元，把消息转换为这条平坦序列。每个模型的格式都不一样：

```
Llama 3:
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are helpful.<|eot_id|><|start_header_id|>user<|end_header_id|>

Hello<|eot_id|><|start_header_id|>assistant<|end_header_id|>

Hi there!<|eot_id|>

ChatGPT:
<|im_start|>system
You are helpful.<|im_end|>
<|im_start|>user
Hello<|im_end|>
<|im_start|>assistant
Hi there!<|im_end|>
```

如果模板不对，模型就会输出乱码。模型只在一种精确格式上训练，任何偏差——少一个换行、交换两个词元、多一个空格——都会让输入偏离训练分布。

### 速度

Python 的速度不足以承担生产级词元化。

tiktoken（OpenAI）使用 Rust 编写，并提供 Python 绑定；HuggingFace tokenizers 同样使用 Rust；SentencePiece 则使用 C++。这些实现比纯 Python 快 10～100 倍。

换个角度看：如果以每秒 100 万个词元（较快的 Python）的速度，为 Llama 3 预训练处理 15 万亿个词元，需要 174 天；若使用 Rust 达到每秒 1 亿个词元，则只需 1.7 天。

你在这里用 Python 构建，是为了理解算法。生产环境中应使用编译后的实现，只接触它的 Python 包装层。

```figure
weight-tying
```

## 动手构建

### 第 1 步：字节级编码

从基础开始。把任意字符串转换为字节序列，将每个字节映射为可打印字符以供展示，再执行逆过程。

```python
def bytes_to_tokens(text):
    return list(text.encode("utf-8"))

def tokens_to_text(token_bytes):
    return bytes(token_bytes).decode("utf-8", errors="replace")
```

用多语言文本测试字节数量：

```python
texts = [
    ("English", "hello"),
    ("Chinese", "你好"),
    ("Emoji", "🔥"),
    ("Mixed", "hello你好🔥"),
]

for label, text in texts:
    b = bytes_to_tokens(text)
    print(f"{label}: {len(text)} chars -> {len(b)} bytes -> {b}")
```

“hello”占 5 个字节，“你好”占 6 个字节（每个字符 3 字节），火焰表情占 4 个字节。字节级词元化器不关心它们属于哪种语言，字节就是字节。

### 第 2 步：使用正则表达式的预词元化器

使用 GPT-2 正则表达式将文本切成片段，再由 BPE 分别处理每个片段。

```python
import re

try:
    import regex
    GPT2_PATTERN = regex.compile(
        r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    )
except ImportError:
    GPT2_PATTERN = re.compile(
        r"""'(?:[sdmt]|ll|ve|re)| ?[a-zA-Z]+| ?[0-9]+| ?[^\s\w]+|\s+(?!\S)|\s+"""
    )

def pre_tokenize(text):
    return [match.group() for match in GPT2_PATTERN.finditer(text)]
```

`regex` 模块支持 Unicode 属性转义（字母使用 `\p{L}`，数字使用 `\p{N}`），标准库 `re` 模块则不支持，因此我们会回退到 ASCII 字符类。生产级多语言词元化器应安装 `regex`。

试着运行：

```python
print(pre_tokenize("Hello, world! Don't stop."))
# [' Hello', ',', ' world', '!', " Don", "'t", ' stop', '.']
```

前导空格仍附着在单词上，缩写会在撇号处分开，标点会成为单独片段。BPE 永远不会跨越这些边界合并词元。

### 第 3 步：在字节序列上运行 BPE

沿用第 01 课的核心算法，但现在要分别处理预词元化后的每个片段。

```python
from collections import Counter

def get_byte_pairs(chunks):
    pairs = Counter()
    for chunk in chunks:
        byte_seq = list(chunk.encode("utf-8"))
        for i in range(len(byte_seq) - 1):
            pairs[(byte_seq[i], byte_seq[i + 1])] += 1
    return pairs

def apply_merge(byte_seq, pair, new_id):
    merged = []
    i = 0
    while i < len(byte_seq):
        if i < len(byte_seq) - 1 and byte_seq[i] == pair[0] and byte_seq[i + 1] == pair[1]:
            merged.append(new_id)
            i += 2
        else:
            merged.append(byte_seq[i])
            i += 1
    return merged
```

### 第 4 步：处理特殊词元

特殊词元需要精确匹配并使用固定 ID，完全绕过 BPE。

```python
class SpecialTokenHandler:
    def __init__(self):
        self.special_tokens = {}
        self.pattern = None

    def add_token(self, token_str, token_id):
        self.special_tokens[token_str] = token_id
        escaped = [re.escape(t) for t in sorted(self.special_tokens.keys(), key=len, reverse=True)]
        self.pattern = re.compile("|".join(escaped))

    def split_with_specials(self, text):
        if not self.pattern:
            return [(text, False)]
        parts = []
        last_end = 0
        for match in self.pattern.finditer(text):
            if match.start() > last_end:
                parts.append((text[last_end:match.start()], False))
            parts.append((match.group(), True))
            last_end = match.end()
        if last_end < len(text):
            parts.append((text[last_end:], False))
        return parts
```

### 第 5 步：完整的词元化器类

把所有步骤串起来：规范化、按特殊词元切分、预词元化、BPE 合并、映射为 ID。

```python
import unicodedata

class ProductionTokenizer:
    def __init__(self):
        self.merges = {}
        self.vocab = {i: bytes([i]) for i in range(256)}
        self.special_handler = SpecialTokenHandler()
        self.next_id = 256

    def normalize(self, text):
        return unicodedata.normalize("NFKC", text)

    def train(self, text, num_merges):
        text = self.normalize(text)
        chunks = pre_tokenize(text)
        chunk_bytes = [list(chunk.encode("utf-8")) for chunk in chunks]

        for i in range(num_merges):
            pairs = Counter()
            for seq in chunk_bytes:
                for j in range(len(seq) - 1):
                    pairs[(seq[j], seq[j + 1])] += 1
            if not pairs:
                break
            best = max(pairs, key=pairs.get)
            new_id = self.next_id
            self.next_id += 1
            self.merges[best] = new_id
            self.vocab[new_id] = self.vocab[best[0]] + self.vocab[best[1]]
            chunk_bytes = [apply_merge(seq, best, new_id) for seq in chunk_bytes]

    def add_special_token(self, token_str):
        token_id = self.next_id
        self.next_id += 1
        self.special_handler.add_token(token_str, token_id)
        self.vocab[token_id] = token_str.encode("utf-8")
        return token_id

    def encode(self, text):
        text = self.normalize(text)
        parts = self.special_handler.split_with_specials(text)
        all_ids = []
        for part_text, is_special in parts:
            if is_special:
                all_ids.append(self.special_handler.special_tokens[part_text])
            else:
                for chunk in pre_tokenize(part_text):
                    byte_seq = list(chunk.encode("utf-8"))
                    for pair, new_id in self.merges.items():
                        byte_seq = apply_merge(byte_seq, pair, new_id)
                    all_ids.extend(byte_seq)
        return all_ids

    def decode(self, ids):
        byte_parts = []
        for token_id in ids:
            if token_id in self.vocab:
                byte_parts.append(self.vocab[token_id])
        return b"".join(byte_parts).decode("utf-8", errors="replace")

    def vocab_size(self):
        return len(self.vocab)
```

### 第 6 步：多语言测试

这才是真正的测试：把英语、中文、表情符号和代码全都交给它。

```python
corpus = (
    "The quick brown fox jumps over the lazy dog. "
    "The quick brown fox runs through the forest. "
    "Machine learning models process natural language. "
    "Deep learning transforms how we build software. "
    "def train(model, data): return model.fit(data) "
    "def predict(model, x): return model(x) "
)

tok = ProductionTokenizer()
tok.train(corpus, num_merges=50)

bos = tok.add_special_token("<|begin|>")
eos = tok.add_special_token("<|end|>")

test_texts = [
    "The quick brown fox.",
    "你好世界",
    "Hello 🌍 World",
    "def foo(x): return x + 1",
    f"<|begin|>Hello<|end|>",
]

for text in test_texts:
    ids = tok.encode(text)
    decoded = tok.decode(ids)
    print(f"Input:   {text}")
    print(f"Tokens:  {len(ids)} ids")
    print(f"Decoded: {decoded}")
    print()
```

汉字各产生 3 个字节，表情符号产生 4 个字节。它们都不会让词元化器崩溃，也都不会产生未知词元。这就是字节级 BPE 的力量。

## 学以致用

### 比较真实词元化器

加载 Llama 3、GPT-4 与 Mistral 的真实词元化器，观察它们如何处理同一个多语言段落。

```python
import tiktoken

gpt4_enc = tiktoken.get_encoding("cl100k_base")

test_paragraph = "Machine learning is powerful. 机器学习很强大。 L'apprentissage automatique est puissant. 🤖💪"

tokens = gpt4_enc.encode(test_paragraph)
pieces = [gpt4_enc.decode([t]) for t in tokens]
print(f"GPT-4 ({len(tokens)} tokens): {pieces}")
```

```python
from transformers import AutoTokenizer

llama_tok = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")
mistral_tok = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")

for name, tok in [("Llama 3", llama_tok), ("Mistral", mistral_tok)]:
    tokens = tok.encode(test_paragraph)
    pieces = tok.convert_ids_to_tokens(tokens)
    print(f"{name} ({len(tokens)} tokens): {pieces[:20]}...")
```

同一段文本会产生不同的词元数量。Llama 3 拥有 128K 词表，因此会更积极地合并常见模式；GPT-4 的 100K 词表居中；Mistral 的 32K 词表会产生更多词元，但嵌入层更小。

权衡始终相同：词表越大，序列越短，参数却越多。

## 交付成果

本课会生成一个用于构建和调试生产级词元化器的提示词。参见 `outputs/prompt-tokenizer-builder.md`。

## 练习

1. **简单：** 添加 `get_token_bytes(id)` 方法，显示任意词元 ID 对应的原始字节。用它检查最常用的合并词元究竟表示什么。
2. **中等：** 实现 Llama 风格的预词元化器：按空白和数字切分，但保留前导空格。在相同语料上，将它的词表与 GPT-2 正则表达式方案比较。
3. **困难：** 添加一个聊天模板方法，接收由 `{"role": ..., "content": ...}` 消息组成的列表，并生成符合 Llama 3 聊天格式的正确词元序列。用 HuggingFace 实现验证结果。

## 关键术语

| 术语 | 人们通常怎么说 | 实际含义 |
|------|----------------|----------------------|
| 字节级 BPE | “处理字节的词元化器” | 基础词表包含 256 种字节值的 BPE——可以处理任何输入且不会产生未知词元 |
| 预词元化 | “在 BPE 前切分” | 基于正则或规则的切分，可防止 BPE 跨越词边界合并 |
| NFKC 规范化 | “Unicode 清理” | 先进行规范分解，再进行兼容组合——“fi”连字会变成“fi”，全角“A”会变成“A” |
| 聊天模板 | “消息如何变成词元” | 把角色/内容消息列表转换为平坦词元序列的精确格式——因模型而异，且必须匹配训练格式 |
| 特殊词元 | “控制词元” | 绕过 BPE 的保留词元 ID——[BOS]、[EOS]、[PAD]、聊天标记——在合并前精确匹配 |
| 词元膨胀率（fertility） | “每个单词的词元数” | 输出词元数与输入单词数的比值——GPT-4 的英语约为 1.3，韩语为 2～3；越高意味着浪费的上下文越多 |
| tiktoken | “OpenAI 词元化器” | 带 Python 绑定的 Rust BPE 实现——比纯 Python 快 10～100 倍 |
| 合并表 | “词表” | 训练中学到的有序字节对合并列表——它就是词元化器学到的知识 |

## 延伸阅读

- [OpenAI tiktoken 源码](https://github.com/openai/tiktoken)——GPT-3.5/4 使用的 Rust BPE 实现
- [HuggingFace tokenizers](https://github.com/huggingface/tokenizers)——支持 BPE、WordPiece 与 Unigram 的 Rust 词元化器库
- [Llama 3 论文（Meta，2024）](https://arxiv.org/abs/2407.21783)——128K 词表与词元化器训练细节
- [SentencePiece（Kudo 与 Richardson，2018）](https://arxiv.org/abs/1808.06226)——与语言无关的词元化方法
- [GPT-2 词元化器源码](https://github.com/openai/gpt-2/blob/master/src/encoder.py)——最初的字节到 Unicode 映射
