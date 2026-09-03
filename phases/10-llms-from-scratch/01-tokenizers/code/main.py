from collections import Counter


class CharTokenizer:
    def encode(self, text):
        return [ord(c) for c in text]

    def decode(self, tokens):
        return "".join(chr(t) for t in tokens)


class BPETokenizer:
    def __init__(self):
        self.merges = {}
        self.vocab = {}

    def _get_pairs(self, tokens):
        pairs = Counter()
        for i in range(len(tokens) - 1):
            pairs[(tokens[i], tokens[i + 1])] += 1
        return pairs

    def _merge_pair(self, tokens, pair, new_token):
        merged = []
        i = 0
        while i < len(tokens):
            if i < len(tokens) - 1 and tokens[i] == pair[0] and tokens[i + 1] == pair[1]:
                merged.append(new_token)
                i += 2
            else:
                merged.append(tokens[i])
                i += 1
        return merged

    def train(self, text, num_merges):
        tokens = list(text.encode("utf-8"))
        self.vocab = {i: bytes([i]) for i in range(256)}

        for i in range(num_merges):
            pairs = self._get_pairs(tokens)
            if not pairs:
                break
            best_pair = max(pairs, key=pairs.get)
            new_token = 256 + i
            tokens = self._merge_pair(tokens, best_pair, new_token)
            self.merges[best_pair] = new_token
            self.vocab[new_token] = self.vocab[best_pair[0]] + self.vocab[best_pair[1]]

        return self

    def encode(self, text):
        tokens = list(text.encode("utf-8"))
        for pair, new_token in self.merges.items():
            tokens = self._merge_pair(tokens, pair, new_token)
        return tokens

    def decode(self, tokens):
        byte_sequence = b"".join(self.vocab[t] for t in tokens)
        return byte_sequence.decode("utf-8", errors="replace")

    def vocab_size(self):
        return len(self.vocab)

    def token_to_str(self, token_id):
        return self.vocab.get(token_id, b"<?>").decode("utf-8", errors="replace")


def compression_ratio(tokenizer, text):
    encoded = tokenizer.encode(text)
    raw_bytes = len(text.encode("utf-8"))
    return len(encoded) / raw_bytes


def vocabulary_stats(tokenizer, texts):
    total_tokens = 0
    total_words = 0
    token_usage = Counter()

    for text in texts:
        encoded = tokenizer.encode(text)
        total_tokens += len(encoded)
        total_words += len(text.split())
        for t in encoded:
            token_usage[t] += 1

    avg_tokens_per_word = total_tokens / total_words if total_words > 0 else 0

    print(f"词表大小：{tokenizer.vocab_size()}")
    print(f"每个单词的平均 token 数：{avg_tokens_per_word:.2f}")
    print(f"使用过的唯一 token 总数：{len(token_usage)}")

    print("\n使用频率最高的 10 个 token：")
    for token_id, count in token_usage.most_common(10):
        display = tokenizer.token_to_str(token_id)
        print(f"  {token_id:4d}: '{display}' x{count}")

    unused = tokenizer.vocab_size() - len(token_usage)
    print(f"\n未使用的 token：{unused}/{tokenizer.vocab_size()}")


def demo_char_tokenizer():
    print("=" * 60)
    print("步骤1:字符级 Tokenizer")
    print("=" * 60)

    ct = CharTokenizer()

    texts = ["hello", "Hello, world!", "GPT-4"]
    for text in texts:
        encoded = ct.encode(text)
        decoded = ct.decode(encoded)
        print(f"  '{text}' -> {encoded}")
        print(f"  往返一致性：{'通过' if decoded == text else '失败'}")
        print(f"  token 数：{len(encoded)}")
        print()


def demo_bpe_training():
    print("=" * 60)
    print("步骤 2：BPE 训练")
    print("=" * 60)

    corpus = (
        "The cat sat on the mat. The cat ate the rat. "
        "The dog sat on the log. The dog ate the frog. "
        "Natural language processing is the study of how computers "
        "understand and generate human language. "
        "Tokenization is the first step in any NLP pipeline. "
        "Language models read tokens, not words. "
        "The tokenizer converts text into a sequence of integers. "
        "Each integer maps to a subword in the vocabulary."
    )

    tokenizer = BPETokenizer()
    tokenizer.train(corpus, num_merges=50)

    print(f"\n训练后的词表大小：{tokenizer.vocab_size()}")
    print(f"学到的合并规则数：{len(tokenizer.merges)}")

    return tokenizer, corpus


def demo_encode_decode(tokenizer):
    print("\n" + "=" * 60)
    print("步骤3:编码和解码")
    print("=" * 60)

    test_sentences = [
        "The cat sat on the mat.",
        "Natural language processing",
        "tokenization pipeline",
        "unhappiness",
        "The dog ate the frog.",
    ]

    for sentence in test_sentences:
        encoded = tokenizer.encode(sentence)
        decoded = tokenizer.decode(encoded)
        raw_bytes = len(sentence.encode("utf-8"))
        ratio = len(encoded) / raw_bytes
        roundtrip = "PASS" if decoded == sentence else "FAIL"
        print(f"\n  '{sentence}'")
        print(f"  编码结果：{encoded[:15]}{'...' if len(encoded) > 15 else ''}")
        print(f"  token 数：{len(encoded)}（原文 {raw_bytes} 字节）")
        print(f"  压缩比：{ratio:.2f}")
        print(f"  往返一致性：{roundtrip}")


def demo_tiktoken_comparison(tokenizer):
    print("\n" + "=" * 60)
    print("步骤 4：与 tiktoken 比较")
    print("=" * 60)

    try:
        import tiktoken
    except ImportError:
        print("未安装 tiktoken。请运行：pip install tiktoken")
        return

    enc = tiktoken.get_encoding("cl100k_base")

    texts = [
        "The cat sat on the mat.",
        "unhappiness",
        "Hello, world!",
        "def fibonacci(n): return n if n < 2 else fibonacci(n-1) + fibonacci(n-2)",
        "Geschwindigkeitsbegrenzung",
    ]

    for text in texts:
        our_tokens = tokenizer.encode(text)
        tk_tokens = enc.encode(text)
        tk_pieces = [enc.decode([t]) for t in tk_tokens]
        print(f"\n  '{text}'")
        print(f"  本课程 BPE：{len(our_tokens)} 个 token")
        print(f"  tiktoken：  {len(tk_tokens)} 个 token -> {tk_pieces}")
        ratio = len(our_tokens) / len(tk_tokens) if len(tk_tokens) > 0 else 0
        print(f"  本课程 BPE / tiktoken：{ratio:.1f}x")


def demo_vocabulary_analysis(tokenizer, corpus):
    print("\n" + "=" * 60)
    print("步骤5:词汇分析")
    print("=" * 60)

    test_texts = [
        corpus,
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is a subset of artificial intelligence.",
        "Python is the most popular language for data science.",
    ]

    vocabulary_stats(tokenizer, test_texts)

    print("\n压缩比：")
    for text in test_texts[:3]:
        preview = text[:50] + "..." if len(text) > 50 else text
        ratio = compression_ratio(tokenizer, text)
        print(f"  {ratio:.2f} -- '{preview}'")


if __name__ == "__main__":
    demo_char_tokenizer()
    tokenizer, corpus = demo_bpe_training()
    demo_encode_decode(tokenizer)
    demo_tiktoken_comparison(tokenizer)
    demo_vocabulary_analysis(tokenizer, corpus)
