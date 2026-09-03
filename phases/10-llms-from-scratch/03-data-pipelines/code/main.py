import re
import hashlib
import random
import time
from collections import Counter, defaultdict


def clean_text(text):
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^\x20-\x7E\n]", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r" {2,}", " ", text)
    return text.strip()


def quality_filter(text, min_words=50, max_ratio_caps=0.3, max_ratio_special=0.1):
    words = text.split()
    if len(words) < min_words:
        return False
    caps_ratio = sum(1 for w in words if w.isupper()) / len(words)
    if caps_ratio > max_ratio_caps:
        return False
    special_chars = sum(1 for c in text if not c.isalnum() and not c.isspace())
    if special_chars / max(len(text), 1) > max_ratio_special:
        return False
    return True


def get_shingles(text, k=5):
    words = text.lower().split()
    if len(words) < k:
        return set()
    return {" ".join(words[i:i + k]) for i in range(len(words) - k + 1)}


def minhash_signature(shingles, num_hashes=128):
    signature = []
    for i in range(num_hashes):
        min_hash = float("inf")
        for shingle in shingles:
            h = int(hashlib.sha256(f"{i}:{shingle}".encode()).hexdigest(), 16)
            min_hash = min(min_hash, h)
        if min_hash == float("inf"):
            min_hash = 0
        signature.append(min_hash)
    return signature


def lsh_buckets(signature, bands=16):
    rows_per_band = len(signature) // bands
    buckets = []
    for b in range(bands):
        start = b * rows_per_band
        band_data = tuple(signature[start:start + rows_per_band])
        bucket_hash = hashlib.md5(str(band_data).encode()).hexdigest()
        buckets.append((b, bucket_hash))
    return buckets


def deduplicate(documents, threshold=0.8, num_hashes=128, bands=16):
    signatures = []
    shingle_sets = []
    for doc in documents:
        shingles = get_shingles(doc)
        shingle_sets.append(shingles)
        signatures.append(minhash_signature(shingles, num_hashes))

    bucket_map = defaultdict(list)
    for doc_idx, sig in enumerate(signatures):
        for band_id, bucket_hash in lsh_buckets(sig, bands):
            bucket_map[(band_id, bucket_hash)].append(doc_idx)

    duplicate_pairs = set()
    for bucket_docs in bucket_map.values():
        if len(bucket_docs) < 2:
            continue
        for i in range(len(bucket_docs)):
            for j in range(i + 1, len(bucket_docs)):
                duplicate_pairs.add((bucket_docs[i], bucket_docs[j]))

    removed = set()
    for i, j in duplicate_pairs:
        if i in removed or j in removed:
            continue
        s1, s2 = shingle_sets[i], shingle_sets[j]
        if not s1 or not s2:
            continue
        jaccard = len(s1 & s2) / len(s1 | s2)
        if jaccard >= threshold:
            removed.add(j)

    return [doc for idx, doc in enumerate(documents) if idx not in removed], len(removed)


class SimpleTokenizer:
    def __init__(self, vocab_size=256):
        self.vocab = {i: bytes([i]) for i in range(256)}
        self.merges = {}
        self.next_id = 256
        self.eos_id = None
        self.pad_id = 0

    def train_bpe(self, text, num_merges):
        tokens = list(text.encode("utf-8"))
        for i in range(num_merges):
            pairs = Counter()
            for j in range(len(tokens) - 1):
                pairs[(tokens[j], tokens[j + 1])] += 1
            if not pairs:
                break
            best = max(pairs, key=pairs.get)
            new_id = self.next_id
            self.next_id += 1
            self.merges[best] = new_id
            self.vocab[new_id] = self.vocab[best[0]] + self.vocab[best[1]]
            merged = []
            j = 0
            while j < len(tokens):
                if j < len(tokens) - 1 and tokens[j] == best[0] and tokens[j + 1] == best[1]:
                    merged.append(new_id)
                    j += 2
                else:
                    merged.append(tokens[j])
                    j += 1
            tokens = merged

        self.eos_id = self.next_id
        self.vocab[self.eos_id] = b"<EOS>"
        self.next_id += 1

    def encode(self, text):
        tokens = list(text.encode("utf-8"))
        for pair, new_id in self.merges.items():
            merged = []
            j = 0
            while j < len(tokens):
                if j < len(tokens) - 1 and tokens[j] == pair[0] and tokens[j + 1] == pair[1]:
                    merged.append(new_id)
                    j += 2
                else:
                    merged.append(tokens[j])
                    j += 1
            tokens = merged
        return tokens

    def decode(self, ids):
        byte_parts = []
        for token_id in ids:
            if token_id in self.vocab and token_id != self.eos_id:
                byte_parts.append(self.vocab[token_id])
        return b"".join(byte_parts).decode("utf-8", errors="replace")

    def vocab_size(self):
        return len(self.vocab)


def tokenize_corpus(documents, tokenizer):
    all_tokens = []
    for doc in documents:
        tokens = tokenizer.encode(doc)
        all_tokens.extend(tokens)
        all_tokens.append(tokenizer.eos_id)
    return all_tokens


def pack_sequences(token_ids, seq_length, pad_id=0):
    sequences = []
    attention_masks = []
    for i in range(0, len(token_ids), seq_length):
        seq = token_ids[i:i + seq_length]
        mask = [1] * len(seq)
        if len(seq) < seq_length:
            pad_count = seq_length - len(seq)
            seq = seq + [pad_id] * pad_count
            mask = mask + [0] * pad_count
        sequences.append(seq)
        attention_masks.append(mask)
    return sequences, attention_masks


class PreTrainingDataLoader:
    def __init__(self, sequences, attention_masks, batch_size, shuffle=True):
        self.sequences = sequences
        self.attention_masks = attention_masks
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __len__(self):
        return (len(self.sequences) + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        indices = list(range(len(self.sequences)))
        if self.shuffle:
            random.shuffle(indices)
        for start in range(0, len(indices), self.batch_size):
            batch_idx = indices[start:start + self.batch_size]
            batch_seqs = [self.sequences[i] for i in batch_idx]
            batch_masks = [self.attention_masks[i] for i in batch_idx]
            yield batch_seqs, batch_masks


def compute_statistics(documents, token_ids, sequences, tokenizer_vocab_size):
    total_chars = sum(len(d) for d in documents)
    total_tokens = len(token_ids)
    unique_tokens = len(set(token_ids))
    compression_ratio = total_chars / max(total_tokens, 1)

    doc_lengths = [len(d.split()) for d in documents]
    avg_doc_length = sum(doc_lengths) / max(len(doc_lengths), 1)
    max_doc_length = max(doc_lengths) if doc_lengths else 0
    min_doc_length = min(doc_lengths) if doc_lengths else 0

    token_counts = Counter(token_ids)
    top_tokens = token_counts.most_common(10)

    non_pad_tokens = sum(sum(1 for t in seq if t != 0) for seq in sequences)
    total_positions = sum(len(seq) for seq in sequences)
    utilization = non_pad_tokens / max(total_positions, 1)

    return {
        "total_documents": len(documents),
        "total_characters": total_chars,
        "total_tokens": total_tokens,
        "unique_tokens": unique_tokens,
        "vocab_utilization": unique_tokens / max(tokenizer_vocab_size, 1),
        "compression_ratio": compression_ratio,
        "avg_doc_length_words": avg_doc_length,
        "max_doc_length_words": max_doc_length,
        "min_doc_length_words": min_doc_length,
        "num_sequences": len(sequences),
        "sequence_utilization": utilization,
        "top_10_tokens": top_tokens,
    }


def generate_sample_corpus():
    base_docs = [
        "Machine learning is a subset of artificial intelligence that provides systems the ability "
        "to automatically learn and improve from experience without being explicitly programmed. "
        "Machine learning focuses on the development of computer programs that can access data and "
        "use it to learn for themselves. The process of learning begins with observations or data, "
        "such as examples, direct experience, or instruction, in order to look for patterns in data "
        "and make better decisions in the future based on the examples that we provide.",

        "Deep learning is part of a broader family of machine learning methods based on artificial "
        "neural networks with representation learning. Learning can be supervised, semi-supervised "
        "or unsupervised. Deep learning architectures such as deep neural networks, recurrent neural "
        "networks, convolutional neural networks and transformers have been applied to fields "
        "including natural language processing, speech recognition, computer vision, and many other tasks.",

        "Natural language processing is a subfield of linguistics, computer science, and artificial "
        "intelligence concerned with the interactions between computers and human language, in "
        "particular how to program computers to process and analyze large amounts of natural language "
        "data. The result is a computer capable of understanding the contents of documents, including "
        "the contextual nuances of the language within them.",

        "Transformers are a type of neural network architecture that has become the dominant approach "
        "for natural language processing tasks. The key innovation is the self-attention mechanism, "
        "which allows the model to weigh the importance of different parts of the input when producing "
        "each part of the output. This enables transformers to capture long-range dependencies in text "
        "much more effectively than previous recurrent approaches.",

        "The attention mechanism in neural networks allows the model to focus on relevant parts of "
        "the input sequence when generating each element of the output. In the transformer architecture, "
        "multi-head attention computes attention in parallel across multiple representation subspaces, "
        "enabling the model to jointly attend to information from different representation subspaces "
        "at different positions in the sequence.",

        "Reinforcement learning is an area of machine learning concerned with how intelligent agents "
        "ought to take actions in an environment in order to maximize the notion of cumulative reward. "
        "Reinforcement learning is one of three basic machine learning paradigms, alongside supervised "
        "learning and unsupervised learning. It differs from supervised learning in that correct input "
        "and output pairs need not be presented.",

        "Computer vision is an interdisciplinary scientific field that deals with how computers can "
        "gain high-level understanding from digital images or videos. From the perspective of "
        "engineering, it seeks to understand and automate tasks that the human visual system can do. "
        "Computer vision tasks include methods for acquiring, processing, analyzing and understanding "
        "digital images, and extraction of high-dimensional data from the real world.",

        "Convolutional neural networks are a class of deep learning architecture commonly applied to "
        "analyze visual imagery. They use a variation of multilayer perceptrons designed to require "
        "minimal preprocessing. They are also known as shift invariant or space invariant artificial "
        "neural networks based on their shared-weights architecture and translation invariance "
        "characteristics. Convolutional networks were inspired by biological processes.",

        "Generative adversarial networks consist of two neural networks that contest with each other "
        "in the form of a zero-sum game, where one agent gain is another agent loss. Given a training "
        "set, this technique learns to generate new data with the same statistics as the training set. "
        "For example, a generative adversarial network trained on photographs can generate new "
        "photographs that look authentic to human observers.",

        "Transfer learning is a machine learning method where a model developed for a task is reused "
        "as the starting point for a model on a second task. It is a popular approach in deep learning "
        "where pre-trained models are used as the starting point on computer vision and natural language "
        "processing tasks given the vast compute and time resources required to develop neural network "
        "models on these problems and the large improvements they provide.",
    ]

    near_dup_1 = (
        "Machine learning is a subset of artificial intelligence that provides systems the ability "
        "to automatically learn and improve from experience. Machine learning focuses on developing "
        "computer programs that can access data and use it to learn for themselves. The learning "
        "process begins with observations or data, such as examples or direct experience, in order "
        "to look for patterns and make better decisions based on the examples provided."
    )

    near_dup_2 = (
        "Deep learning is part of a broader family of machine learning methods based on artificial "
        "neural networks with representation learning. Learning can be supervised, semi-supervised "
        "or unsupervised. Deep learning architectures such as deep neural networks, recurrent neural "
        "networks, convolutional neural networks and transformers have been applied to fields "
        "including natural language processing, speech recognition, computer vision, and many other tasks."
    )

    short_doc = "This is too short to be useful."

    html_doc = (
        "<html><body><h1>Title</h1><p>Machine learning is transforming how we build software. "
        "Deep neural networks can learn complex patterns from data. The transformer architecture "
        "has become the dominant approach for language tasks. Self-attention allows models to capture "
        "long-range dependencies. Pre-training on large corpora produces strong foundation models. "
        "Fine-tuning adapts these models to specific tasks with minimal additional data.</p></body></html>"
    )

    spam_doc = "BUY NOW CLICK HERE FREE MONEY GUARANTEED RESULTS " * 20

    docs = base_docs + [near_dup_1, near_dup_2, short_doc, html_doc, spam_doc]
    return docs


def run_pipeline():
    print("=" * 60)
    print("预训的数据管道")
    print("=" * 60)

    raw_docs = generate_sample_corpus()
    print(f"\n原始文档数：{len(raw_docs)}")

    print("\n - - - 阶段1:清理 - -")
    cleaned_docs = [clean_text(doc) for doc in raw_docs]
    print(f"移除 HTML 后：{len(cleaned_docs)} 篇文档")

    print("\n - - - 阶段2: 质量过滤 - -")
    filtered_docs = [doc for doc in cleaned_docs if quality_filter(doc)]
    removed_quality = len(cleaned_docs) - len(filtered_docs)
    print(f"已移除 {removed_quality} 篇低质量文档")
    print(f"剩余：{len(filtered_docs)} 篇文档")

    print("\n-- 第3阶段:重复(MinHash+LSH) ---")
    start = time.time()
    deduped_docs, num_removed = deduplicate(filtered_docs, threshold=0.8)
    dedup_time = time.time() - start
    print(f"在 {dedup_time:.2f} 秒内移除 {num_removed} 篇近重复文档")
    print(f"剩余：{len(deduped_docs)} 篇文档")

    print("\n - - - 阶段4:切换 - -")
    all_text = " ".join(deduped_docs)
    tokenizer = SimpleTokenizer()
    start = time.time()
    tokenizer.train_bpe(all_text, num_merges=100)
    train_time = time.time() - start
    print(f"在 {train_time:.2f} 秒内训练出包含 {tokenizer.vocab_size()} 个 token 的 tokenizer")

    start = time.time()
    token_ids = tokenize_corpus(deduped_docs, tokenizer)
    tok_time = time.time() - start
    print(f"在 {tok_time:.2f} 秒内完成 {len(token_ids):,} 个 token 的分词（{len(token_ids)/max(tok_time, 0.001):,.0f} token/秒）")

    print("\n -- -- 第5阶段:顺序包装 -- --")
    seq_length = 128
    sequences, masks = pack_sequences(token_ids, seq_length, pad_id=0)
    print(f"已打包为 {len(sequences)} 个长度为 {seq_length} 的序列")

    print("\n-- 第6阶段: DataLoader-")
    batch_size = 4
    loader = PreTrainingDataLoader(sequences, masks, batch_size)
    print(f"DataLoader：{len(loader)} 个批次，batch size 为 {batch_size}")

    batch_count = 0
    total_tokens_served = 0
    for batch_seqs, batch_masks in loader:
        batch_count += 1
        total_tokens_served += sum(sum(m) for m in batch_masks)
        if batch_count <= 2:
            print(f"\n  批次 {batch_count}：")
            print(f"    序列数：{len(batch_seqs)}")
            print(f"    第一个序列（前 20 个 token）：{batch_seqs[0][:20]}...")
            print(f"    第一个 mask（前 20 项）：{batch_masks[0][:20]}...")
    print(f"\n  已提供的批次总数：{batch_count}")
    print(f"  已提供的非 padding token 总数：{total_tokens_served:,}")

    print("\n -- -- 数据集统计 -- --")
    stats = compute_statistics(deduped_docs, token_ids, sequences, tokenizer.vocab_size())
    print(f"  文档数：      {stats['total_documents']}")
    print(f"  字符总数：    {stats['total_characters']:,}")
    print(f"  token 总数：  {stats['total_tokens']:,}")
    print(f"  唯一 token 数：{stats['unique_tokens']}")
    print(f"  词表利用率：  {stats['vocab_utilization']:.1%}")
    print(f"  压缩比：      {stats['compression_ratio']:.2f} 字符/token")
    print(f"  平均文档长度：{stats['avg_doc_length_words']:.0f} 个单词")
    print(f"  序列数：      {stats['num_sequences']}")
    print(f"  序列利用率：  {stats['sequence_utilization']:.1%}")

    print("\n -- -- 管道摘要 -- --")
    print(f"  原始文档：   {len(raw_docs)}")
    print(f"  清洗后：     {len(cleaned_docs)}")
    print(f"  质量过滤后： {len(filtered_docs)}（-{removed_quality}）")
    print(f"  去重后：     {len(deduped_docs)}（-{num_removed}）")
    print(f"  最终 token： {len(token_ids):,}")
    print(f"  训练序列：   {len(sequences)}")
    print(f"  训练批次：   {len(loader)}")


if __name__ == "__main__":
    run_pipeline()
