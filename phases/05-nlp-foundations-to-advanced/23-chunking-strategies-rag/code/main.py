import hashlib
import math
import re


def tokenize(text):
    return re.findall(r"[a-z0-9]+", text.lower())


def hash_embed(text, dim=256):
    vec = [0.0] * dim
    for tok in tokenize(text):
        h = hashlib.md5(tok.encode()).digest()
        idx = int.from_bytes(h[:4], "big") % dim
        sign = 1.0 if h[4] % 2 else -1.0
        vec[idx] += sign
    norm = math.sqrt(sum(v * v for v in vec))
    if norm == 0:
        return vec
    return [v / norm for v in vec]


def cosine(a, b):
    return sum(x * y for x, y in zip(a, b))


def chunk_fixed(text, size, overlap=0):
    if size <= 0:
        raise ValueError("size 必须为正数")
    step = size - overlap
    if step <= 0:
        raise ValueError("overlap 必须小于 size")
    return [text[i:i + size] for i in range(0, len(text), step) if text[i:i + size].strip()]


def chunk_recursive(text, size, seps=("\n\n", "\n", ". ", " ")):
    if len(text) <= size:
        return [text.strip()] if text.strip() else []
    for sep in seps:
        if sep not in text:
            continue
        parts = text.split(sep)
        chunks = []
        buf = ""
        for p in parts:
            candidate = buf + sep + p if buf else p
            if len(candidate) <= size:
                buf = candidate
            else:
                if buf:
                    chunks.append(buf.strip())
                buf = p
        if buf:
            chunks.append(buf.strip())
        return [c for c in chunks if c]
    return chunk_fixed(text, size)


def split_sentences(text):
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def chunk_semantic(text, threshold=0.3, min_chars=40):
    sentences = split_sentences(text)
    if not sentences:
        return []
    embs = [hash_embed(s) for s in sentences]
    chunks = [[sentences[0]]]
    for i in range(1, len(sentences)):
        sim = cosine(embs[i], embs[i - 1])
        if sim < threshold and len(" ".join(chunks[-1])) >= min_chars:
            chunks.append([sentences[i]])
        else:
            chunks[-1].append(sentences[i])
    return [" ".join(c) for c in chunks]


def chunk_sentence(text, sentences_per_chunk=3):
    sentences = split_sentences(text)
    return [" ".join(sentences[i:i + sentences_per_chunk])
            for i in range(0, len(sentences), sentences_per_chunk)]


def chunk_parent_child(text, parent_size=800, child_size=200):
    parents = chunk_recursive(text, size=parent_size)
    mapping = []
    for p_idx, parent in enumerate(parents):
        children = chunk_recursive(parent, size=child_size)
        for child in children:
            mapping.append({"child": child, "parent_idx": p_idx, "parent": parent})
    return mapping


def retrieve_recall(chunks, query, gold_substrings, top_k=3):
    chunk_embs = [hash_embed(c) for c in chunks]
    q_emb = hash_embed(query)
    scored = sorted([(cosine(e, q_emb), i) for i, e in enumerate(chunk_embs)], reverse=True)
    top_texts = [chunks[i] for _, i in scored[:top_k]]
    return any(any(g.lower() in c.lower() for g in gold_substrings) for c in top_texts)


def main():
    doc = """Chapter 1. Introduction. This contract is between Acme Corp and Beta Inc. The parties agree to the following terms.

Chapter 2. Payment. Acme will pay Beta thirty thousand dollars on the first of each month. Late payments incur a five percent fee.

Chapter 3. Termination. Either party may terminate this agreement with ninety days written notice. Termination for cause requires only thirty days notice. Breach of payment constitutes cause.

Chapter 4. Confidentiality. Both parties agree to keep trade secrets confidential. This obligation survives termination of the agreement.

Chapter 5. Miscellaneous. This agreement is governed by the laws of the State of California. Disputes shall be resolved by arbitration."""

    print("=== 策略对比 ===")
    print()

    fixed = chunk_fixed(doc, size=300, overlap=50)
    print(f"固定切分（300 字符，重叠 50）：    {len(fixed)} 个分块")

    rec = chunk_recursive(doc, size=300)
    print(f"递归切分（300 字符）：              {len(rec)} 个分块")

    sem = chunk_semantic(doc)
    print(f"语义切分（哈希技巧）：              {len(sem)} 个分块")

    sent = chunk_sentence(doc, sentences_per_chunk=3)
    print(f"按句切分（每块 3 句）：             {len(sent)} 个分块")

    pc = chunk_parent_child(doc, parent_size=800, child_size=200)
    parents = {m["parent_idx"] for m in pc}
    print(f"父子切分（800 / 200）：             {len(pc)} 个子块，{len(parents)} 个父块")

    queries = [
        ("When can either party terminate?", ["ninety days", "thirty days"]),
        ("What is the late payment fee?", ["five percent"]),
        ("Which state laws apply?", ["California"]),
    ]

    print()
    print("=== 3 个查询上的召回率@3 ===")
    for name, chunks in [("fixed", fixed), ("recursive", rec), ("semantic", sem),
                         ("sentence", sent), ("parent", [m["parent"] for m in pc])]:
        hits = sum(retrieve_recall(chunks, q, gold) for q, gold in queries)
        print(f"  {name:<12}: {hits} / {len(queries)}")

    print()
    print("注意：哈希技巧嵌入器的噪声较大。")
    print("生产级嵌入器（BGE、text-3）在相同分块上的召回率会高 20–40 个百分点。")


if __name__ == "__main__":
    main()
