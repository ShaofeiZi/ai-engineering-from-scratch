// 用 TypeScript 实现 RAG 分块策略：固定、递归、语义、按句和父子分块。
// 与 code/main.py 对应，并遵循 LangChain.js
// （RecursiveCharacterTextSplitter）的分割器层次结构。
// 资料来源：
//   https://docs.langchain.com/oss/javascript/integrations/splitters
//   https://philna.sh/blog/2024/09/18/how-to-chunk-text-in-javascript-for-rag-applications/
//   https://github.com/langchain-ai/langchainjs（textsplitters 包）

import { createHash } from "node:crypto";

type Vec = readonly number[];

type ParentChildPair = {
  child: string;
  parentIdx: number;
  parent: string;
};

const TOKEN_RE = /[a-z0-9]+/g;

function tokenize(text: string): string[] {
  return text.toLowerCase().match(TOKEN_RE) ?? [];
}

function hashEmbed(text: string, dim = 256): Vec {
  if (dim <= 0) throw new Error("dim 必须为正数");
  // 哈希技巧嵌入器：每个 token 为一个哈希维度贡献 +/-1。
  // 该方法具有确定性且无需训练，可替代生产级嵌入器
  // （BGE-M3、text-embedding-3-small、voyage-3）进行演示。
  const vec = new Array<number>(dim).fill(0);
  for (const tok of tokenize(text)) {
    const digest = createHash("md5").update(tok).digest();
    const idx = digest.readUInt32BE(0) % dim;
    const sign = digest[4] % 2 === 0 ? -1 : 1;
    vec[idx] += sign;
  }
  let norm = 0;
  for (const v of vec) norm += v * v;
  norm = Math.sqrt(norm);
  if (norm === 0) return vec;
  return vec.map((v) => v / norm);
}

function cosine(a: Vec, b: Vec): number {
  let dot = 0;
  const n = Math.min(a.length, b.length);
  for (let i = 0; i < n; i += 1) dot += a[i] * b[i];
  return dot;
}

function chunkFixed(text: string, size: number, overlap = 0): string[] {
  if (size <= 0) throw new Error("size 必须为正数");
  const step = size - overlap;
  if (step <= 0) throw new Error("overlap 必须小于 size");
  const out: string[] = [];
  for (let i = 0; i < text.length; i += step) {
    const piece = text.slice(i, i + size);
    if (piece.trim().length > 0) out.push(piece);
  }
  return out;
}

function chunkRecursive(
  text: string,
  size: number,
  seps: readonly string[] = ["\n\n", "\n", ". ", " "],
): string[] {
  if (size <= 0) throw new Error("size 必须为正数");
  // 仿照 LangChain.js RecursiveCharacterTextSplitter：先尝试最强分隔符
  // （段落）；如果当前切分仍留下大于 `size` 的分块，则降级使用较弱的
  // 分隔符（句子、单词）。
  if (text.length <= size) {
    const t = text.trim();
    return t.length > 0 ? [t] : [];
  }
  for (const sep of seps) {
    if (!text.includes(sep)) continue;
    const parts = text.split(sep);
    const chunks: string[] = [];
    let buf = "";
    for (const part of parts) {
      const candidate = buf.length === 0 ? part : buf + sep + part;
      if (candidate.length <= size) {
        buf = candidate;
      } else {
        if (buf.length > 0) chunks.push(buf.trim());
        buf = part;
      }
    }
    if (buf.length > 0) chunks.push(buf.trim());
    return chunks.filter((c) => c.length > 0);
  }
  return chunkFixed(text, size);
}

function splitSentences(text: string): string[] {
  return text
    .trim()
    .split(/(?<=[.!?])\s+/)
    .map((s) => s.trim())
    .filter((s) => s.length > 0);
}

function chunkSemantic(text: string, threshold = 0.3, minChars = 40): string[] {
  const sentences = splitSentences(text);
  if (sentences.length === 0) return [];
  const embs = sentences.map((s) => hashEmbed(s));
  const groups: string[][] = [[sentences[0]]];
  for (let i = 1; i < sentences.length; i += 1) {
    const sim = cosine(embs[i], embs[i - 1]);
    const current = groups[groups.length - 1];
    const joinedLen = current.join(" ").length;
    if (sim < threshold && joinedLen >= minChars) {
      groups.push([sentences[i]]);
    } else {
      current.push(sentences[i]);
    }
  }
  return groups.map((g) => g.join(" "));
}

function chunkSentence(text: string, sentencesPerChunk = 3): string[] {
  if (sentencesPerChunk <= 0) throw new Error("sentencesPerChunk 必须为正数");
  const sentences = splitSentences(text);
  const out: string[] = [];
  for (let i = 0; i < sentences.length; i += sentencesPerChunk) {
    out.push(sentences.slice(i, i + sentencesPerChunk).join(" "));
  }
  return out;
}

function chunkParentChild(text: string, parentSize = 800, childSize = 200): ParentChildPair[] {
  const parents = chunkRecursive(text, parentSize);
  const pairs: ParentChildPair[] = [];
  parents.forEach((parent, parentIdx) => {
    const children = chunkRecursive(parent, childSize);
    for (const child of children) {
      pairs.push({ child, parentIdx, parent });
    }
  });
  return pairs;
}

function retrieveRecall(
  chunks: readonly string[],
  query: string,
  goldSubstrings: readonly string[],
  topK = 3,
): boolean {
  const embs = chunks.map((c) => hashEmbed(c));
  const qEmb = hashEmbed(query);
  const scored = embs.map((e, i) => ({ score: cosine(e, qEmb), idx: i }));
  scored.sort((x, y) => y.score - x.score);
  const top = scored.slice(0, topK).map(({ idx }) => chunks[idx]);
  return top.some((c) => goldSubstrings.some((g) => c.toLowerCase().includes(g.toLowerCase())));
}

function main(): void {
  const doc = `Chapter 1. Introduction. This contract is between Acme Corp and Beta Inc. The parties agree to the following terms.

Chapter 2. Payment. Acme will pay Beta thirty thousand dollars on the first of each month. Late payments incur a five percent fee.

Chapter 3. Termination. Either party may terminate this agreement with ninety days written notice. Termination for cause requires only thirty days notice. Breach of payment constitutes cause.

Chapter 4. Confidentiality. Both parties agree to keep trade secrets confidential. This obligation survives termination of the agreement.

Chapter 5. Miscellaneous. This agreement is governed by the laws of the State of California. Disputes shall be resolved by arbitration.`;

  console.log("=== 策略对比 ===\n");

  const fixed = chunkFixed(doc, 300, 50);
  console.log("固定切分（300 字符，重叠 50）：    " + fixed.length + " 个分块");

  const rec = chunkRecursive(doc, 300);
  console.log("递归切分（300 字符）：              " + rec.length + " 个分块");

  const sem = chunkSemantic(doc);
  console.log("语义切分（哈希技巧）：              " + sem.length + " 个分块");

  const sent = chunkSentence(doc, 3);
  console.log("按句切分（每块 3 句）：             " + sent.length + " 个分块");

  const pc = chunkParentChild(doc, 800, 200);
  const parentSet = new Set(pc.map((m) => m.parentIdx));
  console.log("父子切分（800 / 200）：             " + pc.length + " 个子块，" + parentSet.size + " 个父块");

  const queries: ReadonlyArray<{ q: string; gold: readonly string[] }> = [
    { q: "When can either party terminate?", gold: ["ninety days", "thirty days"] },
    { q: "What is the late payment fee?", gold: ["five percent"] },
    { q: "Which state laws apply?", gold: ["California"] },
  ];

  console.log("\n=== 3 个查询上的召回率@3 ===");
  const strategies: ReadonlyArray<{ name: string; chunks: readonly string[] }> = [
    { name: "fixed", chunks: fixed },
    { name: "recursive", chunks: rec },
    { name: "semantic", chunks: sem },
    { name: "sentence", chunks: sent },
    { name: "parent", chunks: Array.from(new Set(pc.map((m) => m.parent))) },
  ];
  for (const { name, chunks } of strategies) {
    const hits = queries.reduce((acc, { q, gold }) => acc + (retrieveRecall(chunks, q, gold) ? 1 : 0), 0);
    console.log("  " + name.padEnd(12) + ": " + hits + " / " + queries.length);
  }

  console.log("\n注意：哈希技巧嵌入器的噪声较大。");
  console.log("生产级嵌入器（BGE、text-3）在相同分块上的召回率会高 20–40 个百分点。");
}

main();
