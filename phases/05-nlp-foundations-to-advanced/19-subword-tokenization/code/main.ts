// 用 TypeScript 实现子词 tokenization：从零完成 BPE 训练和编码。
// 与 code/main.py 对应，推理循环采用 tiktoken 和 microsoft/Tokenizer
// 使用的合并排名字典方法。
// 资料来源：
//   https://github.com/openai/tiktoken（教学版 BPE）
//   https://github.com/microsoft/Tokenizer（tiktoken 的 TypeScript 移植版）
//   https://sebastianraschka.com/blog/2025/bpe-from-scratch.html

type Sym = string;
type Word = readonly Sym[];
type Pair = readonly [Sym, Sym];
type Merge = Pair;

type WordCounts = Map<string, number>;
type Vocab = Map<Word, number>;

const WORD_TOKEN_RE = /[a-zA-Z]+/g;
const END_OF_WORD = "</w>";
const PAIR_SEP = "␟";

function pairKey(a: Sym, b: Sym): string {
  return a + PAIR_SEP + b;
}

function wordCounts(text: string): WordCounts {
  const counts: WordCounts = new Map();
  const matches = text.toLowerCase().match(WORD_TOKEN_RE) ?? [];
  for (const word of matches) {
    counts.set(word, (counts.get(word) ?? 0) + 1);
  }
  return counts;
}

function initVocab(counts: WordCounts): Vocab {
  const vocab: Vocab = new Map();
  for (const [word, freq] of counts) {
    const symbols: Sym[] = [...word, END_OF_WORD];
    vocab.set(Object.freeze(symbols), freq);
  }
  return vocab;
}

type PairCounts = Map<string, { pair: Pair; count: number }>;

function pairCounts(vocab: Vocab): PairCounts {
  const pairs: PairCounts = new Map();
  for (const [symbols, freq] of vocab) {
    for (let i = 0; i < symbols.length - 1; i += 1) {
      const a = symbols[i];
      const b = symbols[i + 1];
      const key = pairKey(a, b);
      const entry = pairs.get(key);
      if (entry) {
        entry.count += freq;
      } else {
        pairs.set(key, { pair: [a, b] as const, count: freq });
      }
    }
  }
  return pairs;
}

function bestPair(pairs: PairCounts): Pair | undefined {
  let best: { pair: Pair; count: number } | undefined;
  for (const entry of pairs.values()) {
    if (!best || entry.count > best.count) {
      best = entry;
    }
  }
  return best?.pair;
}

function mergePair(vocab: Vocab, pair: Pair): Vocab {
  const [a, b] = pair;
  const merged = a + b;
  const next: Vocab = new Map();
  for (const [symbols, freq] of vocab) {
    const out: Sym[] = [];
    let i = 0;
    while (i < symbols.length) {
      if (i < symbols.length - 1 && symbols[i] === a && symbols[i + 1] === b) {
        out.push(merged);
        i += 2;
      } else {
        out.push(symbols[i]);
        i += 1;
      }
    }
    next.set(Object.freeze(out), freq);
  }
  return next;
}

function trainBpe(text: string, numMerges: number): { merges: Merge[]; tokens: Sym[] } {
  const counts = wordCounts(text);
  if (counts.size === 0) {
    throw new Error("wordCounts：语料库未生成任何单词");
  }
  let vocab = initVocab(counts);
  const merges: Merge[] = [];
  for (let step = 0; step < numMerges; step += 1) {
    const pairs = pairCounts(vocab);
    if (pairs.size === 0) break;
    const winner = bestPair(pairs);
    if (!winner) break;
    merges.push(winner);
    vocab = mergePair(vocab, winner);
  }
  const tokens = new Set<Sym>();
  for (const symbols of vocab.keys()) {
    for (const s of symbols) tokens.add(s);
  }
  return { merges, tokens: [...tokens].sort() };
}

function encodeBpe(word: string, merges: readonly Merge[]): Sym[] {
  let symbols: Sym[] = [...word, END_OF_WORD];
  for (const [a, b] of merges) {
    const merged = a + b;
    let i = 0;
    while (i < symbols.length - 1) {
      if (symbols[i] === a && symbols[i + 1] === b) {
        symbols = [...symbols.slice(0, i), merged, ...symbols.slice(i + 2)];
      } else {
        i += 1;
      }
    }
  }
  return symbols;
}

function rankedEncode(word: string, merges: readonly Merge[]): Sym[] {
  // 合并排名查找：生产级 tokenizer（tiktoken、HF）根据每个相邻对在合并
  // 列表中的位置为其评分，并优先合并排名最低者。结果与 encodeBpe 相同，
  // 复杂度近似与词长呈线性关系。
  const ranks: Map<string, number> = new Map();
  merges.forEach(([a, b], idx) => {
    ranks.set(pairKey(a, b), idx);
  });

  let symbols: Sym[] = [...word, END_OF_WORD];
  for (;;) {
    let bestIdx = -1;
    let bestRank = Infinity;
    for (let i = 0; i < symbols.length - 1; i += 1) {
      const rank = ranks.get(pairKey(symbols[i], symbols[i + 1]));
      if (rank !== undefined && rank < bestRank) {
        bestRank = rank;
        bestIdx = i;
      }
    }
    if (bestIdx === -1) break;
    const merged = symbols[bestIdx] + symbols[bestIdx + 1];
    symbols = [...symbols.slice(0, bestIdx), merged, ...symbols.slice(bestIdx + 2)];
  }
  return symbols;
}

function main(): void {
  const corpus = `
    the quick brown fox jumps over the lazy dog
    a stitch in time saves nine
    language models learn from statistical patterns in text
    tokenization splits text into smaller units called tokens
    subword tokenization lets rare words decompose into known pieces
    byte pair encoding is the dominant tokenization algorithm today
    the lazy dog slept while the fox jumped again and again
    patterns of letters in words are learnable and reusable
  `;

  const small = trainBpe(corpus, 30);
  const big = trainBpe(corpus, 150);

  console.log("=== BPE，30 次合并 ===");
  console.log("词表大小：" + small.tokens.length);
  console.log("前 10 次合并：");
  small.merges.slice(0, 10).forEach(([a, b], i) => {
    console.log("  " + i + ": " + JSON.stringify(a) + " + " + JSON.stringify(b) + " -> " + JSON.stringify(a + b));
  });

  console.log("");
  console.log("=== BPE，150 次合并 ===");
  console.log("词表大小：" + big.tokens.length);

  console.log("");
  const heldOut = ["tokenizable", "unlearnable", "foxhound", "languages"];
  console.log("=== 编码留出词（150 次合并的模型）===");
  for (const word of heldOut) {
    const naive = encodeBpe(word, big.merges);
    const ranked = rankedEncode(word, big.merges);
    const tag = naive.length === 1 ? "完整" : "拆分(" + naive.length + ")";
    const equal = naive.length === ranked.length && naive.every((s, i) => s === ranked[i]);
    console.log("  " + word.padEnd(14) + " -> " + naive.join(" | ") + "  [" + tag + "]  排名编码==朴素编码：" + equal);
  }

  console.log("");
  console.log("注意：使用微型示例语料库时，大多数留出词都会被拆分。");
  console.log("生产级词表会在数十亿个 token 上训练。");
}

main();
