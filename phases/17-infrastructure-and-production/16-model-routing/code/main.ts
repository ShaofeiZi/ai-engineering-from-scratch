/**
 * 模型路由——TypeScript 移植版与基于规则的路由器。
 *
 * 分为两部分：
 *   1. ModelRouter：根据模型目录与请求信号进行基于规则的选择。每条规则先按能力
 *      匹配度给候选模型评分，再依据调用方提供的策略权衡延迟、成本与能力。对应
 *      docs/en.md 中的四种信号（任务类别、提示词长度、与困难样本集的相似度、
 *      自置信度）。
 *   2. 与 main.py 一致的成本/质量模拟器：在混合难度工作负载下比较 NO_ROUTE、
 *      PRE_ROUTE 和 CASCADE 模式。
 *
 * 参考资料：
 *   - RouteLLM (LMSYS): https://github.com/lm-sys/RouteLLM
 *   - OpenRouter 推荐/路由原语：https://openrouter.ai/
 *   - 带回退与成本路由的 LiteLLM 路由器配置（在文档中引用）
 *
 * 使用 Node 20+ 标准库运行，无 npm 依赖。
 */

// -- 定价（2026-04 估算） -------------------------------------------------

const CHEAP_INPUT = 0.25;
const CHEAP_OUTPUT = 1.0;
const FRONTIER_INPUT = 3.0;
const FRONTIER_OUTPUT = 15.0;

// -- 模型目录与路由原语 ---------------------------------------------------

type Capability =
  | "chat"
  | "code"
  | "math"
  | "vision"
  | "long-context"
  | "tool-use";

type Model = {
  id: string;
  // 每百万 token 的价格。
  inputPrice: number;
  outputPrice: number;
  // 首 token 延迟 P50（毫秒）。
  latencyMs: number;
  // 最大上下文长度（token）。
  contextWindow: number;
  // 能力集合，用于路由器匹配度评分。
  capabilities: Set<Capability>;
  // 根据文档粗略映射得出的 0～1 主观质量分数。
  qualityFloor: number;
};

const CATALOG: Model[] = [
  {
    id: "haiku-class",
    inputPrice: CHEAP_INPUT,
    outputPrice: CHEAP_OUTPUT,
    latencyMs: 250,
    contextWindow: 200_000,
    capabilities: new Set<Capability>(["chat", "tool-use"]),
    qualityFloor: 0.75,
  },
  {
    id: "sonnet-class",
    inputPrice: 1.0,
    outputPrice: 5.0,
    latencyMs: 450,
    contextWindow: 200_000,
    capabilities: new Set<Capability>([
      "chat",
      "code",
      "tool-use",
      "long-context",
    ]),
    qualityFloor: 0.9,
  },
  {
    id: "frontier",
    inputPrice: FRONTIER_INPUT,
    outputPrice: FRONTIER_OUTPUT,
    latencyMs: 800,
    contextWindow: 1_000_000,
    capabilities: new Set<Capability>([
      "chat",
      "code",
      "math",
      "vision",
      "tool-use",
      "long-context",
    ]),
    qualityFloor: 1.0,
  },
];

type RouteSignals = {
  // 由小型上游分类器得出的任务类别。
  taskClass: "simple" | "medium" | "hard";
  // 估算的提示词 token 数。
  promptTokens: number;
  // 与人工整理的已知困难样本集之间的 0～1 余弦相似度。
  hardSetSimilarity: number;
  // 此请求所需的能力。
  required: Capability[];
};

type RoutePolicy = {
  // 权重之和为 1，表示各维度的重要程度。
  weightCost: number;
  weightLatency: number;
  weightCapability: number;
  // 所选模型必须达到的质量下限。
  minQuality: number;
};

type RouteDecision = {
  model: Model;
  estCost: number;
  reasoning: string;
};

class ModelRouter {
  private readonly catalog: readonly Model[];
  private readonly hardSetThreshold: number;

  constructor(catalog: readonly Model[], hardSetThreshold = 0.88) {
    this.catalog = catalog;
    this.hardSetThreshold = hardSetThreshold;
  }

  // 估算请求在某模型上的综合成本。除非调用方传入真实输出估算值，否则假定
  // 输出 200 个 token。
  estCost(model: Model, promptTokens: number, outputTokens = 200): number {
    return (
      (promptTokens / 1e6) * model.inputPrice +
      (outputTokens / 1e6) * model.outputPrice
    );
  }

  // 筛选模型目录，只保留满足以下条件的模型：
  //  (a) 覆盖所有必需能力；
  //  (b) 上下文窗口容得下提示词；
  //  (c) 达到策略要求的质量下限。
  candidates(signals: RouteSignals, policy: RoutePolicy): Model[] {
    return this.catalog.filter((m) => {
      for (const c of signals.required) if (!m.capabilities.has(c)) return false;
      if (signals.promptTokens > m.contextWindow) return false;
      if (m.qualityFloor < policy.minQuality) return false;
      return true;
    });
  }

  // 加权选择：成本越低、延迟越低、能力匹配度越高越好。
  // 与“困难样本集”的相似度达到阈值时，直接选择前沿模型（与文档规则一致）。
  pick(signals: RouteSignals, policy: RoutePolicy): RouteDecision {
    if (signals.hardSetSimilarity >= this.hardSetThreshold) {
      const frontier = this.catalog.find((m) => m.id === "frontier");
      if (frontier) {
        return {
          model: frontier,
          estCost: this.estCost(frontier, signals.promptTokens),
          reasoning: `困难样本集相似度 ${signals.hardSetSimilarity.toFixed(2)} >= ${this.hardSetThreshold}——固定到前沿模型`,
        };
      }
    }

    const cands = this.candidates(signals, policy);
    if (cands.length === 0) {
      throw new Error("没有候选模型同时满足策略与所需能力");
    }
    // 归一化以实现公平加权。
    const costs = cands.map((m) => this.estCost(m, signals.promptTokens));
    const latencies = cands.map((m) => m.latencyMs);
    const caps = cands.map((m) => m.capabilities.size);
    const maxCost = Math.max(...costs);
    const maxLat = Math.max(...latencies);
    const maxCap = Math.max(...caps);

    let bestIdx = 0;
    let bestScore = -Infinity;
    let bestReason = "";
    for (let i = 0; i < cands.length; i++) {
      const costScore = 1 - costs[i] / (maxCost || 1);
      const latScore = 1 - latencies[i] / (maxLat || 1);
      const capScore = caps[i] / (maxCap || 1);
      const score =
        policy.weightCost * costScore +
        policy.weightLatency * latScore +
        policy.weightCapability * capScore;
      if (score > bestScore) {
        bestScore = score;
        bestIdx = i;
        bestReason =
          `成本=${costScore.toFixed(2)} 延迟=${latScore.toFixed(2)} 能力=${capScore.toFixed(2)} ` +
          `加权分数=${score.toFixed(3)}`;
      }
    }

    return {
      model: cands[bestIdx],
      estCost: costs[bestIdx],
      reasoning: bestReason,
    };
  }
}

// -- 工作负载与模拟器（与 main.py 一致） ---------------------------------

type Difficulty = "simple" | "medium" | "hard";
type Query = {
  difficulty: Difficulty;
  promptTokens: number;
  outputTokens: number;
};

function makeRng(seed: number): () => number {
  let s = seed >>> 0;
  return function () {
    s = (s + 0x6d2b79f5) >>> 0;
    let t = s;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function randint(rng: () => number, lo: number, hi: number): number {
  return lo + Math.floor(rng() * (hi - lo + 1));
}

function makeWorkload(n = 1000, seed = 7): Query[] {
  const rng = makeRng(seed);
  const reqs: Query[] = [];
  for (let i = 0; i < n; i++) {
    const p = rng();
    if (p < 0.6) {
      reqs.push({
        difficulty: "simple",
        promptTokens: randint(rng, 200, 1000),
        outputTokens: randint(rng, 50, 200),
      });
    } else if (p < 0.9) {
      reqs.push({
        difficulty: "medium",
        promptTokens: randint(rng, 800, 3000),
        outputTokens: randint(rng, 100, 400),
      });
    } else {
      reqs.push({
        difficulty: "hard",
        promptTokens: randint(rng, 2000, 8000),
        outputTokens: randint(rng, 200, 1500),
      });
    }
  }
  return reqs;
}

function costOf(route: "cheap" | "frontier", q: Query): number {
  if (route === "cheap") {
    return (
      (q.promptTokens / 1e6) * CHEAP_INPUT +
      (q.outputTokens / 1e6) * CHEAP_OUTPUT
    );
  }
  return (
    (q.promptTokens / 1e6) * FRONTIER_INPUT +
    (q.outputTokens / 1e6) * FRONTIER_OUTPUT
  );
}

function quality(route: "cheap" | "frontier", q: Query): number {
  if (route === "frontier") return 1.0;
  return { simple: 0.99, medium: 0.92, hard: 0.75 }[q.difficulty];
}

type SimRow = {
  pattern: string;
  cost: number;
  meanQuality: number;
  escalated: number;
};

function simulate(pattern: string, reqs: readonly Query[]): SimRow {
  let totalCost = 0;
  let totalQ = 0;
  let escalated = 0;
  const rng = makeRng(11);

  for (const q of reqs) {
    if (pattern === "NO_ROUTE") {
      totalCost += costOf("frontier", q);
      totalQ += 1.0;
    } else if (pattern === "PRE_ROUTE") {
      if (q.difficulty === "simple") {
        totalCost += costOf("cheap", q);
        totalQ += quality("cheap", q);
      } else {
        totalCost += costOf("frontier", q);
        totalQ += 1.0;
      }
    } else if (pattern === "CASCADE") {
      totalCost += costOf("cheap", q);
      const confident =
        q.difficulty === "simple" ||
        (q.difficulty === "medium" && rng() < 0.5);
      if (confident) {
        totalQ += quality("cheap", q);
      } else {
        escalated++;
        totalCost += costOf("frontier", q);
        totalQ += 1.0;
      }
    }
  }

  return {
    pattern,
    cost: totalCost,
    meanQuality: totalQ / reqs.length,
    escalated,
  };
}

function reportRow(row: SimRow, baseline: number): void {
  const save = ((baseline - row.cost) / baseline) * 100;
  console.log(
    `${row.pattern.padEnd(12)}  成本=$${row.cost.toFixed(2).padStart(7)}  ` +
      `节省=${save.toFixed(1).padStart(5)}%  ` +
      `质量=${(row.meanQuality * 100).toFixed(1).padStart(5)}%  ` +
      `升级数=${String(row.escalated).padStart(4)}`,
  );
}

// -- 演示 ------------------------------------------------------------------

function routerDemo(): void {
  console.log("--- 基于规则的 ModelRouter ---");
  const router = new ModelRouter(CATALOG);

  const balanced: RoutePolicy = {
    weightCost: 0.5,
    weightLatency: 0.2,
    weightCapability: 0.3,
    minQuality: 0.7,
  };
  const latencyFirst: RoutePolicy = {
    weightCost: 0.1,
    weightLatency: 0.7,
    weightCapability: 0.2,
    minQuality: 0.7,
  };

  const cases: { name: string; signals: RouteSignals; policy: RoutePolicy }[] = [
    {
      name: "FAQ 风格短提示词（均衡策略）",
      signals: {
        taskClass: "simple",
        promptTokens: 400,
        hardSetSimilarity: 0.2,
        required: ["chat"],
      },
      policy: balanced,
    },
    {
      name: "使用工具的代码生成（均衡策略）",
      signals: {
        taskClass: "medium",
        promptTokens: 2500,
        hardSetSimilarity: 0.4,
        required: ["chat", "code", "tool-use"],
      },
      policy: balanced,
    },
    {
      name: "接近已知困难样本集的数学任务（自动固定到前沿模型）",
      signals: {
        taskClass: "hard",
        promptTokens: 1500,
        hardSetSimilarity: 0.92,
        required: ["chat", "math"],
      },
      policy: balanced,
    },
    {
      name: "80 万 token 的长上下文（仅前沿模型可容纳）",
      signals: {
        taskClass: "hard",
        promptTokens: 800_000,
        hardSetSimilarity: 0.1,
        required: ["chat", "long-context"],
      },
      policy: balanced,
    },
    {
      name: "FAQ 风格短提示词（延迟优先）",
      signals: {
        taskClass: "simple",
        promptTokens: 300,
        hardSetSimilarity: 0.1,
        required: ["chat"],
      },
      policy: latencyFirst,
    },
  ];

  for (const c of cases) {
    const d = router.pick(c.signals, c.policy);
    console.log(`  ${c.name}`);
    console.log(
      `    → ${d.model.id}  估算成本=$${d.estCost.toFixed(5)}  原因=${d.reasoning}`,
    );
  }
}

function patternsDemo(): void {
  console.log("\n" + "=".repeat(80));
  console.log("模型路由——三种模式，1000 个混合难度请求");
  console.log("=".repeat(80));
  const reqs = makeWorkload();
  const baseline = simulate("NO_ROUTE", reqs).cost;
  for (const p of ["NO_ROUTE", "PRE_ROUTE", "CASCADE"]) {
    reportRow(simulate(p, reqs), baseline);
  }
  console.log(
    "\n解读：分类器准确时，PRE_ROUTE 可大幅节省成本。CASCADE",
  );
  console.log(
    "能保障质量下限，但会增加升级请求的延迟。",
  );
}

function main(): void {
  routerDemo();
  patternsDemo();
}

main();
