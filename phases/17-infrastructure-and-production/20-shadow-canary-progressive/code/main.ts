/**
 * 影子流量、金丝雀与渐进式发布——TypeScript 移植版与策略引擎。
 *
 * 三种策略：
 *   1. 影子模式：将每个请求复制给候选版本，记录差异，但绝不向用户返回候选输出。
 *      在影响用户之前发现成本与长度回归。
 *   2. 金丝雀发布：分阶段逐步转移流量，并设置五道 LLM 专项门禁。任一门禁
 *      被突破即停止。
 *   3. 渐进式策略：组合影子流量 → 金丝雀 → 100%，并使用支持数秒而非数小时
 *      回滚的策略标志。
 *
 * 另外还包含与 main.py 相同的金丝雀模拟器（六个阶段、五道门禁、六种回归场景），
 * 使结果可复现。
 *
 * 参考资料：
 *   - Argo Rollouts（Kubernetes 渐进式交付）
 *     https://argo-rollouts.readthedocs.io/
 *   - Flagger（渐进式交付操作器）
 *     https://docs.flagger.app/
 *   - docs/en.md 引用的约 15% 运行间非确定性（GPU 浮点运算不满足结合律、
 *     批次大小变化与采样）。
 *
 * 使用 Node 20+ 标准库运行，无 npm 依赖。
 */

// -- 基线与门禁 -----------------------------------------------------------

type Metrics = {
  latencyP99Ms: number;
  costPerReq: number;
  errorRate: number;
  outputLenP99: number;
  thumbsDownRate: number;
};

const BASELINE: Metrics = {
  latencyP99Ms: 900,
  costPerReq: 0.02,
  errorRate: 0.02,
  outputLenP99: 450,
  thumbsDownRate: 0.03,
};

// 超过基线这些倍数即视为突破门禁。阈值设置得足够高，以避开 LLM 非确定性
// 噪声下限（据 docs/en.md，约为 15%）。
const GATES: Record<keyof Metrics, number> = {
  latencyP99Ms: 1.5,
  costPerReq: 1.2,
  errorRate: 2.0,
  outputLenP99: 1.4,
  thumbsDownRate: 1.5,
};

const STAGES = [0.01, 0.1, 0.25, 0.5, 0.75, 1.0];

// -- Mulberry32 伪随机数生成器 -------------------------------------------

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

function stageSeed(i: number): number {
  return 11 + i * 3;
}

// -- 回归注入器 -----------------------------------------------------------

type Regression = {
  latencyMult: number;
  costMult: number;
  errorMult: number;
  outputLenMult: number;
  thumbsDownMult: number;
};

const NO_REGRESSION: Regression = {
  latencyMult: 1,
  costMult: 1,
  errorMult: 1,
  outputLenMult: 1,
  thumbsDownMult: 1,
};

function measureStage(_stage: number, reg: Regression, seed: number): Metrics {
  const rng = makeRng(seed);
  // 噪声下限是 docs/en.md 所述的非确定性：每次测量约 ±8%。
  const noise = (v: number): number => v * (0.92 + rng() * 0.16);
  return {
    latencyP99Ms: noise(BASELINE.latencyP99Ms * reg.latencyMult),
    costPerReq: noise(BASELINE.costPerReq * reg.costMult),
    errorRate: noise(BASELINE.errorRate * reg.errorMult),
    outputLenP99: noise(BASELINE.outputLenP99 * reg.outputLenMult),
    thumbsDownRate: noise(BASELINE.thumbsDownRate * reg.thumbsDownMult),
  };
}

function checkGates(metrics: Metrics): (keyof Metrics)[] {
  const breaches: (keyof Metrics)[] = [];
  for (const k of Object.keys(GATES) as (keyof Metrics)[]) {
    if (metrics[k] > BASELINE[k] * GATES[k]) breaches.push(k);
  }
  return breaches;
}

// -- 策略引擎 -------------------------------------------------------------

type ShadowSample = {
  baselineCost: number;
  candidateCost: number;
  baselineLatencyMs: number;
  candidateLatencyMs: number;
};

type ShadowReport = {
  n: number;
  meanCostDeltaPct: number;
  meanLatencyDeltaPct: number;
  // 若仅影子阶段的结果就足以在金丝雀发布前停止，则为 true。
  alert: boolean;
  reasons: string[];
};

function shadowEvaluate(samples: ShadowSample[]): ShadowReport {
  if (samples.length === 0) {
    return {
      n: 0,
      meanCostDeltaPct: 0,
      meanLatencyDeltaPct: 0,
      alert: false,
      reasons: [],
    };
  }
  let costDelta = 0;
  let latDelta = 0;
  let costN = 0;
  let latN = 0;
  for (const s of samples) {
    // 跳过基线非正的行，避免单个零值将平均值变为 Infinity/NaN 并破坏门禁判断。
    if (s.baselineCost > 0) {
      costDelta += (s.candidateCost - s.baselineCost) / s.baselineCost;
      costN++;
    }
    if (s.baselineLatencyMs > 0) {
      latDelta += (s.candidateLatencyMs - s.baselineLatencyMs) / s.baselineLatencyMs;
      latN++;
    }
  }
  const meanCost = costN > 0 ? (costDelta / costN) * 100 : 0;
  const meanLat = latN > 0 ? (latDelta / latN) * 100 : 0;
  const reasons: string[] = [];
  if (meanCost > 30) reasons.push(`成本 +${meanCost.toFixed(1)}%（>30%）`);
  if (meanLat > 50) reasons.push(`延迟 +${meanLat.toFixed(1)}%（>50%）`);
  return {
    n: samples.length,
    meanCostDeltaPct: meanCost,
    meanLatencyDeltaPct: meanLat,
    alert: reasons.length > 0,
    reasons,
  };
}

type CanaryDecision = {
  promoted: boolean;
  stagesAdvanced: number;
  breaches: (keyof Metrics)[];
};

function canaryRollout(reg: Regression): CanaryDecision {
  for (let i = 0; i < STAGES.length; i++) {
    const metrics = measureStage(STAGES[i], reg, stageSeed(i));
    const breaches = checkGates(metrics);
    if (breaches.length > 0) {
      return { promoted: false, stagesAdvanced: i, breaches };
    }
  }
  return { promoted: true, stagesAdvanced: STAGES.length, breaches: [] };
}

// PolicyEngine 封装功能标志，可在 O(1) 时间内将 pinnedModel 从候选版本切回基线。
// 这模拟了 LaunchDarkly/Flagsmith/Unleash 的标志翻转回滚。
class PolicyEngine {
  private baselineDigest: string;
  private pinnedDigest: string;
  private rolloutPct = 0;

  constructor(initialDigest: string) {
    this.baselineDigest = initialDigest;
    this.pinnedDigest = initialDigest;
  }

  promote(candidateDigest: string, pct: number): void {
    this.pinnedDigest = candidateDigest;
    this.rolloutPct = pct;
  }

  // 常量时间回滚——运维手册所翻转的操作。重新固定到构造时捕获的基线
  // （或最近一次回滚覆盖值）。
  rollback(baselineDigest?: string): void {
    if (baselineDigest !== undefined) this.baselineDigest = baselineDigest;
    this.pinnedDigest = this.baselineDigest;
    this.rolloutPct = 0;
  }

  pick(rng: () => number): { digest: string; chose: "baseline" | "candidate" } {
    return rng() < this.rolloutPct
      ? { digest: this.pinnedDigest, chose: "candidate" }
      : { digest: this.baselineDigest, chose: "baseline" };
  }
}

// -- 报告 -----------------------------------------------------------------

function rolloutReport(name: string, reg: Regression): void {
  console.log(`\n${name}`);
  console.log(
    `回归倍数：延迟=${reg.latencyMult}，成本=${reg.costMult}，错误=${reg.errorMult}，长度=${reg.outputLenMult}，差评=${reg.thumbsDownMult}`,
  );
  for (let i = 0; i < STAGES.length; i++) {
    const stage = STAGES[i];
    const metrics = measureStage(stage, reg, stageSeed(i));
    const breaches = checkGates(metrics);
    const status =
      breaches.length === 0 ? "通过" : `停止（${breaches.join(",")}）`;
    const pct = Math.round(stage * 100);
    console.log(
      `  阶段 ${String(pct).padStart(3)}%  ` +
        `延迟_p99=${metrics.latencyP99Ms.toFixed(0).padStart(5)}  ` +
        `成本=$${metrics.costPerReq.toFixed(4)}  ` +
        `错误=${(metrics.errorRate * 100).toFixed(1).padStart(4)}%  ` +
        `差评=${(metrics.thumbsDownRate * 100).toFixed(1).padStart(4)}%  ` +
        `${status}`,
    );
    if (breaches.length > 0) {
      console.log("  → 回滚（翻转策略，恢复固定模型）");
      return;
    }
  }
  console.log("  → 已推广至 100%");
}

// -- 演示 ------------------------------------------------------------------

function shadowDemo(): void {
  console.log("--- 影子模式评估（对用户零影响）---");
  // 三种场景：候选版本大致相当、候选版本更便宜、候选版本贵 40%
  // （文档中的典型负面场景）。
  const rng = makeRng(99);
  const mkSamples = (costMult: number, latMult: number): ShadowSample[] =>
    Array.from({ length: 200 }, () => ({
      baselineCost: 0.02 * (0.95 + rng() * 0.1),
      candidateCost: 0.02 * costMult * (0.95 + rng() * 0.1),
      baselineLatencyMs: 800 * (0.95 + rng() * 0.1),
      candidateLatencyMs: 800 * latMult * (0.95 + rng() * 0.1),
    }));

  const scenarios: { name: string; samples: ShadowSample[] }[] = [
    { name: "表现相当的候选版本", samples: mkSamples(1.05, 1.02) },
    { name: "便宜 20% 的候选版本", samples: mkSamples(0.8, 0.95) },
    { name: "贵 40% 的候选版本（回滚案例）", samples: mkSamples(1.4, 1.0) },
  ];

  for (const s of scenarios) {
    const r = shadowEvaluate(s.samples);
    console.log(
      `  ${s.name}：n=${r.n} 成本变化=${r.meanCostDeltaPct.toFixed(1)}%  ` +
        `延迟变化=${r.meanLatencyDeltaPct.toFixed(1)}%  ` +
        `告警=${r.alert}${r.reasons.length ? "  原因=" + r.reasons.join("; ") : ""}`,
    );
  }
}

function policyEngineDemo(): void {
  console.log("\n--- PolicyEngine——先推广，再以 O(1) 回滚 ---");
  const engine = new PolicyEngine("baseline-digest");
  engine.promote("candidate-digest-v2", 0.1);
  const rng = makeRng(42);
  let candidateCount = 0;
  for (let i = 0; i < 1000; i++) {
    if (engine.pick(rng).chose === "candidate") candidateCount++;
  }
  console.log(
    `  推广至 10% 后：1000 次选择中有 ${candidateCount} 次选中候选版本（目标约 100）`,
  );
  engine.rollback();
  let postCount = 0;
  for (let i = 0; i < 1000; i++) {
    if (engine.pick(rng).chose === "candidate") postCount++;
  }
  console.log(`  回滚后：${postCount}/1000（目标为 0）`);
}

function canaryDemo(): void {
  console.log("\n" + "=".repeat(95));
  console.log("金丝雀发布——六个阶段、五道门禁、注入回归");
  console.log("=".repeat(95));

  rolloutReport("无回归推广", NO_REGRESSION);
  rolloutReport("轻微成本回归（10%）——未超门禁", {
    ...NO_REGRESSION,
    costMult: 1.1,
  });
  rolloutReport("成本回归 25%", { ...NO_REGRESSION, costMult: 1.25 });
  rolloutReport("延迟回归 80%", {
    ...NO_REGRESSION,
    latencyMult: 1.8,
  });
  rolloutReport("差评率回归 60%", {
    ...NO_REGRESSION,
    thumbsDownMult: 1.6,
  });
  rolloutReport("质量静默下降且成本缓慢上升", {
    ...NO_REGRESSION,
    costMult: 1.15,
    thumbsDownMult: 1.45,
  });

  // 对相同六种场景输出 canaryRollout() 的程序化判断。
  console.log("\n--- canaryRollout() 程序化判断 ---");
  const scenarios: { name: string; reg: Regression }[] = [
    { name: "无回归", reg: NO_REGRESSION },
    { name: "成本 10%", reg: { ...NO_REGRESSION, costMult: 1.1 } },
    { name: "成本 25%", reg: { ...NO_REGRESSION, costMult: 1.25 } },
    { name: "延迟 80%", reg: { ...NO_REGRESSION, latencyMult: 1.8 } },
    { name: "差评 60%", reg: { ...NO_REGRESSION, thumbsDownMult: 1.6 } },
    {
      name: "成本 15% + 差评 45%",
      reg: { ...NO_REGRESSION, costMult: 1.15, thumbsDownMult: 1.45 },
    },
  ];
  for (const s of scenarios) {
    const d = canaryRollout(s.reg);
    const verdict = d.promoted
      ? "已推广"
      : `在阶段 ${d.stagesAdvanced} 停止，突破项：${d.breaches.join(",")}`;
    console.log(`  ${s.name.padEnd(28)} → ${verdict}`);
  }
}

function main(): void {
  shadowDemo();
  policyEngineDemo();
  canaryDemo();
}

main();
