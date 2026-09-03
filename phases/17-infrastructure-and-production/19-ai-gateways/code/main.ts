/**
 * AI 网关骨架——TypeScript 移植版。
 *
 * 实现 docs/en.md 中的四种核心网关原语：
 *   1. 身份验证：以常量时间比较检查 API 密钥，并解析对应租户。
 *   2. 限流：按租户维护令牌桶，采用 LiteLLM 风格。
 *   3. 重试：对瞬时 429/5xx 错误执行带抖动、有上限的指数退避。
 *   4. 回退链：依次尝试提供商，直到成功。
 *
 * 另外还包含与 main.py 相同的回退模拟器（4 种网关配置、三提供商链路和错误注入），
 * 使结果可复现。
 *
 * 参考资料：
 *   - Kong AI Gateway 基准测试（比 Portkey 高 228%，比 LiteLLM 高 859%）：
 *     https://konghq.com/blog/engineering/ai-gateway-benchmark-kong-ai-gateway-portkey-litellm
 *   - LiteLLM（MIT 开源，支持 100 多家提供商）：https://github.com/BerriAI/litellm
 *   - Portkey（自 2026 年 3 月起采用 Apache 2.0）：https://github.com/Portkey-AI/gateway
 *   - Kong AI Gateway 文档：https://docs.konghq.com/gateway/latest/ai-gateway/
 *
 * 使用 Node 20+ 标准库运行，无 npm 依赖。
 */

import { timingSafeEqual, createHash } from "node:crypto";

// -- 身份验证 -------------------------------------------------------------

type Tenant = {
  id: string;
  // 已签发 API 密钥的 SHA-256 十六进制摘要，绝不以明文存储密钥。
  keyHashHex: string;
  // 租户层级，用于确定限流预算。
  tier: "free" | "trial" | "paid";
};

class AuthService {
  private readonly tenants = new Map<string, Tenant>();
  private readonly hashByKey = new Map<string, Tenant>();

  register(tenant: Tenant): void {
    this.tenants.set(tenant.id, tenant);
    this.hashByKey.set(tenant.keyHashHex, tenant);
  }

  // 通过摘要比较进行常量时间检查。
  authenticate(presentedKey: string): Tenant | undefined {
    const digest = createHash("sha256").update(presentedKey).digest("hex");
    // 遍历所有已知哈希，让未知密钥与已知密钥耗费相同的墙钟时间。
    let match: Tenant | undefined;
    const presented = Buffer.from(digest, "hex");
    for (const t of this.tenants.values()) {
      const stored = Buffer.from(t.keyHashHex, "hex");
      if (
        stored.length === presented.length &&
        timingSafeEqual(stored, presented)
      ) {
        match = t;
      }
    }
    return match;
  }
}

// -- 限流器（令牌桶） -----------------------------------------------------

type Bucket = {
  tokens: number;
  capacity: number;
  refillPerSec: number;
  lastNs: bigint;
};

class TokenBucketLimiter {
  private readonly buckets = new Map<string, Bucket>();
  private readonly tierConfig: Record<
    Tenant["tier"],
    { capacity: number; refillPerSec: number }
  >;
  private readonly now: () => bigint;

  constructor(
    tierConfig: Record<
      Tenant["tier"],
      { capacity: number; refillPerSec: number }
    >,
    now: () => bigint = process.hrtime.bigint,
  ) {
    this.tierConfig = tierConfig;
    this.now = now;
  }

  private getOrCreate(tenant: Tenant): Bucket {
    const existing = this.buckets.get(tenant.id);
    if (existing) return existing;
    const cfg = this.tierConfig[tenant.tier];
    const bucket: Bucket = {
      tokens: cfg.capacity,
      capacity: cfg.capacity,
      refillPerSec: cfg.refillPerSec,
      lastNs: this.now(),
    };
    this.buckets.set(tenant.id, bucket);
    return bucket;
  }

  // 请求未超出令牌桶容量时返回 true，否则返回 false。
  allow(tenant: Tenant, cost = 1): boolean {
    const bucket = this.getOrCreate(tenant);
    const nowNs = this.now();
    const elapsedSec = Number(nowNs - bucket.lastNs) / 1e9;
    bucket.tokens = Math.min(
      bucket.capacity,
      bucket.tokens + elapsedSec * bucket.refillPerSec,
    );
    bucket.lastNs = nowNs;
    if (bucket.tokens >= cost) {
      bucket.tokens -= cost;
      return true;
    }
    return false;
  }
}

// -- 提供商抽象与重试/回退 ------------------------------------------------

type ProviderResponse = {
  provider: string;
  text: string;
  latencyMs: number;
  attempt: number;
};

type ProviderError = {
  retryable: boolean;
  status: 429 | 500 | 502 | 503 | 504 | 400;
  message: string;
};

type Provider = {
  name: string;
  // 真实调用使用 HTTP，因而此调用为异步。返回文本与延迟，或抛出符合
  // ProviderError 结构的值。
  call(prompt: string): Promise<{ text: string; latencyMs: number }>;
};

// 模拟提供商，根据请求计数器确定性地注入错误。
function makeMockProvider(
  name: string,
  baseLatencyMs: number,
  // 决定第 n 次调用是否出错以及如何出错的函数。
  errorPolicy: (n: number) => ProviderError | null,
): Provider {
  let n = 0;
  return {
    name,
    async call(prompt: string): Promise<{ text: string; latencyMs: number }> {
      const callN = ++n;
      const err = errorPolicy(callN);
      // 让出一个微任务，使调用体现真实异步行为。
      await Promise.resolve();
      if (err) {
        throw err;
      }
      return {
        text: `[${name}] ${prompt.slice(0, 60)}`,
        latencyMs: baseLatencyMs,
      };
    },
  };
}

type RetryConfig = {
  maxAttempts: number;
  baseBackoffMs: number;
  // 保证测试和演示的确定性。
  jitter: () => number;
  sleep: (ms: number) => Promise<void>;
};

type RetryOutcome = {
  response: ProviderResponse;
  // 单个提供商的所有重试尝试与退避等待所耗费的墙钟时间。首次尝试成功且
  // 无退避时，它等于 response.latencyMs。
  totalLatencyMs: number;
};

async function callWithRetry(
  provider: Provider,
  prompt: string,
  cfg: RetryConfig,
): Promise<RetryOutcome> {
  let lastErr: ProviderError | undefined;
  let totalLatencyMs = 0;
  for (let attempt = 1; attempt <= cfg.maxAttempts; attempt++) {
    try {
      const r = await provider.call(prompt);
      totalLatencyMs += r.latencyMs;
      return {
        response: {
          provider: provider.name,
          text: r.text,
          latencyMs: r.latencyMs,
          attempt,
        },
        totalLatencyMs,
      };
    } catch (raw) {
      const err = raw as ProviderError;
      lastErr = err;
      if (!err.retryable || attempt === cfg.maxAttempts) break;
      const backoffMs = cfg.baseBackoffMs * 2 ** (attempt - 1) * cfg.jitter();
      totalLatencyMs += backoffMs;
      await cfg.sleep(backoffMs);
    }
  }
  // 将最后一个错误暴露给回退层。
  throw lastErr ?? ({ retryable: false, status: 500, message: "未知错误" } as ProviderError);
}

async function callWithFallback(
  chain: readonly Provider[],
  prompt: string,
  cfg: RetryConfig,
): Promise<{ response: ProviderResponse; fallbackHits: number; totalLatencyMs: number }> {
  let fallbackHits = 0;
  let totalLatencyMs = 0;
  let lastErr: ProviderError | undefined;
  for (let i = 0; i < chain.length; i++) {
    if (i > 0) fallbackHits++;
    try {
      const outcome = await callWithRetry(chain[i], prompt, cfg);
      totalLatencyMs += outcome.totalLatencyMs;
      return { response: outcome.response, fallbackHits, totalLatencyMs };
    } catch (err) {
      lastErr = err as ProviderError;
    }
  }
  throw lastErr ?? { retryable: false, status: 500, message: "没有可用提供商" };
}

// -- 网关 -----------------------------------------------------------------

class AIGateway {
  constructor(
    private readonly auth: AuthService,
    private readonly limiter: TokenBucketLimiter,
    private readonly chain: readonly Provider[],
    private readonly retry: RetryConfig,
    private readonly overheadMs: number,
  ) {}

  async handle(
    presentedKey: string,
    prompt: string,
  ): Promise<
    | { ok: true; response: ProviderResponse; totalLatencyMs: number; fallbackHits: number }
    | { ok: false; status: number; reason: string }
  > {
    const tenant = this.auth.authenticate(presentedKey);
    if (!tenant) return { ok: false, status: 401, reason: "API 密钥无效" };
    if (!this.limiter.allow(tenant)) {
      return { ok: false, status: 429, reason: "超过速率限制" };
    }
    try {
      const { response, fallbackHits, totalLatencyMs } = await callWithFallback(
        this.chain,
        prompt,
        this.retry,
      );
      return {
        ok: true,
        response,
        // 端到端墙钟时间：网关开销 + 每次重试 + 每次退避等待 + 成功提供商之前
        // 所有失败提供商的延迟。
        totalLatencyMs: totalLatencyMs + this.overheadMs,
        fallbackHits,
      };
    } catch (err) {
      const e = err as ProviderError;
      return { ok: false, status: e.status ?? 500, reason: e.message };
    }
  }
}

// -- 模拟器（与 main.py 结构一致） ---------------------------------------

type ProviderProfile = { name: string; baseLatencyMs: number; errorRate: number };

const PROVIDERS: ProviderProfile[] = [
  { name: "OpenAI", baseLatencyMs: 180, errorRate: 0.03 },
  { name: "Anthropic", baseLatencyMs: 220, errorRate: 0.02 },
  { name: "Self-hosted", baseLatencyMs: 100, errorRate: 0.05 },
];

const GATEWAY_OVERHEAD: Record<string, number> = {
  LiteLLM: 10,
  Portkey: 30,
  Kong: 5,
  Cloudflare: 2,
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

type SimRow = {
  gateway: string;
  successRate: number;
  meanLatency: number;
  // 内层每次迭代只尝试一个提供商一次，然后回退；因此这里统计的是失败的
  // 提供商尝试，而不是提供商内部重试。
  providerFailures: number;
  fallbackHits: number;
};

function simulateFallback(gateway: string, n = 1000, seed = 7): SimRow {
  const rng = makeRng(seed);
  let success = 0;
  let totalLatency = 0;
  let providerFailures = 0;
  let fallbackHits = 0;
  const gwOverhead = GATEWAY_OVERHEAD[gateway];

  for (let i = 0; i < n; i++) {
    let reqLatency = gwOverhead;
    let done = false;
    for (let attempt = 0; attempt < PROVIDERS.length; attempt++) {
      const p = PROVIDERS[attempt];
      const errored = rng() < p.errorRate;
      reqLatency += errored ? p.baseLatencyMs * 0.3 : p.baseLatencyMs;
      if (attempt > 0) fallbackHits++;
      if (!errored) {
        success++;
        done = true;
        break;
      }
      providerFailures++;
    }
    void done;
    totalLatency += reqLatency;
  }

  return {
    gateway,
    successRate: success / n,
    meanLatency: totalLatency / n,
    providerFailures,
    fallbackHits,
  };
}

function reportRow(r: SimRow): void {
  console.log(
    `${r.gateway.padEnd(12)}  ` +
      `成功率=${(r.successRate * 100).toFixed(1).padStart(5)}%  ` +
      `平均延迟=${r.meanLatency.toFixed(0).padStart(6)}毫秒  ` +
      `提供商失败=${String(r.providerFailures).padStart(4)}  ` +
      `回退=${String(r.fallbackHits).padStart(4)}`,
  );
}

// -- 演示 ------------------------------------------------------------------

async function liveDemo(): Promise<void> {
  console.log("--- AI 网关原语（身份验证 + 限流 + 重试 + 回退）---");

  const auth = new AuthService();
  // 预先签发两个密钥："secret-paid-key" 对应付费层，"secret-free-key" 对应免费层。
  const paidHash = createHash("sha256").update("secret-paid-key").digest("hex");
  const freeHash = createHash("sha256").update("secret-free-key").digest("hex");
  auth.register({ id: "tenant-paid", keyHashHex: paidHash, tier: "paid" });
  auth.register({ id: "tenant-free", keyHashHex: freeHash, tier: "free" });

  const limiter = new TokenBucketLimiter({
    free: { capacity: 2, refillPerSec: 0.5 },
    trial: { capacity: 5, refillPerSec: 1 },
    paid: { capacity: 100, refillPerSec: 10 },
  });

  // 提供商 1：首次调用返回 429，之后成功。
  const flaky = makeMockProvider("openai", 180, (n) =>
    n === 1
      ? { retryable: true, status: 429, message: "rate_limit_exceeded" }
      : null,
  );
  // 提供商 2：一半调用返回 5xx。
  const wobble = makeMockProvider("anthropic", 220, (n) =>
    n % 2 === 1
      ? { retryable: true, status: 503, message: "upstream_unavailable" }
      : null,
  );
  // 提供商 3：始终健康。
  const healthy = makeMockProvider("self-hosted", 100, () => null);

  const retry: RetryConfig = {
    maxAttempts: 2,
    baseBackoffMs: 1,
    jitter: () => 1.0,
    sleep: (ms: number) => new Promise((res) => setTimeout(res, ms)),
  };

  const gateway = new AIGateway(
    auth,
    limiter,
    [flaky, wobble, healthy],
    retry,
    /* overheadMs */ 5,
  );

  console.log("付费租户——应通过重试或回退成功：");
  for (let i = 0; i < 3; i++) {
    const r = await gateway.handle("secret-paid-key", `你好，世界 ${i}`);
    console.log("  →", JSON.stringify(r));
  }

  console.log("\n免费租户——容量为 2，第三次调用触发限流：");
  for (let i = 0; i < 4; i++) {
    const r = await gateway.handle("secret-free-key", `问题 ${i}`);
    console.log("  →", JSON.stringify(r));
  }

  console.log("\n错误密钥——401：");
  console.log("  →", JSON.stringify(await gateway.handle("nope", "x")));
}

function simulatorDemo(): void {
  console.log("\n" + "=".repeat(80));
  console.log("AI 网关回退——注入错误时的三提供商链路");
  console.log("=".repeat(80));
  const header =
    `${"网关".padEnd(12)}  ` +
    `${"成功率".padStart(7)}         ${"平均延迟".padStart(12)}  提供商失败  回退`;
  console.log(header);
  console.log("-".repeat(header.length));
  for (const gw of ["LiteLLM", "Portkey", "Kong", "Cloudflare"]) {
    reportRow(simulateFallback(gw));
  }
  console.log(
    "\n说明：单一提供商的错误率为 3% 时，成功率为 97%。",
  );
  console.log(
    "双提供商回退的成功率为 99.94%（0.03 × 0.02 的补集）。",
  );
  console.log(
    "三提供商回退的成功率为 99.997%，但回退会增加延迟。",
  );
}

async function main(): Promise<void> {
  await liveDemo();
  simulatorDemo();
}

main().catch((err: unknown) => {
  console.error(err);
  process.exitCode = 1;
});
