/**
 * 批处理 API——TypeScript 移植版与延迟 Future 调度器。
 *
 * 分为两部分：
 *   1. BatchDispatcher：提交 N 个作业，为每个作业返回一个在批次完成时兑现的
 *      Promise。无需网络即可模拟 OpenAI / Anthropic JSONL 批处理生命周期
 *      （in_progress → completed）。调用方采用“延迟 Future”模式：发出后无需等待，
 *      数小时后由 Promise 交付答案。
 *   2. 与 main.py 一致的成本模拟器：在三种工作负载下比较 SYNC、SYNC+CACHE、
 *      BATCH 和 BATCH+CACHE。2026 年 4 月的定价常量见 docs/en.md。
 *
 * 参考资料：
 *   - OpenAI Batch API: platform.openai.com/docs/guides/batch
 *   - Anthropic Message Batches: docs.anthropic.com/en/docs/build-with-claude/batch-processing
 *   - Vertex AI Batch Prediction: cloud.google.com/vertex-ai/generative-ai/docs/model-reference/batch-prediction
 *
 * 使用 Node 20+ 标准库运行，无 npm 依赖。
 */

import { randomUUID } from "node:crypto";

// -- 成本常量（2026-04） --------------------------------------------------

const BASE_INPUT = 3.0;
const BASE_OUTPUT = 15.0;
const CACHED_INPUT = 0.3;
const CACHE_WRITE_5MIN = 1.25 * BASE_INPUT;
const BATCH_DISCOUNT = 0.5;

// -- 带延迟 Future 的批次调度器 -------------------------------------------

type BatchStatus = "queued" | "in_progress" | "completed" | "failed";

type BatchJob<I, O> = {
  id: string;
  input: I;
  promise: Promise<O>;
  // 内部字段：调度时捕获的兑现函数。
  resolve: (out: O) => void;
  reject: (err: Error) => void;
};

type Batch<I, O> = {
  id: string;
  status: BatchStatus;
  createdAt: number;
  completedAt?: number;
  jobs: BatchJob<I, O>[];
};

class BatchDispatcher<I, O> {
  private readonly batches = new Map<string, Batch<I, O>>();
  private readonly processor: (input: I) => Promise<O>;
  // 模拟周转时间。真实提供商承诺 24 小时 SLA，典型 P50 为 2～6 小时。
  // 演示中使用较小的毫秒值，以便快速完成。
  private readonly turnaroundMs: number;

  constructor(
    processor: (input: I) => Promise<O>,
    turnaroundMs: number,
  ) {
    this.processor = processor;
    this.turnaroundMs = turnaroundMs;
  }

  // 新建批次，返回用于追加作业的批次 ID。
  openBatch(): string {
    const id = `batch_${randomUUID().slice(0, 12)}`;
    this.batches.set(id, {
      id,
      status: "queued",
      createdAt: Date.now(),
      jobs: [],
    });
    return id;
  }

  // 向排队中的批次追加作业，返回延迟的 Promise<O>，供调用方在批次关闭并处理后等待。
  // 这与 OpenAI batch.create + retrieve 流程面向用户的形式一致。
  addJob(batchId: string, input: I): Promise<O> {
    const batch = this.requireBatch(batchId);
    if (batch.status !== "queued") {
      return Promise.reject(
        new Error(`批次 ${batchId} 不在队列中（状态=${batch.status}）`),
      );
    }
    // 手工实现延迟对象，以便在处理器循环中兑现。
    let resolve!: (out: O) => void;
    let reject!: (err: Error) => void;
    const promise = new Promise<O>((res, rej) => {
      resolve = res;
      reject = rej;
    });
    batch.jobs.push({
      id: `req_${randomUUID().slice(0, 8)}`,
      input,
      promise,
      resolve,
      reject,
    });
    return promise;
  }

  // 关闭并处理批次，在所有作业兑现或拒绝后返回。
  // 异步迭代模型与真实批次相同：无需逐个等待作业，只需等待整个批次。
  async closeBatch(batchId: string): Promise<Batch<I, O>> {
    const batch = this.requireBatch(batchId);
    batch.status = "in_progress";
    // 模拟提供商调度延迟。
    await new Promise<void>((res) => setTimeout(res, this.turnaroundMs));
    const settlements: Promise<void>[] = batch.jobs.map(async (j) => {
      try {
        j.resolve(await this.processor(j.input));
      } catch (err) {
        j.reject(err instanceof Error ? err : new Error(String(err)));
      }
    });
    await Promise.all(settlements);
    batch.status = "completed";
    batch.completedAt = Date.now();
    return batch;
  }

  getStatus(batchId: string): BatchStatus {
    return this.requireBatch(batchId).status;
  }

  private requireBatch(id: string): Batch<I, O> {
    const b = this.batches.get(id);
    if (!b) throw new Error(`不存在该批次：${id}`);
    return b;
  }
}

// -- 模拟分类处理器（无网络） ---------------------------------------------

type ClassifyIn = { docId: string; text: string };
type ClassifyOut = { docId: string; label: string; confidence: number };

async function fakeClassifier(input: ClassifyIn): Promise<ClassifyOut> {
  // 根据输入长度奇偶性进行确定性分类的简化分类器。
  const label = input.text.length % 2 === 0 ? "positive" : "neutral";
  return {
    docId: input.docId,
    label,
    confidence: 0.5 + (input.text.length % 5) / 10,
  };
}

async function batchDemo(): Promise<void> {
  console.log("--- 带延迟 Future 的批次调度器 ---");
  // 演示中的周转时间设为 50 毫秒（生产环境 SLA 为 24 小时）。
  const dispatcher = new BatchDispatcher<ClassifyIn, ClassifyOut>(
    fakeClassifier,
    50,
  );
  const batchId = dispatcher.openBatch();
  const futures: Promise<ClassifyOut>[] = [];
  for (let i = 0; i < 6; i++) {
    futures.push(
      dispatcher.addJob(batchId, {
        docId: `doc-${i}`,
        text: `文档正文编号 ${i}`,
      }),
    );
  }
  console.log(`关闭前状态：${dispatcher.getStatus(batchId)}`);
  // 调用方等待作业，同时由调度器关闭批次。
  const closePromise = dispatcher.closeBatch(batchId);
  const results = await Promise.all(futures);
  await closePromise;
  console.log(`关闭后状态：${dispatcher.getStatus(batchId)}`);
  for (const r of results) {
    console.log(
      `  ${r.docId} → 标签=${r.label} 置信度=${r.confidence.toFixed(2)}`,
    );
  }
}

// -- 成本模拟器 -----------------------------------------------------------

function costSync(
  docs: number,
  prefixTokens: number,
  perDocTokens: number,
  outTokens: number,
): number {
  let cost = 0;
  for (let i = 0; i < docs; i++) {
    cost += (prefixTokens / 1e6) * BASE_INPUT;
    cost += (perDocTokens / 1e6) * BASE_INPUT;
    cost += (outTokens / 1e6) * BASE_OUTPUT;
  }
  return cost;
}

function costSyncCache(
  docs: number,
  prefixTokens: number,
  perDocTokens: number,
  outTokens: number,
): number {
  let cost = (prefixTokens / 1e6) * CACHE_WRITE_5MIN;
  for (let i = 0; i < docs; i++) {
    if (i > 0) cost += (prefixTokens / 1e6) * CACHED_INPUT;
    cost += (perDocTokens / 1e6) * BASE_INPUT;
    cost += (outTokens / 1e6) * BASE_OUTPUT;
  }
  return cost;
}

function costBatch(
  docs: number,
  prefixTokens: number,
  perDocTokens: number,
  outTokens: number,
): number {
  return costSync(docs, prefixTokens, perDocTokens, outTokens) * BATCH_DISCOUNT;
}

function costBatchCache(
  docs: number,
  prefixTokens: number,
  perDocTokens: number,
  outTokens: number,
): number {
  return (
    costSyncCache(docs, prefixTokens, perDocTokens, outTokens) * BATCH_DISCOUNT
  );
}

function fmtCost(n: number): string {
  return `$${n.toFixed(2)}`.padStart(10);
}

function fmtPct(n: number, baseline: number): string {
  return `${((n / baseline) * 100).toFixed(1)}%`.padStart(5);
}

function runScenario(
  label: string,
  docs: number,
  prefix: number,
  perDoc: number,
  output: number,
): void {
  const sc = costSync(docs, prefix, perDoc, output);
  const scc = costSyncCache(docs, prefix, perDoc, output);
  const bc = costBatch(docs, prefix, perDoc, output);
  const bcc = costBatchCache(docs, prefix, perDoc, output);
  console.log(`\n${label}`);
  console.log(
    `  文档数=${docs}，前缀=${prefix}，每文档=${perDoc}，输出=${output}`,
  );
  console.log(`  SYNC            : ${fmtCost(sc)}  （基线）`);
  console.log(`  SYNC + CACHE    : ${fmtCost(scc)}  （基线的 ${fmtPct(scc, sc)}）`);
  console.log(`  BATCH           : ${fmtCost(bc)}  （基线的 ${fmtPct(bc, sc)}）`);
  console.log(`  BATCH + CACHE   : ${fmtCost(bcc)}  （基线的 ${fmtPct(bcc, sc)}）`);
}

async function main(): Promise<void> {
  await batchDemo();
  console.log("\n" + "=".repeat(80));
  console.log(
    "批处理 API 经济性——批处理叠加提示缓存，成本约为同步调用的 10%",
  );
  console.log("=".repeat(80));
  runScenario(
    "每晚文档摘要（5 万份文档）",
    50_000,
    4000,
    2000,
    200,
  );
  runScenario(
    "内容分类（20 万项，单项较短）",
    200_000,
    1500,
    300,
    50,
  );
  runScenario(
    "大型报告草稿（数量少，单项负载重）",
    1_000,
    6000,
    15_000,
    2000,
  );
}

main().catch((err: unknown) => {
  console.error(err);
  process.exitCode = 1;
});
