"""Batch 与同步调用成本模拟器，使用 Python stdlib。

对包含 5 万份文档的流水线在四种配置下建模：
  SYNC              ：无折扣，无 cache
  SYNC + CACHE      ：首次调用后缓存 system prompt
  BATCH             ：五折，无 cache
  BATCH + CACHE     ：叠加使用（约为 SYNC 账单的 10%）
"""

from __future__ import annotations


BASE_INPUT = 3.00
BASE_OUTPUT = 15.00
CACHED_INPUT = 0.30
CACHE_WRITE_5MIN = 1.25 * BASE_INPUT
BATCH_DISCOUNT = 0.50


def cost_sync(docs: int, prefix_tokens: int, per_doc_tokens: int, out_tokens: int) -> float:
    cost = 0.0
    for _ in range(docs):
        cost += (prefix_tokens / 1e6) * BASE_INPUT
        cost += (per_doc_tokens / 1e6) * BASE_INPUT
        cost += (out_tokens / 1e6) * BASE_OUTPUT
    return cost


def cost_sync_cache(docs: int, prefix_tokens: int, per_doc_tokens: int, out_tokens: int) -> float:
    cost = (prefix_tokens / 1e6) * CACHE_WRITE_5MIN
    for i in range(docs):
        if i > 0:
            cost += (prefix_tokens / 1e6) * CACHED_INPUT
        cost += (per_doc_tokens / 1e6) * BASE_INPUT
        cost += (out_tokens / 1e6) * BASE_OUTPUT
    return cost


def cost_batch(docs: int, prefix_tokens: int, per_doc_tokens: int, out_tokens: int) -> float:
    return cost_sync(docs, prefix_tokens, per_doc_tokens, out_tokens) * BATCH_DISCOUNT


def cost_batch_cache(docs: int, prefix_tokens: int, per_doc_tokens: int, out_tokens: int) -> float:
    return cost_sync_cache(docs, prefix_tokens, per_doc_tokens, out_tokens) * BATCH_DISCOUNT


def run(label: str, docs: int, prefix: int, per_doc: int, output: int) -> None:
    sc = cost_sync(docs, prefix, per_doc, output)
    scc = cost_sync_cache(docs, prefix, per_doc, output)
    bc = cost_batch(docs, prefix, per_doc, output)
    bcc = cost_batch_cache(docs, prefix, per_doc, output)
    print(f"\n{label}")
    print(f"  文档数={docs}，前缀={prefix}，每文档={per_doc}，输出={output}")
    print(f"  SYNC            : ${sc:10.2f}  （基线）")
    print(f"  SYNC + CACHE    : ${scc:10.2f}  （基线的 {scc/sc*100:5.1f}%）")
    print(f"  BATCH           : ${bc:10.2f}  （基线的 {bc/sc*100:5.1f}%）")
    print(f"  BATCH + CACHE   : ${bcc:10.2f}  （基线的 {bcc/sc*100:5.1f}%）")


def main() -> None:
    print("=" * 80)
    print("BATCH API 经济性 — batch 与 prompt caching 叠加后约为同步账单的 10%")
    print("=" * 80)
    run("每晚文档摘要（5 万份文档）",
        docs=50_000, prefix=4000, per_doc=2000, output=200)
    run("内容分类（20 万项，单项较短）",
        docs=200_000, prefix=1500, per_doc=300, output=50)
    run("大型报告草稿（数量少，单项负载重）",
        docs=1_000, prefix=6000, per_doc=15_000, output=2000)


if __name__ == "__main__":
    main()
