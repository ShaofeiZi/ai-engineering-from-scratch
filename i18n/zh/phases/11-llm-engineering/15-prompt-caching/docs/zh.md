# 提示缓存与上下文缓存

> 你的系统提示词有 4,000 个词元，RAG 上下文有 20,000 个词元。每次请求都要发送二者，也每次都要为二者付费。提示缓存允许提供商在服务端保持该前缀的热状态，并在复用时只收取正常价格的 10%。使用得当，它可以把推理成本降低 50%～90%，并把首词元延迟降低 40%～85%。

**Type:** 构建
**Languages:** Python
**Prerequisites:** 阶段 11 · 01（提示工程）、阶段 11 · 05（上下文工程）、阶段 11 · 11（缓存与成本）
**Time:** 约 60 分钟

## 问题

一个编程智能体在对话的每一轮都会向 Claude 发送相同的 15,000 词元系统提示词。按每百万输入词元 3 美元计算，20 轮对话仅输入成本就达到 0.90 美元——还没有算用户真正发出的任何消息。乘以每天 10,000 次对话，这段从不变化的文本每天就会产生 9,000 美元账单。

你无法在不损害质量的前提下缩短提示词，也无法不发送它——模型每一轮都需要这段内容。唯一的办法，是不要再为提供商已经看过的前缀支付全价。

这个办法就是提示缓存。Anthropic 于 2024 年 8 月推出该功能（并在 2025 年增加了 1 小时的延长 TTL 版本），OpenAI 在当年稍晚时将其自动化，Google 则随 Gemini 1.5 一起推出显式上下文缓存。如今，三家都已在前沿模型上将它作为一等功能提供。

## 概念

![提示缓存：写入一次，低价读取](../../../../../../phases/11-llm-engineering/15-prompt-caching/assets/prompt-caching.svg)

**工作机制。** 当一个请求的前缀与近期请求匹配时，提供商会复用上一次运行的 KV 缓存，而不是重新编码这些词元。第一次写入时支付少量溢价，此后每次读取都享受大幅折扣。

**2026 年三种提供商方案。**

| 提供商 | API 风格 | 命中折扣 | 写入溢价 | 默认 TTL | 最小可缓存长度 |
|---------|-----------|--------------|---------------|-------------|---------------|
| Anthropic | 在内容块上显式添加 `cache_control` 标记 | 输入价格降低 90% | 加收 25% | 5 分钟（可延长至 1 小时） | 1,024 个词元（Sonnet/Opus），2,048 个（Haiku） |
| OpenAI | 自动前缀检测 | 输入价格降低 50% | 无 | 最长 1 小时（尽力保留） | 1,024 个词元 |
| Google（Gemini） | 显式 `CachedContent` API | 读取价格约为正常价格的 25%；另收存储费 | 按词元·小时收取存储费 | 用户设置（默认 1 小时） | 4,096 个词元（Flash）/ 32,768 个（Pro） |

**不变量。** 三家都只缓存前缀。如果两次请求之间有任何词元不同，从第一个不同词元开始的全部内容都会缓存未命中。把*稳定*部分放在顶部，把*可变*部分放在底部。

### 缓存友好的布局

```
[system prompt]          <-- cache this
[tool definitions]       <-- cache this
[few-shot examples]      <-- cache this
[retrieved documents]    <-- cache if reused, else don't
[conversation history]   <-- cache up to last turn
[current user message]   <-- never cache (different every time)
```

如果违反这个顺序——把用户消息放在系统提示词上方，或把动态检索内容穿插在少样本示例之间——缓存将永远无法命中。

### 盈亏平衡计算

Anthropic 的 25% 写入溢价意味着，一个缓存块至少要读取两次才能产生净节省。1 次写入 + 1 次读取，平均每次请求的成本为原来的 0.675 倍（节省 32%）；1 次写入 + 10 次读取，平均为 0.205 倍（节省 80%）。经验法则：预计在 TTL 内至少会复用 3 次的内容才值得缓存。

```figure
prompt-cache-hit
```

## 动手构建

### 第 1 步：使用显式标记的 Anthropic 提示缓存

```python
import anthropic

client = anthropic.Anthropic()

SYSTEM = [
    {
        "type": "text",
        "text": "You are a senior Python reviewer. Follow the rubric exactly.\n\n" + RUBRIC_15K_TOKENS,
        "cache_control": {"type": "ephemeral"},
    }
]

def review(code: str):
    return client.messages.create(
        model="claude-opus-4-7",
        max_tokens=1024,
        system=SYSTEM,
        messages=[{"role": "user", "content": code}],
    )
```

`cache_control` 标记告诉 Anthropic 将这个内容块存储 5 分钟。在该时间窗口内复用就会命中；超过时间则过期并重新写入。

**响应用量字段：**

```python
response = review(code_a)
response.usage
# InputTokensUsage(
#     input_tokens=120,
#     cache_creation_input_tokens=15023,   # paid at 1.25x
#     cache_read_input_tokens=0,
#     output_tokens=340,
# )

response_b = review(code_b)
response_b.usage
# cache_creation_input_tokens=0
# cache_read_input_tokens=15023           # paid at 0.1x
```

在 CI 中检查这两个字段——如果多个请求中的 `cache_read_input_tokens` 始终为零，说明缓存键正在发生漂移。

### 第 2 步：一小时延长 TTL

对于长时间运行的批处理任务，默认的 5 分钟会在两次作业之间过期。设置 `ttl`：

```python
{"type": "text", "text": RUBRIC, "cache_control": {"type": "ephemeral", "ttl": "1h"}}
```

1 小时 TTL 的写入溢价是原来的 2 倍（比基准价高 50%，而非 25%），但只要批处理在此期间复用前缀超过 5 次，很快就能回本。

### 第 3 步：OpenAI 自动缓存

OpenAI 不需要任何配置。只要一个超过 1,024 个词元的前缀与近期请求匹配，就会自动享受 50% 折扣。

```python
from openai import OpenAI
client = OpenAI()

resp = client.chat.completions.create(
    model="gpt-5",
    messages=[
        {"role": "system", "content": SYSTEM_PROMPT},   # long and stable
        {"role": "user", "content": user_msg},
    ],
)
resp.usage.prompt_tokens_details.cached_tokens  # the discounted portion
```

同样需要遵循缓存友好的布局规则。有两件不会破坏 Anthropic 缓存、却会破坏 OpenAI 缓存的事情：改变 `user` 字段（它是缓存键的组成部分）和调整工具顺序。

### 第 4 步：Gemini 显式上下文缓存

Gemini 把缓存视为你创建并命名的一等对象：

```python
from google import genai
from google.genai import types

client = genai.Client()

cache = client.caches.create(
    model="gemini-3-pro",
    config=types.CreateCachedContentConfig(
        display_name="rubric-v3",
        system_instruction=RUBRIC,
        contents=[FEW_SHOT_EXAMPLES],
        ttl="3600s",
    ),
)

resp = client.models.generate_content(
    model="gemini-3-pro",
    contents=["Review this code:\n" + code],
    config=types.GenerateContentConfig(cached_content=cache.name),
)
```

只要缓存存在，Gemini 就会按词元·小时收取存储费，而读取价格约为正常输入价格的 25%。如果需要在多天内跨多个会话复用同一份巨型提示词，这种形式最合适。

### 第 5 步：在生产环境中测量命中率

`code/main.py` 包含一个模拟的三提供商计费器，可跟踪写入、读取与未命中次数，并计算每 1,000 次请求的混合成本。应为部署设置目标命中率门禁——大多数生产级 Anthropic 配置在预热后，读取比例应超过 80%。

## 2026 年仍会被带到生产环境的陷阱

- **顶部的动态时间戳。** 把 `"Current time: 2026-04-22 15:30:02"` 放在系统提示词顶部，每次请求都会未命中。应把时间戳移动到缓存断点之后。
- **工具顺序变化。** 以稳定顺序序列化工具——部署间出现一次字典重排，就会破坏所有命中。
- **自由文本的近似重复。** “You are helpful.”与“You are a helpful assistant.”——哪怕只差一个字节，也会完全未命中。
- **内容块太小。** Anthropic 要求至少 1,024 个词元（Haiku 为 2,048）。更小的块会静默地不进入缓存。
- **盲目的成本仪表板。** 把“输入词元”拆分为缓存和未缓存两类，否则流量下降看起来也会像缓存取得了成效。

## 投入使用

2026 年的缓存技术栈：

| 场景 | 选择 |
|-----------|------|
| 智能体拥有稳定的 10k 以上词元系统提示词，且对话轮次较多 | Anthropic `cache_control`，使用 5 分钟 TTL |
| 批处理作业复用前缀超过 30 分钟 | Anthropic，使用 `ttl: "1h"` |
| GPT-5 上的无服务器端点，无自定义基础设施 | OpenAI 自动缓存（只需让前缀稳定且足够长） |
| 多天复用大型代码/文档语料库 | Gemini 显式 `CachedContent` |
| 跨提供商后备 | 在各提供商间保持相同的可缓存前缀布局，让任意一方都能命中 |

把它与语义缓存（阶段 11 · 11）结合，处理用户消息层：提示缓存处理*词元完全相同*的复用，语义缓存处理*含义相同*的复用。

## 交付成果

保存 `outputs/skill-prompt-caching-planner.md`：

```markdown
---
name: prompt-caching-planner
description: Design a cache-friendly prompt layout and pick the right provider caching mode.
version: 1.0.0
phase: 11
lesson: 15
tags: [llm-engineering, caching, cost]
---

Given a prompt (system + tools + few-shot + retrieval + history + user) and a usage profile (requests per hour, TTL needed, provider), output:

1. Layout. Reordered sections with a single cache breakpoint marked; explain which sections are stable, which are volatile.
2. Provider mode. Anthropic cache_control, OpenAI automatic, or Gemini CachedContent. Justify from TTL and reuse pattern.
3. Break-even. Expected reads per write within TTL; net cost vs no-cache with math.
4. Verification plan. CI assertion that cache_read_input_tokens > 0 on the second identical request; dashboard split by cached vs uncached tokens.
5. Failure modes. List the three most likely reasons the cache will miss in this setup (dynamic timestamp, tool reorder, near-duplicate text) and how you will prevent each.

Refuse to ship a cache plan that places a dynamic field above the breakpoint. Refuse to enable 1h TTL without a reuse count that makes the 2x write premium pay back.
```

## 练习

1. **简单。** 使用 Claude 运行一段包含 5,000 词元系统提示词的 10 轮对话。先不使用 `cache_control`，再启用它，分别报告输入词元账单。
2. **中等。** 编写测试工具：给定提示词模板和请求日志，计算各提供商方案（Anthropic 5 分钟、Anthropic 1 小时、OpenAI 自动、Gemini 显式）的预期命中率与节省金额。
3. **困难。** 构建布局优化器：给定一个提示词，以及标记为 `stable=True/False` 的字段列表，在不丢失信息的前提下重写提示词，把单个缓存断点放到最有利于缓存的位置。在真实 Anthropic 端点上验证。

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|-----------------|-----------------------|
| 提示缓存 | “让长提示词变便宜” | 为匹配的前缀复用提供商侧 KV 缓存；重复输入词元可享受 50%～90% 折扣。 |
| `cache_control` | “Anthropic 标记” | 声明“此前所有内容均可缓存”的内容块属性；格式为 `{"type": "ephemeral"}`。 |
| 缓存写入 | “支付溢价” | 填充缓存的第一次请求；Anthropic 按输入价约 1.25 倍计费，OpenAI 免费。 |
| 缓存读取 | “享受折扣” | 匹配前缀的后续请求；分别按正常价格的 10%（Anthropic）、50%（OpenAI）和约 25%（Gemini）计费。 |
| TTL | “能存活多久” | 缓存保持热状态的秒数；Anthropic 默认 5 分钟（可延长至 1 小时），OpenAI 尽力保留最长 1 小时，Gemini 由用户设置。 |
| 延长 TTL | “Anthropic 一小时缓存” | `{"type": "ephemeral", "ttl": "1h"}`；写入溢价为 2 倍，但适合批量复用。 |
| 前缀匹配 | “为什么缓存未命中” | 只有从开头到缓存断点的每个词元逐字节相同时，缓存才会命中。 |
| 上下文缓存（Gemini） | “显式缓存” | Google 提供的具名、按存储计费的缓存对象；最适合多天复用大型语料库。 |

## 延伸阅读

- [Anthropic——提示缓存](https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching)——`cache_control`、1 小时 TTL 与盈亏平衡表。
- [OpenAI——提示缓存](https://platform.openai.com/docs/guides/prompt-caching)——自动前缀匹配。
- [Google——上下文缓存](https://ai.google.dev/gemini-api/docs/caching)——`CachedContent` API 与存储定价。
- [Anthropic 工程——长上下文工作负载的提示缓存](https://www.anthropic.com/news/prompt-caching)——包含延迟数据的最初发布文章。
- 阶段 11 · 05（上下文工程）——如何切分提示词，让缓存能够命中。
- 阶段 11 · 11（缓存与成本）——将提示缓存与用户消息层的语义缓存结合起来。
- [Pope 等，“Efficiently Scaling Transformer Inference”（2022）](https://arxiv.org/abs/2211.05102)——提示缓存向用户暴露的 KV 缓存内存模型；解释为何重新读取缓存前缀的成本约为重新计算的十分之一。
- [Agrawal 等，“SARATHI: Efficient LLM Inference by Piggybacking Decodes with Chunked Prefills”（2023）](https://arxiv.org/abs/2308.16369)——提示缓存跳过的是预填充阶段；本文解释为何命中缓存会显著降低首词元时间（TTFT），却不影响每输出词元时间（TPOT）。
- [Leviathan 等，“Fast Inference from Transformers via Speculative Decoding”（2023）](https://arxiv.org/abs/2211.17192)——提示缓存与推测解码、Flash Attention、MQA/GQA 都是改变推理成本曲线的手段；可通过本文了解另外三种。
