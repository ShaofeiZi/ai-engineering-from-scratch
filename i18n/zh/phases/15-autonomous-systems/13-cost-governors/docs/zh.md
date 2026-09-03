# 动作预算、迭代上限与成本治理器

> 某个中型电商代理在团队启用“order-tracking”技能后，月度 LLM 成本从 $1,200 暴涨到 $4,800。这不是定价 bug，而是代理找到了一个新循环，并持续在里面花钱。Microsoft 的 Agent Governance Toolkit（2026 年 4 月 2 日）把这类问题的防线明确了出来：单次请求的 `max_tokens`、按任务设置的 token 与金额预算、按天/按月限额、迭代上限、分层模型路由、提示缓存（prompt caching）、上下文窗口控制（context windowing）、对昂贵动作设置的 HITL 检查点，以及预算突破后的熔断开关（kill switch）。Anthropic 的 Claude Code Agent SDK 以不同命名暴露了同一套原语。财务速度限制，例如“10 分钟内花费超过 $50 就切断访问”，比月度限额更能快速抓住循环型失控。

**Type:** 学习
**Languages:** Python（stdlib，分层成本治理器模拟器）
**Prerequisites:** 阶段 15 · 10（权限模式），阶段 15 · 12（持久执行）
**Time:** 约 60 分钟

## 问题

自主代理在每一轮都会消耗真实资金。聊天机器人的糟糕输出只是一个糟糕回复；代理的糟糕循环则会变成账单。业内已经把这种失败模式命名为 **Denial of Wallet**：代理不断推理、不断调用工具、不断计费，而系统里没有任何机制能把它停下来，因为压根没人把“停下来”设计进去。

修复方案不是设置一个单独数字，而是在不同时间尺度和不同粒度上叠加一整栈限制：按请求、按任务、按小时、按天、按月。一个设计良好的栈，会在几分钟内抓住失控循环（runaway loop），在几小时内抓住缓慢渗漏（slow leak），在一天之内抓住错误发布（bad release）。当代理是长周期且具备自主性时，正是这套栈让“预算”这个概念依然存在。

这是一节工程课：数学很简单，真正容易失败的是纪律。下面列出的限制项，几乎都能在 Microsoft Agent Governance Toolkit 或 Anthropic Claude Code Agent SDK 文档中找到对应命名。

## 概念

### 成本治理器栈

1. **每次请求的 `max_tokens`。** 最简单的一层，避免任何一次调用吐出无限长回复。
2. **按任务设置 token 预算。** 整个运行过程中，总 token 数不能超过 N。到顶就硬停。
3. **按任务设置金额预算。** 和 token 预算一样，只是以货币计量。Claude Code 中对应 `max_budget_usd`。
4. **按工具设置调用上限。** 例如 `WebFetch` 最多 N 次，`shell_exec` 最多 N 次，等等。
5. **迭代上限（`max_turns`）。** 限制代理循环总轮数，防止无限推理循环。
6. **按分钟 / 小时 / 天 / 月设置上限。** 采用滚动时间窗口（rolling window），在不同时间尺度上抓泄漏。
7. **财务速度限制。** 例如“10 分钟内花费超过 $50 就切断访问”。它能在月度限额触发前先抓住循环烧钱。
8. **分层模型路由。** 默认用便宜模型，只有分类器判断任务值得时，才升级到更贵模型。
9. **Prompt caching。** system prompt 和稳定上下文存到 provider cache；重新发送时的 token 成本接近于零。
10. **Context windowing。** 通过压缩和总结，把活跃上下文压在阈值之下，直接降低 token 成本。
11. **对昂贵动作设置 HITL checkpoint。** 对已知高成本动作，例如长时间工具调用、大文件下载、昂贵模型升级，在执行前要求人工确认。
12. **预算突破后的 kill switch。** 任一上限触发后直接终止 session。触发记录必须保留，重新启用要有单独路径。

### 为什么必须是一整栈，而不是一个 cap

单一月度限额，只会在钱包几乎被烧空之后才发现失控代理。单一按请求限额，在会话级别几乎抓不到任何问题。不同的失败模式需要不同时间尺度来治理：

- **Runaway loop**：代理卡在 5 秒一次的重试里，应该由速度限制抓住。
- **Slow leak**：代理每个任务都做了大约 2 倍预期工作量，应该由每日上限抓住。
- **Bad release**：新版本把 token 消耗拉高到原来的 5 倍，应该由每周或每月上限抓住。
- **Legitimate surge**：需求真实上涨而不是 bug，这时也需要小时或每日上限配合清晰日志来解释增长。

### 一个具体运行框架（harness）的预算暴露面

Claude Code Agent SDK 在公开文档里暴露了这些控制面：

- `max_turns`：迭代上限。
- `max_budget_usd`：美元上限；突破后 session 终止。
- `allowed_tools` / `disallowed_tools`：工具允许列表和拒绝列表。
- 在工具调用前挂 hook，用于自定义成本记账。

它要和第 10 课里的 permission-mode 梯度一起用。一个 `autoMode` 会话如果没有 `max_budget_usd`，本质上就是没有治理的自主运行。Anthropic 也明确把 Auto Mode 描述为必须配预算控制；分类器和成本治理是两条正交控制线，不能互相替代。

### EU AI Act 与 OWASP Agentic Top 10

Microsoft 的 Agent Governance Toolkit 显式覆盖了 OWASP Agentic Top 10，以及 EU AI Act Article 14 关于 human oversight 的要求。对于在欧盟投产的系统来说，日志和上限执行都不是可选项。

### 那个 $1,200 → $4,800 的真实案例

Microsoft 文档里记录了一个真实情况：某个电商代理在新增工具后，月成本直接翻了数倍。这个工具允许代理在每次会话里不断轮询订单状态。系统里没有循环检测（loop detection），没有按工具划分的上限，也没有周环比增长告警。最终的修复是：给该工具单独设上限，再加每日增长告警。

这其实是模板案例：每新增一个工具面，就新增一个潜在循环；每个新工具都需要它自己的上限，也需要它自己的告警。

```figure
cost-governor-stack
```

## 用它

`code/main.py` 会分别模拟“没有成本治理栈”和“有分层成本治理栈”的代理运行。这个模拟代理会在若干轮之后滑进一个轮询循环（polling loop）；分层治理栈会在速度窗口（velocity window）内抓住它，而单一月度限额要等到几天后才会触发。

## 交付成果

`outputs/skill-agent-budget-audit.md` 用来审计一个拟议代理部署的成本治理栈，并标出缺失层。

## 练习

1. 运行 `code/main.py`。确认在 polling-loop 轨迹上，velocity limit 会先于 iteration cap 触发。然后关闭 velocity limit，测量代理在 iteration cap 抓住它之前一共“花掉”多少钱。

2. 为一个浏览器代理（第 11 课）设计按工具划分的上限集合。哪个工具应该设得最紧？哪个工具即使放宽也几乎没有风险？

3. 阅读 Microsoft Agent Governance Toolkit 文档。把工具包里出现的每一种上限类型都列出来，并把它们分别映射到一种失败模式：runaway loop、slow leak、bad release 或 surge。

4. 给一个现实任务估算 overnight unattended run 的价格，例如“在仓库里初筛 50 个 issue”。把 `max_budget_usd` 设为点估计的 2 倍，并解释为什么是 2 倍。

5. Claude Code 的 `max_budget_usd` 只看会话聚合成本。请设计一个你会在外部执行的补充 velocity limit。什么条件会触发切断？重新启用的流程应该是什么样？

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|---|---|---|
| Denial of Wallet | “失控账单” | 代理循环持续花钱，而系统里没有上限去阻止它 |
| max_tokens | “按请求上限” | 单次 completion 大小的天花板 |
| max_turns | “迭代上限” | 一个 session 中代理循环总轮数的上限 |
| max_budget_usd | “美元熔断开关” | 会话成本上限；突破后终止 |
| Velocity limit | “速率上限” | 短时间窗口内的支出限制，例如 $50 / 10 min |
| Tiered routing | “先用小模型” | 默认走便宜模型；只有分类器判断值得时才升级 |
| Prompt caching | “缓存 system prompt” | provider 侧缓存把重复发送的 token 成本压到接近零 |
| HITL checkpoint | “人工审批闸门” | 在昂贵动作前要求人工点确认 |

## 延伸阅读

- [Anthropic Claude Code Agent SDK — agent loop and budgets](https://code.claude.com/docs/en/agent-sdk/agent-loop) — `max_turns`、`max_budget_usd` 与 tool allowlist。
- [Microsoft Agent Framework — human-in-the-loop and governance](https://learn.microsoft.com/en-us/agent-framework/workflows/human-in-the-loop) — 成本治理相关检查点。
- [Anthropic — Claude Managed Agents overview](https://platform.claude.com/docs/en/managed-agents/overview) — provider 侧成本控制。
- [Anthropic — Prompt caching (Claude API docs)](https://platform.claude.com/docs/en/build-with-claude/prompt-caching) — 缓存机制。
- [Anthropic — Measuring agent autonomy in practice](https://www.anthropic.com/research/measuring-agent-autonomy) — 长周期代理的成本画像。
