# 浏览器代理与长周期 Web 任务

> ChatGPT agent（2025 年 7 月）把 Operator 和深度研究（deep research）合并为一个浏览器/终端代理，并在 BrowseComp 上把 SOTA 提到了 68.9%。OpenAI 又在 2025 年 8 月 31 日关闭了独立的 Operator，这是产品层面的整合。Anthropic 收购 Vercept 之后，Claude Sonnet 在 OSWorld 上的成绩从不到 15% 提升到 72.5%。WebArena-Verified（ServiceNow，ICLR 2026）则修正了原始 WebArena 中 11.3 个百分点的假阴性率，并发布了一个 258 任务的 Hard 子集。这些数字都是真的，攻击面也同样真实：OpenAI 的 preparedness 负责人公开表示，浏览器代理里的间接提示注入（indirect prompt injection）“不是一个可以被完全修补的 bug”。2025 到 2026 年间被记录的攻击包括 Tainted Memories（Atlas CSRF）、HashJack（Cato Networks）以及 Perplexity Comet 的单击劫持（one-click hijack）。

**Type:** 学习
**Languages:** Python（stdlib，间接提示注入攻击面模型）
**Prerequisites:** 阶段 15 · 10（权限模式），阶段 15 · 01（长时程代理）
**Time:** 约 45 分钟

## 问题

浏览器代理是一种长周期代理：它会读取不受信任的内容，并据此执行有后果的动作。代理访问的每一个页面，都是用户本人没有亲自写下的输入；每个页面上的每张表单，都可能成为一条命令通道。2025 到 2026 年的攻击样本说明，这不是理论问题。Tainted Memories 允许攻击者借助精心构造的页面，把恶意指令绑定进代理记忆；HashJack 把命令藏进代理访问的 URL fragment；Perplexity Comet 类攻击甚至可以在一次点击里劫持行为。

防守视角并不轻松。OpenAI 的 preparedness 负责人把关键事实说得很直接：间接提示注入“不是一个可以被完全修补的 bug”。原因在于，攻击发生在代理的“读取”和“行动”之间，而这条边界在架构上本来就是模糊的。模型读到的每个 token，理论上都可能被解释成指令。

本课会明确攻击面，梳理基准格局（BrowseComp、OSWorld、WebArena-Verified），并构造一个最小的间接提示注入场景，让你能在第 14 课和第 18 课里继续推导真正可落地的防御方案。

## 概念

### 2026 年格局：每个系统一句话

**ChatGPT agent（OpenAI）。** 2025 年 7 月发布。它把 Operator（浏览）和 Deep Research（多小时研究）统一到一个产品里，并在 2025 年 8 月 31 日关闭了独立的 Operator。在 BrowseComp 上达到 68.9% 的 SOTA，在 OSWorld 和 WebArena-Verified 上也有很强的表现。

**Claude Sonnet + Vercept（Anthropic）。** Anthropic 收购 Vercept 的核心目标是计算机使用（computer-use）能力，这次整合把 Claude Sonnet 在 OSWorld 上的表现从低于 15% 拉升到了 72.5%。Claude Computer Use 以工具 API 的形式对外提供。

**Gemini 3 Pro with Browser Use（DeepMind）。** Browser Use 集成提供了计算机使用控制；FSF v3（2026 年 4 月，第 20 课会讲）则专门跟踪 ML R&D 领域里的自主性风险。

**WebArena-Verified（ServiceNow，ICLR 2026）。** 它修复了一个被充分记录的问题：原始 WebArena 大约有 11.3% 的假阴性率，也就是一些实际上已经完成的任务被误判为失败。Verified 版本用人工策划的成功标准重新评分，并增加了一个 258 任务的 Hard 子集（ICLR 2026 论文，openreview.net/forum?id=94tlGxmqkN）。

### BrowseComp、OSWorld 与 WebArena 的区别

| Benchmark | 它衡量什么 | 时间跨度 |
|---|---|---|
| BrowseComp | 在开放 Web 上顶着时间压力找到具体事实 | 分钟级 |
| OSWorld | 操作完整桌面环境（鼠标、键盘、shell） | 数十分钟 |
| WebArena-Verified | 在模拟网站里完成交易型 Web 任务 | 分钟级 |
| Hard subset | WebArena-Verified 中涉及多页面状态迁移的任务 | 数十分钟 |

它们衡量的是不同维度。BrowseComp 分数高，说明代理擅长找事实；不代表它就会订机票。OSWorld 更接近“它在我的桌面上能不能跑通”。WebArena-Verified 更接近“它能不能把一条流程做完”。要做生产决策，必须选择与你任务分布相匹配的基准。

### 把攻击面逐项命名

1. **间接提示词注入（Indirect prompt injection）。** 不受信任的页面内容里夹带指令；代理读到它们；代理照做。公开例子包括 2024 年 Kai Greshake 等人的工作、2025 年的 Tainted Memories 论文，以及 2026 年的 HashJack（Cato Networks）。
2. **URL fragment / query injection。** 被爬取 URL 的 `#fragment` 或 query string 中夹带命令。它可能根本不会出现在用户肉眼看到的页面里，但仍然进入代理上下文。
3. **Memory-binding attacks。** 页面诱导代理写入持久记忆（第 12 课会讲持久状态）。到下一次会话时，这段记忆会在没有可见触发器的情况下释放恶意载荷（payload）。
4. **CSRF-shaped attacks on authenticated sessions。** Tainted Memories 这一类攻击的典型形态是：代理在某处已登录，攻击者页面诱导代理发出带用户 cookies 的状态变更请求。
5. **One-click hijack。** 表面上无害的可见按钮，背后却承载着代理会继续跟随的恶意载荷。Perplexity Comet 的案例属于这一类。
6. **Agent host surface 上的 Content-Security-Policy 漏洞。** 不只是页面内容本身，渲染层和工具层也可能成为攻击向量；浏览器代理内部再嵌浏览器的这套栈，本来就很宽。

### 为什么“无法被完全修补”

这种攻击和代理能力本身是同构的。代理要完成工作，就必须阅读不受信任的内容；而它读到的任何内容，都可能携带指令；它遵循的任何指令，都可能偏离用户的真实意图。各种防御手段，例如信任边界（trust boundary）、分类器、工具允许列表（tool allowlist）、对后果性动作加 HITL，只能提高攻击成本、压缩影响半径（blast radius），不能把这个类别彻底消灭。

这和第 8 课里 Lob's theorem 的推理模式相似：代理无法证明“下一个 token 是安全的”；它只能通过系统设计，让不安全 token 更容易被发现。

### 真正在生产里会落地的防御姿态

- **Read / write boundary。** 读取永远不直接产生后果。写入，例如提交表单、发布内容、调用有副作用的工具，如果发起依据来自信任边界外，就必须重新走人工批准。
- **每个任务独立的工具允许列表。** 代理可以浏览；但除非你明确为该任务启用转账能力，否则它不该能发起 wire transfer。预算问题会在第 13 课继续展开。
- **Session isolation。** 浏览器代理会话只运行在受限凭证下。不挂生产权限，不连个人邮箱，每个 HTTP 请求都保留审计日志。
- **内容净化器（Content sanitizer）。** 抓取到的 HTML 在拼进模型上下文之前，先去除已知坏模式。它能挡住简单攻击，但挡不住复杂载荷。
- **对后果性动作使用 HITL。** 采用 propose-then-commit 模式（第 15 课）。
- **对记忆设置 canary token。** 如果某条记忆被触发，用户会立刻看见（第 14 课）。

```figure
injection-boundary
```

## 用它

`code/main.py` 模拟了一个极小的浏览器代理运行过程，它会访问三个合成页面。一页是良性的；一页在可见文本里直接嵌入提示注入；另一页把注入放进 URL fragment 中，用户看不见，但代理上下文能看到。脚本会展示：(a) 天真的代理会怎么做；(b) 读写边界能拦住什么；(c) 净化器能拦住什么；(d) 两者都拦不住什么。

## 交付成果

`outputs/skill-browser-agent-trust-boundary.md` 用来界定一个拟议浏览器代理部署的边界：它会接触哪些信任区（trust zone）、被授权写入什么，以及在首次运行前必须具备哪些防御。

## 练习

1. 运行 `code/main.py`。辨认哪一类攻击是净化器能抓住、但读写边界抓不住的；又有哪些攻击只能靠读写边界抓住。

2. 扩展净化器，让它能检测一种 HashJack 风格的 URL-fragment 注入。再测量它在带合法 fragment 的良性 URL 上会产生多少假阳性。

3. 选一个你熟悉的真实浏览器代理流程，例如“预订机票”。把其中每一次读取和每一次写入都列出来，并标出哪些写入必须走 HITL，以及原因是什么。

4. 阅读 WebArena-Verified 的 ICLR 2026 论文。找出原始 WebArena 在哪一类任务上的打分不可靠，并解释 Verified 子集是如何修复这个问题的。

5. 为浏览器代理场景设计一个记忆金丝雀（memory canary）。你会存什么、存在哪里、由什么条件触发告警？

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|---|---|---|
| 间接提示词注入（Indirect prompt injection） | “坏页面文本” | 代理读取的页面里包含不受信任的指令，并最终执行它们 |
| Tainted Memories | “记忆攻击” | 代理把攻击者提供的指令写入持久记忆，并在下次会话被触发 |
| HashJack | “URL fragment 攻击” | 恶意载荷藏在 URL fragment 或 query string 中，进入代理上下文但不一定可见 |
| One-click hijack | “坏按钮” | 表面无害的可见交互，实际携带后续恶意载荷供代理执行 |
| BrowseComp | “Web 搜索基准” | 在开放 Web 上查找具体事实，时间跨度为分钟级 |
| OSWorld | “桌面基准” | 完整 OS 控制下的多步 GUI 任务 |
| WebArena-Verified | “修正后的 Web 任务基准” | ServiceNow 重新评分后的 WebArena，并附带 Hard 子集 |
| 读写边界（Read/write boundary） | “副作用闸门” | 读取永远不直接产生后果；来自信任边界外内容触发的写入需要重新批准 |

## 延伸阅读

- [OpenAI — Introducing ChatGPT agent](https://openai.com/index/introducing-chatgpt-agent/) — Operator 与深度研究（deep research）的合并，以及 BrowseComp SOTA。
- [OpenAI — Computer-Using Agent](https://openai.com/index/computer-using-agent/) — Operator 这条产品线，以及后来演化为 ChatGPT agent 的架构。
- [Zhou et al. — WebArena](https://webarena.dev/) — 原始基准。
- [WebArena-Verified (OpenReview)](https://openreview.net/forum?id=94tlGxmqkN) — ICLR 2026 的修正版子集论文。
- [Anthropic — Measuring agent autonomy in practice](https://www.anthropic.com/research/measuring-agent-autonomy) — 包含对计算机使用代理攻击面的讨论。
