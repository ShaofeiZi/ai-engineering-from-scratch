# Skill Library 与终身学习（Voyager）

> Voyager（Wang 等，TMLR 2024）把可执行代码视为 Skill。Skill 具有名称，可以检索、组合，并可根据环境反馈持续改进。这是 Claude Agent SDK Skill、skillkit 以及 2026 年 Skill Library 模式的参考架构。

**Type:** 构建
**Languages:** Python（标准库）
**Prerequisites:** 阶段 14 · 07（MemGPT）、阶段 14 · 08（Letta Block）
**Time:** 约 75 分钟

## 学习目标

- 说出 Voyager 的三个组件——自动课程、Skill Library、迭代式 prompting——以及各自职责。
- 解释 Voyager 为什么将 action space 设为代码，而不是 primitive command。
- 仅用标准库实现一个支持注册、检索、组合与失败驱动改进的 Skill Library。
- 将 Voyager 模式映射到 2026 年的 Claude Agent SDK Skill 和 skillkit 生态。

## 问题

如果智能体在每个 session 中都从零重建每项能力，就会犯三个错误：

1. **浪费 token。** 每项任务都会再次诱导出相同的推理。
2. **丢失进步。** 在 session A 中学到的修正无法迁移到 session B。
3. **无法完成长时间跨度的组合。** 复杂任务需要能力层级，一次性 prompt 无法表达这些层级。

Voyager 的答案是：把每项可复用能力表示成库中一个具名代码块，可以按相似度检索、与其他 Skill 组合，并根据执行反馈持续改进。

## 概念

### 三个组件

Voyager（arXiv:2305.16291）围绕三个部分构建智能体：

1. **自动课程。** 一个由好奇心驱动的 proposer 根据智能体当前的 Skill 集合与环境状态选择下一个任务。探索自底向上进行。
2. **Skill Library。** 每个 Skill 都是可执行代码。任务成功时加入新 Skill。Skill 根据 query 与 description 的相似度进行检索。
3. **迭代式 prompting 机制。** 失败后，智能体会收到执行错误、环境反馈与自我验证输出，再据此改进 Skill。

Minecraft 评估结果（Wang 等，2024）：与 baseline 相比，获得的独特物品数量是 3.3 倍，制作石制工具快 8.5 倍，制作铁制工具快 6.4 倍，地图探索距离长 2.3 倍。这些数字只适用于 Minecraft，但模式可以迁移。

### Action space = 代码

大多数智能体发出 primitive command。Voyager 发出 JavaScript 函数。一个 Skill 如下：

```
async function craftIronPickaxe(bot) {
  await mineIron(bot, 3);
  await mineStick(bot, 2);
  await placeCraftingTable(bot);
  await craft(bot, 'iron_pickaxe');
}
```

它由子 Skill 组合而成，以 description 和 embedding 为键存储。检索出来的是程序，不是 prompt。

这就是 2026 年 Claude Agent SDK Skill：一个具名、可检索的代码块，加上智能体按需加载的指令。

### Skill 检索

新任务是“制作一把钻石镐”。智能体会：

1. 对任务描述生成 embedding。
2. 在 Skill Library 中查询 top-k 相似 Skill。
3. 取回 `craftIronPickaxe`、`mineDiamond`、`placeCraftingTable` 等。
4. 使用检索到的 primitive 加新逻辑，组合出新的 Skill。

这正是 MCP resource（阶段 13）与 Agent SDK Skill 实现的模式：针对当前任务，在知识/代码表面上进行检索。

### 迭代式改进

Voyager 的反馈循环：

1. 智能体编写一个 Skill。
2. Skill 在环境中运行。
3. 返回三种信号之一：`success`、`error`（附 stack trace）、`self-verification failure`。
4. 智能体以该信号为上下文重写 Skill。
5. 循环执行，直到成功或达到最大轮数。

这是 Self-Refine（第 05 课）应用于代码生成的形式，并由环境提供可信验证。CRITIC（第 05 课）采用相同模式，只是用外部工具作为 verifier。

### 课程与探索

Voyager 的 curriculum 模块会根据智能体已经拥有和尚未完成的能力，提出“在湖边建一处庇护所”之类的任务。proposer 使用环境状态 + Skill inventory，选择略高于当前能力的任务——这是探索的最佳区间。

对生产智能体而言，这对应一个“还缺什么”operator：给定当前 Skill Library 和一个领域，我们尚未覆盖哪些 Skill？团队通常以人工 curriculum review 的形式实现它。

### 这种模式会在哪里出错

- **Skill Library 腐化。** 同一个 Skill 以略有不同的 description 被添加 10 次。写入时去重；检索时只返回一个。
- **组合 Skill 漂移。** 父 Skill 依赖的子 Skill 得到改进。应对 Skill 做版本控制；固定到 v1 的父 Skill 不能悄悄开始使用 v3。
- **检索质量。** 当库增长到数百项以上，仅在 Skill description 上进行 vector retrieval 会退化。用 tag filter 与硬约束补充检索（“只允许 `category=tooling` 的 Skill”）。

```figure
voyager-skills
```

## 构建它

`code/main.py` 实现一个仅依赖标准库的 Skill Library：

- `Skill`——name、description、code（字符串）、version、tag、dependency。
- `SkillLibrary`——register、search（token overlap）、compose（对 dependency 进行 topological sort）以及 refine（更新时提升版本）。
- 一个脚本化智能体：注册三个 primitive Skill，组合出第四个，遇到失败，然后完成改进。

运行：

```
python3 code/main.py
```

trace 会展示 Library 写入、检索、组合、一次执行失败以及 v2 改进——端到端复现 Voyager 循环。

## 使用它

- **Claude Agent SDK Skill**（Anthropic）——2026 年的参考实现：每个 Skill 有 description、code 与 instruction，在智能体 session 中按需加载。
- **skillkit**（npm: skillkit）——面向 32 种以上 AI 编码智能体的跨智能体 Skill 管理。
- **自定义 Skill Library**——面向特定领域（例如数据智能体的 SQL Skill、基础设施智能体的 Terraform Skill）。Voyager 模式也能向小规模场景缩放。
- **OpenAI Agents SDK `tools`**——较轻量的一端；每个工具可视为轻量 Skill。

## 交付它

`outputs/skill-skill-library.md` 可为任意目标运行时生成 Voyager 风格的 Skill Library，并接好注册、检索、版本控制和改进机制。

## 练习

1. 为 `compose()` 添加 dependency cycle detector。当 Skill A 依赖 B、B 又依赖 A 时，会发生什么？应报错还是警告？
2. 实现逐 Skill 版本固定。父 Skill 组合子 Skill `crafting@1` 时，对 `crafting@2` 的改进不得悄悄升级父 Skill。
3. 用 sentence-transformers embedding（或标准库 BM25 实现）替换 token-overlap retrieval。在包含 50 个 Skill 的玩具库上测量 retrieval@5。
4. 添加一个“curriculum”智能体：给定当前 Library 与领域描述，提出 5 个缺失 Skill。每周调用一次。
5. 阅读 Anthropic 的 Claude Agent SDK Skill 文档。把玩具 Library 移植到 SDK 的 Skill schema。discoverability 会发生什么变化？

## 关键术语

| 术语 | 人们通常怎么说 | 它的实际含义 |
|------|----------------|------------------------|
| Skill | “可复用能力” | 带 description 的具名代码块，可按相似度检索 |
| Skill Library | “智能体的操作方法记忆” | 可持久化、可搜索、可组合的 Skill 存储 |
| Curriculum | “任务 proposer” | 由当前能力缺口驱动、自底向上的目标生成器 |
| Composition | “Skill DAG” | Skill 调用 Skill；执行时按拓扑顺序排列 |
| Iterative refinement | “自我纠正循环” | 环境反馈 + 错误 + 自我验证进入下一版本 |
| Action-space-as-code | “程序化 action” | 发出函数而非 primitive command，以实现跨时间扩展的行为 |
| Dedup on write | “Skill 合并” | description 近似重复时合并为一个规范 Skill |

## 延伸阅读

- [Wang 等，Voyager（arXiv:2305.16291）](https://arxiv.org/abs/2305.16291)——原始 Skill Library 论文
- [Claude Agent SDK 概览](https://platform.claude.com/docs/en/agent-sdk/overview)——Skill 在 2026 年的产品化形式
- [Anthropic，使用 Claude Agent SDK 构建智能体](https://www.anthropic.com/engineering/building-agents-with-the-claude-agent-sdk)——Skill 与子智能体实践
- [Madaan 等，Self-Refine（arXiv:2303.17651）](https://arxiv.org/abs/2303.17651)——Voyager 底层的改进循环
