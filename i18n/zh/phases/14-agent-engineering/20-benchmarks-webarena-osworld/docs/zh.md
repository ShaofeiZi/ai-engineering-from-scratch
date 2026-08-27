# 基准评测：WebArena 与 OSWorld

> WebArena 测试的是 agent 在四个自托管 web 应用中的能力。OSWorld 测试的是 agent 在 Ubuntu、Windows、macOS 上操作桌面的能力。在它们发布时（2023–2024），两者都展示了顶级 agent 与人类之间的巨大差距。这个差距正在缩小，但失败模式并没有变。

**Type:** 学习
**Languages:** Python（标准库）
**Prerequisites:** 第 14 阶段 · 19（SWE-bench, GAIA）
**Time:** 约 60 分钟

## 学习目标

- 描述 WebArena 的四个自托管应用，以及为什么基于执行的评估至关重要。
- 解释为什么 OSWorld 使用真实操作系统截图，而不是 accessibility APIs。
- 说出 OSWorld 的两个主要失败模式：GUI grounding 与 operational knowledge。
- 总结 OSWorld-G 和 OSWorld-Human 在基础 benchmark 之上分别增加了什么。

## 问题

通用型 agent 会调用工具，但它们能否在 20 次点击里驱动浏览器完成一次购物结账？能否只靠键盘和鼠标配置一台 Linux 机器？WebArena 和 OSWorld 要回答的就是这些问题。

## 概念

### WebArena（Zhou et al., ICLR 2024）

- 包含 812 个长时程任务，分布在四个自托管 web 应用中：购物网站、论坛、类 GitLab 开发工具，以及企业 CMS。
- 另外还提供 map、calculator、scratchpad 等辅助工具。
- 评估是基于执行结果完成的，通过 gym APIs 检查：订单是否真的提交成功，issue 是否真的关闭，CMS 页面是否真的更新。
- 在发布时，最佳 GPT-4 agent 的成功率是 14.41%，而人类是 78.24%。

自托管这一点很关键，因为 benchmark 的目标应用是固定且可复现的，不会因为线上环境漂移而变得脆弱。

### 扩展版本

- **VisualWebArena**：把视觉 grounding 作为核心，任务是否成功取决于对图像的理解（截图本身就是一等观察对象）。
- **TheAgentCompany**（2024 年 12 月）：加入 terminal 与 coding，更接近真实的远程工作环境。

### OSWorld（Xie et al., NeurIPS 2024）

- 包含 369 个真实计算机任务，覆盖 Ubuntu、Windows、macOS。
- 通过键盘与鼠标自由操作真实应用程序。
- 以 1920×1080 的截图作为观察输入。
- 在发布时，最佳模型只有 12.24%，而人类是 72.36%。

### 主要失败模式

1. **GUI grounding。** 从像素到元素的映射。模型在 1920×1080 的画面中很难稳定定位 UI 元素。
2. **Operational knowledge。** 哪个菜单里有设置、哪个快捷键能用、哪个 preference pane 应该打开。这是人类靠多年使用经验积累出来的知识长尾。

### 后续工作

- **OSWorld-G**：提供 564 条样本的 grounding suite，以及 Jedi training set。它把 grounding 与 planning 拆开，让你能分别测量。
- **OSWorld-Human**：提供人工整理的黄金动作轨迹。它显示顶级 agent 的动作步骤数通常比必要值多出 1.4-2.7x，也就是明显存在轨迹效率差距。

### 为什么这很重要

Claude computer use、OpenAI CUA、Gemini 2.5 Computer Use（Lesson 21）都在 WebArena 与 OSWorld 这类工作负载塑造出来的分布上训练。benchmark 是目标，生产模型则是交付出去的答案。

### 做 benchmark 时常见的错误

- **只看截图驱动评测。** OSWorld 是以截图为核心观察的；如果你评估的是一个依赖 DOM 或 accessibility APIs 的 agent，就等于绕开了 grounding 这个真正难点。
- **忽略轨迹长度。** 只看成功率，会漏掉 OSWorld-Human 揭示出来的 1.4-2.7x 步骤低效问题。
- **让自托管应用失去版本固定。** WebArena 的应用依赖固定版本；如果你更新了版本却没有重新整理任务，结果就不再可比。

```figure
ae-agent-human-gap
```

## 动手构建

`code/main.py` 实现了一个 toy web-agent harness：

- 一个最小化的“购物应用”状态机：list_items、add_to_cart、checkout。
- 3 个任务对应的黄金轨迹。
- 一个脚本化的 agent，尝试完成每个任务。
- 一个基于执行结果的 evaluator（状态检查）以及轨迹效率指标（步骤数对比黄金轨迹）。

运行它:

```
python3 code/main.py
```

输出会给出逐任务成功率和轨迹效率，方法上对应 OSWorld-Human 的思路。

## 如何使用

- **WebArena Verified**：自托管在内部集群中，做持续评估。
- **OSWorld**：放在一组 VM fleet 中，评估桌面 agent。
- **Computer-use agents**（Lesson 21）—— Claude、OpenAI CUA、Gemini —— 都是在这类工作负载上训练出来的。
- **你自己的产品流程**：抓取你最重要的 20 个任务的黄金轨迹，每周让 agent 跑一遍。

## 交付成果

`outputs/skill-web-desktop-harness.md` 会生成一个带“基于执行的评估”和“轨迹效率指标”的 web/desktop agent harness。

## 练习

1. 给这个 toy harness 加上第二个应用（论坛）。写 3 个任务并配好黄金轨迹。
2. 为每个任务增加轨迹效率报告。在你的 toy 里，agent 是黄金轨迹的 1x、2x 还是 3x？
3. 实现一个“干扰项”工具，即黄金轨迹永远不会用到的工具。脚本化 agent 会不会被它诱导？
4. 阅读 OSWorld-G。你会如何在自己的 eval 里把 grounding 失败与 planning 失败拆开测量？
5. 读 WebArena 各应用的 README。如果你升级了其中一个固定版本应用，会有什么东西失效？

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------|----------|
| WebArena | “Web agent benchmark” | 4 个自托管应用上的 812 个任务；gym 风格评估 |
| VisualWebArena | “Visual WebArena” | 强调视觉 grounding 的 WebArena；截图就是观察输入 |
| OSWorld | “Desktop agent benchmark” | 在真实 Ubuntu / Windows / macOS 上完成 369 个任务 |
| GUI grounding | “Pixel-to-element mapping” | 模型在 1920x1080 画面中定位 UI 元素 |
| Operational knowledge | “OS know-how” | 知道该去哪个菜单、哪个快捷键、哪个设置页 |
| OSWorld-G | “Grounding suite” | 564 个只测 grounding 的样本加训练集 |
| OSWorld-Human | “Gold trajectories” | 人工专家动作序列，用于衡量效率 |
| Trajectory efficiency | “Steps over gold” | agent 步骤数除以人类最少步骤数 |

## 延伸阅读

- [Zhou et al., WebArena (arXiv:2307.13854)](https://arxiv.org/abs/2307.13854) — 四应用 web benchmark
- [Xie et al., OSWorld (arXiv:2404.07972)](https://arxiv.org/abs/2404.07972) — 跨操作系统桌面 benchmark
- [Anthropic, Introducing computer use](https://www.anthropic.com/news/3-5-models-and-computer-use) — Claude 的 benchmark 驱动能力
- [OpenAI, Computer-Using Agent](https://openai.com/index/computer-using-agent/) — OSWorld 与 WebArena 成绩
