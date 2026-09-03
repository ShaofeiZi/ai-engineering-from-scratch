---
name: computer-use-safety
description: 为计算机操作智能体构建逐步安全分类器与确认门控，包含导航白名单和注入标记过滤。
version: 1.0.0
phase: 14
lesson: 21
tags: [computer-use, safety, claude, openai-cua, gemini]
---

给定一个计算机操作智能体和一组目标应用，产出一个在每次执行前对每个操作进行分类的安全层。

产出内容：

1. `SafetyClassifier.assess(action, screen) -> SafetyVerdict`，包含字段 `allow`、`reason`、`needs_confirmation`。
2. 智能体可点击的元素标签白名单；不在列表中的则拒绝。
3. 智能体可导航的 URL 白名单；重定向跳出列表的则拒绝。
4. 对 DOM 文本、检索内容和键入文本进行注入标记过滤。任何匹配都会阻止该操作。
5. 针对敏感操作（登录、购买、删除、发布）的确认门控。人在回路回调接口。
6. 追踪发射器：每个决策都记录 (action, verdict, reason)。

硬性拒绝项：

- 仅在第一个操作上运行的安全分类器。每个操作都必须经过分类。
- 形式为 `*` 的白名单。允许一切的白名单不是白名单。
- 因为模型"看起来有信心"而跳过确认。置信度不等于安全性。

拒绝规则：

- 如果智能体拥有计算机操作权限但没有逐步安全保障，拒绝发布。
- 如果智能体可以导航到任意 URL，拒绝。要求使用白名单或黑名单。
- 如果敏感操作在任何模式下绕过确认门控，拒绝。

输出：`classifier.py`、`allowlist.py`、`confirmation.py`、`trace.py`、`README.md`，其中 README 解释门控策略、注入标记和白名单维护流程。以"延伸阅读"结尾，指向第 27 课（提示词注入）和第 23 课（用于安全决策的 OTel span 归因）。
