---
name: multimodal-agent-designer
description: 设计一个多模态智能体（计算机操控、GUI 定位、Web 或移动端），包含动作模式、记忆策略与基准评测方案。
version: 1.0.0
phase: 12
lesson: 25
tags: [multimodal-agents, computer-use, gui-grounding, visualwebarena, agentvista]
---

给定一个计算机操控产品规格（领域、动作集、评测目标），设计智能体循环、记忆策略、定位方式和评测方案。

产出：

1. 动作模式。用 JSON 定义支持的动作（click、type、scroll、drag、select、navigate、done，以及任何视觉工具）。
2. 输入模式。仅截图、无障碍树或混合模式。浏览器默认混合模式；无无障碍钩子的桌面应用使用仅截图模式。
3. 模型选择。Qwen2.5-VL-72B（开源）、Claude Opus 4.7 computer-use（闭源，较强）、GPT-5（闭源，更强）。根据基准和成本进行论证。
4. 记忆策略。每 5 步一次摘要链 + 最近 2 张截图保持实时；超长工作流仅记录日志。
5. 错误恢复。动作失败时，通过 element_desc 语义提示重新定位；最多重试 2 次；回退到重新规划。
6. 评测方案。ScreenSpot-Pro 用于定位，VisualWebArena 用于端到端，AgentVista 用于高难度多步工作流。给出预期分数等级。

硬性否决项：
- 使用自由文本动作输出。必须始终使用 JSON 结构化输出并具有显式模式。
- 声称开源 7B 模型在 AgentVista 上匹敌前沿模型。差距为 10-20 分。
- 依赖跨截图的坐标记忆。截图之间的坐标会发生漂移。

拒绝规则：
- 若产品需要 >50 步的工作流，拒绝单智能体循环，并建议采用分层规划器 + 执行器拆分。
- 若产品运行在无无障碍钩子的受监管平台上，标记仅截图模式的可靠性限制，并提出强验证方案。
- 若任务类别超出已训练分布（专业工业软件），拒绝现成方案，并提出在领域截图上进行微调。

输出：一页智能体设计，包含动作模式、输入模式、模型选择、记忆、恢复、评测。结尾列出 arXiv 2401.10935（SeeClick）、2401.13649（VisualWebArena）、2602.23166（AgentVista）。
