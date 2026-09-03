---
name: skill-invocation-router
description: 为智能体技能目录设计并测试显式人类、隐式模型或智能体、程序化应用、有界技能组合以及测试框架激活策略。
license: MIT
metadata:
  lesson: "25"
---

# 技能调用路由器

当宿主需要可审计的激活策略，而非一个无差别的 `invocable` 标志时，使用此技能。

1. 阅读 `references/invocation-model.md` 并对所请求的通道进行分类。
2. 将 `assets/host-policy.json` 作为适配器配置示例进行审查，而非可移植标准。
3. 运行 `python3 scripts/simulate_invocation.py --policy assets/host-policy.json --actor ACTOR --name NAME --description DESCRIPTION --query QUERY [--explicit-name NAME] [--caller-name NAME] [--depth N] [--user-invocable true|false] [--disable-model-invocation true|false]`。
4. 对于人类、应用、技能或测试框架的请求，要求精确匹配已发现的名称及其通道特定的允许列表。
5. 对于技能调用方，还要求调用方身份、非循环目标以及有界的组合深度。
6. 对于模型或自主智能体请求，移除由该执行者或已识别的宿主扩展判定为不可用的候选项。
7. 仅对剩余的描述评分。选择得分最高的合格匹配，若无合格候选项达到阈值则弃权。
8. 返回包含适配器、通道、分数和策略原因的 JSON 决策。

激活仅加载指令。它不批准工具、文件系统更改、网络访问、机密使用或捆绑脚本。
