---
name: skill-catalog-builder
description: 在显式发现范围内构建一个有界智能体技能目录，并在加载指令主体之前报告冲突。
license: MIT
metadata:
  lesson: "24"
---

# 技能目录构建器

当智能体宿主需要在多个技能目录之间进行确定性发现时，使用此技能。

1. 阅读 `references/discovery-contract.md`。
2. 查看 `assets/scope-policy.json` 中的示例宿主策略；不要假设其顺序是通用的。
3. 运行 `python3 scripts/build_catalog.py project=PATH user=PATH`，范围按从最高优先级到最低优先级的顺序列出。
4. 在激活某个技能之前，检查 JSON 的 `collisions` 和 `omitted` 数组。
5. 仅加载所选的 SKILL.md 主体。仅当该主体明确引用某个直接引用时，才加载该直接引用。

在发现过程中，切勿执行捆绑的脚本。切勿因偶然的文件系统顺序而选择同等优先级的重复项。

返回目录预算、已选条目、冲突解决方案以及省略项。
