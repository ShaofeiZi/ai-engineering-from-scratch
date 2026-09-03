---
name: skill-contract-reviewer
description: 验证一个智能体技能包，并在实现前选择正确的指令、能力或生命周期原语。
license: MIT
metadata:
  lesson: "22"
---

# 技能合约审查器

当一个工作流即将成为可复用的智能体产物时，使用此技能。

1. 将 `SKILL_ROOT` 设置为包含已安装 `SKILL.md` 的绝对目录。不要假设进程工作目录就是该 bundle。
2. 将 `TARGET_ROOT` 设置为原始工作区的绝对工作目录，并在该根目录下解析所提议的技能目录。
3. 读取 `$SKILL_ROOT/references/contract.md` 并验证可移植的 `SKILL.md` 身份字段。
4. 读取 `$SKILL_ROOT/references/decision-model.md`，并区分仓库上下文、可复用方法、外部能力、生命周期时序、确定性逻辑和隔离委托。
5. 在执行前，展示精确解析后的参数向量。运行
   `python3 "$SKILL_ROOT/scripts/check_skill.py" "$TARGET_SKILL"`，其中
   `TARGET_SKILL` 是 `TARGET_ROOT` 下所提议技能的绝对目录。
6. 检查 JSON 报告。在讨论宿主特定扩展之前修复每一个错误。
7. 将所提议的产物与 `$SKILL_ROOT/assets/task-shapes.json` 进行比较，并返回最小可组合的原语集合。

不要声称运行时扩展是可移植合约的一部分。不要将有效技能视为运行脚本或访问工具的许可。

返回验证报告、所选原语，以及解释每个选择的一句话。包含执行证据，其中包括解析后的脚本路径、解析后的目标路径、cwd、精确 argv 和退出码。如果宿主无法暴露其中某个观察项，则将其标记为未验证，而不是编造它。
