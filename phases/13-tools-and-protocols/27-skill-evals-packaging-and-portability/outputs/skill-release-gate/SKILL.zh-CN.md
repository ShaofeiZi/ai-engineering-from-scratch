---
name: skill-release-gate
description: 在发布前评估智能体技能包的结构完整性、触发器质量、产物改进、脚本正确性、安全性、已安装目录树完整性以及目标主机可移植性。
license: MIT
metadata:
  lesson: "27"
---

# 技能发布门禁

在发布或分发智能体技能目录包之前使用此技能。

## 工作流程

1. 将 `SKILL_ROOT` 解析为包含此已安装 `SKILL.md` 的绝对目录。不要假设进程 cwd 就是已安装的包。
2. 从原始工作区工作目录解析 `TARGET_ROOT`，并将用户提供的候选项解析为绝对路径 `TARGET_BUNDLE`。
3. 从 `SKILL_ROOT` 读取 `references/eval-contract.md`。
4. 检查 `TARGET_BUNDLE` 下 `evals/cases.json` 中的正向触发用例和近似命中触发用例。
5. 检查 `TARGET_BUNDLE` 下 `evals/artifacts.json` 中的共享基线和带技能断言。
6. 检查 `TARGET_BUNDLE` 下 `evals/evidence.json` 中的显式脚本和安全结果。
7. 检查 `TARGET_BUNDLE` 下 `assets/hosts.json` 中声明的运行时能力，并根据其 `assets/manifest.json` 验证目标文件哈希。
8. 对于生产环境，将确定性的预测、产物、证据和主机能力替换为捕获的结果；设置全部四种捕获模式；并将每条原始触发观测、两个产物、完整证据集和非空主机矩阵绑定到非空来源及匹配的 SHA-256 溯源摘要。这些本地检查可以设置 `localEvidenceReady`，但本地可重算的哈希不能证明捕获。
9. 获取一个外部 JSON 证明，其 `evidenceRoot` 与报告匹配，同时获取其确切字节的 SHA-256，该摘要须来自独立的可信策略或发布通道。该证明必须是位于目标包之外的常规文件。
10. 执行前，显示解析出的确切 argv。已安装的评估器为 `SKILL_ROOT` 下的 `scripts/evaluate_skill.py`。对于随附的课程夹具，使用 `python3`、该绝对评估器路径、`--fixture-demo` 以及绝对路径 `TARGET_BUNDLE` 构建 argv。对于生产环境，使用同一个已安装脚本，配合 `--attestation`、`--trusted-attestation-sha256` 和绝对路径 `TARGET_BUNDLE`，不使用 `--fixture-demo`。
11. 返回 `checksPassed`、`fixturePassed`、`localEvidenceReady`、`trustAnchorValid`、`productionReady` 和 `passed`，附带证据根、评估模式、失败检查、精确率、召回率、每条原始触发观测、每用例重复运行率、产物比较、脚本和安全证据、已安装目录树验证以及可移植性矩阵。包含解析出的脚本路径、解析出的目标路径、cwd、确切 argv 和退出码。将不可用的观测标记为未验证。

## 输出契约

返回完整的 JSON 评估报告。保留每一层特定的检查及其证据，使通过的聚合结果无法隐藏路由、产物、脚本、安全、已安装目录树或可移植性方面的失败。`fixturePassed` 报告教学夹具是否成功。`localEvidenceReady` 仅报告本地摘要完整性。`passed` 仅在 `productionReady` 同时具有有效的包外信任锚点时才为真。

## 失败行为

如果配置无效、溯源缺失或不匹配、可信证明缺失或无效、文件哈希不一致、所需能力缺失，或任何生产门禁失败，则以非零结果停止并报告失败的层。显式 `--fixture-demo` 路径仅在 `fixturePassed` 为真时才可成功退出，且它从不做出发布声明。绝不自动发布、在他处安装、修复证据、创建信任决策或降低阈值。

不要仅因为 SKILL.md 能解析或一个正向提示词被激活就发布包。当目标丢弃了所需的伴随文件或忽略所需的运行时扩展时，不要将该包标记为可移植。
