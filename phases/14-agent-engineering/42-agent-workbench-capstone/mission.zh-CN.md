# 任务 - 毕业项目：发布可复用的智能体工作台组件包

## 目标
将前十一节课的成果组装为一个带版本号的 `outputs/agent-workbench-pack/` 目录，并附带安装器，能够幂等地部署到任何目标仓库中。

## 输入
- 第 32 至 40 课的 schema、脚本和文档
- 组件包布局：`AGENTS.md`、`docs/`、`schemas/`、`scripts/`、`bin/`、`README.md`、`VERSION`

## 交付物
- `outputs/agent-workbench-pack/`，完整布局已填充
- `bin/install.sh`（或 `bin/install.py`），在没有 `--force` 时拒绝覆盖
- `VERSION` 文件，外加一份 `README.md`，说明哪些内容纳入组件包、哪些不纳入

## 验收标准
- `python3 code/main.py` 退出码为零，并打印组件包目录树
- 重新运行组装器是幂等的
- 将 `bin/install.sh` 安装到全新目标仓库后，留下一个可正常运行的工作台：状态、看板、规则、范围、初始化、运行器、门禁、审查器、交接全部就位

## 范围之外
- 针对具体项目的任务内容。任务应放在目标仓库的看板上，而非组件包中。
- 厂商 SDK 调用。组件包在设计上与框架无关。

## 参考
- `docs/en.md` - 完整课程
- `code/main.py` - 参考实现
- `outputs/skill-workbench-pack.md` - 提取的技能
