# Agent Workbench Pack

适用于任何希望获得可靠智能体工作的仓库的即插即用工作台。

## 你将获得

- `AGENTS.md` 指向 pack 其余部分的简短路由文件。
- `docs/` 规则、可靠性策略、交接协议、评审标准。
- `schemas/` 用于状态、看板和范围契约的 JSON Schema。
- `scripts/` 初始化、反馈运行器、验证关卡、交接生成器。
- `bin/install.sh` 幂等安装脚本。

## 快速开始

```
bin/install.sh
$EDITOR task_board.json
python3 scripts/init_agent.py
```

## 版本管理

`VERSION` 文件即是契约。主版本号升级需要进行状态迁移。
