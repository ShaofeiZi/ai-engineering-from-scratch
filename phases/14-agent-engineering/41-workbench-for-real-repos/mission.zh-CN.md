# 任务 - 在真实仓库上的工作台

## 目标
通过仅提示词的流水线和工作台引导的流水线，对同一个示例应用执行相同的 `/signup` 校验任务，然后产出一份怀疑者也能读懂的前后对比报告。

## 输入
- `sample_app/`，包含 `app.py`（无校验）、`test_app.py`（一个正常路径测试）、`README.md`、`scripts/release.sh` 作为禁区诱饵
- 两条流水线均已完全脚本化，不进行真实 LLM 调用

## 交付物
- `code/main.py`，针对同一 fixture 编排两条流水线
- `before-after-report.md`，包含五项结果表格
- `comparison.json`，用于下游图表生成

## 验收标准
- `python3 code/main.py` 退出码为零
- 报告衡量全部五项结果：测试是否实际运行、验收是否达成、是否触及范围外文件、交接质量、评审者总计
- 工作台流水线在五项中至少四项上优于仅提示词流水线

## 不在范围内
- 接入真实 LLM。流水线已脚本化以确保可复现性。
- 调优模型。该对比通过构造方式保持模型不变。

## 参考资料
- `docs/en.md` - 完整课程
- `code/main.py` - 参考实现
- `outputs/skill-workbench-benchmark.md` - 提取的技能
