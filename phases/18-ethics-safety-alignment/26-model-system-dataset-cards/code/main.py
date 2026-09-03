"""最小化的 model card、datasheet、system card 生成器——仅使用 Python 标准库。

为玩具部署生成三种规范文档：
  - Model Card (Mitchell et al. 2019)
  - Datasheet (Gebru et al. 2018)
  - System Card (Sidhpurwala 2024 / "Blueprints of Trust" 2025)

每份文档都是打印到 stdout 的 Markdown 字符串，各章节遵循规范模板。

用法：python3 code/main.py
"""

from __future__ import annotations


def model_card() -> str:
    return """
# Model Card：ToyClassifier-1.0

## 模型详情
- 开发者：ai-engineering-from-scratch / Phase 18 / Lesson 26
- 版本：1.0.0
- 类型：二元逻辑分类器（玩具示例）
- 许可证：MIT
- 联系方式：phase-18-lesson-26

## 预期用途
- 主要用途：教学演示
- 范围外：任何生产决策

## 因素
- 敏感属性：性别（玩具示例中为二元）、年龄分组
- 环境：受控合成数据

## 指标
- 准确率、demographic parity、equalized odds（参见第 21 课）

## 训练数据
- 合成数据集；参见随附的 Datasheet

## 定量分析
- 总体准确率：0.97
- demographic parity 差距：+0.03（group0 与 group1）
- equalized odds TPR 差距：-0.01

## 伦理考量
- 玩具分类器；未验证可用于现实场景。
- 偏见指标仅为占位示例；任何部署前都应完成全面审计。

## 注意事项与建议
- 使用部署场景特定数据重新训练。
- 若训练数据包含 PII，请应用第 22 课（DP）。
"""


def datasheet() -> str:
    return """
# Datasheet：ToyBinaryClassification-1.0

## 动机
- 为阶段 18 第 26 课的教学演示而创建
- 无资助方；不用于生产环境

## 构成
- 1,500 个合成样本
- 特征：两个连续维度和一个二元敏感属性
- 标签：二元，根据 x[0] + x[1] > 0 规则生成

## 收集过程
- 使用固定种子的 Python random.gauss 合成生成
- 不涉及人类受试者

## 标注
- 标签由程序生成，不存在人工标注错误

## 用途
- 预期用途：讲授公平性指标（第 21 课）和偏见探针（第 20 课）
- 禁止用途：作为任何生产规模数据集的代理

## 分发
- 包含在 Phase 18 / Lesson 26 仓库目录中

## 维护
- 静态数据；每次运行时根据固定种子重新生成
"""


def system_card() -> str:
    return """
# System Card：ToyClassifier 服务

## 部署
- 范围：localhost 教学服务
- 技术栈：单线程 HTTP 服务器后的 ToyClassifier-1.0

## 安全能力
- 提示词注入：不适用（非生成式）
- 数据外泄检测：基础出口速率限制
- 速率限制：每个客户端 100 req/min

## 对齐
- 模型仅反映合成标签规则
- 无 RLHF；无拒绝策略

## 事件响应
- 无生产 SLA；没有升级渠道
- Issue 跟踪器：Phase 18 / Lesson 26

## 监管对齐
- EU AI Act：不适用（玩具示例；未在欧盟部署）
- GPAI Code of Practice：不适用（非 GPAI）
- Transparency Code：不适用（不输出 AI 生成内容）
"""


def main() -> None:
    print("=" * 74)
    print("CARDS 生成器（阶段 18，第 26 课）")
    print("=" * 74)
    print(model_card())
    print(datasheet())
    print(system_card())
    print("=" * 74)
    print("要点：三种规范卡片覆盖三个范围。model card 记录模型，datasheet 记录")
    print("数据，system card 记录部署。到 2026 年，EU AI Act GPAI Code of")
    print("Practice 要求将 model card 作为合规工件。可验证证明（Laminator 2024）")
    print("是下一阶段。")
    print("=" * 74)


if __name__ == "__main__":
    main()
