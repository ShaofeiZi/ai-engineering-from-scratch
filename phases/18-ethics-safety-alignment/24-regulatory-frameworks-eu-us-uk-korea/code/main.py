"""监管框架时间线打印器——仅使用 Python 标准库。

打印 EU AI Act、GPAI Code of Practice、Transparency Code、UK AISI 更名、
US CAISI 更名以及韩国 AI Framework Act 里程碑的统一时间线。

仅供参考；主要来源见 docs/en.md。

用法：python3 code/main.py
"""

from __future__ import annotations


TIMELINE = [
    ("2024-08-01", "EU AI Act 生效"),
    ("2024-12-00", "韩国国会通过 AI Framework Act"),
    ("2025-01-00", "韩国颁布 AI Framework Act（2026 年 1 月生效）"),
    ("2025-02-02", "EU AI Act：禁止性实践和 AI 素养要求开始适用"),
    ("2025-02-00", "UK AISI 更名为 AI Security Institute"),
    ("2025-06-00", "US AISI 更名为 CAISI（Center for AI Standards and Innovation）"),
    ("2025-07-10", "GPAI Code of Practice 发布（3 章，12 项承诺）"),
    ("2025-08-02", "EU AI Act：GPAI 和治理义务开始适用"),
    ("2025-12-17", "Article 50 Transparency Code 初稿"),
    ("2026-01-00", "韩国 AI Framework Act 生效"),
    ("2026-03-00", "Transparency Code 第二稿"),
    ("2026-06-00", "Transparency Code 最终版"),
    ("2026-08-02", "EU AI Act：全面适用 + Article 50 透明度要求 + 处罚"),
    ("2027-08-02", "EU AI Act：旧有 GPAI + 嵌入式高风险系统"),
]


def main() -> None:
    print("=" * 78)
    print("AI 监管时间线（阶段 18，第 24 课）")
    print("=" * 78)
    for date, event in TIMELINE:
        print(f"  {date}  {event}")
    print("\n" + "=" * 78)
    print("要点：EU AI Act 设定了全球标准，并于 2026 年 8 月全面执行。")
    print("英国聚焦前沿安全，美国转向促进增长。韩国建立了亚洲首个综合框架。")
    print("跨司法辖区的部署方需遵守最严格的规则，通常就是欧盟规则。")
    print("=" * 78)


if __name__ == "__main__":
    main()
