"""跨框架合规映射——使用 Python 标准库。

给定一项控制措施，输出它满足的框架。给定客户画像（地区 + 客群），
输出所需框架。
"""

from __future__ import annotations


CONTROL_MAP = {
    "访问日志": ["ISO 27001 A.5.15-5.18", "GDPR 第 32 条", "HIPAA §164.312(a)", "SOC 2 CC6"],
    "变更管理": ["ISO 27001 A.8.32", "PCI DSS 要求 6", "HIPAA 违规通知", "SOC 2 CC8"],
    "传输中加密": ["ISO 27001 A.8.24", "GDPR 第 32 条", "HIPAA §164.312(e)", "PCI DSS 要求 4"],
    "密钥管理": ["ISO 27001 A.8.19", "PCI DSS 要求 8", "SOC 2 CC6.1"],
    "PII 脱敏（推理时）": ["GDPR 第 25 条", "欧盟人工智能法案第 10 条", "HIPAA §164.514"],
    "审计日志保留": ["SOC 2 CC7", "HIPAA §164.312(b)", "ISO 27001 A.8.15"],
    "符合性评估": ["欧盟人工智能法案第 43 条（高风险）"],
    "影响评估": ["科罗拉多州人工智能法案 SB24-205", "欧盟人工智能法案第 27 条"],
    "数据主体权利": ["GDPR 第三章", "CCPA"],
    "已签署 BAA": ["HIPAA §164.504(e)"],
}


PROFILE_MAP = {
    ("美国", "B2B SaaS"):           ["SOC 2 Type II", "ISO 27001", "ISO 42001"],
    ("美国", "医疗"):               ["SOC 2 Type II", "HIPAA", "ISO 27001"],
    ("美国", "金融科技"):           ["SOC 2 Type II", "PCI-DSS", "ISO 27001"],
    ("欧盟", "B2B SaaS"):           ["GDPR", "SOC 2 Type II", "ISO 27001", "欧盟人工智能法案"],
    ("欧盟", "医疗"):               ["GDPR", "SOC 2 Type II", "HIPAA（全球）", "欧盟人工智能法案"],
    ("全球", "企业"):               ["SOC 2 Type II", "ISO 27001", "ISO 42001", "GDPR", "HIPAA", "欧盟人工智能法案"],
    ("美国科罗拉多州", "B2B SaaS"): ["SOC 2 Type II", "科罗拉多州人工智能法案", "ISO 27001"],
}


def main() -> None:
    print("=" * 80)
    print("合规控制映射——一项控制，多个框架")
    print("=" * 80)
    for control, frameworks in CONTROL_MAP.items():
        print(f"\n{control}")
        for f in frameworks:
            print(f"  → {f}")

    print("\n" + "=" * 80)
    print("客户画像映射——各地区与客群所需的框架")
    print("=" * 80)
    for (geo, segment), frameworks in PROFILE_MAP.items():
        print(f"\n{geo} · {segment}")
        for f in frameworks:
            print(f"  · {f}")

    print("\n说明：《欧盟人工智能法案》的高风险条款将于 2026 年 8 月 2 日开始执行。")
    print("罚款最高可达 3500 万欧元或全球年营业额的 7%。")


if __name__ == "__main__":
    main()
