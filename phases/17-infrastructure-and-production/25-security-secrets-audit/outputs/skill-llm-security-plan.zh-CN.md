---
name: llm-security-plan
description: 产出一套 LLM 安全方案，涵盖密钥保险库、PII 清洗与一致性令牌化、网络出口允许列表、审计日志留存以及零信任姿态。
version: 1.0.0
phase: 17
lesson: 25
tags: [security, vault, hashicorp, aws-secrets-manager, pii, presidio, egress, audit-log, zero-trust, ci-cd-supply-chain]
---

给定监管范围（SOC 2、HIPAA、GDPR）、当前凭据状态以及网络/出口姿态，产出一套安全方案。

需产出：

1. 保险库迁移。选择保险库（HashiCorp、AWS Secrets Manager、Azure Key Vault、GCP Secret Manager）。网关模式：应用 → 网关 → 运行时访问保险库。废弃硬编码环境变量和配置文件凭据。
2. 密钥扫描。在每次提交上启用 TruffleHog / GitGuardian / Gitleaks。检测到即阻断 PR。
3. 轮换策略。≤ 90 天。尽可能自动化。CI/CD 凭据须独立轮换（更短——建议 30 天）。
4. PII 清洗。实体识别（Presidio + 正则）。一致性令牌化（同一值 → 同一占位符）以保留语义。
5. 出口允许列表。白名单 LLM 提供商域名、向量数据库、保险库端点。DNS 允许列表解析器。
6. 审计日志。仅追加、不可变。必要字段：用户、租户、Prompt/响应哈希、Token 数、成本、护栏触发记录。按框架留存（SOC 2 为 1 年 / HIPAA 为 6 年）。
7. CI/CD 卫生。OIDC 身份联合（无静态云密钥）。窄范围授予 CI/CD 凭据。引用 2026 年 Vercel 供应链事件作为动机。

硬性拒绝：
- 配置文件中存在静态密钥。拒绝。
- 在审计日志中存储原始 Prompt。拒绝——仅存哈希，除非监管框架明确另行要求。
- 允许出口至 `*` 或"互联网"。拒绝——须用白名单。

拒绝规则：
- 若客户不接受任何保险库（气隙隔离要求），拒绝常规方案并设计基于文件加轮换的备选方案。须明确注明其安全性更低。
- 若以"延迟"为由放弃 PII 清洗，拒绝——延迟通常 <20 ms，而监管风险远超其影响。
- 若要求保险库根 Token 轮换周期 >90 天，拒绝——会成为泄露入口。

输出：一页方案，包含保险库、扫描、轮换、清洗、出口、审计日志、CI/CD 姿态。末尾附单一指标：每月密钥扫描命中数；目标为零。
