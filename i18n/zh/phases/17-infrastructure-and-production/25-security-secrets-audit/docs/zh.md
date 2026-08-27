# 安全：密钥、API Key 轮换、审计日志与护栏

> 2026 年要消除 secrets sprawl，标准做法是使用集中式密钥库，例如 HashiCorp Vault、AWS Secrets Manager、Azure Key Vault。凭据绝不能存进配置文件、版本库里的 env 文件或电子表格。应用优先使用 IAM roles 而不是静态 key；CI/CD 优先使用 OIDC。AI-gateway 模式是 2026 年的主流答案：apps → gateway → model provider，gateway 在运行时从 vault 拉取凭据。只要在 vault 里完成 rotation，所有应用几分钟内就会拿到新 key，不需要 redeploy，也不需要在 Slack 里到处问“谁有新 key”。rotation policy 应该 ≤90 天；每次 commit 都要用 TruffleHog、GitGuardian 或 Gitleaks 做扫描。零信任基线包括 MFA、SSO、RBAC/ABAC、短生命周期 token 和 device posture。PII scrubbing 则依靠 entity recognition 在转发前遮蔽 PHI/PII；consistent tokenization（Mesh approach）把敏感值映射成稳定占位符，以便 LLM 保留代码与关系语义。网络出站方面，LLM 服务应放在独立 VPC/VNet 子网，只允许访问 `api.openai.com`、`api.anthropic.com` 等必要域名，其余一律阻断。2026 年最值得记住的事故驱动案例，是 Vercel 供应链攻击：攻击者通过受损的 CI/CD 凭据，跨数千个客户部署窃取了 env vars。

**Type:** 学习
**Languages:** Python（标准库，玩具级 PII 脱敏器与审计日志写入器）
**Prerequisites:** 阶段 17 · 19（AI 网关）、阶段 17 · 13（可观测性）
**Time:** 约 60 分钟

## 学习目标

- 列出 4 种 secrets management 反模式：把配置文件提交到 VCS、硬编码 env、用电子表格管密钥、长期静态 key，并说出各自替代方案。
- 解释 AI-gateway-pulls-from-vault 这一模式为什么已经成为 2026 年生产环境标准。
- 实现一个带 consistent tokenization 的 PII scrubber，让相同敏感值始终映射到相同占位符，并保留语义关系。
- 说出 2026 年 Vercel 供应链事件，以及它对 CI/CD credential hygiene 的警示。

## 问题

一个实习生把带 API key 的 `.env` 提交进了仓库。虽然很快删掉了，但 key 已经进入 git history。GitGuardian 扫描发现后，你们的 rotation 流程却是：“Slack 通知全员、更新 40 份配置、重部署所有服务。”8 小时后，一半服务已经切到新 key，另一半还在等变更窗口。

与此同时，用户 prompt 里包含了 “My SSN is 123-45-6789.”，请求被直接发给了 OpenAI。你们虽然签了 BAA，但内部 policy 明明要求在转发前屏蔽 PII，结果并没有做到。

再同时，你们 EKS 集群里的 LLM pod 可以访问任意公网主机。于是有人通过指向攻击者域名的 DNS 查询把数据外带了出去，而你们没有任何拦截。

LLM 服务的安全必须同时覆盖这三类问题：vault-backed credentials、PII scrubbing、network egress filtering，以及审计日志。

## 概念

### 集中式密钥库 + IAM role 拉取

**Vault**：HashiCorp Vault、AWS Secrets Manager、Azure Key Vault、GCP Secret Manager。它们共同承担“一处存真相”的角色。

**IAM role**：应用或 gateway 用自己的 IAM 身份完成认证，而不是携带静态 key。vault 再按 token 生命周期把 secret 返回给它。

**AI-gateway 模式**：gateway 在处理请求时，从 vault 动态拉取 `OPENAI_API_KEY`。只要在 vault 中完成 rotation，下一次请求就会自动拿到新 key，不需要 redeploy。

### 轮换周期不超过 90 天

所有 API key、vault root token、CI/CD 凭据都应该遵守这个窗口。能自动轮换的就自动轮换；必须手动轮换的，也要留下日志并可追踪。

### 密钥扫描

- **TruffleHog**：基于 regex + entropy 扫描 commit。
- **GitGuardian**：商业方案，准确率较高。
- **Gitleaks**：开源方案，常见于 CI。

每次 commit 都要扫描。一旦发现新增 secret，就阻断 PR。

### 零信任姿态

- 所有账号都必须启用 MFA。
- SSO 走 SAML 或 OIDC。
- 细粒度访问控制使用 RBAC 或 ABAC。
- token 尽量短命，按小时而不是按天计算。
- 设备姿态要纳入控制，只允许启用磁盘加密的企业设备接入。

### PII / PHI 脱敏

在 prompt 离开你自己的基础设施之前，需要经过下面几个步骤：

1. 做 entity recognition，可以用 spaCy NER、Presidio 或商业方案。
2. 屏蔽命中的实体，例如把 `"My SSN is 123-45-6789"` 变成 `"My SSN is [SSN_TOKEN_A3F]"`。
3. 使用 consistent tokenization（Mesh approach），让相同的敏感值始终映射到同一个 placeholder，以便 LLM 保留关系语义。
4. 如果业务需要，可以为 LLM 输出做可选的 reverse mapping。

静态 regex 过滤器适合抓基础模式，NER 适合抓更复杂的实体。实际生产里两者都要上。

### 输入与输出护栏

输入侧：阻断已知 jailbreak、禁止话题，并按用户做 rate limit。

输出侧：用 regex 检查是否泄露 secret，例如 API key pattern，或在拒答上下文里是否出现 email pattern；再叠加 classifier 检测 policy violation。

### 网络出站白名单

把 LLM 服务放进独立子网，并把出站策略收紧到 allowlist：

- 允许访问：`api.openai.com`、`api.anthropic.com`、vector DB endpoint、vault endpoint。
- 其他所有外联全部 drop。
- DNS 也通过只允许 allowlist 的 resolver，以防 DNS tunneling exfiltration。

### 审计日志

每一次 LLM 调用都应该留下不可变日志，至少包括：

- Timestamp
- User / tenant
- Prompt hash，而不是原始 prompt，以降低隐私风险
- Model + version
- Token counts
- Cost
- Response hash
- 任何 guardrail 触发记录

日志保留时间按监管要求来做，例如 SOC 2 通常保 1 年，HIPAA 通常保 6 年。

### 2026 年 Vercel 事件

这是一次典型的供应链攻击：攻击者利用被攻陷的 CI/CD 凭据，从数千个客户部署中外带 env vars。结论非常直接：CI/CD 凭据就是 production-equivalent credential。它们必须放进 vault、尽可能缩小权限范围，并且激进轮换。

### 你应该记住的数字

- Rotation policy：≤ 90 天。
- 每次 commit 都要跑扫描：TruffleHog、GitGuardian、Gitleaks。
- Vercel 2026 事件：CI/CD 凭据失陷，导致数千个客户的 env vars 泄露。
- Audit log retention：SOC 2 = 1 年，HIPAA = 6 年。

```figure
i4-vault-rotation
```

## 用起来

`code/main.py` 实现了一个玩具版 PII scrubber，带有 consistent tokenization 和 append-only audit log。

## 交付物

这一课产出 `outputs/skill-llm-security-plan.md`。给定监管范围和当前状态，它会规划 vault migration、scrubber、egress policy 和 audit log。

## 练习

1. 运行 `code/main.py`。发送两个引用同一个 SSN 的 prompt，确认它们都被替换成同一个 placeholder。
2. 为一个调用 OpenAI + Anthropic + Weaviate 的 vLLM-on-EKS 部署设计 network egress policy。
3. 你在 git history 里发现了一把 2 年前的 key。正确动作是什么：只 rotation、只清 history，还是两者都做？说明理由。
4. 如果 audit log 每天增长 10 GB，设计一个分层保留策略，例如 hot 30d、warm 12mo、cold 6yr。
5. 论证 reverse-tokenization 是否值得这套复杂度，还是直接让 placeholder 保持可见更合理。

## 关键术语

| 术语 | 人们常说什么 | 实际含义 |
|------|----------------|------------------------|
| Vault | “密钥存储” | 集中式凭据管理服务 |
| IAM role | “基于身份的认证” | 由应用承担的角色，用于换取短生命周期凭据 |
| OIDC for CI/CD | “云签发 token” | CI 中不放静态 key，而是用 OIDC 做身份交换 |
| TruffleHog / GitGuardian / Gitleaks | “密钥扫描器” | 在 commit 时发现 secrets 的扫描器 |
| RBAC / ABAC | “访问控制” | 基于角色与基于属性的授权方式 |
| PII scrubbing | “数据脱敏” | 删除或 token 化敏感实体 |
| Consistent tokenization | “稳定占位符” | 相同值每次都映射为同一个 token |
| Mesh approach | “Mesh 令牌化” | 一种尽量保留语义关系的 tokenization 模式 |
| Egress whitelist | “出站 allowlist” | 只允许访问明确批准的域名 |
| Audit log | “不可变历史” | 用于合规和审计的 append-only 记录 |

## 延伸阅读

- [Doppler — 高级 LLM 安全](https://www.doppler.com/blog/advanced-llm-security)
- [Portkey — 使用 secret references 管理 LLM API 密钥](https://portkey.ai/blog/secret-references-ai-api-key-management/)
- [Datadog — LLM Guardrails 最佳实践](https://www.datadoghq.com/blog/llm-guardrails-best-practices/)
- [JumpServer — 2026 Secrets Management 最佳实践](https://www.jumpserver.com/blog/secret-management-best-practices-2026)
- [Microsoft Presidio](https://github.com/microsoft/presidio)：PII 检测与匿名化工具。
- [HashiCorp Vault docs](https://developer.hashicorp.com/vault/docs)
