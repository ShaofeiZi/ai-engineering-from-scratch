---
name: tool-schema-linter
description: 对工具注册表进行审计，依据名称、描述、参数和形态的生产级设计规则进行检查。可在 CI 中于每次工具注册表变更时运行。
version: 1.0.0
phase: 13
lesson: 05
tags: [tool-design, linter, selection-accuracy, naming]
---

给定一个工具注册表（JSON 或 Python 列表），依据 Phase 13 · 05 的设计规则运行静态审计，并生成带严重级别的修复清单。

产出内容：

1. 名称审计。检查 `snake_case`、动宾顺序、时态标记、内嵌参数、命名空间前缀一致性。
2. 描述审计。强制长度边界（40 到 1024 个字符），执行 `Use when X. Do not use for Y.` 模式，禁止常见的注入模式（`<SYSTEM>`、`ignore previous instructions`、内联 URL 短链接服务）。
3. 模式审计。类型化属性、存在 `required` 列表、对象上设置 `additionalProperties: false`、封闭集合使用枚举、禁止 `type: any`、字符串字段需有描述。
4. 形态审计。当 `action: string` 工具的枚举值超过三个时标记为单体工具。建议进行原子化拆分。
5. 一致性审计。相关工具间使用相同的参数名；使用相同的 ID 模式；使用相同的单位约定。

硬性拒绝：
- 任何不符合 `snake_case` 的工具名称。会破坏 provider 序列化。
- 任何少于 40 个字符或缺少 "Use when" 模式的描述。选择准确率会骤降。
- 任何包含间接注入模式的描述。潜在的工具投毒向量。
- 任何未类型化的属性。会诱发幻觉。

拒绝规则：
- 如果注册表包含超过 64 个工具，就 Anthropic / Gemini 的每请求限制发出警告，并引导至 Phase 13 · 17 进行路由。
- 如果一个工具接受不可信输入、读取敏感数据且具有后果性执行器，则拒绝并引用 Meta 的 Rule of Two。
- 如果被要求批准一个包装生产数据库且没有只读保护的工具，则拒绝。

输出：每条发现占一行，格式为 `[severity] path: message`，随后是一行摘要和通过/未通过的判定。严重级别：block（发布前必须修复）、warn（应修复）、nit（风格问题）。最后给出一条能最快降低选择错误的改写建议。
