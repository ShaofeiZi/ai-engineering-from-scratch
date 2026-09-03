---
name: parallel-call-safety-check
description: 审计工具注册表以判定其并行安全性。为每个工具标记 parallel_safe，注明排序依赖，并标记下游速率限制风险。
version: 1.0.0
phase: 13
lesson: 03
tags: [parallel-tool-calls, streaming, correlation, rate-limits]
---

给定一个工具注册表（包含名称、描述和执行器的工具列表），返回一份添加了 `parallel_safe: bool`、`ordering_deps: [tool_name]` 和 `rate_limit_group: name` 字段的标注副本。

需要产出：

1. 逐工具分类。为每个工具判定：在同一轮次中可安全并行运行（纯读操作、访问不同资源）；不安全（写操作、共享资源、外部速率限制）。
2. 依赖图。识别某个工具的输出应作为另一个工具输入的配对。无法在同一轮次内并行。用 `ordering_deps` 标记。
3. 速率限制分组。访问同一下游 API 的工具共享一个分组。宿主应按组而非按工具限制并发。
4. 安全建议。针对每个不安全的工具，说明是否应在该轮次禁用并行、排队，还是按资源分片。
5. 供应商特定标志。当集合中存在任何不安全的工具时，建议在 OpenAI 上设置 `parallel_tool_calls=false`，或在 Anthropic 上设置 `disable_parallel_tool_use=true`。

硬性拒绝：
- 审计后没有任何分类的注册表。默认拒绝；未知即不安全。
- 对共享资源进行写路径的工具被标记为 `parallel_safe: true`。会导致竞态条件。
- 任何访问受速率限制的外部 API 但未设置 `rate_limit_group` 的工具。

拒绝规则：
- 若被要求不经检查即标记所有工具为并行安全，则拒绝。
- 若注册表包含针对同一资源的关键性工具（如同一路径上的 `delete_file` 和 `write_file`），拒绝并行并引导至 Phase 14 · 09 了解沙箱级别的序列化。
- 若用户声称其工具绝不会发生竞态，则拒绝并要求提供证据（测试、日志或形式化论证）。竞态在生产环境中往往悄然发生。

输出：一份修订后的注册表，以 JSON blob 形式呈现，每个工具包含三个新增字段，随后是一段简短总结，指出风险最高的并行化选择及建议的缓解措施。最后给出针对当前轮次的 `tool_choice` 覆盖建议。
