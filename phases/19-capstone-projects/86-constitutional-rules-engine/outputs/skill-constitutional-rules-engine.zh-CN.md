---
name: skill-constitutional-rules-engine
description: 声明式 YAML 规则引擎，用于输出约束，支持严重度、解释、修复操作和结构化差异
version: 1.0.0
phase: 19
lesson: 86
tags: [safety, rules, constitutional]
---

# 宪法规则引擎

宪法（constitution）是一个 YAML 文件。每条规则包含 `name`、`severity`（low | medium | high）、`applies_when`（谓词）、`must`（谓词）、`explanation` 以及可选的 `fix`。

## 谓词

原子谓词：

- `contains_regex` / `not_contains_regex`
- `starts_with_regex` / `ends_with_regex`
- `max_words` / `min_words`

组合谓词：

- `all_of: [...predicates]`
- `any_of: [...predicates]`
- `not_: predicate`

## 修复操作

- `append_if_missing: <suffix>`
- `prepend_if_missing: <prefix>`
- `replace_regex: { pattern: <regex>, replacement: <text> }`

## 引擎输出

`Engine.evaluate(text) -> EngineReport` 对每条规则返回一个 `RuleResult`，其 `status` 取值为 `pass`、`violation`、`not_applicable`。`report.violations()` 过滤出违规项，`report.max_severity()` 返回当前最严重的严重度。

## 产物

`outputs/rules_report.json` 包含每个用例的草稿、修订稿和结构化差异。
