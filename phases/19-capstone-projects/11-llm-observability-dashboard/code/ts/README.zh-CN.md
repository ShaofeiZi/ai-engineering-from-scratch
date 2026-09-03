# LLM 可观测性仪表板（TypeScript 骨架）

用于 LLM 可观测性仪表板毕业项目的多文件 TypeScript 骨架。
Hono 服务器接收 OpenTelemetry GenAI span，将其保存在 10k 环形缓冲区中，
并渲染 p50/p95/p99 延迟和每模型成本。

## 布局

- `src/index.ts` — 入口点，注入合成 span 并可选地提供 HTTP 服务。
- `src/server.ts` — `/trace`、`/`、`/dashboard`、`/dashboard.json`、`/healthz` 的 Hono 路由。
- `src/spans.ts` — `RingBuffer` 和 `ObservabilityStore`（默认 10k span）。
- `src/rollup.ts` — `percentile` 和 `rollUpByModel`。
- `src/pricing.ts` — 2026 年各模型价格和成本辅助函数。
- `src/types.ts` — 共享类型。
- `tests/*.test.ts` — `node --test` 风格测试，通过 `tsx` 运行。

## 安装

```bash
npm install
```

## 运行

```bash
npm start         # 生成 1200 个合成 span 并打印汇总
npm run serve     # 同时在 PORT 上提供 HTTP 摄取端点和仪表板（默认 8011）
```

## 验证

```bash
npm run typecheck
npm test
```

## 规格参考

- 源课程：`phases/19-capstone-projects/11-llm-observability-dashboard/docs/en.md`
- [OpenTelemetry GenAI 语义约定](https://opentelemetry.io/docs/specs/semconv/gen-ai/)
