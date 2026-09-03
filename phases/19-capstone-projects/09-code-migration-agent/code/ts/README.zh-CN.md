# 代码迁移智能体看板（TypeScript 骨架）

代码迁移智能体毕业项目的看板层多文件 TypeScript 骨架。智能体
（Python）运行在沙箱中；本服务器为操作者渲染进度。

## 目录结构

- `src/index.ts` — 入口，模拟 tick 并可选地提供 HTTP 服务。
- `src/server.ts` — Hono 路由：`/`、`/dashboard`、`/migrations`、`/migrations/:id`。
- `src/migrations.ts` — 逐文件状态机和种子数据。
- `src/cost.ts` — 轮次计数和美元预算强制执行。
- `src/types.ts` — 共享类型。
- `tests/*.test.ts` — `node --test` 风格测试，通过 `tsx` 运行。

## 安装

```bash
npm install
```

## 运行

```bash
npm start         # 离线模拟 40 个时钟周期并打印汇总
npm run serve     # 在 PORT 上提供 HTML 仪表板（默认 8009）
```

## 验证

```bash
npm run typecheck
npm test
```

## 规格参考

- 源课程：`phases/19-capstone-projects/09-code-migration-agent/docs/en.md`
- 配方参考：[OpenRewrite](https://docs.openrewrite.org)、libcst。
