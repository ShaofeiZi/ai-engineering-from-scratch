# 毕业项目 19/02 — 代码库 RAG（TypeScript）

多文件 TypeScript 代码搜索 API，实现了 `../docs/en.md` 中描述的
混合检索流水线。离线、确定性、六块样本语料，
node:http 搭载在 hono fetch 处理器之后。

## 目录结构

```text
src/
  index.ts        entry point; boots node:http + self-probe + exits 0
  server.ts       hono routes (/healthz, /query) with zod-validated POST body
  retrieval.ts    runQuery + RRF merge over dense and BM25
  index_store.ts  FNV-1a hash embedder, cosine, field-weighted BM25
  corpus.ts       six-chunk sample (uploader / auth / client / catalog)
  types.ts        Chunk, RankedChunk, QueryResponse, anchor()
tests/
  index_store.test.ts
  retrieval.test.ts
  server.test.ts
```

## 运行

```bash
npm install
npm start                # 启动 API、探测三个查询，随后以状态码 0 退出
npm start -- --serve     # 保持服务器运行；按 Ctrl-C 停止
npm test                 # 通过 tsx 运行 node --test
npm run typecheck        # 运行 tsc --noEmit
```

非交互式 `npm start` 路径会断言 `/healthz` 返回 200，
且每个探测查询至少返回一条引用。路由：

- `GET /healthz` — 返回 `{ok, corpus}`。
- `GET /query?q=...` — 运行混合查询。
- `POST /query` — JSON `{q, topK?}`，经 zod 校验（`topK` 上限为 50）。
