# 毕业项目 04 — 多模态文档问答（TypeScript）

查看器骨架，返回页面图像 URL 及引用的边界框 JSON 列表。HTML 响应内联了一个小型 canvas 覆盖脚本，在页面图像上绘制引用区域。与 `../main.py` 中的 Python 流水线配套使用。

## 目录结构

```text
ts/
  package.json
  tsconfig.json
  src/
    index.ts        # 入口，包含演示和 HTTP 服务器
    server.ts       # Hono 应用，提供 /health、/、/document/:id
    fixtures.ts     # 10-K 表格和 Nature 图表夹具
    render.ts       # HTML 索引和逐文档叠加层渲染器
    types.ts        # 类型定义：DocumentFixture、EvidenceRegion、BoundingBox
  tests/
    fixtures.test.ts
    render.test.ts
    server.test.ts
```

## 运行

```bash
npm install
npm run typecheck
npm test
npm start          # 执行一次自检，随后以状态码 0 退出
npm run serve      # 在 127.0.0.1:<port> 上启动交互式 HTTP 服务器
```

交互式服务器在 `PORT` 未设置时自动选择可用端口，并在标准输出打印所选 URL。访问 `/` 查看首页，`/document/10k-acme-2025` 查看演示覆盖，或设置 `accept: application/json` 获取结构化响应。

## 测试

通过 tsx 运行 `node --test`。测试覆盖夹具查找（正向 + 负向）、五个恶意字符的 HTML 转义、文档 HTML 载荷结构，以及 hono 路由（200、404、内容协商）。
