// Phase 13 第 01 课——TypeScript 工具接口回归测试。
//
// 课程：../../docs/en.md
// 通过公开的决策接缝验证离线路由器。
// 运行：npx tsx --test code/tests/test_main.ts
// Node 测试运行器：https://nodejs.org/api/test.html

import assert from "node:assert/strict";
import test from "node:test";

import { fakeDecide } from "../main";

function singleToolCall(message: string) {
  const decision = fakeDecide(message, [{ role: "user", content: message }]);
  assert.ok("toolCalls" in decision);
  assert.equal(decision.toolCalls.length, 1);
  return decision.toolCalls[0];
}

test("routes a Chinese city weather request", () => {
  const call = singleToolCall("北京 的天气");

  assert.equal(call.name, "get_weather");
  assert.deepEqual(call.arguments, { city: "北京", units: "celsius" });
});

test("keeps routing the original English weather request", () => {
  const call = singleToolCall("tell me the weather in Bengaluru");

  assert.equal(call.name, "get_weather");
  assert.deepEqual(call.arguments, { city: "Bengaluru", units: "celsius" });
});

test("normalizes a lowercase English city", () => {
  const call = singleToolCall("weather in tokyo");

  assert.deepEqual(call.arguments, { city: "Tokyo", units: "celsius" });
});

test("keeps routing Chinese addition requests", () => {
  const call = singleToolCall("请计算 7 加 35");

  assert.equal(call.name, "add");
  assert.deepEqual(call.arguments, { a: 7, b: 35 });
});

test("leaves unsupported requests unrouted", () => {
  const decision = fakeDecide("写一首关于茶的俳句", [
    { role: "user", content: "写一首关于茶的俳句" },
  ]);

  assert.deepEqual(decision, { content: "我无法将该请求路由到任何已注册的工具。" });
});
