// 综合项目 19/03：实时语音 Web 客户端（多文件 TypeScript）。
//
// 资料来源：
//   本课程的 docs/en.md（WebRTC 客户端 + VAD + 插话客户端 UX）
//   RFC 6455 WebSocket protocol  https://datatracker.ietf.org/doc/html/rfc6455
//   ws (Node WebSocket library)  https://github.com/websockets/ws
//   Silero VAD v5 model card     https://github.com/snakers4/silero-vad
//
// 流水线拆分为多个模块：vad.ts（轮次完成分数 + 合成帧生成器）、
// orchestrator.ts（支持插话的 IDLE -> LISTENING -> WAITING -> THINKING ->
// SPEAKING 状态机）、protocol.ts（经 zod 验证的帧 envelope）、server.ts
//（Hono /healthz + ws 升级），以及本入口；本入口运行两个离线会话、启动实时
// ws 服务器、执行探测并以状态码 0 退出。

import WebSocket from "ws";
import { runSession, renderToConsole, summarize } from "./orchestrator.ts";
import { synthCall } from "./vad.ts";
import { decodeFrame } from "./protocol.ts";
import { buildServer } from "./server.ts";
import type { Frame } from "./protocol.ts";

async function probeWs(
  port: number,
  timeoutMs = 3000,
): Promise<{ events: number; gotSummary: boolean }> {
  return await new Promise<{ events: number; gotSummary: boolean }>((resolve, reject) => {
    const ws = new WebSocket(`ws://127.0.0.1:${port}`);
    let events = 0;
    let gotSummary = false;
    let settled = false;
    const finish = (val: { events: number; gotSummary: boolean }): void => {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      resolve(val);
    };
    const timer = setTimeout(() => {
      if (settled) return;
      ws.removeAllListeners();
      try {
        ws.close();
      } catch {
        // 已在关闭
      }
      finish({ events, gotSummary });
    }, timeoutMs);
    ws.on("message", (raw) => {
      try {
        const f: Frame = decodeFrame(raw.toString("utf8"));
        if (f.type === "event") events += 1;
        else if (f.type === "summary") gotSummary = true;
      } catch {
        // 探测时忽略格式错误的帧
      }
    });
    ws.on("close", () => finish({ events, gotSummary }));
    ws.on("error", (err) => {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      reject(err);
    });
  });
}

async function main(): Promise<void> {
  // 预检：驱动两个离线会话通过状态机。
  const clean = runSession(synthCall("what is the weather in tokyo tomorrow"), {
    useTool: true,
    bargeInAtMs: null,
  });
  renderToConsole("session 1: clean call with tool (weather)", clean);
  if (clean.turnCompleteMs <= 0 || clean.firstAudioOutMs <= 0) {
    throw new Error("正常会话未产生首次音频输出");
  }

  const bargeFrames = synthCall("tell me a long story about");
  if (bargeFrames.length === 0) {
    throw new Error("synthCall 未返回任何帧");
  }
  const anchorIdx = Math.max(0, bargeFrames.length - 20);
  const anchorFrame = bargeFrames[anchorIdx] ?? bargeFrames[bargeFrames.length - 1];
  for (let i = 0; i < 8; i++) {
    const idx = anchorIdx + i;
    if (idx >= 0 && idx < bargeFrames.length) {
      bargeFrames[idx] = {
        tMs: bargeFrames[idx].tMs,
        isSpeech: true,
        partial: bargeFrames[idx].partial,
      };
    }
  }
  const bargeIn = runSession(bargeFrames, {
    useTool: false,
    bargeInAtMs: anchorFrame.tMs - 60,
  });
  renderToConsole("session 2: user barges in mid-response", bargeIn);
  if (bargeIn.bargeIns === 0) {
    throw new Error("插话会话未记录任何插话事件");
  }

  // 实时流程：启动 WS 服务器，通过它驱动一个会话，然后关闭。
  const { server } = buildServer();
  await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", () => resolve()));
  const addr = server.address();
  if (!addr || typeof addr === "string") throw new Error("地址不可用");
  console.log(`语音客户端骨架 ws://127.0.0.1:${addr.port}`);
  if (process.argv.includes("--serve")) {
    process.on("SIGINT", () => server.close(() => process.exit(0)));
    return;
  }
  const probe = await probeWs(addr.port);
  console.log(`[ws 探测] 已接收帧数：${probe.events + (probe.gotSummary ? 1 : 0)}`);
  console.log(`[ws 探测] 摘要：${probe.gotSummary ? "已收到" : "缺失"}`);
  console.log(`[ws 探测] 示例摘要：${JSON.stringify(summarize(clean))}`);
  await new Promise<void>((resolve) => server.close(() => resolve()));
  if (!probe.gotSummary) throw new Error("ws 探测未收到摘要帧");
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
