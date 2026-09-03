import type {
  AudioChunk,
  Metrics,
  SessionOptions,
  SessionSummary,
  State,
  Tool,
} from "./types.ts";
import { turnCompletionScore } from "./vad.ts";

export const WEATHER: Tool = {
  name: "weather.tokyo_tomorrow",
  latencyMs: 420,
  result: "68/52，局部多云",
};

export function newMetrics(): Metrics {
  return {
    events: [],
    turnCompleteMs: 0,
    firstLlmTokenMs: 0,
    firstAudioOutMs: 0,
    bargeIns: 0,
  };
}

export function turnLatencyMs(m: Metrics): number {
  if (m.turnCompleteMs && m.firstAudioOutMs) return m.firstAudioOutMs - m.turnCompleteMs;
  return -1;
}

export function summarize(m: Metrics): SessionSummary {
  return {
    turnCompleteMs: m.turnCompleteMs,
    firstLlmTokenMs: m.firstLlmTokenMs,
    firstAudioOutMs: m.firstAudioOutMs,
    turnLatencyMs: turnLatencyMs(m),
    bargeIns: m.bargeIns,
  };
}

export function runSession(frames: AudioChunk[], opts: SessionOptions): Metrics {
  const m = newMetrics();
  let state: State = "IDLE";
  let silenceRunMs = 0;
  let finalPartial = "";
  let llmStartedAt = -1;
  let ttsStartedAt = -1;
  let toolStartedAt = -1;
  let fillerEmitted = false;
  let toolPhase: "none" | "running" | "done" = "none";

  const log = (line: string): void => {
    m.events.push(line);
    opts.onEvent?.(line);
  };

  for (const f of frames) {
    if (
      opts.bargeInAtMs !== null &&
      f.tMs >= opts.bargeInAtMs &&
      (state === "SPEAKING" || state === "THINKING") &&
      f.isSpeech
    ) {
      m.bargeIns += 1;
      log(`${f.tMs}ms 插话：取消 TTS，重新启用 ASR`);
      state = "LISTENING";
      silenceRunMs = 0;
      finalPartial = "";
      toolPhase = "none";
      toolStartedAt = -1;
      fillerEmitted = false;
      ttsStartedAt = -1;
      llmStartedAt = -1;
      continue;
    }

    if (state === "IDLE") {
      if (f.isSpeech) {
        state = "LISTENING";
        log(`${f.tMs}ms LISTENING`);
      }
      continue;
    }

    if (state === "LISTENING") {
      if (f.isSpeech) {
        silenceRunMs = 0;
        finalPartial = f.partial || finalPartial;
      } else {
        silenceRunMs += 20;
        if (silenceRunMs >= 500) {
          const score = turnCompletionScore(finalPartial);
          if (score >= 0.6) {
            state = "WAITING";
            m.turnCompleteMs = f.tMs;
            log(
              `${f.tMs}ms 轮次完成（分数=${score.toFixed(2)}）部分结果='${finalPartial}'`,
            );
          } else {
            log(`${f.tMs}ms 检测到静音，但分数=${score.toFixed(2)}，继续等待`);
          }
        }
      }
    }

    if (state === "WAITING") {
      if (opts.useTool && toolPhase === "none") {
        toolStartedAt = f.tMs;
        toolPhase = "running";
        log(`${f.tMs}ms 已发起工具调用：${WEATHER.name}`);
        state = "THINKING";
      } else {
        llmStartedAt = f.tMs + 140;
        state = "THINKING";
        log(`${f.tMs}ms 已发起 LLM 调用`);
      }
      continue;
    }

    if (state === "THINKING") {
      if (toolPhase === "running") {
        if (!fillerEmitted && f.tMs - toolStartedAt >= 300) {
          fillerEmitted = true;
          log(`${f.tMs}ms 填充语：'稍等，让我查一下'`);
        }
        if (f.tMs - toolStartedAt >= WEATHER.latencyMs) {
          toolPhase = "done";
          log(`${f.tMs}ms 工具结果：${WEATHER.result}`);
          llmStartedAt = f.tMs + 140;
        }
      } else if (llmStartedAt > 0 && f.tMs >= llmStartedAt) {
        if (m.firstLlmTokenMs === 0) {
          m.firstLlmTokenMs = f.tMs;
          log(`${f.tMs}ms LLM 首个 token`);
        }
        ttsStartedAt = f.tMs + 180;
        state = "SPEAKING";
      }
      continue;
    }

    if (state === "SPEAKING") {
      if (ttsStartedAt > 0 && f.tMs >= ttsStartedAt && m.firstAudioOutMs === 0) {
        m.firstAudioOutMs = f.tMs;
        log(`${f.tMs}ms TTS 首次音频输出`);
      }
    }
  }
  return m;
}

export function renderToConsole(label: string, m: Metrics): void {
  console.log(`=== ${label} ===`);
  for (const line of m.events) console.log(" ", line);
  console.log(`  轮次完成        @ ${m.turnCompleteMs}ms`);
  console.log(`  LLM 首个 token  @ ${m.firstLlmTokenMs}ms`);
  console.log(`  首次音频输出     @ ${m.firstAudioOutMs}ms`);
  console.log(`  轮次延迟         = ${turnLatencyMs(m)}ms`);
  console.log(`  插话次数         = ${m.bargeIns}`);
  console.log("");
}
