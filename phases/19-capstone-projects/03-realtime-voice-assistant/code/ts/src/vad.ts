import type { AudioChunk } from "./types.ts";

export function turnCompletionScore(partial: string): number {
  // LiveKit 轮次检测模型的小型替代实现。
  if (!partial) return 0;
  const tail = partial.trimEnd();
  if (tail.endsWith("?") || tail.endsWith(".") || tail.endsWith("!")) return 0.95;
  const n = partial.split(/\s+/).filter(Boolean).length;
  if (n < 3) return 0.2;
  if (n < 6) return 0.55;
  return 0.75;
}

export function synthCall(script: string, startMs = 0, noise = 0): AudioChunk[] {
  // 生成每帧 20ms 的“音频”：先是前导静音，再逐词生成语音，
  // 最后追加较长的尾部静音，使状态机能够端到端运行。
  const words = script.trim().split(/\s+/).filter(Boolean);
  const frames: AudioChunk[] = [];
  let t = startMs;
  for (let i = 0; i < 6; i++) {
    frames.push({ tMs: t, isSpeech: Math.random() < noise, partial: "" });
    t += 20;
  }
  let partial = "";
  for (const w of words) {
    partial = (partial ? partial + " " : "") + w;
    for (let i = 0; i < 16; i++) {
      frames.push({ tMs: t, isSpeech: true, partial });
      t += 20;
    }
  }
  for (let i = 0; i < 110; i++) {
    frames.push({ tMs: t, isSpeech: false, partial });
    t += 20;
  }
  return frames;
}
