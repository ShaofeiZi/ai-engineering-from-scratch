"""阶段 13 第 03 课 - 并行与流式工具调用。

两个演示，仅使用标准库：
  1. Three-city 天气查询，顺序与并行对比（线程池）。
     测量 wall-clock 并展示最大值与求和模式。
  2. out-of-order 参数块的流式累加器。
     回放一个由三次交错并行调用组成的模拟 OpenAI-shaped 流，
     并在执行前重新组装每个 per-id。

运行：python code/main.py
"""

from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field


# ------------------------------------------------------------------
# 演示 1：顺序与并行天气查询
# ------------------------------------------------------------------

SIMULATED_LATENCY_MS = {"Bengaluru": 400, "Tokyo": 600, "Zurich": 800}


def executor_weather(city: str) -> dict:
    latency = SIMULATED_LATENCY_MS.get(city, 500)
    time.sleep(latency / 1000.0)
    return {"city": city, "temp_c": hash(city) % 35}


def run_sequential(cities: list[str]) -> tuple[float, list[dict]]:
    start = time.perf_counter()
    results = [executor_weather(c) for c in cities]
    dt_ms = (time.perf_counter() - start) * 1000
    return dt_ms, results


def run_parallel(cities: list[str]) -> tuple[float, list[dict]]:
    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=len(cities)) as pool:
        results = list(pool.map(executor_weather, cities))
    dt_ms = (time.perf_counter() - start) * 1000
    return dt_ms, results


# ------------------------------------------------------------------
# 演示 2：流式累加器
# ------------------------------------------------------------------

@dataclass
class CallBuffer:
    id: str
    name: str = ""
    args_buf: str = ""
    done: bool = False

    def try_parse(self) -> dict | None:
        if not self.done:
            return None
        return json.loads(self.args_buf)


@dataclass
class StreamAccumulator:
    buffers: dict[str, CallBuffer] = field(default_factory=dict)

    def on_event(self, event: dict) -> list[CallBuffer]:
        kind = event["type"]
        idx = event.get("id")
        completed: list[CallBuffer] = []
        if kind == "call_start":
            self.buffers[idx] = CallBuffer(id=idx, name=event["name"])
        elif kind == "args_delta":
            buf = self.buffers[idx]
            buf.args_buf += event["chunk"]
        elif kind == "call_stop":
            buf = self.buffers[idx]
            buf.done = True
            completed.append(buf)
        return completed


def fake_openai_stream():
    """三次交错的并行调用。真实的流就是这样的。"""
    yield {"type": "call_start", "id": "call_A", "name": "get_weather"}
    yield {"type": "call_start", "id": "call_B", "name": "get_weather"}
    yield {"type": "call_start", "id": "call_C", "name": "get_weather"}
    yield {"type": "args_delta", "id": "call_A", "chunk": '{"city"'}
    yield {"type": "args_delta", "id": "call_B", "chunk": '{"city'}
    yield {"type": "args_delta", "id": "call_A", "chunk": ':"Beng'}
    yield {"type": "args_delta", "id": "call_C", "chunk": '{"city":"Zu'}
    yield {"type": "args_delta", "id": "call_A", "chunk": 'aluru"}'}
    yield {"type": "call_stop", "id": "call_A"}
    yield {"type": "args_delta", "id": "call_B", "chunk": '":"Tokyo"}'}
    yield {"type": "call_stop", "id": "call_B"}
    yield {"type": "args_delta", "id": "call_C", "chunk": 'rich"}'}
    yield {"type": "call_stop", "id": "call_C"}


def replay_and_execute() -> dict[str, dict]:
    acc = StreamAccumulator()
    results: dict[str, dict] = {}
    in_flight: dict[str, "Future"] = {}  # type: ignore
    with ThreadPoolExecutor(max_workers=4) as pool:
        for event in fake_openai_stream():
            completed = acc.on_event(event)
            for buf in completed:
                args = buf.try_parse()
                print(f"  调用 {buf.id} 参数完成 -> {args}")
                in_flight[buf.id] = pool.submit(executor_weather, args["city"])
        for cid, fut in in_flight.items():
            results[cid] = fut.result()
    return results


# ------------------------------------------------------------------
# 主函数
# ------------------------------------------------------------------

def main() -> None:
    print("=" * 72)
    print("第 13 阶段第 03 课 - 并行与流式工具调用")
    print("=" * 72)

    cities = ["Bengaluru", "Tokyo", "Zurich"]
    sum_lat = sum(SIMULATED_LATENCY_MS.values())
    max_lat = max(SIMULATED_LATENCY_MS.values())

    print("\n--- 演示 1：三城市天气查询（模拟） ---")
    print(f"各城市模拟延迟 : {SIMULATED_LATENCY_MS}")
    print(f"理论顺序耗时           : {sum_lat} 毫秒  (求和)")
    print(f"理论并行耗时           : {max_lat} 毫秒  (最大值)")

    seq_ms, seq_res = run_sequential(cities)
    par_ms, par_res = run_parallel(cities)
    print(f"\n实际顺序耗时 : {seq_ms:.0f} 毫秒")
    print(f"实际并行耗时   : {par_ms:.0f} 毫秒")
    speedup = seq_ms / par_ms if par_ms else 0
    print(f"加速比           : {speedup:.2f}x")

    print("\n--- 演示 2：流式累加器 ---")
    print("回放由三次并行调用组成的模拟交错流 ……")
    results = replay_and_execute()
    print("\n最终结果（按 tool_call_id 索引）：")
    for cid, r in results.items():
        print(f"  {cid} -> {r}")


if __name__ == "__main__":
    main()
