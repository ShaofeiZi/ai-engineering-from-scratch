"""Swarm 架构演示：worker 从共享队列中拉取任务。

对比可变时长工作负载下的三种调度策略：
  - 顺序执行（1 个 worker 处理所有任务）
  - 固定分配（每个任务预先分配给特定 worker）
  - swarm（4 个 worker 从共享队列中拉取任务）

Swarm 会自动平衡负载；固定分配会让较快的 worker 闲置。
"""
from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass


@dataclass
class Task:
    task_id: int
    duration: float
    pre_assigned: int  # 用于固定分配基线


def fake_work(task: Task) -> str:
    time.sleep(task.duration)
    return f"task-{task.task_id}-done"


def run_sequential(tasks: list[Task]) -> tuple[float, dict[int, int]]:
    t0 = time.time()
    counts: dict[int, int] = {0: 0}
    for t in tasks:
        fake_work(t)
        counts[0] += 1
    return time.time() - t0, counts


def run_fixed_assignment(tasks: list[Task], n_workers: int) -> tuple[float, dict[int, int]]:
    """每个任务都预先分配给一个 worker ID。worker 串行处理自己的任务。"""
    per_worker: dict[int, list[Task]] = {i: [] for i in range(n_workers)}
    for t in tasks:
        per_worker[t.pre_assigned].append(t)
    counts: dict[int, int] = {i: 0 for i in range(n_workers)}

    def worker(wid: int) -> None:
        for t in per_worker[wid]:
            fake_work(t)
            counts[wid] += 1

    t0 = time.time()
    threads = [threading.Thread(target=worker, args=(i,)) for i in range(n_workers)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    return time.time() - t0, counts


def run_swarm(tasks: list[Task], n_workers: int) -> tuple[float, dict[int, int]]:
    """worker 从共享队列中拉取任务。"""
    q: queue.Queue = queue.Queue()
    for t in tasks:
        q.put(t)
    counts: dict[int, int] = {i: 0 for i in range(n_workers)}
    lock = threading.Lock()

    def worker(wid: int) -> None:
        while True:
            try:
                task = q.get_nowait()
            except queue.Empty:
                return
            fake_work(task)
            with lock:
                counts[wid] += 1
            q.task_done()

    t0 = time.time()
    threads = [threading.Thread(target=worker, args=(i,)) for i in range(n_workers)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    return time.time() - t0, counts


def make_tasks(n_workers: int = 4) -> list[Task]:
    """8 个任务：一半较快（0.1 秒），一半较慢（0.4 秒）。预分配采用最差情况：
    worker 0 获得所有慢任务，其他 worker 获得快任务。"""
    tasks: list[Task] = []
    for i in range(8):
        is_slow = i < 4
        tasks.append(
            Task(
                task_id=i,
                duration=0.4 if is_slow else 0.1,
                pre_assigned=0 if is_slow else (i - 3) % n_workers,
            )
        )
    return tasks


def main() -> None:
    print("Swarm 架构演示 — 可变时长工作负载")
    print("-" * 56)
    n_workers = 4

    tasks = make_tasks(n_workers)
    total_work = sum(t.duration for t in tasks)
    print(f"{len(tasks)} 个任务，4 个慢任务（0.4 秒）+ 4 个快任务（0.1 秒）")
    print(f"总工作时间：{total_work:.2f} 秒")
    print(f"使用 {n_workers} 个 worker 时的理想并行时间：{total_work / n_workers:.2f} 秒")

    seq_time, seq_counts = run_sequential(tasks)
    print(f"\n顺序执行（1 个 worker）：实际耗时={seq_time:.2f} 秒，计数={seq_counts}")

    fixed_time, fixed_counts = run_fixed_assignment(tasks, n_workers)
    print(f"固定分配（{n_workers} 个 worker）：实际耗时={fixed_time:.2f} 秒，计数={fixed_counts}")
    print("  worker 0 获得全部 4 个慢任务；其他 worker 完成快任务后处于闲置状态。")

    swarm_time, swarm_counts = run_swarm(tasks, n_workers)
    print(f"Swarm（{n_workers} 个 worker）：实际耗时={swarm_time:.2f} 秒，计数={swarm_counts}")
    print("  负载会自动均衡：先完成任务的 worker 会拉取下一个任务。")

    speedup_vs_seq = seq_time / swarm_time if swarm_time > 0 else float("inf")
    speedup_vs_fixed = fixed_time / swarm_time if swarm_time > 0 else float("inf")
    print(f"\nSwarm 相对顺序执行的加速比：{speedup_vs_seq:.2f}x")
    print(f"Swarm 相对固定分配的加速比：{speedup_vs_fixed:.2f}x")
    print("\n要点：当任务时长不一且难以预先分配时，swarm 更有优势。")
    print("权衡：没有集中式轨迹；调试需要逐任务 ID 和持久化日志。")


if __name__ == "__main__":
    main()
