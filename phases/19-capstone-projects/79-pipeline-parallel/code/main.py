"""流水线并行，含 GPipe 调度与 bubble 分析。

将一个顺序 MLP 拆分为 N 个 stage。调度模拟每个 stage 的 forward 与
backward 的挂钟时间，随后打印甘特图，并对照闭式解 (N-1)/(M+N-1) 计算
bubble 占比。

第二个演示在 torch.distributed gloo 上搭建一个 2-stage 真实流水线：
rank 0 拥有 stage 0，rank 1 拥有 stage 1，激活通过 send/recv 流转，
该调度训练一个小型 MLP 若干步以验证布线正确。

运行：python3 code/main.py
"""

from __future__ import annotations

import multiprocessing as mp
import os
import sys
import tempfile

import torch
import torch.distributed as dist
import torch.nn as nn


SEED = 23
NUM_STAGES = 4
NUM_MICROBATCHES = 8
FORWARD_UNITS = 1
BACKWARD_UNITS = 2


def _loopback_iface() -> str:
    return "lo0" if sys.platform == "darwin" else "lo"


def bubble_fraction(num_stages: int, num_microbatches: int) -> float:
    """GPipe 每个 stage 的闭式 bubble 占比。

    forward 每 stage 耗时 M + N - 1 个周期（M 个有效 + N - 1 个空闲预热）。
    backward 每 stage 耗时 M + N - 1 个周期（M 个有效 + N - 1 个空闲排空）。
    总周期数 = 2(M + N - 1)；每 stage 有效 = 2M。
    bubble 占比 = 2(N - 1) / 2(M + N - 1) = (N - 1) / (M + N - 1)。
    """
    n = num_stages
    m = num_microbatches
    return (n - 1) / (m + n - 1)


def gpipe_schedule(num_stages: int, num_microbatches: int) -> list:
    """以 (cycle, stage, microbatch, phase) 列表形式返回 GPipe 调度。

    phase 为 'F' 表示 forward，'B' 表示 backward，'.' 表示空闲。
    cycle 为整数时间槽；microbatch 为 microbatch 索引。
    """
    n = num_stages
    m = num_microbatches
    schedule = []
    # 前向传播：微批次 i 在周期 i 进入阶段 0，在周期 i+k 进入阶段 k。
    for mb in range(m):
        for stage in range(n):
            cycle = mb + stage
            schedule.append((cycle, stage, mb, "F"))
    # 反向传播：微批次 i 在阶段 n-1 的周期 i+n-1 完成前向传播，
    # 随后在周期 m+n-1+i 从阶段 n-1 开始反向传播，并逐步回到阶段 0。
    forward_end = m + n - 1
    for mb in range(m):
        for stage in reversed(range(n)):
            cycle = forward_end + (m - 1 - mb) + (n - 1 - stage)
            schedule.append((cycle, stage, mb, "B"))
    return schedule


def render_gantt(schedule: list, num_stages: int, num_microbatches: int) -> str:
    """将调度渲染为按阶段和周期排列的文本甘特图。"""
    n = num_stages
    m = num_microbatches
    max_cycle = max(c for c, _, _, _ in schedule)
    grid = [["." for _ in range(max_cycle + 1)] for _ in range(n)]
    for cycle, stage, mb, phase in schedule:
        grid[stage][cycle] = f"{phase}{mb}" if phase != "." else "."
    lines = []
    header = "stage \\ cycle  " + " ".join(f"{c:>2}" for c in range(max_cycle + 1))
    lines.append(header)
    for s, row in enumerate(grid):
        lines.append(f"stage {s}         " + " ".join(f"{cell:>2}" for cell in row))
    return "\n".join(lines)


def measure_bubble(num_stages: int, num_microbatches: int) -> float:
    """经验气泡率：统计已渲染调度中的空闲槽位。"""
    schedule = gpipe_schedule(num_stages, num_microbatches)
    max_cycle = max(c for c, _, _, _ in schedule)
    total_slots = num_stages * (max_cycle + 1)
    used = len(schedule)
    return (total_slots - used) / total_slots


class StageMLP(nn.Module):
    """顺序 MLP 的一个阶段。"""

    def __init__(self, in_dim: int, hid_dim: int, out_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hid_dim)
        self.fc2 = nn.Linear(hid_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.fc2(torch.relu(self.fc1(x))))


def _pipe_worker(rank: int, world_size: int, init_file: str, iface: str,
                 steps: int, batch: int, microbatches: int, out_queue) -> None:
    """双 rank 流水线：rank 0 持有阶段 0，rank 1 持有阶段 1。

    前向传播时，rank 0 在微批次上运行阶段 0，并将激活值发送给 rank 1。
    rank 1 运行阶段 1、计算 loss、执行反向传播，再把梯度发回 rank 0。
    rank 0 在阶段 0 完成反向传播。每个微批次重复此流程。
    """
    os.environ["GLOO_SOCKET_IFNAME"] = iface
    dist.init_process_group(
        backend="gloo", init_method=f"file://{init_file}",
        rank=rank, world_size=world_size,
    )
    torch.manual_seed(SEED + rank)
    in_dim, hid_dim, mid_dim, out_dim = 16, 32, 16, 4
    if rank == 0:
        stage = StageMLP(in_dim, hid_dim, mid_dim)
    else:
        stage = StageMLP(mid_dim, hid_dim, out_dim)
    optim = torch.optim.SGD(stage.parameters(), lr=0.05)
    loss_fn = nn.MSELoss()
    g = torch.Generator().manual_seed(SEED + 99)
    losses = []
    for step in range(steps):
        optim.zero_grad(set_to_none=True)
        for _ in range(microbatches):
            if rank == 0:
                x = torch.randn(batch, in_dim, generator=g)
                act = stage(x)
                dist.send(act.detach(), dst=1)
                grad = torch.zeros_like(act)
                dist.recv(grad, src=1)
                act.backward(grad)
            else:
                act = torch.zeros(batch, mid_dim, requires_grad=True)
                buf = torch.zeros(batch, mid_dim)
                dist.recv(buf, src=0)
                act = buf.detach().requires_grad_(True)
                pred = stage(act)
                y = torch.zeros(batch, out_dim)
                loss = loss_fn(pred, y)
                loss.backward()
                dist.send(act.grad.detach(), dst=0)
                losses.append(loss.item())
        optim.step()
    norm = sum(p.detach().pow(2).sum().item() for p in stage.parameters()) ** 0.5
    out_queue.put((rank, losses, norm))
    out_queue.close()
    out_queue.join_thread()
    os._exit(0)


def run_pipeline(steps: int = 5, batch: int = 8, microbatches: int = 4) -> dict:
    """启动双 rank 流水线；返回各 rank 的 loss（仅 rank 1 报告）与范数。"""
    ctx = mp.get_context("spawn")
    out_queue = ctx.Queue()
    init_dir = tempfile.mkdtemp(prefix="aie_pipe_")
    init_file = os.path.join(init_dir, "rendezvous")
    iface = _loopback_iface()
    world_size = 2
    procs = []
    try:
        for r in range(world_size):
            p = ctx.Process(
                target=_pipe_worker,
                args=(r, world_size, init_file, iface, steps, batch, microbatches, out_queue),
            )
            p.start()
            procs.append(p)
        results = {}
        for _ in range(world_size):
            rank, losses, norm = out_queue.get(timeout=120)
            results[rank] = (losses, norm)
        return results
    finally:
        for p in procs:
            p.join(timeout=5)
            if p.is_alive():
                p.terminate()
                p.join(timeout=2)
        try:
            os.remove(init_file)
        except FileNotFoundError:
            pass
        try:
            os.rmdir(init_dir)
        except OSError:
            pass


def main() -> int:
    print(f"GPipe 调度分析：stages={NUM_STAGES}, microbatches={NUM_MICROBATCHES}")
    schedule = gpipe_schedule(NUM_STAGES, NUM_MICROBATCHES)
    print(render_gantt(schedule, NUM_STAGES, NUM_MICROBATCHES))
    closed = bubble_fraction(NUM_STAGES, NUM_MICROBATCHES)
    measured = measure_bubble(NUM_STAGES, NUM_MICROBATCHES)
    print(f"\n闭式气泡率：{closed * 100:.2f}%")
    print(f"实测气泡率：{measured * 100:.2f}%")
    print("\n微批次数与气泡率对比（N=4）：")
    print(f"{'M':<6}{'气泡率 %':<10}")
    for m in (1, 2, 4, 8, 16, 32, 64):
        print(f"{m:<6}{bubble_fraction(4, m)*100:<10.2f}")
    print("\n正在通过 gloo 运行真实的双阶段流水线……")
    results = run_pipeline(steps=3, batch=8, microbatches=4)
    rank1_losses = results[1][0]
    print(f"rank 1 得到 {len(rank1_losses)} 个微批次损失；最终范数 rank 0 = {results[0][1]:.4f}，rank 1 = {results[1][1]:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
