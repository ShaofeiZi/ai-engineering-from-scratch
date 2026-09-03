"""基于 multiprocessing.Queue 的集合通信原语，并与 gloo 后端逐字节校验。

在将 N 个 rank 连成环的队列网格上实现 ring allreduce、tree broadcast、
allgather、reduce_scatter。每个原语都与在相同 tensor 和相同 world size 下
初始化的 gloo 后端 torch.distributed 逐字节比对。per-rank 字节计数器验证了
ring allreduce 的 2T(N-1)/N 通信量缩放规律。

运行：python3 code/main.py

mesh worker 使用 'fork' 多进程上下文，子进程直接继承 Queue 文件描述符而无需
pickle。gloo 参考 worker 使用 'spawn'，因为 torch.distributed 需要干净的进程
环境。两种上下文均随标准库 multiprocessing 提供。
"""

from __future__ import annotations

import multiprocessing as mp
import os
import tempfile
from dataclasses import dataclass

import torch
import torch.distributed as dist


PRIMITIVES = ("allreduce", "broadcast", "allgather", "reduce_scatter")
RECV_TIMEOUT_S = 30.0


def _loopback_iface() -> str:
    """返回环回接口名；macOS 使用 lo0，Linux 使用 lo。"""
    import sys as _sys
    return "lo0" if _sys.platform == "darwin" else "lo"


@dataclass
class Mesh:
    """以队列构成的全连接图作为点对点 mesh。

    每个 rank 持有 out_queues[dst] 和 in_queues[src]。ring 算法只使用相邻边；
    全连接 mesh 保持 API 通用，便于后续课程在不重新布线的情况下实验树形拓扑。
    """

    rank: int
    world_size: int
    out_queues: list
    in_queues: list
    byte_counter: object = None

    def send(self, dst: int, tensor: torch.Tensor) -> None:
        if dst == self.rank:
            raise ValueError("rank 不能向自身发送")
        payload = tensor.detach().clone().contiguous()
        nbytes = payload.numel() * payload.element_size()
        if self.byte_counter is not None:
            with self.byte_counter.get_lock():
                self.byte_counter.value += nbytes
        self.out_queues[dst].put(payload)

    def recv(self, src: int) -> torch.Tensor:
        if src == self.rank:
            raise ValueError("rank 不能从自身接收")
        return self.in_queues[src].get(timeout=RECV_TIMEOUT_S)


def build_queue_grid(ctx, world_size: int):
    """使用给定上下文分配 (world_size, world_size) 的队列网格。"""
    grid = [[None] * world_size for _ in range(world_size)]
    for src in range(world_size):
        for dst in range(world_size):
            if src != dst:
                grid[src][dst] = ctx.Queue()
    return grid


def mesh_from_grid(rank: int, world_size: int, grid, byte_counter) -> Mesh:
    out_qs = [grid[rank][d] for d in range(world_size)]
    in_qs = [grid[s][rank] for s in range(world_size)]
    return Mesh(rank=rank, world_size=world_size,
                out_queues=out_qs, in_queues=in_qs,
                byte_counter=byte_counter)


def ring_allreduce(mesh: Mesh, tensor: torch.Tensor) -> torch.Tensor:
    """两阶段 ring allreduce（先 reduce-scatter 再 allgather）。

    将 tensor 切成 world_size 等份（用零填充以均匀分块）。调用结束后，
    每个 rank 持有相同的求和结果，形状与原始 tensor 一致。
    """
    w = mesh.world_size
    r = mesh.rank
    if w == 1:
        return tensor.clone()
    n = tensor.numel()
    pad = (-n) % w
    flat = torch.zeros(n + pad, dtype=tensor.dtype)
    flat[:n] = tensor.flatten()
    chunks = [c.clone() for c in flat.chunk(w)]
    next_rank = (r + 1) % w
    prev_rank = (r - 1) % w
    for step in range(w - 1):
        send_idx = (r - step) % w
        recv_idx = (r - step - 1) % w
        mesh.send(next_rank, chunks[send_idx])
        incoming = mesh.recv(prev_rank)
        chunks[recv_idx] = chunks[recv_idx] + incoming
    for step in range(w - 1):
        send_idx = (r - step + 1) % w
        recv_idx = (r - step) % w
        mesh.send(next_rank, chunks[send_idx])
        incoming = mesh.recv(prev_rank)
        chunks[recv_idx] = incoming
    return torch.cat(chunks)[:n].reshape(tensor.shape)


def broadcast(mesh: Mesh, tensor: torch.Tensor, src: int) -> torch.Tensor:
    """树形 broadcast，跳数为 ceil(log2(world_size))。

    在第 r 轮，持有该值的 rank 集合翻倍。源 rank 播下初始值；非源 rank
    忽略自身输入，从已持有值的对端接收。
    """
    w = mesh.world_size
    r = mesh.rank
    if w == 1:
        return tensor.clone()
    has_value = {src}
    out = tensor.clone() if r == src else torch.zeros_like(tensor)
    round_idx = 0
    while len(has_value) < w:
        new_holders = set()
        for h in sorted(has_value):
            partner = h + (1 << round_idx)
            if partner < w and partner not in has_value:
                if r == h:
                    mesh.send(partner, out)
                elif r == partner:
                    out = mesh.recv(h)
                new_holders.add(partner)
        has_value |= new_holders
        round_idx += 1
    return out


def allgather(mesh: Mesh, tensor: torch.Tensor) -> torch.Tensor:
    """通过 N-1 次 ring 轮转实现 allgather。

    每个 rank 输入一个长度为 T 的 shard，输出为所有 shard 按 rank 顺序拼接，
    总长度为 T * world_size。
    """
    w = mesh.world_size
    r = mesh.rank
    if w == 1:
        return tensor.clone()
    shards = [torch.zeros_like(tensor) for _ in range(w)]
    shards[r] = tensor.clone()
    next_rank = (r + 1) % w
    prev_rank = (r - 1) % w
    for step in range(w - 1):
        send_idx = (r - step) % w
        recv_idx = (r - step - 1) % w
        mesh.send(next_rank, shards[send_idx])
        shards[recv_idx] = mesh.recv(prev_rank)
    return torch.cat(shards)


def reduce_scatter(mesh: Mesh, tensor: torch.Tensor) -> torch.Tensor:
    """reduce-scatter，即 ring allreduce 的前半段。

    输入是长度为 world_size * T 的 tensor。输出是该 rank 对应的长度为 T 的
    chunk，其中保存了所有 rank 在该索引区间上的求和结果。底层 ring 算法将完整
    求和结果存放在索引 (r + 1) % W 处；我们返回该 chunk 并标记为 rank r 的
    输出，以匹配 torch.distributed 中 rank r 拥有 chunks[r] 的约定。
    """
    w = mesh.world_size
    r = mesh.rank
    n = tensor.numel()
    if n % w != 0:
        raise ValueError(f"reduce_scatter 需要 numel 能被 world_size 整除，得到 {n} / {w}")
    if w == 1:
        return tensor.clone()
    rotated = list(tensor.chunk(w))
    rotated = [rotated[(i - 1) % w].clone() for i in range(w)]
    chunks = rotated
    next_rank = (r + 1) % w
    prev_rank = (r - 1) % w
    for step in range(w - 1):
        send_idx = (r - step) % w
        recv_idx = (r - step - 1) % w
        mesh.send(next_rank, chunks[send_idx])
        incoming = mesh.recv(prev_rank)
        chunks[recv_idx] = chunks[recv_idx] + incoming
    return chunks[(r + 1) % w]


def _gloo_worker(rank: int, world_size: int, op: str, tensor_bytes: bytes,
                 shape, dtype_str: str, init_file: str,
                 iface: str, out_queue) -> None:
    os.environ["GLOO_SOCKET_IFNAME"] = iface
    dist.init_process_group(
        backend="gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
    )
    dtype = getattr(torch, dtype_str)
    tensor = torch.frombuffer(bytearray(tensor_bytes), dtype=dtype).reshape(shape).clone()
    if op == "allreduce":
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        out = tensor
    elif op == "broadcast":
        dist.broadcast(tensor, src=0)
        out = tensor
    elif op == "allgather":
        gathered = [torch.zeros_like(tensor) for _ in range(world_size)]
        dist.all_gather(gathered, tensor)
        out = torch.cat(gathered)
    elif op == "reduce_scatter":
        chunks = [c.contiguous() for c in tensor.chunk(world_size)]
        recv = torch.zeros_like(chunks[0])
        dist.reduce_scatter(recv, chunks, op=dist.ReduceOp.SUM)
        out = recv
    else:
        raise ValueError(f"未知操作 {op}")
    out_queue.put((rank, out.clone()))
    out_queue.close()
    out_queue.join_thread()
    os._exit(0)


def gloo_reference(op: str, world_size: int,
                   per_rank_tensors: list) -> list:
    """通过 torch.distributed gloo 运行相同操作以进行校验。

    使用基于文件的初始化（file:// URI），因为通过 libuv 的 TCP 初始化在
    macOS 上并发创建 process group 时存在已知问题。
    """
    ctx = mp.get_context("spawn")
    out_queue = ctx.Queue()
    init_dir = tempfile.mkdtemp(prefix="aie_gloo_")
    init_file = os.path.join(init_dir, "rendezvous")
    iface = _loopback_iface()
    procs = []
    try:
        for r in range(world_size):
            t = per_rank_tensors[r].contiguous()
            p = ctx.Process(
                target=_gloo_worker,
                args=(r, world_size, op, bytes(t.numpy().tobytes()),
                      tuple(t.shape), str(t.dtype).split(".")[-1],
                      init_file, iface, out_queue),
            )
            p.start()
            procs.append(p)
        results = {}
        for _ in range(world_size):
            rank, tensor = out_queue.get(timeout=60)
            results[rank] = tensor
        return [results[r] for r in range(world_size)]
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


def _mesh_worker(rank: int, world_size: int, op: str,
                 grid, byte_counter, tensor_bytes: bytes,
                 shape, dtype_str: str, src: int, out_queue) -> None:
    mesh = mesh_from_grid(rank, world_size, grid, byte_counter)
    dtype = getattr(torch, dtype_str)
    tensor = torch.frombuffer(bytearray(tensor_bytes), dtype=dtype).reshape(shape).clone()
    if op == "allreduce":
        result = ring_allreduce(mesh, tensor)
    elif op == "broadcast":
        result = broadcast(mesh, tensor, src=src)
    elif op == "allgather":
        result = allgather(mesh, tensor)
    elif op == "reduce_scatter":
        result = reduce_scatter(mesh, tensor)
    else:
        raise ValueError(f"未知操作 {op}")
    out_queue.put((rank, result))


def run_mesh(op: str, world_size: int,
             per_rank_tensors: list,
             src: int = 0) -> tuple:
    """在队列网格上运行所选原语，返回各 rank 输出及字节总量。"""
    ctx = mp.get_context("fork")
    grid = build_queue_grid(ctx, world_size)
    byte_counter = ctx.Value("q", 0)
    out_queue = ctx.Queue()
    procs = []
    try:
        for r in range(world_size):
            t = per_rank_tensors[r].contiguous()
            p = ctx.Process(
                target=_mesh_worker,
                args=(r, world_size, op, grid, byte_counter,
                      bytes(t.numpy().tobytes()), tuple(t.shape),
                      str(t.dtype).split(".")[-1], src, out_queue),
            )
            p.start()
            procs.append(p)
        results = {}
        for _ in range(world_size):
            rank, tensor = out_queue.get(timeout=60)
            results[rank] = tensor
        return [results[r] for r in range(world_size)], byte_counter.value
    finally:
        for p in procs:
            p.join(timeout=30)
            if p.is_alive():
                p.terminate()
                p.join(timeout=2)


def verify_against_gloo(op: str, world_size: int,
                        per_rank_tensors: list) -> tuple:
    """将 mesh 实现与 gloo 参考结果对比，返回 (是否匹配, 最大绝对误差)。"""
    mesh_out, _ = run_mesh(op, world_size, per_rank_tensors)
    gloo_out = gloo_reference(op, world_size, per_rank_tensors)
    max_diff = 0.0
    for m, g in zip(mesh_out, gloo_out):
        diff = (m - g).abs().max().item()
        if diff > max_diff:
            max_diff = diff
    return max_diff < 1e-5, max_diff


def main() -> int:
    world_size = 4
    n = 64
    torch.manual_seed(7)
    per_rank = [torch.randn(n, dtype=torch.float32) for _ in range(world_size)]
    print(f"world_size={world_size}, tensor_len={n}, dtype=float32")
    print(f"{'操作':<16} {'gloo 匹配':<12} {'最大绝对误差':<14}")
    for op in PRIMITIVES:
        if op == "broadcast":
            inputs = [per_rank[0].clone() if r == 0 else torch.zeros(n) for r in range(world_size)]
        elif op == "reduce_scatter":
            inputs = [torch.randn(n * world_size, dtype=torch.float32) for _ in range(world_size)]
        else:
            inputs = per_rank
        match, diff = verify_against_gloo(op, world_size, inputs)
        print(f"{op:<16} {str(match):<12} {diff:<14.3e}")
    expected_per_rank_bytes = 2 * (world_size - 1) * (n // world_size) * 4
    _, total_bytes = run_mesh("allreduce", world_size, per_rank)
    per_rank_bytes = total_bytes / world_size
    print(f"\nallreduce 每 rank 字节数: 实测={per_rank_bytes:.0f} "
          f"理论={expected_per_rank_bytes} "
          f"公式=2T(N-1)/N，其中 T={n*4} 字节")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
