"""AdamW 配合余弦学习率调度与线性预热。

实现：
- CosineWithWarmup，一个无状态的调度器，其 lr(step) 严格遵循预热、峰值
  和衰减边界。
- TrainState，将 AdamW 优化器与调度器连接，逐个训练步骤运行，并记录
  学习率与梯度的 L2 范数。
- plot_schedule_ascii 和 write_schedule_csv，确定性辅助函数，分别生成
  文本图表和 CSV，供流水线其余部分读取。

底部的演示构建一个微小的 torch.nn.Linear 模型，在固定批次上训练 20
步，打印逐步日志，并渲染调度曲线。
运行：python3 code/main.py
"""

from __future__ import annotations

import csv
import dataclasses
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable

try:
    import torch
    from torch import nn
except ImportError as exc:
    raise SystemExit(
        "本课程需要 torch。请运行以下命令安装：pip install torch"
    ) from exc


PLOT_HEIGHT = 12
PLOT_WIDTH = 60


@dataclass
class CosineWithWarmup:
    """无状态的余弦学习率调度器，包含预热阶段。

    步数索引约定：第 0 步是第一次训练更新。此时学习率恰好为零（预热斜坡
    从这里开始）；在 warmup_steps 步恰好为 lr_max；在 total_steps 步恰好
    为 lr_min；超过 total_steps 后保持为 lr_min。
    """

    warmup_steps: int
    total_steps: int
    lr_max: float
    lr_min: float = 0.0

    def __post_init__(self) -> None:
        if self.warmup_steps < 0:
            raise ValueError("warmup_steps must be non-negative")
        if self.total_steps <= 0:
            raise ValueError("total_steps must be positive")
        if self.warmup_steps >= self.total_steps:
            raise ValueError("warmup_steps must be less than total_steps")
        if self.lr_max <= 0:
            raise ValueError("lr_max must be positive")
        if self.lr_min < 0:
            raise ValueError("lr_min must be non-negative")
        if self.lr_min > self.lr_max:
            raise ValueError("lr_min must not exceed lr_max")

    def lr(self, step: int) -> float:
        if step < 0:
            raise ValueError(f"step must be non-negative, got {step}")
        if self.warmup_steps > 0 and step <= self.warmup_steps:
            return self.lr_max * (step / self.warmup_steps)
        if step >= self.total_steps:
            return self.lr_min
        decay_span = max(1, self.total_steps - self.warmup_steps)
        progress = (step - self.warmup_steps) / decay_span
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.lr_min + (self.lr_max - self.lr_min) * cosine

    def points(self, num_steps: int | None = None) -> list[tuple[int, float]]:
        upper = self.total_steps if num_steps is None else num_steps
        if upper <= 0:
            return []
        return [(step, self.lr(step)) for step in range(upper + 1)]


@dataclass
class StepLog:
    """逐步训练日志中的一行。"""

    step: int
    lr: float
    grad_l2_norm: float
    loss: float

    def to_csv_row(self) -> list[str]:
        return [
            str(self.step),
            f"{self.lr:.10f}",
            f"{self.grad_l2_norm:.10f}",
            f"{self.loss:.10f}",
        ]


def gradient_l2_norm(parameters: Iterable[torch.nn.Parameter]) -> float:
    """返回拼接后梯度向量的 L2 范数。

    在梯度场景下复现 ``torch.nn.utils.get_total_norm``，使课程无需依赖
    某个可能提供或不提供该辅助函数的特定 PyTorch 版本。
    """

    squared_sum = 0.0
    for param in parameters:
        if param.grad is None:
            continue
        grad = param.grad.detach()
        squared_sum += float(grad.pow(2).sum().item())
    return math.sqrt(squared_sum)


class TrainState:
    """将模型、AdamW 优化器、调度器和损失函数绑定在一起。

    该类自行维护步数计数器，使调度器的步数轴成为持久状态。
    """

    def __init__(
        self,
        model: nn.Module,
        schedule: CosineWithWarmup,
        loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        weight_decay: float = 0.01,
        betas: tuple[float, float] = (0.9, 0.95),
        eps: float = 1e-8,
    ) -> None:
        self.model = model
        self.schedule = schedule
        self.loss_fn = loss_fn
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=schedule.lr(0),
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
        )
        self.global_step = 0
        self._log: list[StepLog] = []

    def set_lr(self, lr: float) -> None:
        for group in self.optimizer.param_groups:
            group["lr"] = lr

    @property
    def log(self) -> list[StepLog]:
        return list(self._log)

    def step(self, batch_inputs: torch.Tensor, batch_targets: torch.Tensor) -> StepLog:
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        predictions = self.model(batch_inputs)
        loss = self.loss_fn(predictions, batch_targets)
        loss.backward()
        grad_norm = gradient_l2_norm(self.model.parameters())
        rate = self.schedule.lr(self.global_step)
        self.set_lr(rate)
        self.optimizer.step()
        record = StepLog(
            step=self.global_step,
            lr=rate,
            grad_l2_norm=grad_norm,
            loss=float(loss.detach().item()),
        )
        self._log.append(record)
        self.global_step += 1
        return record


def plot_schedule_ascii(
    schedule: CosineWithWarmup,
    width: int = PLOT_WIDTH,
    height: int = PLOT_HEIGHT,
) -> str:
    """返回调度器在 ``[0, total_steps]`` 范围内的文本图。"""

    if width <= 2 or height <= 2:
        raise ValueError("width and height must be at least 3")
    total = schedule.total_steps
    step_axis = [
        int(round(i * total / max(1, width - 1))) for i in range(width)
    ]
    rates = [schedule.lr(step) for step in step_axis]
    upper = max(rates)
    if upper <= 0:
        upper = 1.0

    grid = [[" "] * width for _ in range(height)]
    for col, rate in enumerate(rates):
        row = int(round((height - 1) * (1.0 - rate / upper)))
        row = max(0, min(height - 1, row))
        grid[row][col] = "*"

    rows: list[str] = []
    for r, row in enumerate(grid):
        label = upper * (1.0 - r / max(1, height - 1))
        rows.append(f"{label:8.6f} | " + "".join(row))
    axis = " " * 10 + "+" + "-" * width
    last_label = (
        " " * 11
        + f"step 0"
        + " " * (width - len("step 0") - len(f"step {total}"))
        + f"step {total}"
    )
    return "\n".join(rows + [axis, last_label])


def write_schedule_csv(schedule: CosineWithWarmup, path: Path) -> None:
    """将每一步写入 CSV，列为 ``(step, lr)``。"""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["step", "lr"])
        for step, rate in schedule.points():
            writer.writerow([step, f"{rate:.10f}"])


def write_step_log_csv(log: Iterable[StepLog], path: Path) -> None:
    """使用规范 schema 将训练日志写入 CSV。"""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["step", "lr", "grad_l2_norm", "loss"])
        for row in log:
            writer.writerow(row.to_csv_row())


@dataclass
class LinearWarmupConstant:
    """替代调度器：线性预热后保持在 lr_max 平台。

    可作为余弦变体消融实验的基线。契约相同：预热步数非零时 lr(0) 为零，
    超过 warmup_steps 后学习率保持在 lr_max。
    """

    warmup_steps: int
    lr_max: float

    def __post_init__(self) -> None:
        if self.warmup_steps < 0:
            raise ValueError("warmup_steps must be non-negative")
        if self.lr_max <= 0:
            raise ValueError("lr_max must be positive")

    def lr(self, step: int) -> float:
        if step < 0:
            raise ValueError(f"step must be non-negative, got {step}")
        if self.warmup_steps == 0:
            return self.lr_max
        if step >= self.warmup_steps:
            return self.lr_max
        return self.lr_max * (step / self.warmup_steps)


@dataclass
class InverseSqrtWarmup:
    """线性预热后进行平方根倒数衰减。

    当 step > warmup_steps 时，按 ``lr_max * sqrt(warmup_steps / step)``
    衰减。该方法过去常用于 Transformer 训练，可作为对比基线。
    """

    warmup_steps: int
    lr_max: float

    def __post_init__(self) -> None:
        if self.warmup_steps <= 0:
            raise ValueError("inverse-sqrt warmup requires warmup_steps > 0")
        if self.lr_max <= 0:
            raise ValueError("lr_max must be positive")

    def lr(self, step: int) -> float:
        if step < 0:
            raise ValueError(f"step must be non-negative, got {step}")
        if step <= self.warmup_steps:
            return self.lr_max * (step / self.warmup_steps)
        return self.lr_max * math.sqrt(self.warmup_steps / step)


@dataclass
class EWMA:
    """标量的指数加权移动平均，可用于平滑梯度范数。"""

    beta: float
    value: float = 0.0
    initialized: bool = False

    def __post_init__(self) -> None:
        if not 0 < self.beta < 1:
            raise ValueError("beta must be in (0, 1)")

    def update(self, sample: float) -> float:
        if not self.initialized:
            self.value = float(sample)
            self.initialized = True
            return self.value
        self.value = self.beta * self.value + (1.0 - self.beta) * float(sample)
        return self.value


@dataclass
class StepLogSummary:
    """将逐步日志归约为审查者优先关注的数值。"""

    steps: int
    lr_peak: float
    lr_final: float
    grad_l2_peak: float
    loss_initial: float
    loss_final: float
    loss_delta: float


def summarize_step_log(log: Iterable[StepLog]) -> StepLogSummary:
    rows = list(log)
    if not rows:
        raise ValueError("step log is empty")
    return StepLogSummary(
        steps=len(rows),
        lr_peak=max(row.lr for row in rows),
        lr_final=rows[-1].lr,
        grad_l2_peak=max(row.grad_l2_norm for row in rows),
        loss_initial=rows[0].loss,
        loss_final=rows[-1].loss,
        loss_delta=rows[-1].loss - rows[0].loss,
    )


def split_decay_groups(
    model: nn.Module,
    weight_decay: float = 0.01,
    no_decay_names: tuple[str, ...] = ("bias", "LayerNorm.weight", "layer_norm.weight"),
) -> list[dict[str, object]]:
    """将模型参数分为衰减组与不衰减组。

    Transformer 训练的惯例是对稠密权重矩阵应用权重衰减，而不对偏置或
    LayerNorm 增益参数应用。本辅助函数返回 AdamW 接受的两个参数组 dict。
    """

    decay_params: list[nn.Parameter] = []
    no_decay_params: list[nn.Parameter] = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if any(needle in name for needle in no_decay_names):
            no_decay_params.append(param)
        else:
            decay_params.append(param)
    groups: list[dict[str, object]] = []
    if decay_params:
        groups.append({"params": decay_params, "weight_decay": weight_decay})
    if no_decay_params:
        groups.append({"params": no_decay_params, "weight_decay": 0.0})
    return groups


def build_toy_model(
    in_dim: int = 16,
    out_dim: int = 4,
    seed: int = 7,
) -> tuple[nn.Module, torch.Tensor, torch.Tensor]:
    """用于演示的微型线性模型和固定批次。"""

    torch.manual_seed(seed)
    model = nn.Sequential(nn.Linear(in_dim, 32), nn.GELU(), nn.Linear(32, out_dim))
    inputs = torch.randn(8, in_dim)
    targets = torch.randn(8, out_dim)
    return model, inputs, targets


def run_demo() -> int:
    """在玩具模型上运行 20 个训练步骤并渲染调度曲线。"""

    model, inputs, targets = build_toy_model()
    schedule = CosineWithWarmup(
        warmup_steps=4,
        total_steps=20,
        lr_max=1e-2,
        lr_min=1e-4,
    )
    state = TrainState(
        model=model,
        schedule=schedule,
        loss_fn=nn.functional.mse_loss,
    )
    for _ in range(20):
        record = state.step(inputs, targets)
        print(
            f"step={record.step:>3} lr={record.lr:.6f} "
            f"grad_l2={record.grad_l2_norm:.6f} loss={record.loss:.6f}"
        )
    print()
    print("学习率调度：")
    print(plot_schedule_ascii(schedule, width=40, height=10))
    summary = summarize_step_log(state.log)
    print()
    print(
        f"摘要：steps={summary.steps} lr_peak={summary.lr_peak:.6f} "
        f"grad_l2_peak={summary.grad_l2_peak:.6f} loss_delta={summary.loss_delta:.6f}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(run_demo())
