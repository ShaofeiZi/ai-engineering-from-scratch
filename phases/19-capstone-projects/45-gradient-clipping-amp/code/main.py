"""梯度裁剪与混合精度训练步骤。

实现：
- clip_global_l2_norm：对 torch.nn.utils.clip_grad_norm_ 的封装，
  同时返回裁剪前的范数和显式的裁剪后范数。
- has_non_finite_grad：扫描梯度中是否存在 NaN 和 Inf 的辅助函数。
- AmpTrainState：训练步骤编排器，将 AdamW 优化器、autocast 和
  GradScaler 串联为一个安全的步骤。
- StepLog 与 SkipLog：结构化的逐步骤记录。

底部的演示会用一个小的 torch.nn.Linear 模型训练 20 步，并在指定步骤
注入一个非有限梯度，以触发跳过路径。
运行：python3 code/main.py
"""

from __future__ import annotations

import csv
import math
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Iterable

try:
    import torch
    from torch import nn
except ImportError as exc:
    raise SystemExit(
        "本课程需要 torch。请使用以下命令安装：pip install torch"
    ) from exc


DEFAULT_MAX_NORM = 1.0
DEFAULT_DEVICE = "cpu"
NORM_TYPE = 2.0


@dataclass
class StepLog:
    """逐步骤训练日志中的一行。"""

    step: int
    lr: float
    grad_l2_pre_clip: float
    grad_l2_post_clip: float
    loss: float
    skipped: bool
    skip_reason: str
    scaler_scale: float

    def to_csv_row(self) -> list[str]:
        return [
            str(self.step),
            f"{self.lr:.10f}",
            f"{self.grad_l2_pre_clip:.10f}",
            f"{self.grad_l2_post_clip:.10f}",
            f"{self.loss:.10f}",
            "1" if self.skipped else "0",
            self.skip_reason,
            f"{self.scaler_scale:.6f}",
        ]


@dataclass
class SkipLog:
    """被跳过步骤的独立记录，用于告警和事后追溯。"""

    step: int
    reason: str
    pre_clip_norm: float
    loss: float
    scaler_scale: float


def has_non_finite_grad(parameters: Iterable[torch.nn.Parameter]) -> bool:
    """如果任意梯度中包含 NaN 或 Inf，则返回 True。"""

    for param in parameters:
        if param.grad is None:
            continue
        grad = param.grad.detach()
        if not torch.isfinite(grad).all().item():
            return True
    return False


def compute_global_l2_norm(parameters: Iterable[torch.nn.Parameter]) -> float:
    """计算全部梯度的欧几里得范数，但不执行裁剪。"""

    squared_sum = 0.0
    for param in parameters:
        if param.grad is None:
            continue
        grad = param.grad.detach()
        squared_sum += float(grad.pow(2).sum().item())
    return math.sqrt(squared_sum)


def clip_global_l2_norm(
    parameters: list[torch.nn.Parameter],
    max_norm: float,
) -> tuple[float, float]:
    """就地将梯度裁剪到 max_norm，并返回 ``(pre_clip, post_clip)``。

    当 ``pre_clip <= max_norm`` 时不修改梯度，且 ``post_clip == pre_clip``。
    当 ``pre_clip > max_norm`` 时，梯度按 ``max_norm / pre_clip`` 缩放，且
    ``post_clip == max_norm``。
    """

    if max_norm <= 0:
        raise ValueError("max_norm must be positive")
    pre_clip = compute_global_l2_norm(parameters)
    if not math.isfinite(pre_clip):
        return pre_clip, pre_clip
    if pre_clip <= max_norm:
        return pre_clip, pre_clip
    scale = max_norm / (pre_clip + 1e-12)
    for param in parameters:
        if param.grad is not None:
            param.grad.detach().mul_(scale)
    return pre_clip, max_norm


class AmpTrainState:
    """包含混合精度与梯度裁剪的训练步骤。

    将模型、AdamW 优化器、GradScaler 和 autocast 设备连接起来。公开的
    ``step(inputs, targets)`` 会执行：

      1. 在 autocast 下前向传播。
      2. 检查 loss 是否有限；非有限 loss 会跳过反向传播。
      3. 通过 scaler.scale(loss) 反向传播。
      4. 调用 scaler.unscale_(optimizer)。
      5. 检查梯度是否有限；非有限梯度会跳过优化器步骤。
      6. 裁剪到 max_norm。
      7. 调用 scaler.step(optimizer) 和 scaler.update()。
    """

    def __init__(
        self,
        model: nn.Module,
        lr: float = 1e-2,
        max_norm: float = DEFAULT_MAX_NORM,
        device_type: str = DEFAULT_DEVICE,
        weight_decay: float = 0.01,
        amp_dtype: torch.dtype | None = None,
    ) -> None:
        if max_norm <= 0:
            raise ValueError("max_norm must be positive")
        if device_type not in ("cpu", "cuda"):
            raise ValueError(f"device_type must be 'cpu' or 'cuda', got {device_type}")
        self.model = model
        self.max_norm = max_norm
        self.device_type = device_type
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )
        scaler_enabled = device_type == "cuda"
        self.scaler = torch.amp.GradScaler(device_type, enabled=scaler_enabled)
        if amp_dtype is None:
            amp_dtype = torch.bfloat16 if device_type == "cpu" else torch.float16
        self.amp_dtype = amp_dtype
        self.global_step = 0
        self._log: list[StepLog] = []
        self._skip_log: list[SkipLog] = []
        self._loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = nn.functional.mse_loss

    @property
    def log(self) -> list[StepLog]:
        return list(self._log)

    @property
    def skip_log(self) -> list[SkipLog]:
        return list(self._skip_log)

    @property
    def skip_count(self) -> int:
        return len(self._skip_log)

    def set_loss_fn(self, fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]) -> None:
        self._loss_fn = fn

    def set_lr(self, lr: float) -> None:
        for group in self.optimizer.param_groups:
            group["lr"] = lr

    def _current_lr(self) -> float:
        return float(self.optimizer.param_groups[0]["lr"])

    def step(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        gradient_corruptor: Callable[[nn.Module], None] | None = None,
    ) -> StepLog:
        """运行一个训练步骤，并可选择注入梯度损坏以便测试。

        ``gradient_corruptor`` 允许演示在反向传播后、unscale 之前注入非有限
        梯度。生产调用方将其保留为 None；测试会传入一个把 Inf 写入某个参数
        梯度的闭包。
        """

        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type=self.device_type, dtype=self.amp_dtype):
            predictions = self.model(inputs)
            loss = self._loss_fn(predictions, targets)

        if not torch.isfinite(loss).all().item():
            # 跳过且不调用 scaler.update()：本步骤从未调用
            # scaler.scale(loss).backward()，此时调用 update() 会违反
            # GradScaler 要求的调用顺序。
            return self._record_skip(
                loss_value=float(loss.detach().cpu().item()),
                reason="non_finite_loss",
                pre_clip=0.0,
                update_scaler=False,
            )

        self.scaler.scale(loss).backward()
        if gradient_corruptor is not None:
            gradient_corruptor(self.model)
        self.scaler.unscale_(self.optimizer)

        if has_non_finite_grad(self.model.parameters()):
            scale_before = float(self.scaler.get_scale())
            self.scaler.update()
            record = StepLog(
                step=self.global_step,
                lr=self._current_lr(),
                grad_l2_pre_clip=float("inf"),
                grad_l2_post_clip=float("inf"),
                loss=float(loss.detach().item()),
                skipped=True,
                skip_reason="non_finite_grad",
                scaler_scale=scale_before,
            )
            self._log.append(record)
            self._skip_log.append(
                SkipLog(
                    step=self.global_step,
                    reason="non_finite_grad",
                    pre_clip_norm=float("inf"),
                    loss=float(loss.detach().item()),
                    scaler_scale=scale_before,
                )
            )
            self.global_step += 1
            return record

        pre_clip, post_clip = clip_global_l2_norm(list(self.model.parameters()), self.max_norm)

        self.scaler.step(self.optimizer)
        self.scaler.update()
        record = StepLog(
            step=self.global_step,
            lr=self._current_lr(),
            grad_l2_pre_clip=pre_clip,
            grad_l2_post_clip=post_clip,
            loss=float(loss.detach().item()),
            skipped=False,
            skip_reason="",
            scaler_scale=float(self.scaler.get_scale()),
        )
        self._log.append(record)
        self.global_step += 1
        return record

    def _record_skip(
        self,
        loss_value: float,
        reason: str,
        pre_clip: float,
        update_scaler: bool = True,
    ) -> StepLog:
        record = StepLog(
            step=self.global_step,
            lr=self._current_lr(),
            grad_l2_pre_clip=pre_clip,
            grad_l2_post_clip=pre_clip,
            loss=loss_value,
            skipped=True,
            skip_reason=reason,
            scaler_scale=float(self.scaler.get_scale()),
        )
        self._log.append(record)
        self._skip_log.append(
            SkipLog(
                step=self.global_step,
                reason=reason,
                pre_clip_norm=pre_clip,
                loss=loss_value,
                scaler_scale=float(self.scaler.get_scale()),
            )
        )
        self.global_step += 1
        if update_scaler:
            self.scaler.update()
        return record


def rolling_skip_rate(log: Iterable[StepLog], window: int = 1000) -> list[float]:
    """返回每一步最近 ``window`` 步的滚动跳过率。"""

    if window <= 0:
        raise ValueError("window must be positive")
    rows = list(log)
    rates: list[float] = []
    skipped: list[int] = []
    for row in rows:
        skipped.append(1 if row.skipped else 0)
        if len(skipped) > window:
            skipped = skipped[-window:]
        rates.append(sum(skipped) / len(skipped))
    return rates


def write_step_log_csv(log: Iterable[StepLog], path: Path) -> None:
    """写入采用规范 schema 的训练步骤 CSV。

    列为：step、lr、grad_l2_pre_clip、grad_l2_post_clip、loss、skipped、
    skip_reason、scaler_scale。
    """

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            [
                "step",
                "lr",
                "grad_l2_pre_clip",
                "grad_l2_post_clip",
                "loss",
                "skipped",
                "skip_reason",
                "scaler_scale",
            ]
        )
        for row in log:
            writer.writerow(row.to_csv_row())


def build_toy_model(
    in_dim: int = 16,
    out_dim: int = 4,
    seed: int = 7,
) -> tuple[nn.Module, torch.Tensor, torch.Tensor]:
    torch.manual_seed(seed)
    model = nn.Sequential(nn.Linear(in_dim, 32), nn.GELU(), nn.Linear(32, out_dim))
    inputs = torch.randn(8, in_dim)
    targets = torch.randn(8, out_dim)
    return model, inputs, targets


def inject_inf_into_first_grad(model: nn.Module) -> None:
    """仅用于测试：将 +Inf 写入第一个参数的梯度。"""

    for param in model.parameters():
        if param.grad is not None:
            param.grad.data[...] = float("inf")
            return


def run_demo() -> int:
    """训练 20 步，并在已知步骤注入非有限梯度。"""

    model, inputs, targets = build_toy_model()
    state = AmpTrainState(model=model, lr=1e-2, max_norm=1.0, device_type="cpu")
    for index in range(20):
        corruptor: Callable[[nn.Module], None] | None = None
        if index == 5:
            corruptor = inject_inf_into_first_grad
        record = state.step(inputs, targets, gradient_corruptor=corruptor)
        marker = "SKIP" if record.skipped else "STEP"
        print(
            f"{marker} step={record.step:>3} lr={record.lr:.6f} "
            f"pre_clip={record.grad_l2_pre_clip:>10.6f} "
            f"post_clip={record.grad_l2_post_clip:>10.6f} "
            f"loss={record.loss:.6f} scale={record.scaler_scale:.1f} "
            f"reason={record.skip_reason or '-'}"
        )
    print()
    print(
        f"skip_count={state.skip_count} "
        f"final_skip_rate={rolling_skip_rate(state.log, window=10)[-1]:.4f}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(run_demo())
