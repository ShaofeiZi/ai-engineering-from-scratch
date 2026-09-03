"""LLaVA-OneVision token 预算 + 课程规划器 — 标准库。

给定每个样本的视觉 token 总预算和任务混合（单图、多图、视频比例），分配：
  - 单图的 AnyRes 切片数量与池化因子
  - 多图的每样本图像数和单图分辨率
  - 视频的每样本帧数和逐帧池化

打印逐阶段训练计划，并给出每个样本的预期 FLOPs。
在各场景下保持预算大致恒定，以确保 LLM 不会超出上下文。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Budget:
    single_image_tokens: int
    multi_image_tokens: int
    video_tokens: int

    def max(self) -> int:
        return max(self.single_image_tokens, self.multi_image_tokens, self.video_tokens)

    def min(self) -> int:
        return min(self.single_image_tokens, self.multi_image_tokens, self.video_tokens)


def anyres_tokens(tiles: int, per_tile: int) -> int:
    return (tiles + 1) * per_tile


def per_tile_tokens(resolution: int, patch: int, pool: int) -> int:
    g = resolution // patch
    pooled = g // pool
    return pooled * pooled


def plan_single_image(budget: int) -> dict:
    for tiles in [9, 4, 1]:
        for per_tile_size in [(384, 14, 2), (384, 14, 1), (336, 14, 2)]:
            res, patch, pool = per_tile_size
            per = per_tile_tokens(res, patch, pool)
            total = anyres_tokens(tiles, per)
            if total <= budget:
                return {
                    "scenario": "single-image",
                    "tiles": tiles,
                    "tile_res": res,
                    "pool": pool,
                    "per_tile": per,
                    "total": total,
                }
    return {"scenario": "single-image", "tiles": 1, "per_tile": 81, "total": 162}


def plan_multi_image(budget: int) -> dict:
    for n_images in [8, 6, 4, 2]:
        for res_pool in [(384, 2), (384, 1), (336, 2)]:
            res, pool = res_pool
            per = per_tile_tokens(res, 14, pool)
            total = n_images * per
            if total <= budget:
                return {
                    "scenario": "multi-image",
                    "n_images": n_images,
                    "resolution": res,
                    "pool": pool,
                    "per_image": per,
                    "total": total,
                }
    return {"scenario": "multi-image", "n_images": 2, "per_image": 81, "total": 162}


def plan_video(budget: int) -> dict:
    for n_frames in [32, 16, 8]:
        for res_pool in [(384, 3), (384, 2), (336, 2)]:
            res, pool = res_pool
            per = per_tile_tokens(res, 14, pool)
            total = n_frames * per
            if total <= budget:
                return {
                    "scenario": "video",
                    "n_frames": n_frames,
                    "resolution": res,
                    "pool": pool,
                    "per_frame": per,
                    "total": total,
                }
    return {"scenario": "video", "n_frames": 8, "per_frame": 64, "total": 512}


def print_plan(plan: dict, budget: int) -> None:
    pct = 100 * plan["total"] / budget
    scenario_names = {"single-image": "单图", "multi-image": "多图", "video": "视频"}
    field_names = {
        "tiles": "瓦片数", "tile_res": "瓦片分辨率", "pool": "池化因子",
        "per_tile": "每瓦片 token", "n_images": "图像数",
        "resolution": "分辨率", "per_image": "每图 token",
        "n_frames": "帧数", "per_frame": "每帧 token",
    }
    print(f"\n{scenario_names[plan['scenario']]:<12} 预算目标 {budget:>5}，已用 {plan['total']:>5}  ({pct:>5.1f}%)")
    for k, v in plan.items():
        if k in ("scenario", "total"):
            continue
        print(f"    {field_names.get(k, k):<12}：{v}")


def curriculum_stages(mix: dict) -> None:
    print("\n课程调度（三阶段）")
    print("-" * 60)
    stages = [
        ("阶段 SI   ", 1.0, 0.0, 0.0, "仅单图，AnyRes 高分辨率"),
        ("阶段 OV   ", 0.5, 0.3, 0.2, "OneVision 混合，统一预算"),
        ("阶段 TT   ", mix["single"], mix["multi"], mix["video"],
         "目标任务微调"),
    ]
    print(f"{'阶段':<12}{'单图':>8}{'多图':>8}{'视频':>8}   备注")
    for name, s, m, v, note in stages:
        print(f"{name:<12}{s:>8.2f}{m:>8.2f}{v:>8.2f}   {note}")
    print("\n顺序很重要：根据 LLaVA-OneVision 消融实验，倒序训练（视频优先）"
          "会使 MMMU 低 2-4 分。")


def main() -> None:
    print("=" * 60)
    print("LLAVA-ONEVISION TOKEN 预算与课程（第 12 阶段，第 08 课）")
    print("=" * 60)

    budget = 4096

    si = plan_single_image(budget)
    mi = plan_multi_image(budget)
    vi = plan_video(budget)

    print(f"\n共享的每样本视觉 token 预算：{budget}")
    for p in (si, mi, vi):
        print_plan(p, budget)

    spread = max(si["total"], mi["total"], vi["total"]) - min(si["total"], mi["total"], vi["total"])
    print(f"\n各场景的预算跨度：{spread} 个 token "
          f"（占预算的 {100*spread/budget:.1f}%）")
    print("LLaVA-OneVision 目标：将偏差控制在 30% 以下，以获得可预测的 LLM 成本。")

    mix = {"single": 0.4, "multi": 0.3, "video": 0.3}
    curriculum_stages(mix)

    print("\n涌现能力（报告于 LLaVA-OneVision 第 4.3 节）")
    print("-" * 60)
    print("  多摄像头推理    ：结合多图与视频课程")
    print("  标记集提示      ：空间定位 + 多图引用")
    print("  iPhone 截图智能体：UI 截图 + 视频工作流迁移")
    print("  三者均未出现在 SI 阶段数据中；课程学习可解锁这些能力。")


if __name__ == "__main__":
    main()
