---
name: skill-mot-evaluator
description: 为 MOTA / IDF1 / HOTA 编写完整的评测工具，用于与真值轨迹对比
version: 1.0.0
phase: 4
lesson: 27
tags: [mot, evaluation, tracking, metrics]
---

# MOT 评测器

将你的跟踪器输出封装到标准的 MOTA/IDF1/HOTA 流水线中，以便与文献中的结果进行公平比较。

## 何时使用

- 在 MOT17 / MOT20 / DanceTrack / SportsMOT 上对新的跟踪器进行基准测试。
- 在你自己的视频上将 ByteTrack 与 BoT-SORT、SAM 2 进行对比。
- 为论文或 PR 描述生成可复现的数值。

## 输入

- `predictions`：按帧给出的 `(track_id, x, y, w, h, confidence)` 元组列表。
- `ground_truth`：按帧给出的 `(gt_id, x, y, w, h)` 元组列表。
- `iou_threshold`：MOTA 通常为 0.5；HOTA 采用扫描方式。
- `evaluator`：`py-motmetrics`（MOTA、IDF1）或 `TrackEval`（HOTA）。

## 输出格式约定

`py-motmetrics` 和 `TrackEval` 都期望一种特定的磁盘文件格式：

```
# predictions.txt
<frame>,<track_id>,<x>,<y>,<w>,<h>,<confidence>,-1,-1,-1

# ground_truth.txt
<frame>,<gt_id>,<x>,<y>,<w>,<h>,1,-1,-1,-1
```

帧从 1 开始编号，边界框格式为 (x, y, w, h)，而不是 (x1, y1, x2, y2)。格式转换是大多数集成 bug 的藏身之处。

## 步骤

1. 将你的跟踪器输出转换为 MOT Challenge 文本格式。
2. 对两个文件运行 `py-motmetrics.io.loadtxt`。
3. 使用 `mm.metrics.create().compute()` 计算 MOTA 和 IDF1。
4. 对于 HOTA，使用相同的文件调用 `TrackEval` 并设置 `Metrics: HOTA`。
5. 将结果保存为 JSON，供仪表盘使用。

## 实现草图

```python
import motmetrics as mm

def evaluate_mota_idf1(pred_path, gt_path):
    gt = mm.io.loadtxt(gt_path, fmt="mot15-2D")
    pred = mm.io.loadtxt(pred_path, fmt="mot15-2D")
    acc = mm.utils.compare_to_groundtruth(gt, pred, dist="iou", distth=0.5)
    metrics = mm.metrics.create().compute(
        acc, metrics=["num_frames", "mota", "motp", "idf1", "idp", "idr", "num_switches"]
    )
    return metrics


def write_mot_txt(predictions, path):
    with open(path, "w") as f:
        for frame_idx, detections in enumerate(predictions, start=1):
            for tid, x, y, w, h, conf in detections:
                f.write(f"{frame_idx},{tid},{x:.2f},{y:.2f},{w:.2f},{h:.2f},{conf:.3f},-1,-1,-1\n")
```

## 报告

```
[mot evaluation]
  frames:     <int>
  gt tracks:  <int>
  pred tracks: <int>

[metrics]
  MOTA:       <float>
  MOTP:       <float>
  IDF1:       <float>
  IDP/IDR:    <float/float>
  ID switches: <int>
  HOTA:       <float>  (from TrackEval)
```

## 规则

- 输出文本文件中的帧始终使用 1 起始编号；MOT 工具期望如此。
- 在写入之前将 (x1, y1, x2, y2) 转换为 (x, y, w, h)。
- 在现代比较中不要只报告 MOTA；应同时包含 IDF1 和 HOTA。
- 注意 MOT17 上的私有检测与公开检测——它们是分开评测的，混用会虚高得分。
- 记录每个序列的得分；汇总数据会掩盖单个困难序列上的失败。
