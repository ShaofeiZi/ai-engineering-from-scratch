---
name: skill-cmer-monitor
description: 为生产环境 VLM 端点接入跨模态错误率（CMER）监控、仪表盘和告警
version: 1.0.0
phase: 4
lesson: 25
tags: [vlm, production, monitoring, hallucination]
---

# CMER 监控

将跨模态对齐作为一等的生产环境 KPI。

## 何时使用

- 部署任何基于图像生成文本的 VLM 端点时。
- 调查关于幻觉响应的报告时。
- 追踪输入分布偏移是否导致模型接地性下降时。

## 输入

- `vlm_output`：生成的文本。
- `text_confidence`：softmax 后每个 token 概率的平均值，取值范围为 `[0, 1]`。计算方式为 `exp(mean(log_probs))`。不要传入原始 logits；原始 logits 无界，且 `conf_threshold` 假定输入为概率值。
- `image_embedding`：图像的 CLIP 系列嵌入（DINOv3、SigLIP、CLIP）。
- `text_embedding`：生成文本的 CLIP 系列嵌入。
- 可选 `prompt_type`：用于分组的标签（vqa / ocr / captioning / agent）。

## 单次请求计算

```python
import torch

def cmer_flag(image_emb, text_emb, text_conf, sim_thr=0.25, conf_thr=0.8):
    if image_emb.shape != text_emb.shape:
        raise ValueError(f"emb shape mismatch: {image_emb.shape} vs {text_emb.shape}")
    image_emb = image_emb / (image_emb.norm() + 1e-8)
    text_emb = text_emb / (text_emb.norm() + 1e-8)
    sim = float((image_emb * text_emb).sum())
    flagged = (text_conf > conf_thr) and (sim < sim_thr)
    return {"sim": sim, "flagged": flagged}
```

嵌入是来自独立 CLIP 系列编码器的一维 PyTorch 张量（`torch.float32`）。如果使用 NumPy 数组，请将 `.norm()` 替换为 `np.linalg.norm(...)`，并相应地转换输出类型。

将 `sim`、`text_conf`、`flagged`、`prompt_type`、`timestamp`、`model_version`、`request_id` 存入你的监控流水线（Prometheus、DataDog、OpenTelemetry）。

## 聚合指标

```
CMER = (flagged requests in window) / (total requests in window)
```

按端点、按 prompt_type、按模型版本分别上报。

## 告警阈值

- 基线 CMER：在 7 天的正常流量上建立。
- 警告：CMER >= 1.5 倍基线，持续 1 小时。
- 严重：CMER >= 2 倍基线，持续 30 分钟；或在任意窗口内绝对值 > 15%。

## 仪表盘面板

1. CMER 随时间变化（5 分钟分桶，7 天窗口）。
2. 按 prompt_type 分组的 CMER（堆叠柱状图）。
3. 每小时 `sim` 的分布（直方图）。
4. 排名靠前的幻觉输出（每天采样 20 条被标记的响应供人工审查）。

## CMER 飙升时的处置

1. 对被标记的请求进行采样。
2. 确认模型版本没有被意外更改。
3. 检查输入分布（新的文件格式？新的图像来源？压缩方式不同？）。
4. 在飙升消除前，将受影响的流量路由到人工审查。
5. 如果飙升持续存在，微调或替换模型；不要抑制告警。

## 规则

- 绝不要使用 VLM 自身的嵌入来计算 CMER；必须使用独立编码器（DINOv3、SigLIP 或 CLIP-L/14）。否则你衡量的是模型的自洽性，而非对齐性。
- 始终记录原始 `sim` 值，而不仅仅是 `flagged` 标志位；分布偏移会先在低分位数显现，随后才会反映到标记率上。
- 不要在未接入 CMER 监控的情况下上线 VLM 端点；幻觉是生产环境的主要失效模式，没有该指标时它是静默的。
- 对于敏感领域（医疗、法律、金融），将 `sim_threshold` 提高到 0.35 或更高；标记条件为 `sim < sim_threshold`，因此更高的阈值会将更多输出判定为潜在未接地——这才是高风险场景应有的默认值。
