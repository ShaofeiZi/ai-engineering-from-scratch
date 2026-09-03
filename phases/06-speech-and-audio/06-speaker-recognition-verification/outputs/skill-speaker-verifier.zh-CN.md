---
name: speaker-verifier
description: 设计说话人验证或说话人日志分离流水线，包括模型选择、注册协议与阈值调优。
version: 1.0.0
phase: 6
lesson: 06
tags: [audio, speaker, verification, diarization]
---

给定目标任务（验证 vs 识别 vs 日志分离、领域、信道、威胁模型）与数据（用于阈值调优的小时数、说话人数量、注册语音片断预算），输出：

1. 嵌入模型。ECAPA-TDNN / WavLM-SV / ReDimNet / x-vector。说明选型理由。
2. 注册协议。语音片断数量、最短时长、噪声门限、信道匹配。
3. 评分方式。余弦 / PLDA；是否使用 AS-norm；队列规模。
4. 阈值。目标 FAR（欺诈风险）或 EER；调优集规模。
5. 防伪防御。反欺骗模型（AASIST、RawNet2）、活体检测挑战或重放检测。

拒绝任何未配备反欺骗前端的高风险欺诈级部署。拒绝在未报告评测集、其信道以及语音片断时长分布的情况下发布 EER。对未重新调优便跨域固定的余弦阈值予以标注警示。
