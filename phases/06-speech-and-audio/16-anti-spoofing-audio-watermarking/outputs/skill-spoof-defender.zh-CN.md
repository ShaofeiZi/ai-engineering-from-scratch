---
name: spoof-defender
description: 为语音生成 / 语音认证部署选择检测模型、水印、来源清单和运维操作手册。
version: 1.0.0
phase: 6
lesson: 16
tags: [anti-spoofing, watermark, audioseal, asvspoof, c2pa, voice-fraud]
---

根据工作负载(语音生成 vs 语音认证、部署规模、合规区域、对手画像),输出:

1. 检测 (CM)。AASIST · RawNet2 · NeXt-TDNN + WavLM · 商用方案(Pindrop、Validsoft)。训练数据:ASVspoof 2019 / ASVspoof 5 / 领域专属数据。目标 EER。
2. 水印(出站生成)。AudioSeal 16 位载荷编码 `(model_id, user_id, generation_ts)` · WaveVerify(备选)· 无(需附理由)。检测器在 CI 中对每一条出货前的输出运行检测。
3. 来源。使用部署方密钥签名的 C2PA 清单 · IPTC 元数据 · 无(用于非消费者音频)。
4. 语音认证防护(如适用)。活体挑战(随机短语 TTS' 并转写)、重放攻击检测(AASIST + PA 模型)、按通道进行生物识别阈值校准。
5. 运维。审计日志留存、同意凭证留存(7 年以上)、滥用检测信号(突发流量尖峰、命名实体提示)、熔断(kill-switch)流程。

拒绝不含 AudioSeal(或等效水印)的语音生成部署。拒绝不含反欺骗检测的语音生物识别部署——语音克隆使仅基于余弦相似度的认证可被轻易绕过。拒绝仅依赖来源清单的部署(可被剥离)。拒绝将检测阈值训练于 ASVspoof 2019 并直接用于真实世界部署、且未做通道校准扫描的方案。

示例输入:"银行客服 IVR。语音生物识别解锁 + AI 生成的语音坐席。每月 1000 万通电话。美国 + 欧盟。"

示例输出:
- 检测:Pindrop 商用方案(首选)或 NeXt-TDNN + WavLM 开源方案。在 ASVspoof 5 + 10 万条银行专属通话样本上训练。目标 EER &lt; 0.5%(域内数据)。
- 水印:对每条出站 TTS 话语施加 AudioSeal 16 位载荷;载荷编码 bank_id + session_id + 时间戳。检测器在发送前进行验证。
- 来源:在面向客户的音频导出流程中添加 C2PA 清单;仅内部通话跳过。
- 语音认证:每次认证均进行活体挑战(TTS 随机 4 位数字短语;用户复述 + 检测器 + 转写器)。每次入站认证尝试均运行反欺骗检测。生物识别阈值设为 FAR 0.1%、FRR 1%。
- 运维:同意凭证 + 审计日志在区域内留存 7 年(EU 数据驻留于欧盟)。克隆请求量突发 &gt; 2σ 时告警;触发滥用检测时执行熔断。
