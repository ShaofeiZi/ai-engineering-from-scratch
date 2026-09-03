# 更新日志

课程的新变化。最新内容排在最前。

格式大致遵循 [Keep a Changelog](https://keepachangelog.com/)。每条记录都会注明阶段、课程以及变更内容，方便学习者直接定位到改动点。

## [Unreleased]

### Added
- `scripts/scaffold-lesson.sh` — 脚手架工具，用于创建 `phases/NN-phase/NN-lesson/`，包含完整的目录结构，并根据 `LESSON_TEMPLATE.md` 预填一份 `docs/en.md` 骨架。
- `.github/PULL_REQUEST_TEMPLATE.md` — 贡献者清单（代码可运行、无代码注释、坚持从零开始构建、每节课原子化提交、ROADMAP 行使用 markdown 链接）。
- `.github/ISSUE_TEMPLATE/bug_report.md` 和 `new_lesson_proposal.md` — 用于提交 bug 报告和课程提案的结构化入口。
- 本 `CHANGELOG.md`。

## 2026-04 — Phase 4:Computer Vision 完成

### Added
- Phase 4 全部 28 节课程，涵盖从图像基础到多模态视觉（VLM、3D、视频、自监督）。
- `ROADMAP.md` 中 Phase 4 的各行以 markdown 链接指向课程文件夹，使网站能正确展示。

### Fixed
- 对 Phase 4 中 15 余节课程进行精度修订：
  - `phase-4/02`：形状计算器明确了自适应池化、展平和线性层中 RF/stride 的处理方式。
  - `phase-4/03`：骨干网络选择器描述列出了所有覆盖的模型族；为 OCR、医疗、工业场景补充了检测头指引。
  - `phase-4/04`：分类诊断针对每种失败模式使用量化阈值；对未定义指标声明 `n/a`；增加了类别数少于 3 的保护判断。
  - `phase-4/06`：检测指标解读使用 `AP@0.5`（而非 `mAP@0.5`）；声明每类召回率为可选项；锚框设计器澄清了步长截断和每层单锚路径。
  - `phase-4/10`：采样器选择器声明 `unet_forward_ms` 为输入；ControlNet 检查提升为规则 0。
  - `phase-4/14`：ViT 检查器与拒绝规则保持一致——对端口尝试行为进行审计，而非背书。
  - `phase-4/24`：开放词表技术栈选择器具有明确的规则优先级和许可证过滤语义；概念设计器解决了 step-5 与 rule-80 之间的冲突。
  - `phase-4/25`：VLM 文档的 `_merge` 在占位符不匹配时抛出描述性的 `ValueError`；CMER 在内部进行归一化。
  - `phase-4/27`：`synthetic_frames` 将 GT 框裁剪到帧的 H/W 范围内。
  - `phase-4/28`：`rope_3d` 校验维度切分；从 DiT block 示例中移除未使用的 `F` 导入。

## 2026-Q1 及更早

### Added
- Phase 0（Setup & Tooling）：全部 12 节课程。
- Phase 1（Math Foundations）：全部 22 节课程。
- Phase 2（ML Fundamentals）：全部 18 节课程。
- Phase 3（Deep Learning Core）：核心课程直至 perceptron、backprop、optimizers。
- 内置 Claude Code 技能：`find-your-level`（分班测验）和 `check-understanding`（分阶段测验）。
- 网站 `aiengineeringfromscratch.com`：目录、分课程页面、路线图、277 词术语表。
- 全部 20 个阶段的初始脚手架（`phases/00-*` 到 `phases/19-*`）。
- `LESSON_TEMPLATE.md`、`CONTRIBUTING.md`、`ROADMAP.md`、`README.md`。

[Unreleased]: https://github.com/rohitg00/ai-engineering-from-scratch/compare/HEAD...HEAD
