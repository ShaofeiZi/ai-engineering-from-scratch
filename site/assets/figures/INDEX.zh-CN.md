# 图表索引

`site/assets/figures/` 下随仓库发布的每一幅图都列在下表中。FIG 编号在全仓库范围内唯一、单调递增，且绝不重复使用。

图表的视觉风格记录在 `blueprint-diagram` Claude Code skill 中。根据项目“仓库中不存放供应商或工具制品”的规则，该 skill 与本仓库分开发行。安装后，其源文件位于 `~/.claude/skills/blueprint-diagram/`；如需获取安装路径，请咨询维护者，或者按照下方[如何添加](#如何添加)一节，在不依赖该 skill 的情况下手动完成。

| FIG | slug | 阶段 | 课程 | 添加日期 | 说明 |
|---|---|---|---|---|---|
| 000 | (curriculum stack — embedded in the README banner) | — | — | 2026-05-09 | 首页主视觉，位于 `assets/banner.svg`，不在本目录中 |
| 001 | exploded-view-floppy | — | — | 2026-05-09 | skill 的参考示例，位于 `~/.claude/skills/blueprint-diagram/references/examples/` |
| 001.A | prompts | — | — | 2026-05-13 | README 中“every lesson ships something”卡片——prompt 制品图标 |
| 001.B | skills | — | — | 2026-05-13 | README 卡片——可直接放入项目的 `SKILL.md` 图标 |
| 001.C | agents | — | — | 2026-05-13 | README 卡片——ReAct 风格的 agent 循环图标 |
| 001.D | mcp-servers | — | — | 2026-05-13 | README 卡片——包含 tools/resources/prompts 的 MCP server 机架图标 |
| 002 | kernel-surface-gaussian | — | — | 2026-05-09 | skill 的参考示例 |
| 003 | pixel-vector-bezier | — | — | 2026-05-09 | skill 的参考示例 |
| 004 | gaussian-kernel-blur | 1 | 8 | 2026-05-09 | “Optimization: Gradient Descent Family”课程的 Gaussian blur 可视化 |
| 005 | transformer-attention-heads | 7 | 1 | 2026-05-09 | multi-head attention 模块的分解视图 |
| 006 | ai-engineering-learning-paths | all | core learning paths | 2026-08-23 | 用于浏览课程体系的四条相互连接的领域路径 |
| 006.M | ai-engineering-learning-paths-mobile | all | core learning paths | 2026-08-23 | 四条相互连接的领域路径在窄屏上的纵向视图 |

## 编号规则

- `001`–`099`：为早期课程图表保留（阶段 0–7）。
- `100`+：按照创作顺序依次分配。
- 子图使用字母后缀，例如 `004.A`、`004.B`；子图与其父图共用同一行记录。

## 如何添加

如果已经安装 `blueprint-diagram` skill：

1. 使用概念描述运行该 skill。
2. skill 会将 SVG 写入 `site/assets/figures/NNN-slug.svg`，使用下一个可用编号在此处追加一行，并在有要求时通过 `![FIG_NNN](path)` 将图表接入相应课程 Markdown。

如果没有安装该 skill，请手动完成：

1. 按照奶油色纸张与蓝图风格创作 SVG：纸张使用奶油色 `#fafaf5`，线条使用蓝图蓝 `#3553ff`，标签采用带引导线的 JetBrains Mono 大写字母，不使用其他彩色强调。
2. 从上表选择下一个可用 FIG 编号，并以 `site/assets/figures/<NNN>-<slug>.svg` 保存。
3. 在此表中新增一行，填写 FIG 编号、slug、目标阶段与课程、当天日期，以及一行说明。
4. 在课程 Markdown 中以 `![FIG_NNN](../../site/assets/figures/<NNN>-<slug>.svg)` 引用该图。
5. 分别在 480 / 720 / 1200 px 视口宽度下验证：标签不得与几何图形重叠，引导线必须准确到达目标。

## 许可证

图表按仓库的 MIT 许可证发布。分发 SVG 源文件时，MIT 许可证要求保留版权声明；如果只是复用渲染后的图像（例如嵌入博客文章或演示文稿），则无需额外署名。
