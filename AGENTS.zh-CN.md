# AGENTS.md

面向贡献者和 AI 代理的操作手册,适用于所有接触本仓库的人。在提交 PR 之前请先阅读本文档。

本仓库是一套课程,而非一个 SaaS 应用。课程本身即产品。以下每一条规则都是为了确保 435 节课程在长期演进中保持一致。

---

## 理念

435 节课程。20 个阶段。每一个算法都先从原始数学推导入手,再引入任何框架。你会用 Python、TypeScript、Rust 或 Julia 手写反向传播、分词器、注意力机制和代理循环,然后用生产库跑一遍同样的运算,让框架不再是黑盒。"构建它 / 使用它"(Build It / Use It)的拆分是主干。每节课程都产出一个可复用的制品,能直接接入你的日常工作流。

---

## 仓库布局

```text
phases/
  NN-phase-slug/
    NN-lesson-slug/
      docs/en.md              # 课程讲解
      code/                   # 实现与测试
      quiz.json               # 6 道题
      outputs/                # 可复用制品（技能 / 提示词 / 智能体 / MCP 服务器）
README.md                     # 项目主页；课程计数自动同步
ROADMAP.md                    # 阶段/课程状态
glossary/terms.md             # 规范术语定义
site/
  build.js                    # 解析 README + ROADMAP + glossary -> data.js
  data.js                     # 生成文件；推送到 main 时由 CI 重建
certifications/claude/
  program.json                # 项目元数据、来源政策、官方链接
  tracks/*.json               # 考试蓝图、有序路线、学习计划
  lessons/NN-slug/            # 共享认证课程契约
  assessments/<exam-code>/    # 原创诊断测验和完整模考
scripts/                      # 自动化脚本
.github/workflows/
  curriculum.yml              # 不变量检查与自动同步工作流
```

---

## 硬性规则

1. **每个课程目录一个提交。** 绝不把多节课程合并到一个提交里。一个 10 节课程的 PR 应有 10 个提交。
2. **约定式提交标题** ≤72 字符:`feat(phase-NN/MM): <slug>`。提交正文解释"为什么",而不是"做了什么"。
3. **图表仅使用 Mermaid 或 SVG。** 不使用 ASCII / Unicode 画框字符。
4. **每个围栏代码块都要带语言标签。** 按需使用 `text`、`json`、`python`、`typescript`、`rust`、`julia`、`bash`、`console`、`mermaid`、`yaml`。
5. **只允许原创实现。** 不要在文档、代码注释或提交文本中引用外部课程仓库。当 RFC、官方规范和学术论文是权威来源时,可引用它们。
6. **依赖白名单**(见下文 `Dependencies`)。标准库优先。
7. **绝不提交生成文件**:`catalog.json` 已被 gitignore,`site/data.js` 由 CI 重建,`package-lock.json` 永不纳入版本控制。

---

## 依赖

| 语言        | 允许                                                                     |
|-------------|--------------------------------------------------------------------------|
| Python      | `numpy`, `torch`, `h5py`, `zstandard`, `safetensors`, stdlib              |
| TypeScript  | `hono`、`zod`、`ws`（仅在需要 WebSocket 时）、`@hono/node-server`、Node 20+ 标准库 |
| Rust        | 仅标准库（单文件 `rustc --edition 2021`）                                |
| Julia       | `Random`、`Statistics`、`LinearAlgebra`、`Printf`（Julia 标准库）         |

如果某条发现指向一个被禁用的依赖,请跳过并注明理由"为保持教学清晰性而坚持标准库优先"。

---

## 课程契约

### docs/en.md frontmatter

```markdown
# <Title>

> <One-line hook>

**Type:** <Learn | Build | Reference>
**Languages:** <comma-list matching the main.* files in code/>
**Prerequisites:** <comma-list of upstream lessons, or "None">
**Time:** ~<estimate in minutes>

## Learning Objectives
- <4-6 bullet points starting with a verb>
```

`**Languages:**` 字段必须与 `code/` 中存在 `main.*` 文件的语言一致。

### quiz.json schema

```json
{
  "lesson": "<dir-slug>",
  "title": "<Lesson Title>",
  "questions": [
    {"stage": "pre",   "question": "...", "options": ["a","b","c","d"], "correct": 0, "explanation": ""},
    {"stage": "check", "question": "...", "options": ["a","b","c","d"], "correct": 1, "explanation": ""},
    {"stage": "check", "question": "...", "options": ["a","b","c","d"], "correct": 2, "explanation": ""},
    {"stage": "check", "question": "...", "options": ["a","b","c","d"], "correct": 1, "explanation": ""},
    {"stage": "post",  "question": "...", "options": ["a","b","c","d"], "correct": 3, "explanation": ""},
    {"stage": "post",  "question": "...", "options": ["a","b","c","d"], "correct": 0, "explanation": ""}
  ]
}
```

恰好 6 道题:1 道 pre + 3 道 check + 2 道 post。`correct` 从 0 开始索引。站点渲染器只识别这种结构——旧版 `q/choices/answer` 模式会静默崩溃。

### Claude 认证契约

位于 `certifications/claude/lessons/` 下的认证课程遵循与阶段课程相同的文档、测验、图表、依赖以及"每课程一提交"规则。每节认证课程都需要一个可运行的 main 文件和至少五项确定性测试。轨道引用稳定的课程路径,因此一节课程可服务于多个认证而不重复。概念性课程仍需实操:使用场景运行器、策略评分器、制品校验器、审批模拟器、威胁模型检查器或证据评分器,而非伪造的供应商 API 代码。一条轨道也可将现有的 `phases/` 课程作为可选的深度拓展引用。

完全对齐的认证课程采用与最强阶段课程相同的"解释、操作、构建、交付、验证"循环。每节认证课程必须包含精确的 `Interactive Lab`、`Practice Lab`、`Shipped Artifact`、`Verify It` 和 `Capstone Connection` 小节;嵌入已注册的 `figure` 机制;在 `outputs/` 下至少交付一个文件;并提供可运行的场景、模拟器、评分器或制品校验器及测试。概念性课程中的代码必须体现该课程的判断要点。不要为了满足可运行表面而硬塞一个假 API 集成。治理类课程可使用模拟事件、策略评分器、威胁模型检查、ADR 校验、审批工作流或证据包评分器。

`program.json` 负责独立课程免责声明、验证日期和官方链接。`prerequisites.json` 负责机器可读的认证课程依赖图。每条必需的轨道路线在进入消费它们的课程之前,必须包含那些内部前置课程。`tracks/` 下的每个文件负责一项公开考试蓝图、其精确的领域权重、有序的课程路线、评估声明和学习计划。考试事实必须来自当前官方指南。产品与模型细节必须标注日期,并与当前官方文档核对。

诊断测验和模考使用单独的评估模式,因为它们支持多选题:

```json
{
  "id": "claude-ccar-f-diagnostic",
  "version": 1,
  "track": "claude-ccar-f",
  "kind": "diagnostic",
  "title": "Architect Foundations Diagnostic",
  "timeLimitMinutes": 30,
  "questions": [
    {
      "id": "ccar-f-agent-001",
      "domain": "agentic-architecture-orchestration",
      "objective": "choose-an-orchestration-pattern",
      "type": "single",
      "prompt": "A self-contained original scenario...",
      "options": ["a", "b", "c", "d"],
      "correct": [1],
      "explanation": "Why the decision fits and the alternatives do not.",
      "references": ["certifications/claude/lessons/16-multi-agent-orchestration-and-delegation"]
    }
  ]
}
```

`correct` 始终是一个数组。`single` 题恰好有一个索引;`multiple` 题至少有两个。题目必须原创、映射到公开考点、包含实质性解释,且绝不复制或试图还原保密的考试内容。练习百分比是原始得分,而非 Anthropic 的量表分数,本课程绝不保证通过。公开的认证页面和课程上下文还必须声明:这是一套独立的社区课程,与 Anthropic 不存在隶属、背书、赞助或授权关系。

### AI 原生认证学习者模式

当用户请求选择、开始、继续、学习、练习或评估某项 Claude 认证时,在授课前请先阅读并遵循 `skills/claude-certification/SKILL.md`。本规则适用于 Codex 以及任何读取 `AGENTS.md` 的其他执行框架;Claude Code 还会在 `.claude/skills/` 下发现对应的封装。

在学习者模式下,把本仓库当作一位交互式导师。读取所选轨道清单,每次讲授路线中的一节课程,运行其真实场景与测试,要求学习者在 `learning-artifacts/` 下产出自己拥有的制品,对存储的测验或评估评分,并将进度保留在 `CLAUDE-CERTIFICATION.md` 中。不要把已检入的参考制品当作学习者作品来修改。认证课程通过 GitHub 和网站交付,刻意置于图书生成流水线之外。它保持仅英文,且刻意也置于机器翻译流水线之外。

### code/

- 在该语言的规范命令下能端到端运行并以退出码 0 结束。
- 自终止演示。不进入无限 stdin 循环,不因缺少 API key 而挂起。
- 4-6 行头部注释,引用本课程的 `docs/en.md` 路径以及任何规范或 RFC 来源。

### code/tests/

- 至少 5 个单元测试。
- 通过语言自带的标准库运行器运行(`python3 -m unittest discover`、`npx tsx --test`,Rust/Julia 内联)。

---

## PR 前验证

推送前在本地运行:

```bash
python3 scripts/audit_lessons.py
python3 scripts/audit_certifications.py
python3 scripts/check_readme_counts.py        # 提示性检查——CI 会在合并时修复

# 对每节有改动的课程执行：
cd phases/NN-phase/MM-lesson/code
python3 main.py && python3 -m unittest discover tests -v   # 或对应语言的等效命令
```

CI 门禁(`.github/workflows/curriculum.yml`):

| Job                              | Trigger      | Behavior                                              |
|----------------------------------|--------------|-------------------------------------------------------|
| `audit`                          | push + PR    | 运行 `audit_lessons.py`。阻断式。                       |
| `readme-counts-sync` (main only) | push to main | 重建 catalog 并自动修复 README 计数。                     |
| `site-rebuild` (main only)       | push to main | 重新运行 `node site/build.js`,提交 `site/data.js`。       |
| `readme-counts-drift`            | PR           | 仅供参考——main 分支会在合并时自愈。                         |

---

## 自动化契约

**CI 自动处理——请勿在你的 PR 中触碰:**

| 范围                 | 自动化任务                     | 时机                |
|----------------------|--------------------------------|---------------------|
| `catalog.json`       | 按需重建（已 gitignore）       | 每次 CI 任务        |
| `README.md` 计数     | `readme-counts-sync`           | 推送到 main 时      |
| `site/data.js`       | `site-rebuild`                 | 推送到 main 时      |

**由你处理:**

| 范围                          | 时机                                                             |
|-------------------------------|------------------------------------------------------------------|
| `README.md` 课程链接行        | 添加新课程时——链接格式为 `[Title](phases/NN-phase/MM-lesson/)`   |
| `ROADMAP.md` 状态             | 将课程标记为已完成或 WIP 时                                      |
| `glossary/terms.md`           | 引入供多节课程使用的术语时                                       |

**常见 Bug**:合并后若 `grep -c 'tree/main/phases/NN-' site/data.js` 为 0,说明 Phase NN README 的行是纯文本,缺少 `[Title](phases/NN-...)` markdown 链接。`site/build.js` 从该链接推导 URL。

---

## 冲突解决

```bash
git fetch origin main
git merge --no-edit origin/main

# catalog 冲突（仅限旧分支——catalog.json 现已被 gitignore）：
git rm catalog.json
git commit --no-edit

# README 计数冲突：
git checkout --theirs README.md
python3 scripts/build_catalog.py
python3 scripts/check_readme_counts.py --fix
git add README.md && git commit --no-edit

# site/data.js 冲突：
git checkout --theirs site/data.js
node site/build.js
git add site/data.js && git commit --no-edit

git push origin <your-branch>
```

不要对有未关闭评审评论的分支使用 `git push --force`。强制推送会使评论脱离。

---

## 新课程接入

```bash
mkdir -p phases/NN-phase-slug/MM-new-lesson/{docs,code/tests,outputs}

# 1. 按上方 frontmatter 编写 docs/en.md。
# 2. 编写 code/main.<lang>，并添加 4-6 行头部注释。
# 3. 编写 code/tests/test_main.*，包含至少 5 个测试。
# 4. 按上方 schema 编写 quiz.json。
# 5.（可选）如果课程交付技能，添加 outputs/skill-<slug>.md。

# 6. 添加到 README.md：
#    | MM | [Lesson Title](phases/NN-phase-slug/MM-new-lesson/) | Type | Lang |

# 7. 更新 ROADMAP.md 状态行。

# 8. 在本地验证。

# 9. 创建原子提交：
git add phases/NN-phase-slug/MM-new-lesson README.md ROADMAP.md
git commit -m "feat(phase-NN/MM): add <slug>"
git push -u origin <your-branch>
gh pr create --title "feat(phase-NN/MM): add <slug>" --body "<5-line summary>"
```

`site/data.js` 在合并时重新生成——留给 CI 处理。

---

最近审阅:2026-05-27。
