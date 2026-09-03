---
name: cross-policy-diff
description: 使用 OpenAI Preparedness Framework v2、Anthropic RSP v3.0 和 DeepMind FSF v3 作为参考，针对特定能力生成跨政策对比。
version: 1.0.0
phase: 15
lesson: 20
tags: [preparedness-framework, fsf, rsp, cross-policy, scaling-policy]
---

给定一项特定的前沿能力（例如"长程自主性"、"自主复制与适应"、"R&D 自动化"），生成一份跨政策对比，展示三套框架分别如何对该能力进行分类，以及会触发哪些缓解措施。

产出：

1. **OpenAI PF v2 分类。** Tracked 或 Research。若为 Tracked，指出 Capabilities + Safeguards Report 的触发条件。若为 Research，注明政策用语为"潜在"缓解措施。
2. **Anthropic RSP v3.0 分类。** 属于哪个阈值（ASL-3、AI R&D-4、硬编码禁令）？属于哪类缓解措施（肯定性论证、安全 + 部署）？确认该承诺属于 Anthropic 单边层级还是行业建议层级。
3. **DeepMind FSF v3 分类。** 属于哪个领域（Cyber、Bio、ML R&D、CBRN）？属于哪个 CCL 或 Tracked Capability Level？是否启用了欺骗性对齐监控？
4. **一致性总结。** 三套政策是否就该能力的严重程度达成一致，还是存在实质性分歧？哪个分类最严格，哪个最宽松？
5. **度量依赖。** 每项分类都依赖于能力度量。指出该能力如何被度量，以及哪个评测提供方（METR、Apollo、内部、第三方）负责该度量。

硬性拒绝条件：
- 仅基于公告语言的相似性就声称存在跨政策一致性，而没有文档级证据。
- 任何无法指向源文档中具体条款的分类。
- 将 OpenAI 的"Research Category"等同于"Tracked Category"——二者具有不同的操作后果。

拒绝规则：
- 如果用户无法为每项分类提供源文档段落，则拒绝并要求先提供引用。
- 如果用户将政策的存在视为缓解措施已在实践中执行的证据，则拒绝并要求提供具体缓解措施已触发的证据。
- 如果某项能力被声称已被某框架"覆盖"，但该词未出现在文档中，则拒绝并要求提供具体条款引用。

输出格式：

返回一份对比文档，包含：
- **能力定义**（一句话）
- **OpenAI PF v2 行**（分类、触发条件、源条款）
- **Anthropic RSP v3.0 行**（分类、触发条件、单边 vs 建议）
- **DeepMind FSF v3 行**（领域、CCL / TCL、欺骗性对齐涉及情况）
- **一致性总结**（一致点 + 实质性分歧）
- **度量归属**（评测提供方、评测频率）
- **读者建议**（最严格、最宽松，并说明理由）
