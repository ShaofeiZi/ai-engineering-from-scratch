---
name: fairness-criterion
description: 识别某项主张所援引的公平性准则，并审计相关假设。
version: 1.0.0
phase: 18
lesson: 21
tags: [fairness, demographic-parity, equalized-odds, counterfactual-fairness, impossibility]
---

给定一项公平性主张或政策，识别其援引的准则、该主张所依赖的假设，以及不可能性定理对其余准则的含义。

产出：

1. 准则识别。将主张标记为以下之一：群体平等（demographic parity）、均等化赔率（equalized odds）、条件使用准确率平等（conditional use accuracy equality）、个体公平性（individual fairness）、反事实公平性（counterfactual fairness）。含糊的主张必须在继续之前解决。
2. 基线率审计。部署环境中各群体的基线率是多少？在基线率不等的情况下，Chouldechova / KMR 2017 不可能性适用：没有模型能同时满足三个群体准则。
3. 因果 DAG 依赖。如果主张是反事实公平性，其因果 DAG 是什么？反事实公平性的合理性仅与 DAG 的合理性相当。缺乏 DAG 将使该主张无效。
4. 相似性度量。如果主张是个体公平性，其相似性度量 d 是什么？该选择是任务特定的，是一个政策决策而非统计决策。
5. 干预合法性。如果主张使用反事实推理，是否涉及对受保护属性的干预？如果是，考虑使用回溯反事实（arXiv:2401.13935）以规避法律问题。

硬性否决：
- 任何没有准则识别的"公平"主张。
- 在基线率不等的情况下，任何声称"满足所有公平性准则"但未承认 Chouldechova / KMR 2017 的主张。
- 任何没有公开因果 DAG 的反事实公平性主张。

拒绝规则：
- 如果用户询问哪个公平性准则是"正确的"，拒绝排名并解释这是一个政策选择。
- 如果用户询问某个模型是否"公平"，拒绝二元主张；公平性是相对于准则的。

输出：一页纸的审计，填写上述五个部分，在适用时标记不可能性，并命名主张中隐含的政策选择。视情况各引用 Dwork 等人 2012、Kusner 等人 2017、Chouldechova 2017 一次。
