# Dataset v0 构建方法与依据

## 目标

Dataset v0 面向论文结构化摘要、基于证据问答和引用/论断核对三个用例。数据构建采用
“多源发现 → OA 下载 → 原文切块 → Hy3 生成 → Hy3 自审 → 规则校验 → 人工批准”的
流程。模型输出始终是候选标注，不是金标准。

## 为什么采用生成—过滤—人工审核

Self-Instruct 表明，大模型可以生成任务指令和实例，再通过规则与模型过滤形成可用候选
数据；本项目借鉴其生成—过滤结构，但不把自生成数据直接用于最终评价。HaluEval 使用
模型构造具有迷惑性的错误内容，再结合人工标注形成幻觉评测样本；因此本项目明确区分
普通样本、限定条件难例和 Unsupported/Contradicted 反例，并要求人工逐条核对。

参考：

- Yizhong Wang et al., *Self-Instruct: Aligning Language Models with
  Self-Generated Instructions*, ACL 2023, DOI: 10.18653/v1/2023.acl-long.754.
- Junyi Li et al., *HaluEval: A Large-Scale Hallucination Evaluation Benchmark
  for Large Language Models*, EMNLP 2023, DOI: 10.18653/v1/2023.emnlp-main.397.

## 样本配方

最小 Dataset 包含七种配方：

1. 标准结构化摘要；
2. 跨段、保留限定条件的摘要难例；
3. 单证据或双证据问答；
4. 跨段/跨论文比较问答难例；
5. 原文直接支持的正确论断；
6. 部分事实正确但限定条件被改写的难例；
7. 原文不支持或直接矛盾的反例。

`gold_evidence_ids` 必须来自本地 `workspace.json`。结构校验会检查未知论文、未知证据、
重复 case ID、任务分布和反例标签。Hy3 自审失败的样本标记为 `needs_revision`；即使自动
检查通过，人工状态仍保持 `pending`。

## 文献来源与去重

- OpenAlex：主题检索、引用数和 `best_oa_location`；
- Crossref：DOI、出版信息和跨学科元数据；
- arXiv：计算机、物理和数学预印本及合法 PDF；
- Hy3 联网搜索：发现当前网页来源，仅作补充审计线索。

去重先使用规范化 DOI；任一记录缺少 DOI 时，使用规范化标题、第一作者姓氏以及标题
Jaccard 相似度 `>= 0.90`。合并时保留各来源 provenance，优先完整元数据，并单独保留
可靠的 OA PDF 位置。

## 下载边界

只处理明确标记开放获取且存在 PDF URL 的记录。Crossref 链接还必须带 Creative Commons
许可。每篇 PDF 都检查公网 URL、重定向目标、`%PDF`、最大大小、非零页数、前三页可提取
文本和 SHA-256。程序不会绕过付费墙、登录、CAPTCHA、Cloudflare、DRM 或出版商机器人
检查，也不自动下载补充材料。

## 批量评测

每个样本输出七维分数：事实准确性、证据可追溯性、术语正确性、覆盖完整性、论断判断、
安全边界和清晰度。证据可追溯性由预测证据 ID 与 Gold Evidence 的精确率、召回率和 F1
直接计算；论断核对任务的标签分数由精确匹配计算，其他维度由证据约束的 Hy3 Judge 给出。
最终同时导出逐样本 CSV、失败样本 JSONL 和按错误类型汇总的 Markdown 报告。

评测设计还沿用项目已有的 ReportBench Claim 级核验、DeepResearch Bench 多维评价和
CALM 表面/事实扰动分离原则，详见 `research_basis.md`。
