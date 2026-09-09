# 论文依据与代码映射

## 1. OpenScholar：检索、引用感知生成与自反馈

Asai 等人的 OpenScholar 将科学文献综合建模为：从领域文献库检索段落，生成带行内
引用的长答案，并用 self-feedback 迭代改进事实性、覆盖度和引用准确性。

本项目对应实现：

- `ingest.py`：把 PDF 转成带论文、页码、章节和 Chunk 编号的 Evidence Database；
- `retrieval.py`：先检索与问题最相关的原文片段，不把全部长文直接塞给模型；
- `prompts.py::GENERATOR_SYSTEM`：强制每个可核查事实使用真实 `evidence_id`；
- `pipeline.py::generate_review`：生成后再执行一次证据约束的 self-feedback 改写。

论文：Akari Asai et al., *Synthesizing scientific literature with
retrieval-augmented language models*, Nature 650, 857–863 (2026),
DOI: 10.1038/s41586-025-10072-4.

## 2. ReportBench：Statement/Claim 级事实与引用核验

ReportBench 不只看整体文风，而是抽取报告中的引用和陈述：带引用的陈述回到原始来源
检查 faithfulness，无引用事实则独立验证 veracity。

本项目对应实现：

- `pipeline.py::extract_claims`：把综述拆成原子 Claim；
- `pipeline.py::verify_claim`：从显式引用论文中优先检索证据；
- 四级判定：`Supported`、`Partially Supported`、`Unsupported`、
  `Contradicted`；
- 当置信度低于阈值时扩大检索范围并用 Hy3 `high` 强度二次核验。

论文：Minghao Li et al., *ReportBench: Evaluating Deep Research Agents via
Academic Survey Tasks*, arXiv:2508.15804 (2025).

## 3. DeepResearch Bench：RACE 与 FACT

DeepResearch Bench 提出 RACE（参考答案、自适应评价准则、动态权重）和 FACT（事实丰度
与引用可信度）。论文的人类一致性实验表明，开放式报告评价应关注排序、相关性和人工
一致性，而不是迷信绝对分数。

本项目对应实现：

- `pipeline.py::adaptive_weights`：根据研究问题和论文集合分配七维权重；
- `citation_faithfulness`、`citation_completeness` 与 Claim 证据核验直接关联；
- 维度分数与 Claim/覆盖统计融合，避免完全依赖单一 Judge；
- JSON 报告保留每个 Claim 和证据位置，便于后续做 Spearman、排序准确率和人工一致性。

论文：Mingxuan Du et al., *DeepResearch Bench: A Comprehensive Benchmark for
Deep Research Agents*, arXiv:2506.11763 (2025).

## 4. 双向证据追溯：正确性与完整性分开测

正向 `Claim -> Evidence` 只能证明“写出来的内容是否有依据”，不能发现“重要内容是否
没写”。因此项目增加反向 `Evidence -> Review`：先从各论文抽取关键方法、结果和局限，
再检查这些 Key Evidence Points 是否在综述中得到实质覆盖。

本项目对应实现：

- `pipeline.py::extract_key_points`；
- `pipeline.py::audit_coverage`；
- 基于关键点 importance 的加权 Coverage 分数。

这是方案书在 OpenScholar、ReportBench 和 FACT 基础上的组合性设计，不声称是上述任一
论文的原样复现。

## 5. CALM：Judge 偏差与对抗校准

CALM 系统讨论位置、冗长、权威、干扰、自增强等 12 类 LLM Judge 偏差，并用原则化
扰动前后的判断一致性量化鲁棒性。

本项目对应实现：

- `adversarial.py` 构造表面扰动：篇幅膨胀、术语堆砌；
- 构造事实扰动：引用替换、数字篡改、过度结论；
- 表面扰动要求得分基本稳定（绝对变化不超过 8 分）；
- 事实扰动要求显著降分（至少下降 15 分）；
- 输出每个扰动的分数变化和总体 robustness rate。

论文：Jiayi Ye et al., *Justice or Prejudice? Quantifying Biases in
LLM-as-a-Judge*, ICLR 2025, arXiv:2410.02736.

## 6. Hy3 官方接口约束

本项目使用腾讯 Hy3 的 OpenAI Chat Completions 兼容接口：

- 模型参数：`hy3`；
- 中国大陆 TokenHub Base URL：`https://tokenhub.tencentmaas.com/v1`；
- 推理强度：`reasoning_effort` 为 `no_think`、`low` 或 `high`；
- 生成、自反馈、困难 Claim 核验和维度评估使用 `high`；
- Claim 拆分与自适应权重等较轻任务使用 `low`；
- 若 API 偶发只返回 `reasoning_content` 而 `content` 为空，客户端会记录警告并采用兼容
  回退，避免静默产生空结果。

官方资料：

- Tencent-Hunyuan/Hy3: https://github.com/Tencent-Hunyuan/Hy3
- TokenHub API: https://cloud.tencent.com/document/product/1823/130078
- 语言模型调用概览: https://cloud.tencent.com/document/product/1823/130079

## 7. Dataset v0：模型生成、难负例与人工审核

Dataset 构建参考 Self-Instruct 的“模型生成候选 + 过滤”结构，以及 HaluEval 的“生成难负例
+ 人工标注”设计。代码不会将同一个 Hy3 的生成与自审结果直接当作最终真值：所有样本
均以 `pending` 保存，必须由人工界面逐条查看原始 Evidence 后批准。

本项目对应实现：

- `literature.py`：OpenAlex、Crossref、arXiv 多源检索，DOI/标题作者去重和 OA 下载校验；
- `dataset_builder.py`：固定配方生成摘要、问答、正确论断、限定条件难例和反例；
- `dataset_review_app.py`：人工批准、修改或拒绝；
- `dataset_evaluator.py`：批量推理、七维 rubric、证据 F1、CSV 与失败归因。

论文：

- Yizhong Wang et al., *Self-Instruct: Aligning Language Models with
  Self-Generated Instructions*, ACL 2023, DOI: 10.18653/v1/2023.acl-long.754.
- Junyi Li et al., *HaluEval: A Large-Scale Hallucination Evaluation Benchmark
  for Large Language Models*, EMNLP 2023, DOI: 10.18653/v1/2023.emnlp-main.397.

完整的数据字段、样本配方和合规下载边界见 `docs/dataset_v0_method.md`。
