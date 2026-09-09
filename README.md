# Hy3Scholar

Hy3Scholar 是一个可运行的研究原型：上传多篇 PDF，围绕研究问题调用腾讯 Hy3
生成带证据编号的结构化综述，再执行 Claim 级事实/引用核验、反向内容覆盖审计、
七维可信评价和 Judge 对抗校准。

这不是“让模型凭记忆给综述打分”的薄封装。所有事实判断都以本地 PDF 中的
`evidence_id`（例如 `P001:p3:c2`）为锚点，最终输出 Claim、原始证据位置、判定、
置信度和多维得分。

项目同时提供 Dataset v0 构建闭环：Hy3 规划检索词并可调用 TokenHub 联网搜索，
OpenAlex/Crossref/arXiv 返回可追溯元数据，系统只下载明确开放获取的 PDF，再由 Hy3
基于原文 Evidence 构造标准样本、难例和反例。所有机器样本默认 `pending`，人工批准后
才进入正式批量评测。

## 方法依据

实现前核对了以下论文和官方资料；详细的论文—代码映射见
[`docs/research_basis.md`](docs/research_basis.md)。

- OpenScholar：领域检索、引用感知生成、自反馈改写。
- DeepResearch Bench：RACE 自适应维度权重与 FACT 引用准确性/有效引用评价。
- ReportBench：将长报告拆成 Statement/Claim，分别验证带引用和无引用事实。
- CALM（Justice or Prejudice?）：用表面扰动与事实扰动检查 LLM Judge 偏差。
- 腾讯 Hy3 官方仓库与 TokenHub 文档：`hy3` 模型名、OpenAI Chat Completions
  兼容接口，以及 `no_think` / `low` / `high` 推理强度。
- Self-Instruct 与 HaluEval：模型生成候选数据、自动过滤、难负例和人工审核。

## 系统流程

```text
PDFs
  -> page/section-aware chunks + evidence_id
  -> Chinese/English BM25 Evidence Database
  -> Hy3 citation-aware generation
  -> Hy3 self-feedback refinement
  -> atomic Claim extraction
  -> Claim -> Evidence verification
  -> Evidence -> Review coverage audit
  -> 7 specialized Hy3 evaluators + adaptive weights
  -> evidence-constrained aggregation
  -> adversarial Judge calibration
```

七个评价维度为：`factual_accuracy`、`citation_faithfulness`、
`citation_completeness`、`coverage`、`comparative_depth`、
`logical_coherence` 和 `academic_integrity`。

事实、引用和覆盖三类分数不是完全采用 Judge 的主观分数，而是将结构化 Judge
结果与可计算的 Claim/证据统计按 40%/60% 融合，以减弱篇幅、术语和语气偏差。

## 安装

需要 Python 3.11+。

```powershell
cd E:\HIT\腾讯犀牛鸟\code
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
Copy-Item .env.example .env
```

编辑 `.env` 或直接设置环境变量：

```powershell
$env:HY3_API_KEY = "你的 TokenHub API Key"
$env:HY3_BASE_URL = "https://tokenhub.tencentmaas.com/v1"
$env:HY3_MODEL = "hy3"
```

密钥只从环境变量读取；`.env` 已被 `.gitignore` 排除。中国大陆 TokenHub 默认
Base URL 为 `https://tokenhub.tencentmaas.com/v1`。若使用本地 vLLM/SGLang：

```powershell
$env:HY3_BASE_URL = "http://127.0.0.1:8000/v1"
$env:HY3_API_KEY = "EMPTY"
```

## 命令行

先构建本地证据库：

```powershell
hy3scholar ingest .\papers\paper1.pdf .\papers\paper2.pdf
```

记录输出的 `workspace_id`，再执行完整流程：

```powershell
hy3scholar run `
  --workspace YOUR_WORKSPACE_ID `
  --question "比较这些论文的主要方法、实验结果与局限"
```

结果保存在 `data/<workspace_id>/results/`：

- `review.md`：带证据编号的综述；
- `evaluation.md`：可读的可信评估报告；
- `latest.json`：包含 Claim、证据、覆盖和维度分数的完整结构化结果。

也可以评估已有综述：

```powershell
hy3scholar evaluate `
  --workspace YOUR_WORKSPACE_ID `
  --question "研究问题" `
  --review .\existing_review.md
```

## Dataset v0

如需调用 Hy3 联网搜索，请先在 TokenHub 的“平台管理 → 工具管理”中开通搜索服务。
默认使用 `lite`。OpenAlex 小规模查询可以匿名运行，建议申请免费 Key 并配置联系邮箱：

```dotenv
HY3_WEB_SEARCH_SOURCE=lite
OPENALEX_API_KEY=
OPENALEX_MAILTO=you@example.com
CROSSREF_MAILTO=you@example.com
HY3SCHOLAR_MAX_DOWNLOADS=10
```

先检索三个正规来源并查看候选论文：

```powershell
hy3scholar literature-search `
  --query "LLM agent memory evaluation" `
  --limit 10 `
  --output .\literature_results
```

加 `--download` 时只下载带明确开放获取位置的 PDF。下载器会阻止本机/内网 URL，手动
检查每次重定向，并验证 PDF 魔数、文件大小、页数、可提取文本和 SHA-256。遇到登录、
付费墙、验证码、Cloudflare 或出版商机器人检查时停止，不尝试绕过。

构建包含三种业务用例、难例和反例的 Dataset：

```powershell
hy3scholar dataset-build `
  --topic "LLM agent memory and self-evolving context" `
  --papers 6 `
  --cases 12 `
  --output .\dataset_v0
```

输出包括 `cases.jsonl`、`annotations.jsonl`、`papers.jsonl`、
`discovered_papers.jsonl`、`download_manifest.jsonl`、`web_sources.jsonl`、
`generation_output.json`、`generation_audit.json`、`workspace.json`、
`build_report.json`、`DATASET_CARD.md` 和通过校验的开放 PDF。

人工审核与正式批量评测：

```powershell
streamlit run dataset_review_app.py
hy3scholar dataset-validate --dataset .\dataset_v0
hy3scholar dataset-evaluate `
  --dataset .\dataset_v0 `
  --output .\evaluation_results
```

评测输出 `predictions.jsonl`、`scores.csv`、`failure_cases.jsonl` 和 `report.md`。
调试阶段可以加 `--allow-pending`，但未审核样本的结果不能作为正式结论。完整方法见
[`docs/dataset_v0_method.md`](docs/dataset_v0_method.md)。

## Web 界面与 API

Streamlit：

```powershell
streamlit run streamlit_app.py
```

FastAPI：

```powershell
uvicorn hy3scholar.api:app --reload
```

主要接口：

- `POST /workspaces`：上传一个或多个 PDF；
- `POST /workspaces/{id}/run`：生成并评价综述；
- `POST /workspaces/{id}/evaluate`：评价已有 Markdown；
- `POST /workspaces/{id}/adversarial-test`：Judge 对抗校准；
- `GET /health`：检查服务和 API Key 配置状态。

Swagger UI 位于 `http://127.0.0.1:8000/docs`。

## 验证

```powershell
pytest
```

测试使用确定性的 Fake Hy3，不消耗 API 配额。真实联调需要配置 API Key 后再运行
命令行或 Web 界面。

## 已知边界

- 当前证据检索是无需额外模型下载的中英文 BM25 基线；接口已将检索层隔离，后续可
  接入向量模型与神经重排序器。它复现 OpenScholar 的“先检索、后证据约束生成”原则，
  但不声称复现其 4500 万论文数据仓库或训练过的专用 retriever。
- 扫描型 PDF 需要先做 OCR；系统会对无可提取文本的 PDF 明确报错。
- 自动评分必须通过人工一致性实验校准后才能作为研究结论，不能把单次 LLM Judge
  分数当作金标准。
- Hy3 联网搜索只作为网页发现来源；DOI、作者、年份和 OA 状态仍由
  OpenAlex/Crossref/arXiv 结构化记录核对。
- Crossref 的 PDF 链接只有同时存在明确 Creative Commons 许可时才会进入自动下载。
- 默认批次为 5–10 篇，程序硬限制最多 20 篇，不支持整刊或无限关键词批量下载。
- 没有 API Key 时仍可完成 PDF 解析和证据库构建，但 Hy3 生成/评价会停止并给出明确
  配置错误。
