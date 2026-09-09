from __future__ import annotations

from hashlib import sha1
import json
from pathlib import Path
import re
from typing import Any

from .client import LLMClient
from .config import Settings
from .dataset_io import validate_dataset, write_jsonl
from .dataset_models import (
    DatasetBuildReport,
    DatasetCase,
    LiteratureRecord,
    SearchFailure,
    WebSearchSource,
)
from .ingest import ingest_pdfs
from .json_utils import extract_json
from .literature import (
    MultiSourceLiteratureSearcher,
    OpenAccessDownloader,
    parse_web_search_sources,
)
from .retrieval import SparseEvidenceRetriever, format_evidence_context
from .schemas import RetrievedEvidence, Workspace


SEARCH_PLAN_SYSTEM = """[TASK:dataset_search_plan]
你为学术评测集规划检索词。只输出 JSON：
{"queries":["...","...","..."]}
给出 3 个适合 OpenAlex、Crossref、arXiv 的英文检索式；每个检索式应聚焦论文主题、
核心方法或评测，不得生成 URL，不得声称找到具体论文。"""

WEB_DISCOVERY_SYSTEM = """[TASK:dataset_web_discovery]
联网查找与主题直接相关的学术论文。优先返回 DOI、arXiv、PubMed Central、出版社或
大学机构仓储页面；排除博客、新闻、聚合转载和未说明来源的 PDF。简要列出标题、年份、
稳定标识符和为什么相关。不要把网页搜索结果当作已经人工核验的事实。"""

CASE_GENERATION_SYSTEM = """[TASK:dataset_case_generation]
你正在构造学术论文阅读助手的 Dataset v0。只能根据提供的 Evidence 生成样本，不得使用
模型记忆补事实。只输出 JSON：{"cases":[...]}
每个 case 必须包含：recipe_id、instruction、reference_answer、gold_evidence_ids、
claim_label（非论断核对任务为 null）、challenge_tags、generation_notes。
gold_evidence_ids 必须逐字取自 Evidence；reference_answer 中每个可核查结论都应能被这些
证据支持。hard 样本应考查限定条件、相似方法、跨段整合或数字边界；counterexample 必须
是证据不支持或直接矛盾的论断，参考答案要解释错误点。不要生成纯文风难例。"""

CASE_AUDIT_SYSTEM = """[TASK:dataset_case_audit]
逐项审核合成评测样本是否忠实于 Evidence，检查引用 ID、限定条件、数字、任务标签和反例
标签。只输出 JSON：
{"audits":[{"case_id":"...","accepted":true,"issues":[],
"corrected_reference_answer":null,"corrected_gold_evidence_ids":null,
"corrected_claim_label":null}]}
必须原样复制输入中的 case_id，并为每个输入样本返回且只返回一条 audit；无法确定时
accepted=false；不得用模型记忆替样本背书。"""


def _normalize_audits(payload: Any) -> list[dict[str, Any]]:
    """Accept small, common JSON-shape deviations without weakening the audit."""

    raw: Any = payload
    if isinstance(payload, dict):
        for key in ("audits", "results", "cases", "items"):
            if key in payload:
                raw = payload[key]
                break
    if isinstance(raw, dict):
        expanded: list[dict[str, Any]] = []
        for key, value in raw.items():
            if isinstance(value, dict):
                expanded.append({"case_id": key, **value})
        raw = expanded
    if not isinstance(raw, list):
        return []

    normalized: list[dict[str, Any]] = []
    for value in raw:
        if not isinstance(value, dict):
            continue
        item = dict(value)
        case_id = item.get("case_id") or item.get("id")
        if not case_id:
            continue
        item["case_id"] = str(case_id)
        issues = item.get("issues", [])
        if isinstance(issues, str):
            item["issues"] = [issues] if issues.strip() else []
        elif not isinstance(issues, list):
            item["issues"] = []
        accepted = item.get("accepted")
        if isinstance(accepted, str):
            lowered = accepted.strip().lower()
            if lowered in {"true", "yes", "accepted", "approved", "pass", "passed"}:
                item["accepted"] = True
            elif lowered in {"false", "no", "rejected", "fail", "failed"}:
                item["accepted"] = False
        if "accepted" not in item:
            if "pass" in item:
                item["accepted"] = bool(item["pass"])
            else:
                status = str(item.get("status") or "").strip().lower()
                if status in {"accepted", "approved", "ok", "pass", "passed"}:
                    item["accepted"] = True
                elif status in {"rejected", "fail", "failed", "needs_revision"}:
                    item["accepted"] = False
        normalized.append(item)
    return normalized


def _normalize_generated_cases(payload: Any) -> list[dict[str, Any]]:
    """Extract generated cases from common model JSON wrappers."""

    raw: Any = payload
    if isinstance(payload, dict):
        for key in ("cases", "results", "items", "dataset", "data"):
            if key in payload:
                raw = payload[key]
                break
    if isinstance(raw, dict):
        for key in ("cases", "results", "items"):
            if key in raw:
                raw = raw[key]
                break
    if isinstance(raw, dict):
        expanded: list[dict[str, Any]] = []
        for key, value in raw.items():
            if isinstance(value, dict):
                expanded.append({"recipe_id": key, **value})
        raw = expanded
    if not isinstance(raw, list):
        return []
    return [dict(item) for item in raw if isinstance(item, dict)]


def _recipes(case_count: int) -> list[dict[str, str]]:
    if case_count < 7:
        raise ValueError("Dataset v0 至少需要 7 个样本，才能覆盖三类任务、难例和反例")
    base = [
        ("structured_summary", "standard", "概括研究问题、方法、结果和局限"),
        ("structured_summary", "hard", "跨段整合并保留适用范围和限定条件"),
        ("evidence_qa", "standard", "可由一个或两个证据块直接回答"),
        ("evidence_qa", "hard", "需要跨论文或跨段比较，避免混淆相似术语"),
        ("claim_check", "standard", "Supported：构造一个被原文直接支持的论断"),
        ("claim_check", "hard", "Partially Supported：保留一部分事实但篡改限定条件"),
        ("claim_check", "counterexample", "Unsupported 或 Contradicted：构造可解释的反例"),
    ]
    extra = [
        ("evidence_qa", "hard", "数字、实验设置或边界条件"),
        ("claim_check", "counterexample", "夸大因果、范围或跨论文错配"),
        ("structured_summary", "standard", "聚焦单篇论文的结构化摘要"),
        ("claim_check", "standard", "Supported：精确保留原文限定语"),
        ("evidence_qa", "standard", "证据不足时应明确拒绝推断"),
    ]
    entries = (base + extra * ((case_count - len(base) + len(extra) - 1) // len(extra)))[
        :case_count
    ]
    return [
        {
            "recipe_id": f"R{index:03d}",
            "task_type": task_type,
            "difficulty": difficulty,
            "requirement": requirement,
        }
        for index, (task_type, difficulty, requirement) in enumerate(entries, 1)
    ]


def _evidence_context(workspace: Workspace, topic: str, max_items: int = 24) -> str:
    retriever = SparseEvidenceRetriever(workspace.chunks)
    selected = retriever.search(
        topic, top_k=min(max_items, len(workspace.chunks)), max_per_paper=6
    )
    seen_papers = {item.chunk.paper_id for item in selected}
    for paper in workspace.papers:
        if paper.paper_id in seen_papers:
            continue
        chunk = next(item for item in workspace.chunks if item.paper_id == paper.paper_id)
        selected.append(RetrievedEvidence(chunk=chunk, score=0.0))
    unique: dict[str, RetrievedEvidence] = {
        item.chunk.evidence_id: item for item in selected
    }
    return format_evidence_context(list(unique.values()), max_chars=52_000)


class DatasetCaseGenerator:
    def __init__(self, client: LLMClient, settings: Settings) -> None:
        self.client = client
        self.settings = settings
        self.generation_payload: Any = None
        self.audit_payload: Any = None

    def _call_json(
        self,
        *,
        system: str,
        user: str,
        max_tokens: int,
        reasoning_effort: str = "high",
    ) -> Any:
        response = self.client.chat(
            system=system,
            user=user,
            reasoning_effort=reasoning_effort,
            temperature=0.1,
            max_tokens=max_tokens,
        )
        return extract_json(response.content)

    def generate(
        self, workspace: Workspace, topic: str, *, case_count: int
    ) -> list[DatasetCase]:
        recipes = _recipes(case_count)
        context = _evidence_context(workspace, topic)
        user = (
            f"TOPIC={topic}\n\nRECIPES=\n"
            f"{json.dumps(recipes, ensure_ascii=False)}\n\nEVIDENCE=\n{context}"
        )
        attempts: list[Any] = []
        raw_cases: list[dict[str, Any]] = []
        selected_attempt: int | None = None
        for attempt in range(2):
            payload = self._call_json(
                system=CASE_GENERATION_SYSTEM,
                user=user,
                max_tokens=9000,
                reasoning_effort="low",
            )
            attempts.append(payload)
            candidate = _normalize_generated_cases(payload)
            if len(candidate) > len(raw_cases):
                raw_cases = candidate
                selected_attempt = attempt + 1
            if len(raw_cases) >= len(recipes):
                break
        self.generation_payload = {
            "selected_attempt": selected_attempt,
            "attempts": attempts,
        }
        if len(raw_cases) < len(recipes):
            raise RuntimeError(
                f"Hy3 只生成了 {len(raw_cases)} 个样本，期望 {len(recipes)} 个；"
                "请缩小论文数量或重试"
            )
        valid_evidence = {chunk.evidence_id for chunk in workspace.chunks}
        valid_papers = {paper.paper_id for paper in workspace.papers}
        by_recipe = {
            str(item.get("recipe_id")): item
            for item in raw_cases
            if isinstance(item, dict)
        }
        cases: list[DatasetCase] = []
        for index, recipe in enumerate(recipes, 1):
            item = by_recipe.get(recipe["recipe_id"]) or raw_cases[index - 1]
            evidence_ids = [
                str(value)
                for value in item.get("gold_evidence_ids", [])
                if str(value) in valid_evidence
            ]
            issues: list[str] = []
            if not evidence_ids:
                issues.append("没有有效的 gold_evidence_ids")
                evidence_ids = [workspace.chunks[(index - 1) % len(workspace.chunks)].evidence_id]
            paper_ids = list(
                dict.fromkeys(value.split(":", 1)[0] for value in evidence_ids)
            )
            paper_ids = [value for value in paper_ids if value in valid_papers]
            instruction = str(item.get("instruction") or "").strip()
            reference = str(item.get("reference_answer") or "").strip()
            if len(instruction) < 8:
                issues.append("instruction 过短")
            if len(reference) < 20:
                issues.append("reference_answer 过短")
            claim_label = item.get("claim_label")
            if recipe["task_type"] != "claim_check":
                claim_label = None
            elif recipe["difficulty"] == "standard":
                claim_label = "Supported"
            elif recipe["difficulty"] == "hard":
                claim_label = "Partially Supported"
            elif claim_label not in {"Unsupported", "Contradicted"}:
                claim_label = "Unsupported"
            cases.append(
                DatasetCase(
                    case_id=f"V0-{index:04d}",
                    topic=topic,
                    task_type=recipe["task_type"],
                    difficulty=recipe["difficulty"],
                    instruction=instruction or f"请完成 {recipe['task_type']} 任务。",
                    reference_answer=reference or "自动生成失败，必须由人工依据证据补充参考答案。",
                    paper_ids=paper_ids or [workspace.papers[0].paper_id],
                    gold_evidence_ids=evidence_ids,
                    claim_label=claim_label,
                    challenge_tags=[str(value) for value in item.get("challenge_tags", [])],
                    generation_notes=str(item.get("generation_notes") or recipe["requirement"]),
                    auto_validation_status="needs_revision" if issues else "passed",
                    auto_validation_issues=issues,
                    human_review_status="pending",
                    generated_by=self.settings.model,
                )
            )
        return self._audit(cases, context)

    def _audit(self, cases: list[DatasetCase], context: str) -> list[DatasetCase]:
        user = (
            "CASES=\n"
            + json.dumps(
                [case.model_dump(mode="json") for case in cases],
                ensure_ascii=False,
            )
            + "\n\nEVIDENCE=\n"
            + context
        )
        attempts: list[Any] = []
        audits: list[dict[str, Any]] = []
        expected_ids = {case.case_id for case in cases}
        selected_attempt: int | None = None
        for attempt in range(2):
            payload = self._call_json(
                system=CASE_AUDIT_SYSTEM,
                user=user,
                max_tokens=6000,
                reasoning_effort="low",
            )
            attempts.append(payload)
            candidate = _normalize_audits(payload)
            returned_ids = {str(item.get("case_id")) for item in candidate}
            if len(returned_ids & expected_ids) > len(
                {str(item.get("case_id")) for item in audits} & expected_ids
            ):
                audits = candidate
                selected_attempt = attempt + 1
            if expected_ids <= returned_ids:
                break
        self.audit_payload = {
            "selected_attempt": selected_attempt,
            "attempts": attempts,
        }
        by_id = {
            str(item.get("case_id")): item
            for item in audits
            if isinstance(item, dict)
        }
        valid_ids = set(
            re.findall(r"P\d{3}:p\d+:c\d+", context)
        )
        output: list[DatasetCase] = []
        for case in cases:
            audit = by_id.get(case.case_id)
            if not audit:
                output.append(
                    case.model_copy(
                        update={
                            "auto_validation_status": "needs_revision",
                            "auto_validation_issues": case.auto_validation_issues
                            + ["Hy3 自审缺少该样本结果"],
                        }
                    )
                )
                continue
            issues = case.auto_validation_issues + [
                str(value) for value in audit.get("issues", [])
            ]
            if not audit.get("accepted") and not audit.get("issues"):
                issues.append("Hy3 自审未通过，但未给出具体原因")
            update: dict[str, Any] = {
                "auto_validation_status": (
                    "passed" if audit.get("accepted") and not issues else "needs_revision"
                ),
                "auto_validation_issues": issues,
            }
            corrected_answer = audit.get("corrected_reference_answer")
            if corrected_answer:
                update["reference_answer"] = str(corrected_answer)
            corrected_ids = audit.get("corrected_gold_evidence_ids")
            if corrected_ids:
                filtered = [str(value) for value in corrected_ids if str(value) in valid_ids]
                if filtered:
                    update["gold_evidence_ids"] = filtered
                    update["paper_ids"] = list(
                        dict.fromkeys(value.split(":", 1)[0] for value in filtered)
                    )
            corrected_label = audit.get("corrected_claim_label")
            if case.task_type == "claim_check" and corrected_label in {
                "Supported", "Partially Supported", "Unsupported", "Contradicted"
            }:
                update["claim_label"] = corrected_label
            output.append(case.model_copy(update=update))
        return output


class DatasetV0Builder:
    def __init__(
        self,
        settings: Settings,
        client: LLMClient,
        *,
        searcher: MultiSourceLiteratureSearcher | None = None,
        downloader: OpenAccessDownloader | None = None,
    ) -> None:
        self.settings = settings
        self.client = client
        self.searcher = searcher or MultiSourceLiteratureSearcher(settings)
        self.downloader = downloader or OpenAccessDownloader(settings)

    def close(self) -> None:
        for component in (self.searcher, self.downloader):
            close = getattr(component, "close", None)
            if close:
                close()

    def plan_queries(self, topic: str) -> list[str]:
        try:
            response = self.client.chat(
                system=SEARCH_PLAN_SYSTEM,
                user=f"研究主题：{topic}",
                reasoning_effort="low",
                temperature=0.1,
                max_tokens=800,
            )
            payload = extract_json(response.content)
            queries = [
                str(value).strip()
                for value in (payload.get("queries", []) if isinstance(payload, dict) else [])
                if str(value).strip()
            ]
            return list(dict.fromkeys(queries))[:3] or [topic]
        except Exception:
            return [topic]

    def web_discover(self, topic: str) -> tuple[list[WebSearchSource], SearchFailure | None]:
        try:
            response = self.client.chat(
                system=WEB_DISCOVERY_SYSTEM,
                user=f"研究主题：{topic}",
                reasoning_effort="low",
                temperature=0.1,
                max_tokens=1800,
                web_search=True,
                search_source=self.settings.web_search_source,
            )
            sources = parse_web_search_sources(response.search_results)
            if not sources:
                return [], SearchFailure(
                    source="hy3_web_search",
                    query=topic,
                    error=(
                        "TokenHub 接受了联网搜索请求，但本次未返回 search_results；"
                        "请检查工具管理中的联网搜索服务，或重试"
                    ),
                )
            return sources, None
        except Exception as exc:
            return [], SearchFailure(source="hy3_web_search", query=topic, error=str(exc))

    def build(
        self,
        *,
        topic: str,
        output_dir: Path,
        paper_count: int = 6,
        case_count: int = 12,
        use_hy3_web: bool = True,
    ) -> DatasetBuildReport:
        if paper_count < 2 or paper_count > self.settings.max_downloads:
            raise ValueError(
                f"paper_count 必须在 2 到 {self.settings.max_downloads} 之间"
            )
        if (output_dir / "cases.jsonl").exists():
            raise FileExistsError(
                f"{output_dir} 已包含 Dataset；为避免覆盖人工标注，请换一个输出目录"
            )
        output_dir.mkdir(parents=True, exist_ok=True)
        queries = self.plan_queries(topic)
        web_sources: list[WebSearchSource] = []
        web_failure: SearchFailure | None = None
        if use_hy3_web:
            web_sources, web_failure = self.web_discover(topic)
        write_jsonl(output_dir / "web_sources.jsonl", web_sources)
        search_report = self.searcher.search(queries, limit_per_source=max(4, paper_count))
        failures = list(search_report.failures)
        if web_failure:
            failures.append(web_failure)
        write_jsonl(output_dir / "discovered_papers.jsonl", search_report.records)
        oa_candidates = [
            record
            for record in search_report.records
            if record.is_open_access and record.pdf_url
        ]
        manifests = []
        downloaded_records: list[LiteratureRecord] = []
        pdf_paths: list[Path] = []
        for record in oa_candidates:
            if len(pdf_paths) >= paper_count:
                break
            entry = self.downloader.download(record, output_dir / "papers")
            manifests.append(entry)
            if entry.status == "open_access_downloaded" and entry.local_path:
                pdf_paths.append(Path(entry.local_path))
                downloaded_records.append(record)
        write_jsonl(output_dir / "download_manifest.jsonl", manifests)
        if len(pdf_paths) < 2:
            raise RuntimeError(
                f"仅获得 {len(pdf_paths)} 篇通过校验的开放获取 PDF；"
                f"已把 {len(manifests)} 条下载状态写入 download_manifest.jsonl"
            )
        workspace_id = "dataset-" + sha1(topic.encode("utf-8")).hexdigest()[:10]
        workspace = ingest_pdfs(
            pdf_paths,
            workspace_id=workspace_id,
            chunk_chars=self.settings.chunk_chars,
            overlap_chars=self.settings.chunk_overlap_chars,
        )
        (output_dir / "workspace.json").write_text(
            workspace.model_dump_json(indent=2), encoding="utf-8"
        )
        papers: list[LiteratureRecord] = []
        updated_manifests = []
        for index, (record, manifest) in enumerate(
            zip(downloaded_records, [m for m in manifests if m.status == "open_access_downloaded"]),
            1,
        ):
            paper_id = f"P{index:03d}"
            papers.append(record.model_copy(update={"paper_id": paper_id}))
            updated_manifests.append(manifest.model_copy(update={"paper_id": paper_id}))
        remaining = [m for m in manifests if m.status != "open_access_downloaded"]
        write_jsonl(output_dir / "papers.jsonl", papers)
        write_jsonl(output_dir / "download_manifest.jsonl", updated_manifests + remaining)
        generator = DatasetCaseGenerator(self.client, self.settings)
        cases = generator.generate(workspace, topic, case_count=case_count)
        (output_dir / "generation_output.json").write_text(
            json.dumps(generator.generation_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (output_dir / "generation_audit.json").write_text(
            json.dumps(generator.audit_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        write_jsonl(output_dir / "cases.jsonl", cases)
        write_jsonl(output_dir / "annotations.jsonl", [])
        self._write_dataset_card(output_dir, topic, workspace, cases)
        validation = validate_dataset(output_dir)
        report = DatasetBuildReport(
            topic=topic,
            queries=queries,
            discovered_records=len(search_report.records),
            oa_candidates=len(oa_candidates),
            downloaded_papers=len(pdf_paths),
            generated_cases=len(cases),
            dataset_dir=str(output_dir.resolve()),
            search_failures=failures,
            validation=validation,
        )
        (output_dir / "build_report.json").write_text(
            report.model_dump_json(indent=2), encoding="utf-8"
        )
        return report

    @staticmethod
    def _write_dataset_card(
        output_dir: Path, topic: str, workspace: Workspace, cases: list[DatasetCase]
    ) -> None:
        counts: dict[str, int] = {}
        for case in cases:
            key = f"{case.task_type}/{case.difficulty}"
            counts[key] = counts.get(key, 0) + 1
        distribution = "\n".join(f"- `{key}`: {value}" for key, value in sorted(counts.items()))
        text = f"""# Hy3Scholar Dataset v0

- 主题：{topic}
- 论文数：{len(workspace.papers)}
- 样本数：{len(cases)}
- 当前状态：**机器生成，等待人工审核；不是金标准**

## 样本分布

{distribution}

## 构建方法

1. Hy3 生成 3 个检索式，并可使用 TokenHub 联网搜索扩展候选来源。
2. OpenAlex、Crossref、arXiv 提供结构化元数据，DOI 优先、标题与第一作者兜底去重。
3. 仅下载明确开放获取的 PDF；验证 `%PDF`、页数、可提取文本、大小和 SHA-256。
4. Hy3 基于带页码的 Evidence 生成摘要、证据问答和论断核对样本。
5. 固定配方覆盖标准样本、限定条件难例和 Unsupported/Contradicted 反例。
6. Hy3 自审只作为过滤信号；所有样本仍保持 `pending`，必须在人工审核页面批准。

## 文件

- `cases.jsonl`：机器生成样本；
- `workspace.json`：原文 Evidence 与稳定证据 ID；
- `papers.jsonl`：论文元数据；
- `download_manifest.jsonl`：下载、许可和校验状态；
- `web_sources.jsonl`：Hy3 联网搜索返回的网页来源；
- `generation_output.json`：Hy3 样本生成的原始结构化结果；
- `generation_audit.json`：Hy3 自审原始结构化结果；
- `annotations.jsonl`：人工审核覆盖层；
- `build_report.json`：构建与结构校验报告。

## 论文依据

- Self-Instruct (Wang et al., ACL 2023)：模型生成候选指令后执行过滤；
- HaluEval (Li et al., EMNLP 2023)：模型生成难负例并进行人工标注；
- ReportBench (Li et al., 2025)：原子 Claim 与原始证据核验；
- DeepResearch Bench (Du et al., 2025)：多维 rubric、事实与引用联合评价；
- CALM (Ye et al., ICLR 2025)：表面扰动和事实扰动分离校准。

## 使用限制

机器生成答案会继承 Hy3 偏差，不能绕过人工复核。下载器不绕过付费墙、登录、验证码、
Cloudflare 或出版商反自动化检查，也不下载补充材料。
"""
        (output_dir / "DATASET_CARD.md").write_text(text, encoding="utf-8")
