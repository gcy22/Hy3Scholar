from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import re
from statistics import mean
from typing import Any

from .client import LLMClient
from .config import Settings
from .json_utils import extract_json
from .prompts import (
    CLAIM_SYSTEM,
    COVERAGE_SYSTEM,
    DIMENSION_SYSTEM,
    GENERATOR_SYSTEM,
    KEY_POINT_SYSTEM,
    REFINER_SYSTEM,
    VERIFY_SYSTEM,
    WEIGHT_SYSTEM,
)
from .retrieval import SparseEvidenceRetriever, format_evidence_context
from .schemas import (
    Claim,
    ClaimEvaluation,
    CoverageEvaluation,
    DimensionEvaluation,
    EvaluationReport,
    KeyEvidencePoint,
    PipelineResult,
    ReviewDraft,
    Workspace,
)


DIMENSION_RUBRICS: dict[str, str] = {
    "factual_accuracy": "方法、实验、数字、因果和结论是否与原文证据一致。",
    "citation_faithfulness": "每个行内引用是否直接支持相邻 Claim，是否存在错引或过度推断。",
    "citation_completeness": "所有需要外部证据的学术 Claim 是否都有可追溯引用。",
    "coverage": "是否覆盖论文集合中的关键方法、主要结果、分歧和重要局限。",
    "comparative_depth": "是否有跨论文的对照、演化关系、适用边界与冲突分析，而非摘要拼接。",
    "logical_coherence": "结构、论点组织、术语与前后推理是否一致。",
    "academic_integrity": "是否存在虚构论文/方法/数据、错误归因、伪造引用或隐藏不确定性。",
}

DEFAULT_WEIGHTS = {
    "factual_accuracy": 0.20,
    "citation_faithfulness": 0.17,
    "citation_completeness": 0.12,
    "coverage": 0.17,
    "comparative_depth": 0.12,
    "logical_coherence": 0.10,
    "academic_integrity": 0.12,
}

_EVIDENCE_ID_RE = re.compile(r"P\d{3}:p\d+:c\d+")
_SENTENCE_RE = re.compile(r"(?<=[。！？.!?])\s+")


def _clamp(value: Any, low: float, high: float, default: float) -> float:
    try:
        return max(low, min(high, float(value)))
    except (TypeError, ValueError):
        return default


def _as_strings(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [str(item) for item in value]
    return []


class Hy3ScholarPipeline:
    def __init__(
        self,
        *,
        client: LLMClient,
        workspace: Workspace,
        settings: Settings | None = None,
    ) -> None:
        self.client = client
        self.workspace = workspace
        self.settings = settings or Settings()
        self.retriever = SparseEvidenceRetriever(workspace.chunks)
        self.warnings: list[str] = []

    def _chat(
        self,
        *,
        system: str,
        user: str,
        reasoning_effort: str = "high",
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        response = self.client.chat(
            system=system,
            user=user,
            reasoning_effort=reasoning_effort,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        if response.warning and response.warning not in self.warnings:
            self.warnings.append(response.warning)
        return response.content

    def _chat_json(self, **kwargs: Any) -> Any:
        return extract_json(self._chat(**kwargs))

    def generate_review(
        self, question: str, *, top_k: int | None = None, refine: bool = True
    ) -> ReviewDraft:
        top_k = top_k or self.settings.default_top_k
        evidence = self.retriever.search(question, top_k=top_k)
        if not evidence:
            raise ValueError("没有检索到与问题相关的证据，请换一个更具体的问题。")
        context = format_evidence_context(
            evidence, max_chars=self.settings.max_context_chars
        )
        prompt = f"研究问题：\n{question}\n\n原始论文证据：\n{context}"
        draft = self._chat(
            system=GENERATOR_SYSTEM,
            user=prompt,
            reasoning_effort="high",
            temperature=0.2,
        ).strip()
        refined = False
        if refine:
            draft = self._chat(
                system=REFINER_SYSTEM,
                user=(
                    f"研究问题：\n{question}\n\n证据：\n{context}\n\n"
                    f"待审查草稿：\n{draft}"
                ),
                reasoning_effort="high",
                temperature=0.1,
            ).strip()
            refined = True
        evidence_ids = sorted(set(_EVIDENCE_ID_RE.findall(draft)))
        valid_ids = {chunk.evidence_id for chunk in self.workspace.chunks}
        invalid_ids = [item for item in evidence_ids if item not in valid_ids]
        if invalid_ids:
            self.warnings.append(
                "生成文本包含不存在的 evidence_id：" + ", ".join(invalid_ids)
            )
        return ReviewDraft(
            question=question,
            markdown=draft,
            evidence_ids=[item for item in evidence_ids if item in valid_ids],
            refined=refined,
        )

    def extract_claims(self, review: ReviewDraft) -> list[Claim]:
        try:
            payload = self._chat_json(
                system=CLAIM_SYSTEM,
                user=review.markdown,
                reasoning_effort="low",
                temperature=0,
            )
            raw_claims = payload.get("claims", []) if isinstance(payload, dict) else []
            claims: list[Claim] = []
            for index, item in enumerate(
                raw_claims[: self.settings.max_claims], 1
            ):
                if not isinstance(item, dict) or not str(item.get("text", "")).strip():
                    continue
                item = dict(item)
                item["claim_id"] = f"C{index:03d}"
                item["citations"] = sorted(
                    set(
                        _EVIDENCE_ID_RE.findall(
                            " ".join(_as_strings(item.get("citations")))
                        )
                    )
                )
                item["importance"] = _clamp(item.get("importance"), 0, 1, 0.5)
                claims.append(Claim.model_validate(item))
            if claims:
                return claims
        except (ValueError, TypeError):
            self.warnings.append("Claim JSON 解析失败，已使用保守句子级回退。")
        return self._fallback_claims(review.markdown)

    def _fallback_claims(self, markdown: str) -> list[Claim]:
        plain = re.sub(r"^#+\s+.*$", "", markdown, flags=re.M)
        sentences = [item.strip() for item in _SENTENCE_RE.split(plain) if item.strip()]
        claims: list[Claim] = []
        for sentence in sentences:
            if len(sentence) < 20:
                continue
            citations = sorted(set(_EVIDENCE_ID_RE.findall(sentence)))
            clean = re.sub(r"\[(?:P\d{3}:p\d+:c\d+)\]", "", sentence).strip()
            claims.append(
                Claim(
                    claim_id=f"C{len(claims) + 1:03d}",
                    text=clean,
                    citations=citations,
                    importance=0.5,
                )
            )
        return claims

    def verify_claim(self, claim: Claim) -> ClaimEvaluation:
        cited_papers = {citation.split(":", 1)[0] for citation in claim.citations}
        evidence = self.retriever.search(
            claim.text,
            top_k=6,
            paper_ids=cited_papers or None,
            max_per_paper=6,
        )
        if not evidence and cited_papers:
            evidence = self.retriever.search(claim.text, top_k=6, max_per_paper=6)
        context = format_evidence_context(evidence, max_chars=18_000)
        item = self._verify_payload(claim, context, effort="low")
        escalated = False
        if item.confidence < self.settings.uncertainty_threshold:
            broader = self.retriever.search(
                claim.text,
                top_k=12,
                paper_ids=cited_papers or None,
                max_per_paper=8,
            )
            deep_context = format_evidence_context(broader, max_chars=36_000)
            item = self._verify_payload(claim, deep_context, effort="high")
            escalated = True
        return item.model_copy(update={"claim_id": claim.claim_id, "escalated": escalated})

    def _verify_payload(
        self, claim: Claim, context: str, *, effort: str
    ) -> ClaimEvaluation:
        if not context:
            return ClaimEvaluation(
                claim_id=claim.claim_id,
                status="Unsupported",
                score=0,
                confidence=1,
                evidence_ids=[],
                rationale="未检索到可用于验证该 Claim 的原始论文证据。",
            )
        payload = self._chat_json(
            system=VERIFY_SYSTEM,
            user=(
                f"Claim ID: {claim.claim_id}\nClaim: {claim.text}\n"
                f"正文显式引用: {json.dumps(claim.citations, ensure_ascii=False)}\n\n"
                f"候选原始证据：\n{context}"
            ),
            reasoning_effort=effort,
            temperature=0,
        )
        if not isinstance(payload, dict):
            raise ValueError("Claim 核验结果不是 JSON 对象")
        valid_statuses = {
            "Supported",
            "Partially Supported",
            "Unsupported",
            "Contradicted",
        }
        status = payload.get("status")
        if status not in valid_statuses:
            status = "Unsupported"
        return ClaimEvaluation(
            claim_id=claim.claim_id,
            status=status,
            score=_clamp(payload.get("score"), 0, 1, 0),
            confidence=_clamp(payload.get("confidence"), 0, 1, 0.5),
            evidence_ids=sorted(
                set(
                    _EVIDENCE_ID_RE.findall(
                        " ".join(_as_strings(payload.get("evidence_ids")))
                    )
                )
            ),
            rationale=str(payload.get("rationale", "未提供理由")),
        )

    def extract_key_points(self, question: str) -> list[KeyEvidencePoint]:
        selected = []
        for paper in self.workspace.papers:
            paper_evidence = self.retriever.search(
                question,
                top_k=4,
                paper_ids={paper.paper_id},
                max_per_paper=4,
            )
            if not paper_evidence:
                paper_evidence = [
                    chunk
                    for chunk in self.workspace.chunks
                    if chunk.paper_id == paper.paper_id
                ][:3]
            selected.extend(paper_evidence)
        context = format_evidence_context(selected, max_chars=48_000)
        payload = self._chat_json(
            system=KEY_POINT_SYSTEM,
            user=f"研究问题：{question}\n\n论文证据：\n{context}",
            reasoning_effort="high",
            temperature=0,
        )
        raw_points = payload.get("key_points", []) if isinstance(payload, dict) else []
        points: list[KeyEvidencePoint] = []
        valid_papers = {paper.paper_id for paper in self.workspace.papers}
        for index, item in enumerate(
            raw_points[: self.settings.max_key_points], 1
        ):
            if (
                not isinstance(item, dict)
                or item.get("paper_id") not in valid_papers
                or not str(item.get("text", "")).strip()
            ):
                continue
            item = dict(item)
            item["key_point_id"] = f"K{index:03d}"
            item["importance"] = _clamp(item.get("importance"), 0, 1, 0.5)
            item["evidence_ids"] = sorted(
                set(
                    _EVIDENCE_ID_RE.findall(
                        " ".join(_as_strings(item.get("evidence_ids")))
                    )
                )
            )
            points.append(KeyEvidencePoint.model_validate(item))
        return points

    def audit_coverage(
        self,
        review: ReviewDraft,
        claims: list[Claim],
        points: list[KeyEvidencePoint],
    ) -> list[CoverageEvaluation]:
        if not points:
            return []
        payload = self._chat_json(
            system=COVERAGE_SYSTEM,
            user=(
                "关键点：\n"
                + json.dumps(
                    [point.model_dump() for point in points],
                    ensure_ascii=False,
                )
                + "\n\n综述 Claims：\n"
                + json.dumps(
                    [claim.model_dump() for claim in claims],
                    ensure_ascii=False,
                )
                + "\n\n综述全文：\n"
                + review.markdown
            ),
            reasoning_effort="high",
            temperature=0,
        )
        raw_items = payload.get("coverage", []) if isinstance(payload, dict) else []
        by_id = {
            str(item.get("key_point_id")): item
            for item in raw_items
            if isinstance(item, dict)
        }
        output: list[CoverageEvaluation] = []
        for point in points:
            item = by_id.get(point.key_point_id, {})
            score = _clamp(item.get("score"), 0, 1, 0)
            output.append(
                CoverageEvaluation(
                    key_point_id=point.key_point_id,
                    covered=bool(item.get("covered", score >= 0.6)),
                    score=score,
                    matched_claim_ids=[
                        value
                        for value in item.get("matched_claim_ids", [])
                        if isinstance(value, str) and value.startswith("C")
                    ],
                    rationale=str(item.get("rationale", "模型未返回该关键点的覆盖判断")),
                )
            )
        return output

    def _evaluate_dimension(
        self,
        dimension: str,
        review: ReviewDraft,
        claims: list[Claim],
        claim_evaluations: list[ClaimEvaluation],
        coverage: list[CoverageEvaluation],
    ) -> DimensionEvaluation:
        payload = self._chat_json(
            system=DIMENSION_SYSTEM,
            user=(
                f"DIMENSION={dimension}\nRUBRIC={DIMENSION_RUBRICS[dimension]}\n\n"
                f"综述：\n{review.markdown}\n\n"
                "Claims 与正向证据核验：\n"
                + json.dumps(
                    {
                        "claims": [item.model_dump() for item in claims],
                        "verification": [
                            item.model_dump() for item in claim_evaluations
                        ],
                        "coverage": [item.model_dump() for item in coverage],
                    },
                    ensure_ascii=False,
                )
            ),
            reasoning_effort="high",
            temperature=0,
        )
        if not isinstance(payload, dict):
            payload = {}
        return DimensionEvaluation(
            dimension=dimension,
            score=_clamp(payload.get("score"), 0, 100, 0),
            confidence=_clamp(payload.get("confidence"), 0, 1, 0.5),
            problems=_as_strings(payload.get("problems")),
            recommendations=_as_strings(payload.get("recommendations")),
            evidence_ids=sorted(
                set(
                    _EVIDENCE_ID_RE.findall(
                        " ".join(_as_strings(payload.get("evidence_ids")))
                    )
                )
            ),
        )

    def evaluate_dimensions(
        self,
        review: ReviewDraft,
        claims: list[Claim],
        claim_evaluations: list[ClaimEvaluation],
        coverage: list[CoverageEvaluation],
    ) -> list[DimensionEvaluation]:
        output: dict[str, DimensionEvaluation] = {}
        with ThreadPoolExecutor(
            max_workers=max(1, self.settings.max_parallel_evaluators)
        ) as executor:
            futures = {
                executor.submit(
                    self._evaluate_dimension,
                    dimension,
                    review,
                    claims,
                    claim_evaluations,
                    coverage,
                ): dimension
                for dimension in DIMENSION_RUBRICS
            }
            for future in as_completed(futures):
                dimension = futures[future]
                try:
                    output[dimension] = future.result()
                except Exception as exc:
                    self.warnings.append(f"{dimension} 评估器失败：{exc}")
                    output[dimension] = DimensionEvaluation(
                        dimension=dimension,
                        score=0,
                        confidence=0,
                        problems=["评估器调用失败"],
                        recommendations=["检查 Hy3 API 后重试"],
                    )
        return [output[name] for name in DIMENSION_RUBRICS]

    def adaptive_weights(self, question: str) -> dict[str, float]:
        try:
            payload = self._chat_json(
                system=WEIGHT_SYSTEM,
                user=(
                    f"研究问题：{question}\n论文数量：{len(self.workspace.papers)}\n"
                    f"论文标题：{[paper.title for paper in self.workspace.papers]}"
                ),
                reasoning_effort="low",
                temperature=0,
            )
            raw = payload.get("weights", {}) if isinstance(payload, dict) else {}
            weights = {
                name: max(0.0, float(raw.get(name, 0)))
                for name in DIMENSION_RUBRICS
            }
            total = sum(weights.values())
            if total <= 0:
                raise ValueError("权重和为 0")
            return {name: value / total for name, value in weights.items()}
        except (ValueError, TypeError):
            self.warnings.append("自适应权重无效，已使用预设权重。")
            return dict(DEFAULT_WEIGHTS)

    def _empirical_scores(
        self,
        claims: list[Claim],
        evaluations: list[ClaimEvaluation],
        points: list[KeyEvidencePoint],
        coverage: list[CoverageEvaluation],
    ) -> dict[str, float]:
        factual = 100 * mean([item.score for item in evaluations] or [0])
        cited_claim_ids = {claim.claim_id for claim in claims if claim.citations}
        cited_scores = [
            item.score for item in evaluations if item.claim_id in cited_claim_ids
        ]
        faithfulness = 100 * mean(cited_scores or [0])
        claims_needing_citation = [
            claim for claim in claims if claim.claim_type != "other" or claim.importance >= 0.5
        ]
        completeness = 100 * (
            sum(bool(claim.citations) for claim in claims_needing_citation)
            / max(1, len(claims_needing_citation))
        )
        point_importance = {point.key_point_id: point.importance for point in points}
        numerator = sum(
            item.score * point_importance.get(item.key_point_id, 0.5)
            for item in coverage
        )
        denominator = sum(point_importance.get(item.key_point_id, 0.5) for item in coverage)
        coverage_score = 100 * numerator / denominator if denominator else 0
        severe = sum(
            item.status in {"Unsupported", "Contradicted"} for item in evaluations
        )
        integrity = max(0.0, 100 - 30 * severe)
        return {
            "factual_accuracy": factual,
            "citation_faithfulness": faithfulness,
            "citation_completeness": completeness,
            "coverage": coverage_score,
            "academic_integrity": integrity,
        }

    def evaluate_review(self, review: ReviewDraft) -> EvaluationReport:
        claims = self.extract_claims(review)
        by_claim_id: dict[str, ClaimEvaluation] = {}
        with ThreadPoolExecutor(
            max_workers=max(1, self.settings.max_parallel_evaluators)
        ) as executor:
            futures = {
                executor.submit(self.verify_claim, claim): claim for claim in claims
            }
            for future in as_completed(futures):
                claim = futures[future]
                try:
                    by_claim_id[claim.claim_id] = future.result()
                except Exception as exc:
                    self.warnings.append(f"{claim.claim_id} 核验失败：{exc}")
                    by_claim_id[claim.claim_id] = ClaimEvaluation(
                        claim_id=claim.claim_id,
                        status="Unsupported",
                        score=0,
                        confidence=0,
                        rationale="核验器调用失败",
                    )
        claim_evaluations = [by_claim_id[claim.claim_id] for claim in claims]
        try:
            key_points = self.extract_key_points(review.question)
            coverage = self.audit_coverage(review, claims, key_points)
        except Exception as exc:
            self.warnings.append(f"反向覆盖审计失败：{exc}")
            key_points, coverage = [], []
        dimensions = self.evaluate_dimensions(
            review, claims, claim_evaluations, coverage
        )
        empirical = self._empirical_scores(
            claims, claim_evaluations, key_points, coverage
        )
        dimension_scores = {item.dimension: item.score for item in dimensions}
        for name, measured in empirical.items():
            dimension_scores[name] = 0.6 * measured + 0.4 * dimension_scores[name]
        weights = self.adaptive_weights(review.question)
        overall = sum(dimension_scores[name] * weights[name] for name in weights)
        return EvaluationReport(
            overall_score=round(overall, 2),
            dimension_scores={
                name: round(score, 2) for name, score in dimension_scores.items()
            },
            dimension_weights=weights,
            dimensions=dimensions,
            claims=claims,
            claim_evaluations=claim_evaluations,
            key_evidence_points=key_points,
            coverage_evaluations=coverage,
            warnings=list(self.warnings),
        )

    def run(
        self, question: str, *, top_k: int | None = None, refine: bool = True
    ) -> PipelineResult:
        review = self.generate_review(question, top_k=top_k, refine=refine)
        evaluation = self.evaluate_review(review)
        return PipelineResult(
            workspace_id=self.workspace.workspace_id,
            review=review,
            evaluation=evaluation,
        )
