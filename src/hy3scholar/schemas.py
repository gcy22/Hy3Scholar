from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field


class EvidenceChunk(BaseModel):
    evidence_id: str
    paper_id: str
    paper_title: str
    file_name: str
    page: int
    chunk_index: int
    section: str | None = None
    text: str


class PaperRecord(BaseModel):
    paper_id: str
    title: str
    file_name: str
    page_count: int
    chunk_count: int


class Workspace(BaseModel):
    workspace_id: str
    papers: list[PaperRecord]
    chunks: list[EvidenceChunk]
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )


class RetrievedEvidence(BaseModel):
    chunk: EvidenceChunk
    score: float


class ReviewDraft(BaseModel):
    question: str
    markdown: str
    evidence_ids: list[str] = Field(default_factory=list)
    refined: bool = False


class Claim(BaseModel):
    claim_id: str
    text: str
    citations: list[str] = Field(default_factory=list)
    claim_type: Literal["method", "result", "comparison", "limitation", "other"] = (
        "other"
    )
    importance: float = Field(default=0.5, ge=0, le=1)


class ClaimEvaluation(BaseModel):
    claim_id: str
    status: Literal[
        "Supported", "Partially Supported", "Unsupported", "Contradicted"
    ]
    score: float = Field(ge=0, le=1)
    confidence: float = Field(ge=0, le=1)
    evidence_ids: list[str] = Field(default_factory=list)
    rationale: str
    escalated: bool = False


class KeyEvidencePoint(BaseModel):
    key_point_id: str
    paper_id: str
    text: str
    importance: float = Field(default=0.5, ge=0, le=1)
    evidence_ids: list[str] = Field(default_factory=list)


class CoverageEvaluation(BaseModel):
    key_point_id: str
    covered: bool
    score: float = Field(ge=0, le=1)
    matched_claim_ids: list[str] = Field(default_factory=list)
    rationale: str


class DimensionEvaluation(BaseModel):
    dimension: str
    score: float = Field(ge=0, le=100)
    confidence: float = Field(ge=0, le=1)
    problems: list[str] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)


class EvaluationReport(BaseModel):
    overall_score: float = Field(ge=0, le=100)
    dimension_scores: dict[str, float]
    dimension_weights: dict[str, float]
    dimensions: list[DimensionEvaluation]
    claims: list[Claim]
    claim_evaluations: list[ClaimEvaluation]
    key_evidence_points: list[KeyEvidencePoint]
    coverage_evaluations: list[CoverageEvaluation]
    warnings: list[str] = Field(default_factory=list)


class PipelineResult(BaseModel):
    workspace_id: str
    review: ReviewDraft
    evaluation: EvaluationReport


class PerturbationResult(BaseModel):
    name: str
    kind: Literal["surface", "factual"]
    score: float = Field(ge=0, le=100)
    delta: float
    passed: bool
    explanation: str


class AdversarialReport(BaseModel):
    baseline_score: float = Field(ge=0, le=100)
    perturbations: list[PerturbationResult]
    robustness_rate: float = Field(ge=0, le=1)

