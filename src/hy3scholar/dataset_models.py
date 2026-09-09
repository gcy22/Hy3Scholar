from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


DownloadStatus = Literal[
    "open_access_downloaded",
    "available_not_downloaded",
    "no_full_text_link",
    "no_authorized_pdf_found",
    "publisher_blocked_waiting_user",
    "publisher_verification_waiting_user",
    "pdf_fetch_failed",
    "failed_after_retry",
]
TaskType = Literal["structured_summary", "evidence_qa", "claim_check"]
Difficulty = Literal["standard", "hard", "counterexample"]
ClaimLabel = Literal[
    "Supported", "Partially Supported", "Unsupported", "Contradicted"
]
ReviewStatus = Literal["pending", "approved", "needs_revision", "rejected"]


class LiteratureRecord(BaseModel):
    record_id: str
    paper_id: str | None = None
    title: str
    authors: list[str] = Field(default_factory=list)
    year: int | None = None
    doi: str | None = None
    arxiv_id: str | None = None
    abstract: str = ""
    venue: str | None = None
    landing_url: str | None = None
    pdf_url: str | None = None
    is_open_access: bool = False
    license: str | None = None
    citation_count: int = Field(default=0, ge=0)
    relevance_score: float = Field(default=0.0, ge=0.0, le=1.0)
    sources: list[str] = Field(default_factory=list)
    provenance: list[dict[str, Any]] = Field(default_factory=list)


class SearchFailure(BaseModel):
    source: str
    query: str
    error: str


class LiteratureSearchReport(BaseModel):
    queries: list[str]
    records: list[LiteratureRecord]
    failures: list[SearchFailure] = Field(default_factory=list)


class WebSearchSource(BaseModel):
    index: int | None = None
    url: str
    name: str = ""
    snippet: str = ""
    site: str = ""


class DownloadManifestEntry(BaseModel):
    record_id: str
    paper_id: str | None = None
    title: str
    doi: str | None = None
    source_url: str | None = None
    status: DownloadStatus
    local_path: str | None = None
    bytes: int = Field(default=0, ge=0)
    sha256: str | None = None
    page_count: int | None = None
    verified_text_chars: int = Field(default=0, ge=0)
    license: str | None = None
    downloaded_at: datetime | None = None
    error: str | None = None
    si_status: Literal["not_checked", "available_not_downloaded"] = "not_checked"


class DatasetCase(BaseModel):
    schema_version: str = "0.1"
    case_id: str
    topic: str
    task_type: TaskType
    difficulty: Difficulty
    instruction: str
    reference_answer: str
    paper_ids: list[str] = Field(min_length=1)
    gold_evidence_ids: list[str] = Field(min_length=1)
    claim_label: ClaimLabel | None = None
    challenge_tags: list[str] = Field(default_factory=list)
    generation_notes: str = ""
    auto_validation_status: Literal["passed", "needs_revision"] = "passed"
    auto_validation_issues: list[str] = Field(default_factory=list)
    human_review_status: ReviewStatus = "pending"
    generated_by: str = "hy3"
    created_at: datetime = Field(default_factory=utc_now)


class HumanAnnotation(BaseModel):
    case_id: str
    status: ReviewStatus
    reviewer: str
    notes: str = ""
    corrected_instruction: str | None = None
    corrected_reference_answer: str | None = None
    corrected_gold_evidence_ids: list[str] | None = None
    corrected_claim_label: ClaimLabel | None = None
    reviewed_at: datetime = Field(default_factory=utc_now)


class TaskResponse(BaseModel):
    answer: str
    decision: ClaimLabel | None = None
    evidence_ids: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.5, ge=0, le=1)
    risk_notes: list[str] = Field(default_factory=list)


class BatchEvaluationRecord(BaseModel):
    case_id: str
    task_type: TaskType
    difficulty: Difficulty
    human_review_status: ReviewStatus
    prediction: TaskResponse
    dimension_scores: dict[str, float]
    overall_score: float = Field(ge=0, le=100)
    evidence_precision: float = Field(ge=0, le=1)
    evidence_recall: float = Field(ge=0, le=1)
    evidence_f1: float = Field(ge=0, le=1)
    claim_label_correct: bool | None = None
    failure_types: list[str] = Field(default_factory=list)
    rationale: str = ""


class DatasetValidationReport(BaseModel):
    valid: bool
    case_count: int = 0
    approved_count: int = 0
    pending_count: int = 0
    task_counts: dict[str, int] = Field(default_factory=dict)
    difficulty_counts: dict[str, int] = Field(default_factory=dict)
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class DatasetBuildReport(BaseModel):
    topic: str
    queries: list[str]
    discovered_records: int
    oa_candidates: int
    downloaded_papers: int
    generated_cases: int
    dataset_dir: str
    search_failures: list[SearchFailure] = Field(default_factory=list)
    validation: DatasetValidationReport
    created_at: datetime = Field(default_factory=utc_now)
