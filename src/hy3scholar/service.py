from __future__ import annotations

from pathlib import Path

from .adversarial import run_adversarial_calibration
from .client import Hy3Client, LLMClient
from .config import Settings
from .ingest import ingest_pdfs
from .pipeline import Hy3ScholarPipeline
from .retrieval import SparseEvidenceRetriever, format_evidence_context
from .schemas import AdversarialReport, PipelineResult, ReviewDraft, Workspace
from .storage import WorkspaceStore


class Hy3ScholarService:
    def __init__(
        self,
        settings: Settings | None = None,
        *,
        client: LLMClient | None = None,
    ) -> None:
        self.settings = settings or Settings()
        self.store = WorkspaceStore(self.settings.storage_dir)
        self._client = client

    @property
    def client(self) -> LLMClient:
        if self._client is None:
            self._client = Hy3Client(self.settings)
        return self._client

    def create_workspace(self, pdf_paths: list[Path]) -> Workspace:
        workspace = ingest_pdfs(
            pdf_paths,
            chunk_chars=self.settings.chunk_chars,
            overlap_chars=self.settings.chunk_overlap_chars,
        )
        self.store.save_workspace(workspace)
        return workspace

    def run(
        self,
        workspace_id: str,
        question: str,
        *,
        top_k: int | None = None,
        refine: bool = True,
    ) -> PipelineResult:
        workspace = self.store.load_workspace(workspace_id)
        pipeline = Hy3ScholarPipeline(
            client=self.client, workspace=workspace, settings=self.settings
        )
        result = pipeline.run(question, top_k=top_k, refine=refine)
        self.store.save_result(result)
        return result

    def evaluate(
        self, workspace_id: str, question: str, markdown: str
    ) -> PipelineResult:
        workspace = self.store.load_workspace(workspace_id)
        pipeline = Hy3ScholarPipeline(
            client=self.client, workspace=workspace, settings=self.settings
        )
        review = ReviewDraft(
            question=question,
            markdown=markdown,
            evidence_ids=[],
            refined=False,
        )
        report = pipeline.evaluate_review(review)
        result = PipelineResult(
            workspace_id=workspace_id, review=review, evaluation=report
        )
        self.store.save_result(result)
        return result

    def adversarial_test(
        self, workspace_id: str, review: str, question: str
    ) -> AdversarialReport:
        workspace = self.store.load_workspace(workspace_id)
        retriever = SparseEvidenceRetriever(workspace.chunks)
        evidence = retriever.search(question, top_k=12)
        report = run_adversarial_calibration(
            client=self.client,
            review=review,
            evidence_context=format_evidence_context(evidence),
        )
        self.store.save_adversarial(workspace_id, report)
        return report

