from __future__ import annotations

import json
import re

from hy3scholar.client import Hy3Response
from hy3scholar.config import Settings
from hy3scholar.pipeline import Hy3ScholarPipeline
from hy3scholar.schemas import EvidenceChunk, PaperRecord, Workspace


class FakeHy3:
    def chat(self, *, system: str, user: str, **_: object) -> Hy3Response:
        if "[TASK:generate]" in system:
            return Hy3Response(
                "# 综述\n图记忆方法提升长期信息组织 [P001:p1:c1]。"
                "注意力基线使用密集注意力 [P002:p1:c1]。"
            )
        if "[TASK:self_refine]" in system:
            draft = user.split("待审查草稿：\n", 1)[1]
            return Hy3Response(draft)
        if "[TASK:extract_claims]" in system:
            return Hy3Response(
                json.dumps(
                    {
                        "claims": [
                            {
                                "text": "图记忆方法提升长期信息组织",
                                "citations": ["P001:p1:c1"],
                                "claim_type": "method",
                                "importance": 0.9,
                            },
                            {
                                "text": "注意力基线使用密集注意力",
                                "citations": ["P002:p1:c1"],
                                "claim_type": "method",
                                "importance": 0.7,
                            },
                        ]
                    },
                    ensure_ascii=False,
                )
            )
        if "[TASK:verify_claim]" in system:
            evidence_id = re.search(r"P\d{3}:p\d+:c\d+", user).group(0)
            return Hy3Response(
                json.dumps(
                    {
                        "status": "Supported",
                        "score": 0.95,
                        "confidence": 0.92,
                        "evidence_ids": [evidence_id],
                        "rationale": "原文直接支持",
                    },
                    ensure_ascii=False,
                )
            )
        if "[TASK:extract_key_points]" in system:
            return Hy3Response(
                json.dumps(
                    {
                        "key_points": [
                            {
                                "paper_id": "P001",
                                "text": "图记忆组织长期信息",
                                "importance": 0.9,
                                "evidence_ids": ["P001:p1:c1"],
                            },
                            {
                                "paper_id": "P002",
                                "text": "密集注意力基线",
                                "importance": 0.7,
                                "evidence_ids": ["P002:p1:c1"],
                            },
                        ]
                    },
                    ensure_ascii=False,
                )
            )
        if "[TASK:coverage_audit]" in system:
            return Hy3Response(
                json.dumps(
                    {
                        "coverage": [
                            {
                                "key_point_id": "K001",
                                "covered": True,
                                "score": 1,
                                "matched_claim_ids": ["C001"],
                                "rationale": "已覆盖",
                            },
                            {
                                "key_point_id": "K002",
                                "covered": True,
                                "score": 1,
                                "matched_claim_ids": ["C002"],
                                "rationale": "已覆盖",
                            },
                        ]
                    },
                    ensure_ascii=False,
                )
            )
        if "[TASK:dimension_eval]" in system:
            dimension = re.search(r"DIMENSION=(\w+)", user).group(1)
            return Hy3Response(
                json.dumps(
                    {
                        "dimension": dimension,
                        "score": 88,
                        "confidence": 0.9,
                        "problems": [],
                        "recommendations": [],
                        "evidence_ids": [],
                    }
                )
            )
        if "[TASK:adaptive_weights]" in system:
            return Hy3Response(
                json.dumps(
                    {"weights": {name: 1 for name in [
                        "factual_accuracy", "citation_faithfulness",
                        "citation_completeness", "coverage", "comparative_depth",
                        "logical_coherence", "academic_integrity"
                    ]}}
                )
            )
        raise AssertionError(system)


def workspace() -> Workspace:
    chunks = [
        EvidenceChunk(
            evidence_id="P001:p1:c1",
            paper_id="P001",
            paper_title="Graph Memory",
            file_name="a.pdf",
            page=1,
            chunk_index=1,
            text="Graph memory 图记忆方法提升长期信息组织。",
        ),
        EvidenceChunk(
            evidence_id="P002:p1:c1",
            paper_id="P002",
            paper_title="Attention Baseline",
            file_name="b.pdf",
            page=1,
            chunk_index=1,
            text="Graph memory comparison uses a dense attention baseline 密集注意力。",
        ),
    ]
    return Workspace(
        workspace_id="test",
        papers=[
            PaperRecord(
                paper_id="P001", title="Graph Memory", file_name="a.pdf",
                page_count=1, chunk_count=1
            ),
            PaperRecord(
                paper_id="P002", title="Attention", file_name="b.pdf",
                page_count=1, chunk_count=1
            ),
        ],
        chunks=chunks,
    )


def test_end_to_end_pipeline_with_fake_hy3() -> None:
    pipeline = Hy3ScholarPipeline(
        client=FakeHy3(),
        workspace=workspace(),
        settings=Settings(max_parallel_evaluators=2),
    )
    result = pipeline.run("graph memory 图记忆方法比较")
    assert result.review.refined is True
    assert len(result.evaluation.claims) == 2
    assert all(item.status == "Supported" for item in result.evaluation.claim_evaluations)
    assert len(result.evaluation.dimensions) == 7
    assert result.evaluation.dimension_scores["coverage"] > 90
    assert result.evaluation.overall_score > 85

