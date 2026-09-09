from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path

from hy3scholar.client import Hy3Response
from hy3scholar.config import Settings
from hy3scholar.dataset_builder import (
    DatasetV0Builder,
    _normalize_audits,
    _normalize_generated_cases,
)
from hy3scholar.dataset_evaluator import DatasetBatchEvaluator
from hy3scholar.dataset_io import load_cases, validate_dataset
from hy3scholar.dataset_models import (
    DownloadManifestEntry,
    LiteratureRecord,
    LiteratureSearchReport,
    WebSearchSource,
)


def _minimal_pdf(text: str) -> bytes:
    stream = f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET".encode("ascii")
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>",
        b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    data = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for index, obj in enumerate(objects, 1):
        offsets.append(len(data))
        data.extend(f"{index} 0 obj\n".encode() + obj + b"\nendobj\n")
    xref = len(data)
    data.extend(f"xref\n0 {len(objects) + 1}\n".encode())
    data.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        data.extend(f"{offset:010d} 00000 n \n".encode())
    data.extend(
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    )
    return bytes(data)


class FakeDatasetHy3:
    def chat(self, *, system: str, user: str, **_: object) -> Hy3Response:
        if "[TASK:dataset_search_plan]" in system:
            return Hy3Response(json.dumps({"queries": ["agent memory", "memory benchmark"]}))
        if "[TASK:dataset_web_discovery]" in system:
            return Hy3Response(
                "found",
                search_results=[
                    {
                        "index": 1,
                        "url": "https://arxiv.org/abs/2501.01234",
                        "name": "Agent Memory",
                        "snippet": "paper",
                        "site": "arXiv",
                    }
                ],
            )
        if "[TASK:dataset_case_generation]" in system:
            cases = []
            for index in range(1, 8):
                cases.append(
                    {
                        "recipe_id": f"R{index:03d}",
                        "instruction": f"请依据论文证据完成第 {index} 个评测任务并保留限定条件。",
                        "reference_answer": "论文证据明确说明该方法、实验结果及其适用范围，不能扩展到未测试场景。",
                        "gold_evidence_ids": [f"P{1 + index % 2:03d}:p1:c1"],
                        "claim_label": "Unsupported" if index == 7 else "Supported",
                        "challenge_tags": ["scope"],
                        "generation_notes": "fixture",
                    }
                )
            return Hy3Response(json.dumps({"cases": cases}, ensure_ascii=False))
        if "[TASK:dataset_case_audit]" in system:
            return Hy3Response(
                json.dumps(
                    {
                        "audits": [
                            {"case_id": f"V0-{index:04d}", "accepted": True, "issues": []}
                            for index in range(1, 8)
                        ]
                    }
                )
            )
        if "[TASK:dataset_inference]" in system:
            decision = "Supported" if "TASK_TYPE=claim_check" in user else None
            return Hy3Response(
                json.dumps(
                    {
                        "answer": "依据原文证据作答，并保留论文限定条件。",
                        "decision": decision,
                        "evidence_ids": ["P001:p1:c1"],
                        "confidence": 0.8,
                        "risk_notes": [],
                    },
                    ensure_ascii=False,
                )
            )
        if "[TASK:dataset_rubric_evaluation]" in system:
            return Hy3Response(
                json.dumps(
                    {
                        "dimension_scores": {
                            "factual_accuracy": 90,
                            "evidence_traceability": 90,
                            "terminology_correctness": 90,
                            "coverage_completeness": 85,
                            "claim_judgement": 90,
                            "safety_boundary": 90,
                            "clarity": 90,
                        },
                        "failure_types": [],
                        "rationale": "fixture",
                    }
                )
            )
        raise AssertionError(system)


class FakeSearcher:
    def search(self, queries: list[str], *, limit_per_source: int) -> LiteratureSearchReport:
        records = [
            LiteratureRecord(
                record_id=f"L{index}",
                title=f"Agent Memory Paper {index}",
                authors=["Alice Smith"],
                year=2025,
                arxiv_id=f"2501.0000{index}",
                abstract="Evidence grounded agent memory evaluation.",
                landing_url=f"https://arxiv.org/abs/2501.0000{index}",
                pdf_url=f"https://arxiv.org/pdf/2501.0000{index}",
                is_open_access=True,
                license="arxiv-nonexclusive-distribution",
                relevance_score=0.9,
                sources=["arxiv"],
            )
            for index in (1, 2)
        ]
        return LiteratureSearchReport(queries=queries, records=records)

    def close(self) -> None:
        pass


class FakeDownloader:
    def download(self, record: LiteratureRecord, output_dir: Path) -> DownloadManifestEntry:
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / f"{record.record_id}.pdf"
        data = _minimal_pdf(
            f"{record.title} presents grounded evidence, methods, results, limitations, and evaluation."
        )
        path.write_bytes(data)
        return DownloadManifestEntry(
            record_id=record.record_id,
            title=record.title,
            source_url=record.pdf_url,
            status="open_access_downloaded",
            local_path=str(path.resolve()),
            bytes=len(data),
            sha256=sha256(data).hexdigest(),
            page_count=1,
            verified_text_chars=90,
            license=record.license,
        )

    def close(self) -> None:
        pass


def test_build_validate_and_batch_evaluate_dataset(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "dataset"
    client = FakeDatasetHy3()
    builder = DatasetV0Builder(
        Settings(arxiv_min_interval_seconds=0),
        client,
        searcher=FakeSearcher(),
        downloader=FakeDownloader(),
    )
    report = builder.build(
        topic="LLM agent memory",
        output_dir=dataset_dir,
        paper_count=2,
        case_count=7,
    )
    builder.close()
    assert report.validation.valid
    assert report.downloaded_papers == 2
    assert len(load_cases(dataset_dir)) == 7
    assert len(
        [
            item
            for item in json.loads(report.model_dump_json())["validation"]["warnings"]
            if "人工审核" in item
        ]
    ) == 1
    assert len(
        [line for line in (dataset_dir / "web_sources.jsonl").read_text().splitlines() if line]
    ) == 1

    results_dir = tmp_path / "results"
    results = DatasetBatchEvaluator(client, Settings()).evaluate(
        dataset_dir, results_dir, allow_pending=True
    )
    assert len(results) == 7
    assert (results_dir / "scores.csv").exists()
    assert (results_dir / "report.md").exists()
    assert validate_dataset(dataset_dir).case_count == 7
    assert (dataset_dir / "generation_output.json").exists()
    assert (dataset_dir / "generation_audit.json").exists()


def test_normalize_audits_accepts_common_shape_variants() -> None:
    results_shape = {
        "results": [
            {"id": "V0-0001", "status": "passed", "issues": ""},
            {"id": "V0-0002", "pass": False, "issues": "数字不一致"},
        ]
    }
    normalized = _normalize_audits(results_shape)
    assert normalized[0]["case_id"] == "V0-0001"
    assert normalized[0]["accepted"] is True
    assert normalized[0]["issues"] == []
    assert normalized[1]["accepted"] is False
    assert normalized[1]["issues"] == ["数字不一致"]

    mapping_shape = {"audits": {"V0-0003": {"accepted": True, "issues": []}}}
    assert _normalize_audits(mapping_shape)[0]["case_id"] == "V0-0003"
    assert _normalize_audits(
        {"audits": [{"case_id": "V0-0004", "accepted": "false"}]}
    )[0]["accepted"] is False


def test_normalize_generated_cases_accepts_wrappers_and_mappings() -> None:
    assert _normalize_generated_cases(
        {"data": {"cases": [{"recipe_id": "R001", "instruction": "x"}]}}
    )[0]["recipe_id"] == "R001"
    assert _normalize_generated_cases(
        {"results": {"R002": {"instruction": "y"}}}
    )[0]["recipe_id"] == "R002"
