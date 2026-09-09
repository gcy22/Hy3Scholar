from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, TypeVar

from pydantic import BaseModel

from .dataset_models import (
    DatasetCase,
    DatasetValidationReport,
    DownloadManifestEntry,
    HumanAnnotation,
    LiteratureRecord,
)
from .schemas import Workspace


ModelT = TypeVar("ModelT", bound=BaseModel)


def write_jsonl(path: Path, items: Iterable[BaseModel]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        json.dumps(item.model_dump(mode="json"), ensure_ascii=False)
        for item in items
    ]
    path.write_text(("\n".join(lines) + "\n") if lines else "", encoding="utf-8")


def read_jsonl(path: Path, model: type[ModelT]) -> list[ModelT]:
    if not path.exists():
        return []
    output: list[ModelT] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            output.append(model.model_validate_json(line))
        except Exception as exc:
            raise ValueError(f"{path.name} 第 {line_number} 行无效：{exc}") from exc
    return output


def load_workspace(dataset_dir: Path) -> Workspace:
    path = dataset_dir / "workspace.json"
    if not path.exists():
        raise FileNotFoundError(f"缺少 {path}")
    return Workspace.model_validate_json(path.read_text(encoding="utf-8"))


def load_annotations(dataset_dir: Path) -> dict[str, HumanAnnotation]:
    annotations = read_jsonl(dataset_dir / "annotations.jsonl", HumanAnnotation)
    return {item.case_id: item for item in annotations}


def apply_annotation(case: DatasetCase, annotation: HumanAnnotation) -> DatasetCase:
    payload = case.model_dump()
    update: dict[str, object] = {"human_review_status": annotation.status}
    if annotation.corrected_instruction:
        update["instruction"] = annotation.corrected_instruction
    if annotation.corrected_reference_answer:
        update["reference_answer"] = annotation.corrected_reference_answer
    if annotation.corrected_gold_evidence_ids is not None:
        update["gold_evidence_ids"] = annotation.corrected_gold_evidence_ids
    if annotation.corrected_claim_label is not None:
        update["claim_label"] = annotation.corrected_claim_label
    payload.update(update)
    return DatasetCase.model_validate(payload)


def load_cases(
    dataset_dir: Path, *, apply_annotations: bool = True
) -> list[DatasetCase]:
    cases = read_jsonl(dataset_dir / "cases.jsonl", DatasetCase)
    if not apply_annotations:
        return cases
    annotations = load_annotations(dataset_dir)
    return [
        apply_annotation(case, annotations[case.case_id])
        if case.case_id in annotations
        else case
        for case in cases
    ]


def upsert_annotation(dataset_dir: Path, annotation: HumanAnnotation) -> None:
    items = load_annotations(dataset_dir)
    items[annotation.case_id] = annotation
    write_jsonl(
        dataset_dir / "annotations.jsonl",
        sorted(items.values(), key=lambda item: item.case_id),
    )


def validate_dataset(dataset_dir: Path) -> DatasetValidationReport:
    errors: list[str] = []
    warnings: list[str] = []
    try:
        workspace = load_workspace(dataset_dir)
    except Exception as exc:
        return DatasetValidationReport(valid=False, errors=[str(exc)])
    try:
        cases = load_cases(dataset_dir)
    except Exception as exc:
        return DatasetValidationReport(valid=False, errors=[str(exc)])
    if not cases:
        errors.append("cases.jsonl 中没有样本")
    evidence_ids = {chunk.evidence_id for chunk in workspace.chunks}
    paper_ids = {paper.paper_id for paper in workspace.papers}
    seen: set[str] = set()
    for case in cases:
        if case.case_id in seen:
            errors.append(f"重复 case_id：{case.case_id}")
        seen.add(case.case_id)
        unknown_evidence = sorted(set(case.gold_evidence_ids) - evidence_ids)
        if unknown_evidence:
            errors.append(f"{case.case_id} 使用未知证据：{', '.join(unknown_evidence)}")
        unknown_papers = sorted(set(case.paper_ids) - paper_ids)
        if unknown_papers:
            errors.append(f"{case.case_id} 使用未知论文：{', '.join(unknown_papers)}")
        if case.task_type == "claim_check" and case.claim_label is None:
            errors.append(f"{case.case_id} 是论断核对样本但缺少 claim_label")
        if case.difficulty == "counterexample" and case.claim_label not in {
            "Unsupported", "Contradicted"
        }:
            errors.append(f"{case.case_id} 反例必须标注 Unsupported 或 Contradicted")
        if case.auto_validation_status != "passed":
            warnings.append(f"{case.case_id} 自动检查未通过，必须人工修改")
    task_counts: dict[str, int] = {}
    difficulty_counts: dict[str, int] = {}
    for case in cases:
        task_counts[case.task_type] = task_counts.get(case.task_type, 0) + 1
        difficulty_counts[case.difficulty] = difficulty_counts.get(case.difficulty, 0) + 1
    for task_type in ("structured_summary", "evidence_qa", "claim_check"):
        if not task_counts.get(task_type):
            warnings.append(f"缺少任务类型：{task_type}")
    for difficulty in ("standard", "hard", "counterexample"):
        if not difficulty_counts.get(difficulty):
            warnings.append(f"缺少难度类型：{difficulty}")
    pending = sum(case.human_review_status in {"pending", "needs_revision"} for case in cases)
    approved = sum(case.human_review_status == "approved" for case in cases)
    if pending:
        warnings.append(f"仍有 {pending} 个样本未完成人工审核")
    manifests = read_jsonl(dataset_dir / "download_manifest.jsonl", DownloadManifestEntry)
    papers = read_jsonl(dataset_dir / "papers.jsonl", LiteratureRecord)
    if not manifests:
        errors.append("缺少下载清单或清单为空")
    if len(papers) != len(workspace.papers):
        errors.append("papers.jsonl 与 workspace.json 的论文数量不一致")
    return DatasetValidationReport(
        valid=not errors,
        case_count=len(cases),
        approved_count=approved,
        pending_count=pending,
        task_counts=task_counts,
        difficulty_counts=difficulty_counts,
        errors=errors,
        warnings=warnings,
    )
