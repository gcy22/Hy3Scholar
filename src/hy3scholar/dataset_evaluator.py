from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from .client import LLMClient
from .config import Settings
from .dataset_io import load_cases, load_workspace, write_jsonl
from .dataset_models import BatchEvaluationRecord, DatasetCase, TaskResponse
from .json_utils import extract_json
from .retrieval import SparseEvidenceRetriever, format_evidence_context
from .schemas import Workspace


TASK_SYSTEM = """[TASK:dataset_inference]
你是基于论文原文的阅读助手。只能使用 Evidence 回答。只输出 JSON：
{"answer":"...","decision":null,"evidence_ids":["P001:p1:c1"],
"confidence":0.0,"risk_notes":[]}
structured_summary 要覆盖问题、方法、结果和局限；evidence_qa 直接回答并说明证据不足；
claim_check 的 decision 必须是 Supported、Partially Supported、Unsupported、Contradicted
之一。引用 ID 必须来自 Evidence。"""

JUDGE_SYSTEM = """[TASK:dataset_rubric_evaluation]
你是严格的学术评测器。根据参考答案、Gold Evidence 和模型输出进行评价。只输出 JSON：
{"dimension_scores":{"factual_accuracy":0,"evidence_traceability":0,
"terminology_correctness":0,"coverage_completeness":0,"claim_judgement":0,
"safety_boundary":0,"clarity":0},"failure_types":[],"rationale":"..."}
各维度 0-100。不要因为答案长、术语多或语气自信而加分；证据不足时谨慎回答应得分，编造
事实、篡改限定条件、错误引用和错误论断标签应明确扣分。"""


DIMENSIONS = (
    "factual_accuracy",
    "evidence_traceability",
    "terminology_correctness",
    "coverage_completeness",
    "claim_judgement",
    "safety_boundary",
    "clarity",
)
WEIGHTS = {
    "factual_accuracy": 0.28,
    "evidence_traceability": 0.24,
    "terminology_correctness": 0.10,
    "coverage_completeness": 0.14,
    "claim_judgement": 0.10,
    "safety_boundary": 0.06,
    "clarity": 0.08,
}


def _score(value: Any, default: float = 0.0) -> float:
    try:
        return round(max(0.0, min(100.0, float(value))), 2)
    except (TypeError, ValueError):
        return default


def _evidence_metrics(predicted: list[str], gold: list[str]) -> tuple[float, float, float]:
    predicted_set, gold_set = set(predicted), set(gold)
    overlap = len(predicted_set & gold_set)
    precision = overlap / len(predicted_set) if predicted_set else 0.0
    recall = overlap / len(gold_set) if gold_set else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


class DatasetBatchEvaluator:
    def __init__(self, client: LLMClient, settings: Settings) -> None:
        self.client = client
        self.settings = settings

    def _run_case(self, case: DatasetCase, workspace: Workspace) -> TaskResponse:
        retriever = SparseEvidenceRetriever(workspace.chunks)
        evidence = retriever.search(case.instruction, top_k=10, max_per_paper=5)
        context = format_evidence_context(evidence, max_chars=32_000)
        response = self.client.chat(
            system=TASK_SYSTEM,
            user=(
                f"TASK_TYPE={case.task_type}\nDIFFICULTY={case.difficulty}\n"
                f"INSTRUCTION={case.instruction}\n\nEVIDENCE=\n{context}"
            ),
            reasoning_effort="high" if case.difficulty != "standard" else "low",
            temperature=0.1,
            max_tokens=2500,
        )
        payload = extract_json(response.content)
        if not isinstance(payload, dict):
            raise RuntimeError("Hy3 推理结果不是 JSON 对象")
        valid_ids = {chunk.evidence_id for chunk in workspace.chunks}
        evidence_ids = [
            str(value)
            for value in payload.get("evidence_ids", [])
            if str(value) in valid_ids
        ]
        decision = payload.get("decision")
        if decision not in {
            "Supported", "Partially Supported", "Unsupported", "Contradicted"
        }:
            decision = None
        try:
            confidence = float(payload.get("confidence") or 0.0)
        except (TypeError, ValueError):
            confidence = 0.0
        return TaskResponse(
            answer=str(payload.get("answer") or "").strip(),
            decision=decision,
            evidence_ids=evidence_ids,
            confidence=max(0.0, min(1.0, confidence)),
            risk_notes=[str(value) for value in payload.get("risk_notes", [])],
        )

    def _judge(
        self,
        case: DatasetCase,
        prediction: TaskResponse,
        workspace: Workspace,
    ) -> BatchEvaluationRecord:
        evidence_map = {chunk.evidence_id: chunk.text for chunk in workspace.chunks}
        gold_context = "\n\n".join(
            f"[{evidence_id}] {evidence_map[evidence_id]}"
            for evidence_id in case.gold_evidence_ids
            if evidence_id in evidence_map
        )
        response = self.client.chat(
            system=JUDGE_SYSTEM,
            user=(
                "CASE=\n"
                + json.dumps(case.model_dump(mode="json"), ensure_ascii=False)
                + "\n\nPREDICTION=\n"
                + json.dumps(prediction.model_dump(mode="json"), ensure_ascii=False)
                + "\n\nGOLD_EVIDENCE=\n"
                + gold_context
            ),
            reasoning_effort="high",
            temperature=0.0,
            max_tokens=2200,
        )
        payload = extract_json(response.content)
        if not isinstance(payload, dict):
            payload = {}
        raw_scores = payload.get("dimension_scores", {})
        scores = {name: _score(raw_scores.get(name)) for name in DIMENSIONS}
        precision, recall, f1 = _evidence_metrics(
            prediction.evidence_ids, case.gold_evidence_ids
        )
        scores["evidence_traceability"] = round(f1 * 100, 2)
        label_correct: bool | None = None
        failures = [str(value) for value in payload.get("failure_types", [])]
        if case.task_type == "claim_check":
            label_correct = prediction.decision == case.claim_label
            scores["claim_judgement"] = 100.0 if label_correct else 0.0
            if not label_correct:
                failures.append("wrong_claim_label")
        if not prediction.evidence_ids:
            failures.append("missing_evidence_ids")
        if precision < 1.0 and prediction.evidence_ids:
            failures.append("invalid_or_irrelevant_evidence")
        active_weights = dict(WEIGHTS)
        if case.task_type != "claim_check":
            active_weights.pop("claim_judgement")
        weight_sum = sum(active_weights.values())
        overall = (
            sum(scores[name] * weight for name, weight in active_weights.items())
            / weight_sum
        )
        return BatchEvaluationRecord(
            case_id=case.case_id,
            task_type=case.task_type,
            difficulty=case.difficulty,
            human_review_status=case.human_review_status,
            prediction=prediction,
            dimension_scores=scores,
            overall_score=round(overall, 2),
            evidence_precision=round(precision, 4),
            evidence_recall=round(recall, 4),
            evidence_f1=round(f1, 4),
            claim_label_correct=label_correct,
            failure_types=list(dict.fromkeys(failures)),
            rationale=str(payload.get("rationale") or ""),
        )

    def evaluate(
        self,
        dataset_dir: Path,
        output_dir: Path,
        *,
        allow_pending: bool = False,
    ) -> list[BatchEvaluationRecord]:
        workspace = load_workspace(dataset_dir)
        cases = [case for case in load_cases(dataset_dir) if case.human_review_status != "rejected"]
        if not allow_pending:
            cases = [case for case in cases if case.human_review_status == "approved"]
            if not cases:
                raise RuntimeError(
                    "没有已批准样本；先运行 dataset-review 完成人工审核，"
                    "或仅在调试时使用 --allow-pending"
                )
        output_dir.mkdir(parents=True, exist_ok=True)
        results: list[BatchEvaluationRecord] = []
        for case in cases:
            prediction = self._run_case(case, workspace)
            results.append(self._judge(case, prediction, workspace))
            write_jsonl(output_dir / "predictions.jsonl", results)
        self._write_csv(output_dir / "scores.csv", results)
        failures = [item for item in results if item.overall_score < 70 or item.failure_types]
        write_jsonl(output_dir / "failure_cases.jsonl", failures)
        self._write_report(output_dir / "report.md", results)
        return results

    @staticmethod
    def _write_csv(path: Path, results: list[BatchEvaluationRecord]) -> None:
        with path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "case_id", "task_type", "difficulty", "human_review_status",
                    "overall_score", *DIMENSIONS, "evidence_precision",
                    "evidence_recall", "evidence_f1", "claim_label_correct",
                    "failure_types",
                ],
            )
            writer.writeheader()
            for item in results:
                row = {
                    "case_id": item.case_id,
                    "task_type": item.task_type,
                    "difficulty": item.difficulty,
                    "human_review_status": item.human_review_status,
                    "overall_score": item.overall_score,
                    **item.dimension_scores,
                    "evidence_precision": item.evidence_precision,
                    "evidence_recall": item.evidence_recall,
                    "evidence_f1": item.evidence_f1,
                    "claim_label_correct": item.claim_label_correct,
                    "failure_types": ";".join(item.failure_types),
                }
                writer.writerow(row)

    @staticmethod
    def _write_report(path: Path, results: list[BatchEvaluationRecord]) -> None:
        mean = sum(item.overall_score for item in results) / max(1, len(results))
        by_task: dict[str, list[float]] = {}
        failures: dict[str, int] = {}
        for item in results:
            by_task.setdefault(item.task_type, []).append(item.overall_score)
            for failure in item.failure_types:
                failures[failure] = failures.get(failure, 0) + 1
        lines = [
            "# Dataset 批量评测报告", "", f"- 样本数：{len(results)}",
            f"- 平均分：**{mean:.2f}/100**", "", "## 分任务结果", "",
            "| 任务 | 样本数 | 平均分 |", "|---|---:|---:|",
        ]
        for task, values in sorted(by_task.items()):
            lines.append(f"| {task} | {len(values)} | {sum(values) / len(values):.2f} |")
        lines.extend(["", "## 失败归因", ""])
        if failures:
            lines.extend(f"- `{name}`：{count}" for name, count in sorted(failures.items()))
        else:
            lines.append("- 未记录失败类型。")
        lines.extend(["", "## 低分样本", ""])
        for item in sorted(results, key=lambda value: value.overall_score):
            if item.overall_score < 70:
                lines.append(
                    f"- `{item.case_id}`：{item.overall_score:.2f}；"
                    f"{', '.join(item.failure_types) or item.rationale}"
                )
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
