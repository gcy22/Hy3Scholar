from __future__ import annotations

import json
from pathlib import Path

from .schemas import AdversarialReport, PipelineResult, Workspace


class WorkspaceStore:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def workspace_dir(self, workspace_id: str) -> Path:
        if not workspace_id.replace("-", "").isalnum():
            raise ValueError("非法 workspace_id")
        return self.root / workspace_id

    def save_workspace(self, workspace: Workspace) -> Path:
        directory = self.workspace_dir(workspace.workspace_id)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "workspace.json"
        path.write_text(
            workspace.model_dump_json(indent=2), encoding="utf-8"
        )
        return path

    def load_workspace(self, workspace_id: str) -> Workspace:
        path = self.workspace_dir(workspace_id) / "workspace.json"
        if not path.exists():
            raise FileNotFoundError(f"Workspace 不存在：{workspace_id}")
        return Workspace.model_validate_json(path.read_text(encoding="utf-8"))

    def save_result(self, result: PipelineResult) -> Path:
        directory = self.workspace_dir(result.workspace_id) / "results"
        directory.mkdir(parents=True, exist_ok=True)
        json_path = directory / "latest.json"
        json_path.write_text(result.model_dump_json(indent=2), encoding="utf-8")
        (directory / "review.md").write_text(result.review.markdown, encoding="utf-8")
        (directory / "evaluation.md").write_text(
            evaluation_to_markdown(result), encoding="utf-8"
        )
        return json_path

    def save_adversarial(
        self, workspace_id: str, report: AdversarialReport
    ) -> Path:
        directory = self.workspace_dir(workspace_id) / "results"
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "adversarial.json"
        path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
        return path


def evaluation_to_markdown(result: PipelineResult) -> str:
    report = result.evaluation
    lines = [
        "# Hy3Scholar 可信评估报告",
        "",
        f"- Workspace: `{result.workspace_id}`",
        f"- 研究问题：{result.review.question}",
        f"- 总体可信度：**{report.overall_score:.1f}/100**",
        "",
        "## 维度评分",
        "",
        "| 维度 | 得分 | 权重 |",
        "|---|---:|---:|",
    ]
    for name, score in report.dimension_scores.items():
        lines.append(
            f"| {name} | {score:.1f} | {report.dimension_weights.get(name, 0):.1%} |"
        )
    lines.extend(["", "## Claim 核验", ""])
    claim_map = {claim.claim_id: claim for claim in report.claims}
    for item in report.claim_evaluations:
        claim = claim_map.get(item.claim_id)
        lines.append(
            f"- **{item.claim_id} · {item.status} · {item.score * 100:.0f}**："
            f"{claim.text if claim else ''} — {item.rationale} "
            f"（证据：{', '.join(item.evidence_ids) or '无'}）"
        )
    lines.extend(["", "## 反向覆盖审计", ""])
    point_map = {point.key_point_id: point for point in report.key_evidence_points}
    for item in report.coverage_evaluations:
        point = point_map.get(item.key_point_id)
        status = "已覆盖" if item.covered else "遗漏/覆盖不足"
        lines.append(
            f"- **{item.key_point_id} · {status}**："
            f"{point.text if point else ''} — {item.rationale}"
        )
    if report.warnings:
        lines.extend(["", "## 警告", ""])
        lines.extend(f"- {warning}" for warning in report.warnings)
    return "\n".join(lines) + "\n"

