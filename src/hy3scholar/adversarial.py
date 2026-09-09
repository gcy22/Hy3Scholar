from __future__ import annotations

import json
import re

from .client import LLMClient
from .json_utils import extract_json
from .prompts import ADVERSARIAL_SYSTEM
from .schemas import AdversarialReport, PerturbationResult


def build_perturbations(review: str) -> dict[str, tuple[str, str]]:
    evidence_ids = re.findall(r"P\d{3}:p\d+:c\d+", review)
    citation_swapped = review
    if len(set(evidence_ids)) >= 2:
        first, second = list(dict.fromkeys(evidence_ids))[:2]
        citation_swapped = review.replace(first, "__SECOND__").replace(second, first)
        citation_swapped = citation_swapped.replace("__SECOND__", second)
    elif evidence_ids:
        citation_swapped = review.replace(evidence_ids[0], "P999:p999:c999", 1)

    numeric_tampered = re.sub(
        r"(?<![A-Za-z:])(\d+(?:\.\d+)?)(?![A-Za-z])",
        lambda match: str(float(match.group(1)) * 1.37).rstrip("0").rstrip("."),
        review,
        count=1,
    )
    overclaimed = review
    replacements = [
        ("可能", "已经证明"),
        ("提示", "决定性地证明"),
        ("may", "definitively"),
        ("suggests", "proves"),
    ]
    for source, target in replacements:
        if source in overclaimed:
            overclaimed = overclaimed.replace(source, target, 1)
            break
    if overclaimed == review:
        overclaimed += "\n\n这些研究已经完全证明该路线在所有场景中均为最优。"

    return {
        "verbosity_expansion": (
            "surface",
            review
            + "\n\n从更广阔的角度看，上述讨论具有重要而深远的理论与实践意义。"
            * 3,
        ),
        "terminology_stuffing": (
            "surface",
            review.replace(
                "方法", "多尺度、全链路、范式驱动且具备协同涌现特征的方法", 2
            ),
        ),
        "citation_swap": ("factual", citation_swapped),
        "numeric_tamper": ("factual", numeric_tampered),
        "overclaim": ("factual", overclaimed),
    }


def run_adversarial_calibration(
    *, client: LLMClient, review: str, evidence_context: str
) -> AdversarialReport:
    perturbations = build_perturbations(review)
    variants = [
        {"name": name, "kind": kind, "review": text}
        for name, (kind, text) in perturbations.items()
    ]
    response = client.chat(
        system=ADVERSARIAL_SYSTEM,
        user=(
            f"原始证据：\n{evidence_context}\n\n基线综述：\n{review}\n\n"
            f"扰动版本：\n{json.dumps(variants, ensure_ascii=False)}"
        ),
        reasoning_effort="high",
        temperature=0,
    )
    payload = extract_json(response.content)
    baseline = max(0.0, min(100.0, float(payload.get("baseline_score", 0))))
    raw_by_name = {
        str(item.get("name")): item
        for item in payload.get("variants", [])
        if isinstance(item, dict)
    }
    results: list[PerturbationResult] = []
    for name, (kind, _) in perturbations.items():
        raw = raw_by_name.get(name, {})
        score = max(0.0, min(100.0, float(raw.get("score", 0))))
        delta = score - baseline
        passed = abs(delta) <= 8 if kind == "surface" else delta <= -15
        results.append(
            PerturbationResult(
                name=name,
                kind=kind,
                score=score,
                delta=round(delta, 2),
                passed=passed,
                explanation=str(raw.get("explanation", "未返回解释")),
            )
        )
    robustness_rate = sum(item.passed for item in results) / max(1, len(results))
    return AdversarialReport(
        baseline_score=baseline,
        perturbations=results,
        robustness_rate=robustness_rate,
    )

