import json

from hy3scholar.adversarial import build_perturbations, run_adversarial_calibration
from hy3scholar.client import Hy3Response


class FakeCalibrator:
    def chat(self, **_: object) -> Hy3Response:
        return Hy3Response(
            json.dumps(
                {
                    "baseline_score": 82,
                    "variants": [
                        {"name": "verbosity_expansion", "score": 81, "explanation": "表面变化"},
                        {"name": "terminology_stuffing", "score": 80, "explanation": "表面变化"},
                        {"name": "citation_swap", "score": 55, "explanation": "引用错误"},
                        {"name": "numeric_tamper", "score": 50, "explanation": "数字错误"},
                        {"name": "overclaim", "score": 60, "explanation": "过度结论"},
                    ],
                },
                ensure_ascii=False,
            )
        )


def test_perturbations_and_calibration() -> None:
    review = "结果可能提升 10% [P001:p1:c1]，基线见 [P002:p1:c1]。"
    variants = build_perturbations(review)
    assert set(variants) == {
        "verbosity_expansion", "terminology_stuffing", "citation_swap",
        "numeric_tamper", "overclaim"
    }
    report = run_adversarial_calibration(
        client=FakeCalibrator(), review=review, evidence_context="evidence"
    )
    assert report.robustness_rate == 1
    assert all(item.passed for item in report.perturbations)

