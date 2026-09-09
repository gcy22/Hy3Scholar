from __future__ import annotations

from pathlib import Path

import streamlit as st

from hy3scholar.dataset_io import (
    load_cases,
    load_workspace,
    upsert_annotation,
    validate_dataset,
)
from hy3scholar.dataset_models import HumanAnnotation


st.set_page_config(page_title="Dataset v0 人工审核", page_icon="✅", layout="wide")
st.title("Hy3Scholar Dataset v0 人工审核")
st.caption("Hy3 只生成候选样本；人工批准后才能进入正式评测。")

dataset_path = Path(st.text_input("Dataset 目录", value="dataset_v0")).expanduser()
if not (dataset_path / "cases.jsonl").exists():
    st.info("请输入包含 cases.jsonl 和 workspace.json 的 Dataset 目录。")
    st.stop()

try:
    cases = load_cases(dataset_path)
    workspace = load_workspace(dataset_path)
except Exception as exc:
    st.error(str(exc))
    st.stop()

if not cases:
    st.warning("Dataset 中没有样本。")
    st.stop()

labels = [
    f"{case.case_id} · {case.task_type} · {case.difficulty} · {case.human_review_status}"
    for case in cases
]
selected_label = st.selectbox("选择样本", labels)
case = cases[labels.index(selected_label)]
evidence_map = {chunk.evidence_id: chunk for chunk in workspace.chunks}

left, right = st.columns([3, 2])
with left:
    st.subheader("样本内容")
    instruction = st.text_area("Instruction", value=case.instruction, height=150)
    reference = st.text_area(
        "Reference answer", value=case.reference_answer, height=260
    )
    selected_evidence = st.multiselect(
        "Gold evidence IDs",
        options=sorted(evidence_map),
        default=[value for value in case.gold_evidence_ids if value in evidence_map],
    )
    claim_options = ["", "Supported", "Partially Supported", "Unsupported", "Contradicted"]
    claim_index = claim_options.index(case.claim_label or "")
    claim_label = st.selectbox(
        "Claim label",
        claim_options,
        index=claim_index,
        disabled=case.task_type != "claim_check",
    )
    st.write("Challenge tags:", case.challenge_tags)
    if case.auto_validation_issues:
        st.warning("；".join(case.auto_validation_issues))

with right:
    st.subheader("原始证据")
    for evidence_id in selected_evidence:
        chunk = evidence_map[evidence_id]
        with st.expander(
            f"{evidence_id} · {chunk.paper_title} · p.{chunk.page}", expanded=True
        ):
            st.write(chunk.text)

st.divider()
reviewer = st.text_input("审核人", placeholder="姓名或代号")
status = st.selectbox(
    "审核结论", ["approved", "needs_revision", "rejected", "pending"],
    index=["approved", "needs_revision", "rejected", "pending"].index(
        case.human_review_status
    ),
)
notes = st.text_area("审核备注", height=100)
if st.button("保存审核", type="primary", disabled=not reviewer.strip()):
    if not selected_evidence:
        st.error("至少选择一条 Gold Evidence。")
    elif case.task_type == "claim_check" and not claim_label:
        st.error("论断核对样本必须选择 Claim label。")
    else:
        annotation = HumanAnnotation(
            case_id=case.case_id,
            status=status,
            reviewer=reviewer.strip(),
            notes=notes.strip(),
            corrected_instruction=(instruction if instruction != case.instruction else None),
            corrected_reference_answer=(
                reference if reference != case.reference_answer else None
            ),
            corrected_gold_evidence_ids=(
                selected_evidence
                if selected_evidence != case.gold_evidence_ids
                else None
            ),
            corrected_claim_label=(claim_label or None),
        )
        upsert_annotation(dataset_path, annotation)
        st.success(f"已保存 {case.case_id}：{status}")

with st.expander("Dataset 校验状态"):
    st.json(validate_dataset(dataset_path).model_dump(mode="json"))
