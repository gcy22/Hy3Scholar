from __future__ import annotations

from pathlib import Path
import tempfile

import streamlit as st

from hy3scholar.config import Settings
from hy3scholar.service import Hy3ScholarService


st.set_page_config(page_title="Hy3Scholar", page_icon="📚", layout="wide")
st.title("Hy3Scholar")
st.caption("基于原始论文证据的 Hy3 文献综述生成与可信评估")

settings = Settings()
service = Hy3ScholarService(settings)
if not settings.api_key:
    st.warning("尚未配置 HY3_API_KEY；可以构建证据库，但不能执行 Hy3 生成与评估。")

uploaded = st.file_uploader("上传 PDF 论文", type=["pdf"], accept_multiple_files=True)
if st.button("构建 Evidence Database", disabled=not uploaded):
    with tempfile.TemporaryDirectory(prefix="hy3scholar-ui-") as tmp:
        paths = []
        for index, item in enumerate(uploaded, 1):
            path = Path(tmp) / f"{index:03d}-{Path(item.name).name}"
            path.write_bytes(item.getvalue())
            paths.append(path)
        workspace = service.create_workspace(paths)
    st.session_state.workspace_id = workspace.workspace_id
    st.success(
        f"Workspace {workspace.workspace_id}：{len(workspace.papers)} 篇论文，"
        f"{len(workspace.chunks)} 个证据块"
    )

workspace_id = st.text_input(
    "Workspace ID", value=st.session_state.get("workspace_id", "")
)
question = st.text_area("研究问题", height=100)
if st.button(
    "用 Hy3 生成并评估",
    type="primary",
    disabled=not (workspace_id and question and settings.api_key),
):
    with st.spinner("Hy3 正在生成、核验并执行多维评估……"):
        result = service.run(workspace_id, question)
    left, right = st.columns([3, 2])
    with left:
        st.subheader("结构化文献综述")
        st.markdown(result.review.markdown)
    with right:
        st.metric("总体可信度", f"{result.evaluation.overall_score:.1f}/100")
        st.bar_chart(result.evaluation.dimension_scores)
        if result.evaluation.warnings:
            st.warning("\n".join(result.evaluation.warnings))
    st.subheader("Claim 级核验")
    st.dataframe(
        [item.model_dump() for item in result.evaluation.claim_evaluations],
        use_container_width=True,
    )
    st.subheader("Evidence→Review 反向覆盖审计")
    st.dataframe(
        [item.model_dump() for item in result.evaluation.coverage_evaluations],
        use_container_width=True,
    )
