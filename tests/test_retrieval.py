from hy3scholar.retrieval import SparseEvidenceRetriever
from hy3scholar.schemas import EvidenceChunk


def chunk(evidence_id: str, paper_id: str, text: str) -> EvidenceChunk:
    return EvidenceChunk(
        evidence_id=evidence_id,
        paper_id=paper_id,
        paper_title=paper_id,
        file_name=f"{paper_id}.pdf",
        page=1,
        chunk_index=1,
        text=text,
    )


def test_chinese_and_english_retrieval() -> None:
    chunks = [
        chunk("P001:p1:c1", "P001", "本文提出静态知识图谱长期记忆方法。"),
        chunk("P002:p1:c1", "P002", "A transformer baseline uses dense attention."),
        chunk("P003:p1:c1", "P003", "无关的图像分类实验。"),
    ]
    retriever = SparseEvidenceRetriever(chunks)
    assert retriever.search("知识图谱长期记忆", top_k=1)[0].chunk.paper_id == "P001"
    assert retriever.search("dense attention transformer", top_k=1)[0].chunk.paper_id == "P002"


def test_paper_filter_is_respected() -> None:
    chunks = [
        chunk("P001:p1:c1", "P001", "graph memory"),
        chunk("P002:p1:c1", "P002", "graph memory"),
    ]
    result = SparseEvidenceRetriever(chunks).search(
        "graph memory", paper_ids={"P002"}, top_k=2
    )
    assert {item.chunk.paper_id for item in result} == {"P002"}

