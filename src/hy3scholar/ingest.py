from __future__ import annotations

from pathlib import Path
import re
import uuid

from pypdf import PdfReader

from .schemas import EvidenceChunk, PaperRecord, Workspace


_SPACE_RE = re.compile(r"[ \t\f\v]+")
_SECTION_RE = re.compile(
    r"^(?:\d+(?:\.\d+)*[.)]?\s*)?"
    r"(abstract|introduction|background|related work|method(?:ology)?|"
    r"experiments?|results?|discussion|limitations?|conclusion|references|"
    r"摘要|引言|背景|相关工作|方法|实验|结果|讨论|局限|结论|参考文献)\b",
    re.I,
)


def normalize_text(text: str) -> str:
    text = text.replace("\x00", "").replace("\u200b", "")
    lines = [_SPACE_RE.sub(" ", line).strip() for line in text.splitlines()]
    output: list[str] = []
    for line in lines:
        if not line:
            if output and output[-1] != "":
                output.append("")
            continue
        if output and output[-1] and output[-1].endswith("-") and line[:1].islower():
            output[-1] = output[-1][:-1] + line
        else:
            output.append(line)
    return "\n".join(output).strip()


def infer_section(text: str) -> str | None:
    for line in text.splitlines()[:10]:
        if len(line) <= 100 and _SECTION_RE.search(line.strip()):
            return line.strip()
    return None


def split_page_text(text: str, chunk_chars: int, overlap_chars: int) -> list[str]:
    if len(text) <= chunk_chars:
        return [text] if text.strip() else []
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
    if len(paragraphs) <= 1:
        paragraphs = [part.strip() for part in text.splitlines() if part.strip()]
    chunks: list[str] = []
    current = ""
    for paragraph in paragraphs:
        if len(paragraph) > chunk_chars:
            if current:
                chunks.append(current.strip())
                current = ""
            start = 0
            step = max(1, chunk_chars - overlap_chars)
            while start < len(paragraph):
                chunks.append(paragraph[start : start + chunk_chars].strip())
                start += step
            continue
        candidate = f"{current}\n\n{paragraph}".strip()
        if current and len(candidate) > chunk_chars:
            chunks.append(current.strip())
            tail = current[-overlap_chars:].strip() if overlap_chars else ""
            current = f"{tail}\n\n{paragraph}".strip()
        else:
            current = candidate
    if current:
        chunks.append(current.strip())
    return [chunk for chunk in chunks if len(chunk) >= 30]


def _paper_title(reader: PdfReader, path: Path) -> str:
    metadata = reader.metadata
    title = getattr(metadata, "title", None) if metadata else None
    if title and str(title).strip() and str(title).lower() != "untitled":
        return normalize_text(str(title))
    return path.stem.replace("_", " ").strip()


def ingest_pdfs(
    paths: list[Path],
    *,
    workspace_id: str | None = None,
    chunk_chars: int = 2200,
    overlap_chars: int = 240,
) -> Workspace:
    if not paths:
        raise ValueError("至少需要一篇 PDF")
    workspace_id = workspace_id or uuid.uuid4().hex[:12]
    papers: list[PaperRecord] = []
    chunks: list[EvidenceChunk] = []
    for paper_index, path in enumerate(paths, 1):
        if path.suffix.lower() != ".pdf":
            raise ValueError(f"仅支持 PDF：{path.name}")
        reader = PdfReader(str(path))
        if reader.is_encrypted:
            try:
                reader.decrypt("")
            except Exception as exc:  # pragma: no cover - depends on PDF encryption
                raise ValueError(f"PDF 已加密且无法读取：{path.name}") from exc
        paper_id = f"P{paper_index:03d}"
        title = _paper_title(reader, path)
        paper_chunk_count = 0
        for page_number, page in enumerate(reader.pages, 1):
            try:
                text = normalize_text(page.extract_text() or "")
            except Exception as exc:
                raise ValueError(
                    f"无法提取 {path.name} 第 {page_number} 页文本"
                ) from exc
            if not text:
                continue
            page_chunks = split_page_text(text, chunk_chars, overlap_chars)
            section = infer_section(text)
            for local_index, chunk_text in enumerate(page_chunks, 1):
                paper_chunk_count += 1
                evidence_id = f"{paper_id}:p{page_number}:c{local_index}"
                chunks.append(
                    EvidenceChunk(
                        evidence_id=evidence_id,
                        paper_id=paper_id,
                        paper_title=title,
                        file_name=path.name,
                        page=page_number,
                        chunk_index=local_index,
                        section=section,
                        text=chunk_text,
                    )
                )
        if paper_chunk_count == 0:
            raise ValueError(
                f"{path.name} 未提取到文本；它可能是扫描件，需要先做 OCR。"
            )
        papers.append(
            PaperRecord(
                paper_id=paper_id,
                title=title,
                file_name=path.name,
                page_count=len(reader.pages),
                chunk_count=paper_chunk_count,
            )
        )
    return Workspace(workspace_id=workspace_id, papers=papers, chunks=chunks)

