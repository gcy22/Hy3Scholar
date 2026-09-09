from __future__ import annotations

import json
from pathlib import Path

import httpx

from hy3scholar.config import Settings
from hy3scholar.dataset_models import LiteratureRecord
from hy3scholar.literature import (
    MultiSourceLiteratureSearcher,
    OpenAccessDownloader,
    deduplicate_records,
    validate_public_url,
)


def _minimal_pdf(text: str) -> bytes:
    escaped = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
    stream = f"BT /F1 12 Tf 72 720 Td ({escaped}) Tj ET".encode("ascii")
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>",
        b"<< /Length " + str(len(stream)).encode("ascii") + b" >>\nstream\n" + stream + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    data = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for index, obj in enumerate(objects, 1):
        offsets.append(len(data))
        data.extend(f"{index} 0 obj\n".encode("ascii") + obj + b"\nendobj\n")
    xref = len(data)
    data.extend(f"xref\n0 {len(objects) + 1}\n".encode("ascii"))
    data.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        data.extend(f"{offset:010d} 00000 n \n".encode("ascii"))
    data.extend(
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode("ascii")
    )
    return bytes(data)


def _record(source: str, *, doi: str | None = None) -> LiteratureRecord:
    return LiteratureRecord(
        record_id=f"L-{source}",
        title="Evidence Grounded Paper Reading",
        authors=["Alice Smith"],
        year=2025,
        doi=doi,
        abstract="A paper about grounded reading.",
        sources=[source],
        relevance_score=0.8,
    )


def test_dedup_prefers_doi_and_merges_oa_location() -> None:
    crossref = _record("crossref", doi="https://doi.org/10.1000/Test")
    openalex = _record("openalex", doi="10.1000/test").model_copy(
        update={
            "is_open_access": True,
            "pdf_url": "https://example.org/paper.pdf",
            "license": "cc-by",
        }
    )
    merged = deduplicate_records([crossref, openalex])
    assert len(merged) == 1
    assert set(merged[0].sources) == {"crossref", "openalex"}
    assert merged[0].is_open_access is True
    assert merged[0].pdf_url.endswith("paper.pdf")


def test_search_parses_three_official_sources() -> None:
    atom = """<?xml version="1.0" encoding="UTF-8"?>
    <feed xmlns="http://www.w3.org/2005/Atom" xmlns:arxiv="http://arxiv.org/schemas/atom">
      <entry><id>https://arxiv.org/abs/2501.01234v1</id><published>2025-01-01T00:00:00Z</published>
      <title>Agent Memory Evaluation</title><summary>Long enough abstract.</summary>
      <author><name>Alice Smith</name></author><link title="pdf" type="application/pdf" href="https://arxiv.org/pdf/2501.01234" /></entry>
    </feed>"""

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "api.openalex.org":
            return httpx.Response(200, json={"results": [{
                "id": "https://openalex.org/W1", "title": "Agent Memory Evaluation",
                "publication_year": 2025, "doi": "https://doi.org/10.1000/memory",
                "authorships": [{"author": {"display_name": "Alice Smith"}}],
                "abstract_inverted_index": {"Agent": [0], "memory": [1]},
                "cited_by_count": 4, "open_access": {"is_oa": True},
                "best_oa_location": {"is_oa": True, "pdf_url": "https://example.org/paper.pdf", "landing_page_url": "https://example.org/paper", "license": "cc-by", "source": {"display_name": "Journal"}},
                "primary_location": {"source": {"display_name": "Journal"}},
            }]})
        if request.url.host == "api.crossref.org":
            return httpx.Response(200, json={"message": {"items": [{
                "title": ["Agent Memory Evaluation"], "DOI": "10.1000/memory",
                "author": [{"given": "Alice", "family": "Smith"}],
                "published": {"date-parts": [[2025]]}, "URL": "https://doi.org/10.1000/memory",
                "is-referenced-by-count": 5,
            }]}})
        if request.url.host == "export.arxiv.org":
            return httpx.Response(200, text=atom)
        raise AssertionError(request.url)

    searcher = MultiSourceLiteratureSearcher(
        Settings(literature_max_retries=1, arxiv_min_interval_seconds=0),
        transport=httpx.MockTransport(handler),
    )
    report = searcher.search(["agent memory"], limit_per_source=2)
    searcher.close()
    assert not report.failures
    assert len(report.records) == 1
    assert {source for record in report.records for source in record.sources} == {
        "openalex", "crossref", "arxiv"
    }


def test_open_access_downloader_verifies_pdf(tmp_path: Path) -> None:
    pdf = _minimal_pdf("Evidence Grounded Dataset Paper with extractable title text")

    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers={"content-type": "application/pdf"}, content=pdf)

    downloader = OpenAccessDownloader(
        Settings(literature_max_retries=1),
        transport=httpx.MockTransport(handler),
        resolver=lambda *_: [(None, None, None, None, ("93.184.216.34", 443))],
    )
    record = _record("openalex", doi="10.1000/pdf").model_copy(
        update={
            "record_id": "Lpdf",
            "is_open_access": True,
            "pdf_url": "https://example.org/paper.pdf",
            "license": "cc-by",
        }
    )
    entry = downloader.download(record, tmp_path)
    downloader.close()
    assert entry.status == "open_access_downloaded"
    assert entry.page_count == 1
    assert entry.sha256
    assert Path(entry.local_path).read_bytes().startswith(b"%PDF")


def test_public_url_blocks_private_addresses() -> None:
    try:
        validate_public_url("http://127.0.0.1/paper.pdf")
    except ValueError as exc:
        assert "公网" in str(exc)
    else:
        raise AssertionError("private URL should be rejected")
