from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from hashlib import sha1, sha256
from io import BytesIO
import ipaddress
import math
from pathlib import Path
import re
import socket
import threading
import time
from typing import Any, Callable, Iterable
from urllib.parse import urljoin, urlparse
import xml.etree.ElementTree as ET

import httpx
from pypdf import PdfReader
from pypdf.errors import PdfReadError

from .config import Settings
from .dataset_models import (
    DownloadManifestEntry,
    LiteratureRecord,
    LiteratureSearchReport,
    SearchFailure,
    WebSearchSource,
)


_DOI_RE = re.compile(r"10\.\d{4,9}/[-._;()/:a-z0-9]+", re.I)
_ARXIV_RE = re.compile(r"(?:arxiv\.org/(?:abs|pdf)/)?([a-z-]+/\d{7}|\d{4}\.\d{4,5})(?:v\d+)?", re.I)
_HTML_RE = re.compile(r"<[^>]+>")
_STOPWORDS = {"a", "an", "the", "in", "of", "for", "on", "to", "and", "with", "by", "et", "al"}


def normalize_doi(value: str | None) -> str | None:
    if not value:
        return None
    match = _DOI_RE.search(value.strip())
    return match.group(0).rstrip(".,;)").lower() if match else None


def normalize_title(value: str) -> str:
    tokens = re.findall(r"[a-z0-9]+|[\u3400-\u9fff]", value.lower())
    return " ".join(token for token in tokens if token not in _STOPWORDS)


def _title_tokens(value: str) -> set[str]:
    normalized = normalize_title(value)
    return set(normalized.split())


def _first_author(record: LiteratureRecord) -> str:
    if not record.authors:
        return ""
    author = record.authors[0].strip().lower()
    return (author.split(",", 1)[0] if "," in author else author.split()[-1]).strip()


def make_record_id(*, doi: str | None, arxiv_id: str | None, title: str) -> str:
    key = normalize_doi(doi) or (arxiv_id or "").lower() or normalize_title(title)
    return "L" + sha1(key.encode("utf-8")).hexdigest()[:12]


def _jaccard(left: str, right: str) -> float:
    a, b = _title_tokens(left), _title_tokens(right)
    union = a | b
    return len(a & b) / len(union) if union else 0.0


def _metadata_completeness(record: LiteratureRecord) -> int:
    return sum(
        bool(value)
        for value in (
            record.doi,
            record.authors,
            record.year,
            record.abstract,
            record.venue,
            record.landing_url,
            record.pdf_url,
            record.license,
        )
    )


def _same_work(left: LiteratureRecord, right: LiteratureRecord) -> bool:
    left_doi, right_doi = normalize_doi(left.doi), normalize_doi(right.doi)
    if left_doi and right_doi:
        return left_doi == right_doi
    if left.arxiv_id and right.arxiv_id:
        return left.arxiv_id.lower() == right.arxiv_id.lower()
    authors_match = bool(_first_author(left)) and _first_author(left) == _first_author(right)
    return authors_match and _jaccard(left.title, right.title) >= 0.90


def _merge_pair(left: LiteratureRecord, right: LiteratureRecord) -> LiteratureRecord:
    preferred, other = sorted(
        (left, right), key=_metadata_completeness, reverse=True
    )
    data = preferred.model_dump()
    for field in (
        "paper_id", "doi", "arxiv_id", "abstract", "venue", "landing_url", "license"
    ):
        if not data.get(field) and getattr(other, field):
            data[field] = getattr(other, field)
    if (not data.get("pdf_url") or not data.get("is_open_access")) and (
        other.pdf_url and other.is_open_access
    ):
        data["pdf_url"] = other.pdf_url
        data["is_open_access"] = True
        data["license"] = other.license or data.get("license")
    data["authors"] = preferred.authors or other.authors
    data["citation_count"] = max(left.citation_count, right.citation_count)
    data["relevance_score"] = max(left.relevance_score, right.relevance_score)
    data["sources"] = list(dict.fromkeys(left.sources + right.sources))
    data["provenance"] = left.provenance + right.provenance
    data["record_id"] = make_record_id(
        doi=data.get("doi"), arxiv_id=data.get("arxiv_id"), title=data["title"]
    )
    return LiteratureRecord.model_validate(data)


def deduplicate_records(records: Iterable[LiteratureRecord]) -> list[LiteratureRecord]:
    merged: list[LiteratureRecord] = []
    for record in records:
        for index, existing in enumerate(merged):
            if _same_work(existing, record):
                merged[index] = _merge_pair(existing, record)
                break
        else:
            merged.append(record)
    return merged


def rank_records(records: list[LiteratureRecord]) -> list[LiteratureRecord]:
    if not records:
        return []
    current_year = datetime.now(timezone.utc).year
    max_citations = max(record.citation_count for record in records)
    output: list[LiteratureRecord] = []
    for record in records:
        recency = 0.5 if not record.year else max(
            0.0, min(1.0, 1.0 - (current_year - record.year) / 15.0)
        )
        impact = (
            math.log1p(record.citation_count) / math.log1p(max_citations)
            if max_citations
            else 0.0
        )
        score = 0.5 * record.relevance_score + 0.3 * recency + 0.2 * impact
        output.append(record.model_copy(update={"relevance_score": round(score, 6)}))
    return sorted(
        output,
        key=lambda item: (
            bool(item.is_open_access and item.pdf_url),
            item.relevance_score,
            item.citation_count,
        ),
        reverse=True,
    )


def _abstract_from_inverted(index: dict[str, list[int]] | None) -> str:
    if not index:
        return ""
    positions = [(position, word) for word, values in index.items() for position in values]
    return " ".join(word for _, word in sorted(positions))


class MultiSourceLiteratureSearcher:
    """Small-batch T1 search over OpenAlex, Crossref and arXiv."""

    def __init__(
        self,
        settings: Settings,
        *,
        transport: httpx.BaseTransport | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.settings = settings
        contact = settings.openalex_mailto or settings.crossref_mailto or "not-provided"
        self.http = httpx.Client(
            timeout=httpx.Timeout(settings.literature_timeout_seconds),
            headers={"User-Agent": f"Hy3Scholar/0.2 (mailto:{contact})"},
            follow_redirects=True,
            transport=transport,
        )
        self.sleep = sleep
        self._arxiv_lock = threading.Lock()
        self._last_arxiv_call = 0.0

    def close(self) -> None:
        self.http.close()

    def _get(self, url: str, *, params: dict[str, Any]) -> httpx.Response:
        last_error: Exception | None = None
        for attempt in range(self.settings.literature_max_retries):
            try:
                response = self.http.get(url, params=params)
                response.raise_for_status()
                return response
            except httpx.HTTPError as exc:
                last_error = exc
                status = exc.response.status_code if isinstance(exc, httpx.HTTPStatusError) else 0
                if status and status != 429 and status < 500:
                    break
                if attempt + 1 < self.settings.literature_max_retries:
                    self.sleep(min(4.0, float(2**attempt)))
        raise RuntimeError(f"请求失败：{url}：{last_error}") from last_error

    def search_openalex(self, query: str, limit: int) -> list[LiteratureRecord]:
        params: dict[str, Any] = {
            "search": query,
            "filter": "is_oa:true",
            "per_page": min(limit, 100),
        }
        if self.settings.openalex_api_key:
            params["api_key"] = self.settings.openalex_api_key
        if self.settings.openalex_mailto:
            params["mailto"] = self.settings.openalex_mailto
        payload = self._get("https://api.openalex.org/works", params=params).json()
        records: list[LiteratureRecord] = []
        for rank, item in enumerate(payload.get("results", []), 1):
            title = (item.get("title") or item.get("display_name") or "").strip()
            if not title:
                continue
            doi = normalize_doi(item.get("doi"))
            best_oa = item.get("best_oa_location") or {}
            primary = item.get("primary_location") or {}
            source = primary.get("source") or best_oa.get("source") or {}
            authors = [
                (entry.get("author") or {}).get("display_name", "").strip()
                for entry in item.get("authorships", [])
                if (entry.get("author") or {}).get("display_name")
            ]
            pdf_url = best_oa.get("pdf_url") if best_oa.get("is_oa") else None
            arxiv_match = _ARXIV_RE.search(str(best_oa.get("landing_page_url") or ""))
            arxiv_id = arxiv_match.group(1) if arxiv_match else None
            records.append(
                LiteratureRecord(
                    record_id=make_record_id(doi=doi, arxiv_id=arxiv_id, title=title),
                    title=title,
                    authors=authors,
                    year=item.get("publication_year"),
                    doi=doi,
                    arxiv_id=arxiv_id,
                    abstract=_abstract_from_inverted(item.get("abstract_inverted_index")),
                    venue=source.get("display_name"),
                    landing_url=best_oa.get("landing_page_url") or primary.get("landing_page_url") or item.get("id"),
                    pdf_url=pdf_url,
                    is_open_access=bool((item.get("open_access") or {}).get("is_oa")),
                    license=best_oa.get("license"),
                    citation_count=int(item.get("cited_by_count") or 0),
                    relevance_score=max(0.1, 1.0 - (rank - 1) / max(limit, 1)),
                    sources=["openalex"],
                    provenance=[{"source": "openalex", "id": item.get("id")}],
                )
            )
        return records

    def search_crossref(self, query: str, limit: int) -> list[LiteratureRecord]:
        params: dict[str, Any] = {"query.bibliographic": query, "rows": min(limit, 100)}
        if self.settings.crossref_mailto:
            params["mailto"] = self.settings.crossref_mailto
        payload = self._get("https://api.crossref.org/works", params=params).json()
        records: list[LiteratureRecord] = []
        for rank, item in enumerate(payload.get("message", {}).get("items", []), 1):
            titles = item.get("title") or []
            title = (titles[0] if titles else "").strip()
            if not title:
                continue
            authors = [
                " ".join(part for part in (author.get("given"), author.get("family")) if part)
                for author in item.get("author", [])
            ]
            date_parts = (
                (item.get("published-print") or item.get("published-online") or item.get("published") or {})
                .get("date-parts", [[None]])
            )
            year = date_parts[0][0] if date_parts and date_parts[0] else None
            doi = normalize_doi(item.get("DOI"))
            licenses = [entry.get("URL", "") for entry in item.get("license", [])]
            oa_license = next((value for value in licenses if "creativecommons.org" in value.lower()), None)
            pdf_link = next(
                (
                    entry.get("URL")
                    for entry in item.get("link", [])
                    if "pdf" in str(entry.get("content-type", "")).lower()
                ),
                None,
            )
            records.append(
                LiteratureRecord(
                    record_id=make_record_id(doi=doi, arxiv_id=None, title=title),
                    title=title,
                    authors=[author for author in authors if author],
                    year=year,
                    doi=doi,
                    abstract=_HTML_RE.sub(" ", item.get("abstract") or "").strip(),
                    venue=((item.get("container-title") or [None])[0]),
                    landing_url=item.get("URL"),
                    pdf_url=pdf_link if oa_license else None,
                    is_open_access=bool(pdf_link and oa_license),
                    license=oa_license,
                    citation_count=int(item.get("is-referenced-by-count") or 0),
                    relevance_score=max(0.1, 1.0 - (rank - 1) / max(limit, 1)),
                    sources=["crossref"],
                    provenance=[{"source": "crossref", "doi": doi, "score": item.get("score")}],
                )
            )
        return records

    def search_arxiv(self, query: str, limit: int) -> list[LiteratureRecord]:
        query_terms = [
            token
            for token in re.findall(r"[A-Za-z0-9-]+", query)
            if token.lower() not in _STOPWORDS
        ][:8]
        arxiv_query = " AND ".join(f"all:{term}" for term in query_terms) or f'all:"{query}"'
        with self._arxiv_lock:
            wait = self.settings.arxiv_min_interval_seconds - (time.monotonic() - self._last_arxiv_call)
            if wait > 0:
                self.sleep(wait)
            response = self._get(
                "https://export.arxiv.org/api/query",
                params={
                    "search_query": arxiv_query,
                    "start": 0,
                    "max_results": min(limit, 100),
                    "sortBy": "relevance",
                    "sortOrder": "descending",
                },
            )
            self._last_arxiv_call = time.monotonic()
        root = ET.fromstring(response.text)
        atom = "{http://www.w3.org/2005/Atom}"
        arxiv = "{http://arxiv.org/schemas/atom}"
        records: list[LiteratureRecord] = []
        for rank, entry in enumerate(root.findall(f"{atom}entry"), 1):
            title = " ".join((entry.findtext(f"{atom}title") or "").split())
            landing = entry.findtext(f"{atom}id") or ""
            match = _ARXIV_RE.search(landing)
            arxiv_id = match.group(1) if match else None
            if not title or not arxiv_id:
                continue
            pdf_url = next(
                (
                    link.attrib.get("href")
                    for link in entry.findall(f"{atom}link")
                    if link.attrib.get("title") == "pdf" or link.attrib.get("type") == "application/pdf"
                ),
                f"https://arxiv.org/pdf/{arxiv_id}",
            )
            published = entry.findtext(f"{atom}published") or ""
            doi = normalize_doi(entry.findtext(f"{arxiv}doi"))
            records.append(
                LiteratureRecord(
                    record_id=make_record_id(doi=doi, arxiv_id=arxiv_id, title=title),
                    title=title,
                    authors=[
                        author.findtext(f"{atom}name") or ""
                        for author in entry.findall(f"{atom}author")
                    ],
                    year=int(published[:4]) if published[:4].isdigit() else None,
                    doi=doi,
                    arxiv_id=arxiv_id,
                    abstract=" ".join((entry.findtext(f"{atom}summary") or "").split()),
                    venue="arXiv",
                    landing_url=landing,
                    pdf_url=pdf_url,
                    is_open_access=True,
                    license="arxiv-nonexclusive-distribution",
                    relevance_score=max(0.1, 1.0 - (rank - 1) / max(limit, 1)),
                    sources=["arxiv"],
                    provenance=[{"source": "arxiv", "id": arxiv_id}],
                )
            )
        return records

    def search(self, queries: list[str], *, limit_per_source: int = 6) -> LiteratureSearchReport:
        queries = list(dict.fromkeys(query.strip() for query in queries if query.strip()))
        records: list[LiteratureRecord] = []
        failures: list[SearchFailure] = []
        jobs: dict[Any, tuple[str, str]] = {}
        with ThreadPoolExecutor(max_workers=3) as executor:
            for query in queries:
                for source, method in (
                    ("openalex", self.search_openalex),
                    ("crossref", self.search_crossref),
                    ("arxiv", self.search_arxiv),
                ):
                    jobs[executor.submit(method, query, limit_per_source)] = (source, query)
            for future in as_completed(jobs):
                source, query = jobs[future]
                try:
                    records.extend(future.result())
                except Exception as exc:
                    failures.append(SearchFailure(source=source, query=query, error=str(exc)))
        return LiteratureSearchReport(
            queries=queries,
            records=rank_records(deduplicate_records(records)),
            failures=failures,
        )


def parse_web_search_sources(items: list[dict[str, Any]] | None) -> list[WebSearchSource]:
    output: list[WebSearchSource] = []
    for item in items or []:
        url = str(item.get("url") or "").strip()
        if not url:
            continue
        output.append(
            WebSearchSource(
                index=item.get("index"),
                url=url,
                name=str(item.get("name") or ""),
                snippet=str(item.get("snippet") or ""),
                site=str(item.get("site") or ""),
            )
        )
    return output


def validate_public_url(
    url: str,
    *,
    resolver: Callable[..., Any] = socket.getaddrinfo,
) -> None:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("只允许有效的 HTTP(S) PDF URL")
    if parsed.port not in {None, 80, 443}:
        raise ValueError("PDF URL 只允许 80/443 端口")
    host = parsed.hostname.lower().rstrip(".")
    if host == "localhost" or host.endswith(".localhost"):
        raise ValueError("拒绝本机地址")
    try:
        addresses = [ipaddress.ip_address(host)]
    except ValueError:
        addresses = [
            ipaddress.ip_address(item[4][0])
            for item in resolver(host, parsed.port or (443 if parsed.scheme == "https" else 80))
        ]
    if not addresses or any(
        address.is_private
        or address.is_loopback
        or address.is_link_local
        or address.is_multicast
        or address.is_reserved
        or address.is_unspecified
        for address in addresses
    ):
        raise ValueError("拒绝非公网 PDF 地址")


def _safe_filename(record: LiteratureRecord) -> str:
    author = re.sub(r"[^\w-]+", "-", _first_author(record) or "Unknown").strip("-")
    year = str(record.year or "n.d.")
    title = re.sub(r"[^\w\u3400-\u9fff-]+", "-", record.title, flags=re.UNICODE).strip("-")
    return f"{author}_{year}_{title[:80]}_{record.record_id}.pdf"


class OpenAccessDownloader:
    def __init__(
        self,
        settings: Settings,
        *,
        transport: httpx.BaseTransport | None = None,
        resolver: Callable[..., Any] = socket.getaddrinfo,
    ) -> None:
        self.settings = settings
        self.resolver = resolver
        self.http = httpx.Client(
            timeout=httpx.Timeout(settings.literature_timeout_seconds),
            headers={"User-Agent": "Hy3Scholar/0.2 open-access-downloader"},
            follow_redirects=False,
            transport=transport,
        )

    def close(self) -> None:
        self.http.close()

    def _fetch(self, url: str) -> tuple[bytes, str]:
        current = url
        for _ in range(4):
            validate_public_url(current, resolver=self.resolver)
            with self.http.stream("GET", current) as response:
                if response.status_code in {301, 302, 303, 307, 308}:
                    location = response.headers.get("location")
                    if not location:
                        raise RuntimeError("重定向缺少 Location")
                    current = urljoin(current, location)
                    continue
                if response.status_code in {401, 403}:
                    raise PermissionError(f"HTTP {response.status_code}")
                response.raise_for_status()
                data = bytearray()
                for chunk in response.iter_bytes():
                    data.extend(chunk)
                    if len(data) > self.settings.max_pdf_bytes:
                        raise ValueError("PDF 超过允许大小")
                return bytes(data), current
        raise RuntimeError("PDF 重定向次数过多")

    def download(self, record: LiteratureRecord, output_dir: Path) -> DownloadManifestEntry:
        base = {
            "record_id": record.record_id,
            "title": record.title,
            "doi": record.doi,
            "source_url": record.pdf_url,
            "license": record.license,
        }
        if not record.is_open_access or not record.pdf_url:
            return DownloadManifestEntry(status="no_authorized_pdf_found", **base)
        try:
            data, final_url = self._fetch(record.pdf_url)
            if not data.startswith(b"%PDF"):
                sample = data[:4096].lower()
                if any(marker in sample for marker in (b"captcha", b"cloudflare", b"are you a robot")):
                    return DownloadManifestEntry(
                        status="publisher_verification_waiting_user",
                        error="开放链接返回了人机验证页面，未尝试绕过",
                        **base,
                    )
                return DownloadManifestEntry(
                    status="no_authorized_pdf_found",
                    error="响应不是 PDF",
                    **base,
                )
            reader = PdfReader(BytesIO(data))
            if not reader.pages:
                raise ValueError("PDF 页数为 0")
            extracted = "\n".join((page.extract_text() or "") for page in reader.pages[:3]).strip()
            if len(extracted) < 40:
                raise ValueError("PDF 无足够可提取文本，可能需要 OCR")
            output_dir.mkdir(parents=True, exist_ok=True)
            destination = output_dir / _safe_filename(record)
            temporary = destination.with_suffix(".pdf.part")
            temporary.write_bytes(data)
            temporary.replace(destination)
            return DownloadManifestEntry(
                status="open_access_downloaded",
                source_url=final_url,
                local_path=str(destination.resolve()),
                bytes=len(data),
                sha256=sha256(data).hexdigest(),
                page_count=len(reader.pages),
                verified_text_chars=len(extracted),
                downloaded_at=datetime.now(timezone.utc),
                record_id=record.record_id,
                title=record.title,
                doi=record.doi,
                license=record.license,
            )
        except PermissionError as exc:
            return DownloadManifestEntry(
                status="publisher_blocked_waiting_user", error=str(exc), **base
            )
        except (httpx.HTTPError, OSError, ValueError, RuntimeError, PdfReadError) as exc:
            return DownloadManifestEntry(status="pdf_fetch_failed", error=str(exc), **base)
