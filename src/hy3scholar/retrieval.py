from __future__ import annotations

from collections import Counter, defaultdict
import math
import re

from .schemas import EvidenceChunk, RetrievedEvidence


_WORD_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.+\-/]*")
_CJK_RE = re.compile(r"[\u3400-\u9fff]")


def tokenize(text: str) -> list[str]:
    lowered = text.lower()
    words = _WORD_RE.findall(lowered)
    cjk = _CJK_RE.findall(lowered)
    cjk_bigrams = ["".join(cjk[i : i + 2]) for i in range(max(0, len(cjk) - 1))]
    return words + cjk + cjk_bigrams


class SparseEvidenceRetriever:
    """Dependency-free BM25 retriever with Chinese character bigrams."""

    def __init__(self, chunks: list[EvidenceChunk]) -> None:
        self.chunks = chunks
        self.term_freqs = [Counter(tokenize(chunk.text)) for chunk in chunks]
        self.lengths = [sum(freq.values()) for freq in self.term_freqs]
        self.avg_length = sum(self.lengths) / max(1, len(self.lengths))
        self.document_frequency: Counter[str] = Counter()
        for freq in self.term_freqs:
            self.document_frequency.update(freq.keys())

    def search(
        self,
        query: str,
        *,
        top_k: int = 10,
        paper_ids: set[str] | None = None,
        max_per_paper: int = 4,
    ) -> list[RetrievedEvidence]:
        query_terms = Counter(tokenize(query))
        if not query_terms or not self.chunks:
            return []
        n_docs = len(self.chunks)
        k1, b = 1.5, 0.75
        scored: list[tuple[float, int]] = []
        for index, (chunk, freq, length) in enumerate(
            zip(self.chunks, self.term_freqs, self.lengths, strict=True)
        ):
            if paper_ids and chunk.paper_id not in paper_ids:
                continue
            score = 0.0
            for term, qtf in query_terms.items():
                tf = freq.get(term, 0)
                if not tf:
                    continue
                df = self.document_frequency[term]
                idf = math.log(1 + (n_docs - df + 0.5) / (df + 0.5))
                norm = tf + k1 * (1 - b + b * length / max(1, self.avg_length))
                score += idf * (tf * (k1 + 1) / norm) * (1 + math.log(qtf))
            if score > 0:
                scored.append((score, index))
        scored.sort(reverse=True)

        selected: list[RetrievedEvidence] = []
        per_paper: dict[str, int] = defaultdict(int)
        for score, index in scored:
            chunk = self.chunks[index]
            if per_paper[chunk.paper_id] >= max_per_paper:
                continue
            selected.append(RetrievedEvidence(chunk=chunk, score=score))
            per_paper[chunk.paper_id] += 1
            if len(selected) >= top_k:
                break
        return selected


def format_evidence_context(
    evidence: list[RetrievedEvidence] | list[EvidenceChunk],
    *,
    max_chars: int = 52_000,
) -> str:
    blocks: list[str] = []
    used = 0
    for item in evidence:
        chunk = item.chunk if isinstance(item, RetrievedEvidence) else item
        header = (
            f"[{chunk.evidence_id}] paper={chunk.paper_id}; "
            f"title={chunk.paper_title}; page={chunk.page}; "
            f"section={chunk.section or 'unknown'}"
        )
        block = f"{header}\n{chunk.text.strip()}"
        if used + len(block) > max_chars:
            break
        blocks.append(block)
        used += len(block)
    return "\n\n---\n\n".join(blocks)

