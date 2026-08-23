"""agentops/ingest/pipeline.py

Three properties carry this design:
  * content addressing: a chunk's identity is the hash of its normalised text
    plus its source, so re-ingesting an unchanged document is a no-op and an
    erasure request is an exact deletion rather than a search
  * structure-aware chunking: tables, headings, and lists are not split
    arbitrarily, because a header separated from its rows is unanswerable
  * a version manifest: the corpus version is what the semantic answer cache
    keys on, so a re-ingest invalidates exactly the affected entries
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Iterable, Iterator


@dataclass(frozen=True, slots=True)
class SourceDocument:
    document_id: str
    tenant_id: str
    uri: str
    media_type: str
    raw_bytes: bytes
    fetched_at: datetime
    provenance: str                 # internal | untrusted


@dataclass(frozen=True, slots=True)
class Chunk:
    chunk_id: str                   # content hash, so identity is the content
    document_id: str
    tenant_id: str
    text: str
    section_path: tuple[str, ...]   # e.g. ("Security", "Access Control")
    page: int | None
    token_count: int
    provenance: str
    ingested_at: datetime


@dataclass
class IngestionReport:
    documents_seen: int = 0
    documents_changed: int = 0
    documents_skipped: int = 0
    chunks_written: int = 0
    chunks_deduplicated: int = 0
    failures: list[tuple[str, str]] = field(default_factory=list)
    corpus_version: str = ""

    def coverage(self) -> float:
        attempted = self.documents_seen
        failed = len(self.failures)
        return (attempted - failed) / max(attempted, 1)


_HEADING = re.compile(r"^(#{1,6})\s+(.*)$")
_TABLE_ROW = re.compile(r"^\s*\|.*\|\s*$")


class IngestionPipeline:
    def __init__(self, normaliser, embedder, chunk_store, vector_index,
                 lexical_index, manifest, target_tokens: int = 380,
                 overlap_tokens: int = 60) -> None:
        self._normalise = normaliser
        self._embed = embedder
        self._chunks = chunk_store
        self._vectors = vector_index
        self._lexical = lexical_index
        self._manifest = manifest
        self._target = target_tokens
        self._overlap = overlap_tokens

    def run(self, documents: Iterable[SourceDocument]) -> IngestionReport:
        report = IngestionReport()
        seen_hashes: set[str] = set()

        for document in documents:
            report.documents_seen += 1
            try:
                text = self._normalise(document)
            except Exception as exc:            # noqa: BLE001
                # A failed document is reported, never silently skipped. A
                # silent skip is how a class of documents disappears from the
                # index and quality collapses without an alert.
                report.failures.append((document.document_id, str(exc)))
                continue

            document_hash = _sha(text)
            if self._manifest.unchanged(document.document_id, document_hash):
                report.documents_skipped += 1
                continue
            report.documents_changed += 1

            for chunk in self._chunk(document, text):
                if chunk.chunk_id in seen_hashes:
                    report.chunks_deduplicated += 1
                    continue
                seen_hashes.add(chunk.chunk_id)
                self._chunks.upsert(chunk)
                self._vectors.upsert(chunk.chunk_id, self._embed(chunk.text),
                                     tenant_id=chunk.tenant_id)
                self._lexical.upsert(chunk.chunk_id, chunk.text,
                                     tenant_id=chunk.tenant_id)
                report.chunks_written += 1

            self._manifest.record(document.document_id, document_hash,
                                  document.provenance)

        report.corpus_version = self._manifest.seal()
        return report

    def _chunk(self, document: SourceDocument, text: str) -> Iterator[Chunk]:
        """Structure-aware chunking.

        Headings define boundaries and are carried into every child chunk as a
        section path, which restores the context a naive splitter destroys.
        Table rows are never split away from the header row above them.
        """
        section: list[str] = []
        buffer: list[str] = []
        table_header: str | None = None
        page = None

        def flush() -> Iterator[Chunk]:
            if not buffer:
                return
            body = "\n".join(buffer).strip()
            if body:
                yield self._make_chunk(document, body, tuple(section), page)
            buffer.clear()

        for line in text.splitlines():
            heading = _HEADING.match(line)
            if heading:
                yield from flush()
                depth = len(heading.group(1))
                section = section[: depth - 1] + [heading.group(2).strip()]
                table_header = None
                continue

            if _TABLE_ROW.match(line):
                if table_header is None:
                    table_header = line
                elif not buffer:
                    # A table continuing into a new chunk carries its header,
                    # otherwise the rows are meaningless on their own.
                    buffer.append(table_header)
            else:
                table_header = None

            buffer.append(line)
            if _estimate_tokens("\n".join(buffer)) >= self._target:
                tail = buffer[-self._overlap_lines():]
                yield from flush()
                buffer.extend(tail)

        yield from flush()

    def _overlap_lines(self) -> int:
        return max(1, self._overlap // 12)

    def _make_chunk(self, document: SourceDocument, body: str,
                    section: tuple[str, ...], page: int | None) -> Chunk:
        prefixed = (" > ".join(section) + "\n" + body) if section else body
        return Chunk(
            chunk_id=_sha(f"{document.document_id}|{prefixed}"),
            document_id=document.document_id,
            tenant_id=document.tenant_id,
            text=prefixed,
            section_path=section,
            page=page,
            token_count=_estimate_tokens(prefixed),
            provenance=document.provenance,
            ingested_at=datetime.now(timezone.utc),
        )


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:32]


def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)
