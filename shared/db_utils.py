"""Shared persistence utilities with multi-tenant isolation."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Sequence


@dataclass
class DocumentChunk:
    chunk_id: str
    tenant_id: str
    content: str
    embedding: list[float] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class InMemoryVectorStore:
    """Multi-tenant vector store with cosine similarity search and metadata filtering."""

    def __init__(self):
        self._chunks: dict[str, list[DocumentChunk]] = {}  # tenant_id -> chunks

    def upsert(self, chunk: DocumentChunk) -> None:
        tenant = chunk.tenant_id
        if tenant not in self._chunks:
            self._chunks[tenant] = []
        # Replace if exists
        self._chunks[tenant] = [c for c in self._chunks[tenant] if c.chunk_id != chunk.chunk_id]
        self._chunks[tenant].append(chunk)

    def search(
        self,
        tenant_id: str,
        query_vector: list[float],
        top_k: int = 5,
        filter_metadata: dict[str, Any] | None = None,
    ) -> list[tuple[DocumentChunk, float]]:
        """Search isolated strictly to tenant_id."""
        tenant_chunks = self._chunks.get(tenant_id, [])
        scored: list[tuple[DocumentChunk, float]] = []

        for chunk in tenant_chunks:
            if filter_metadata:
                match = all(chunk.metadata.get(k) == v for k, v in filter_metadata.items())
                if not match:
                    continue

            score = self._cosine_similarity(query_vector, chunk.embedding)
            scored.append((chunk, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    @staticmethod
    def _cosine_similarity(v1: Sequence[float], v2: Sequence[float]) -> float:
        if not v1 or not v2 or len(v1) != len(v2):
            return 0.0
        dot = sum(a * b for a, b in zip(v1, v2))
        norm1 = math.sqrt(sum(a * a for a in v1))
        norm2 = math.sqrt(sum(b * b for b in v2))
        if norm1 == 0.0 or norm2 == 0.0:
            return 0.0
        return dot / (norm1 * norm2)

    def clear_tenant(self, tenant_id: str) -> None:
        if tenant_id in self._chunks:
            del self._chunks[tenant_id]


vector_store = InMemoryVectorStore()
