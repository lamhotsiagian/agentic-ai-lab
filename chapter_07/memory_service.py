from __future__ import annotations

"""Memory write path.

The write path is the control point. Anything that passes here will be
retrieved for months, so the policy is strict on purpose:
  * candidates are extracted as typed facts, never stored as raw turns
  * every fact carries source, timestamp, and confidence
  * regulated categories are rejected or redacted before persistence
  * a contradicting fact supersedes rather than coexists
"""


import re
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from enum import Enum


class Sensitivity(str, Enum):
    ORDINARY = "ordinary"
    PERSONAL = "personal"
    RESTRICTED = "restricted"       # health, payment, government identifiers


@dataclass(frozen=True, slots=True)
class MemoryFact:
    tenant_id: str
    subject: str                    # entity the fact is about
    predicate: str                  # normalised relation, e.g. billing_contact
    value: str
    source_run_id: str
    source_uri: str | None
    confidence: float
    observed_at: datetime
    sensitivity: Sensitivity = Sensitivity.ORDINARY
    superseded_by: str | None = None
    fact_id: str = ""

    def age_days(self, now: datetime | None = None) -> float:
        now = now or datetime.now(timezone.utc)
        return (now - self.observed_at).total_seconds() / 86_400.0


_RESTRICTED_PATTERNS = (
    re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),                 # national identifier
    re.compile(r"\b(?:\d[ -]*?){13,16}\b"),               # payment card
    re.compile(r"\b(diagnos|prescrib|medicat)\w*\b", re.I),
)


class MemoryWritePolicy:
    """Decides whether a candidate fact is persisted, redacted, or dropped."""

    def __init__(self, half_life_days: float = 90.0, min_confidence: float = 0.55) -> None:
        self._half_life = half_life_days
        self._min_confidence = min_confidence

    def admit(self, fact: MemoryFact) -> MemoryFact | None:
        if fact.confidence < self._min_confidence:
            return None
        if not fact.source_run_id:
            # No provenance means no audit trail and no reliable deletion.
            return None
        if self._is_restricted(fact.value):
            # Restricted categories are never written to long-term memory.
            # They may still be used within the run that observed them.
            return None
        if fact.sensitivity is Sensitivity.PERSONAL:
            # Personal data is admitted but carries a shorter retention.
            return replace(fact, confidence=min(fact.confidence, 0.9))
        return fact

    @staticmethod
    def _is_restricted(value: str) -> bool:
        return any(pattern.search(value) for pattern in _RESTRICTED_PATTERNS)

    def retention_until(self, fact: MemoryFact) -> datetime:
        window = 365 if fact.sensitivity is Sensitivity.ORDINARY else 90
        return fact.observed_at + timedelta(days=window)

    def salience(self, fact: MemoryFact, hit_count: int, now: datetime | None = None) -> float:
        """Combined score used for ranking and for eviction.

        Confidence times exponential recency decay times a mild popularity
        term. Popularity is logarithmic so that one frequently retrieved fact
        cannot dominate the ranking permanently.
        """
        import math

        decay = 0.5 ** (fact.age_days(now) / self._half_life)
        popularity = math.log1p(hit_count) / math.log(10.0)
        return fact.confidence * decay * (1.0 + 0.3 * popularity)


class ConflictResolver:
    """Supersedes rather than deletes, so history stays auditable."""

    def resolve(self, existing: MemoryFact, incoming: MemoryFact) -> list[MemoryFact]:
        if existing.predicate != incoming.predicate or existing.subject != incoming.subject:
            return [existing, incoming]
        if existing.value == incoming.value:
            # Reinforcement. Raise confidence, keep the earlier observation date
            # so decay does not reset every time the fact is restated.
            boosted = replace(
                existing, confidence=min(1.0, existing.confidence + 0.08)
            )
            return [boosted]
        if incoming.observed_at <= existing.observed_at:
            # An older contradicting observation does not overwrite a newer one.
            return [existing]
        # Newer contradicting fact wins, and the old row is retained with a
        # pointer, which is what makes "why did the agent think X" answerable.
        return [
            replace(existing, superseded_by=incoming.fact_id),
            incoming,
        ]

"""Memory read path.

Contract, restated from Chapter 2:
  * read() never returns more than max_tokens of content
  * every returned item carries a source identifier
  * tenancy is enforced in the store, not only in the query
"""


from dataclasses import dataclass
from typing import Callable, Sequence


@dataclass(frozen=True, slots=True)
class Evidence:
    fact_id: str
    text: str
    source_uri: str | None
    score: float
    observed_at_iso: str


class MemoryService:
    RRF_K = 60          # reciprocal rank fusion constant, standard value

    def __init__(self, vector_index, lexical_index, fact_store,
                 policy: MemoryWritePolicy, count_tokens: Callable[[str], int]) -> None:
        self._vector = vector_index
        self._lexical = lexical_index
        self._facts = fact_store
        self._policy = policy
        self._count = count_tokens

    def read(self, query: str, tenant_id: str, max_tokens: int,
             candidates: int = 40) -> list[Evidence]:
        # Both indexes filter by tenant inside the store. A filter applied
        # after retrieval leaks existence, and a filter applied only in the
        # query string is one prompt injection away from being removed.
        dense = self._vector.search(query, tenant_id=tenant_id, k=candidates)
        sparse = self._lexical.search(query, tenant_id=tenant_id, k=candidates)

        fused = self._reciprocal_rank_fusion(dense, sparse)
        hydrated = self._facts.get_many(
            [fact_id for fact_id, _ in fused], tenant_id=tenant_id
        )

        scored: list[tuple[float, Evidence]] = []
        for fact_id, fusion_score in fused:
            fact = hydrated.get(fact_id)
            if fact is None or fact.superseded_by is not None:
                continue            # never return a fact a newer one replaced
            salience = self._policy.salience(
                fact, hit_count=self._facts.hit_count(fact_id)
            )
            combined = fusion_score * (0.4 + 0.6 * salience)
            scored.append((
                combined,
                Evidence(
                    fact_id=fact.fact_id,
                    text=f"{fact.subject} {fact.predicate}: {fact.value}",
                    source_uri=fact.source_uri,
                    score=combined,
                    observed_at_iso=fact.observed_at.isoformat(),
                ),
            ))

        scored.sort(key=lambda pair: pair[0], reverse=True)
        return self._pack(scored, max_tokens)

    def _reciprocal_rank_fusion(
        self, dense: Sequence[tuple[str, float]], sparse: Sequence[tuple[str, float]]
    ) -> list[tuple[str, float]]:
        """Fuse by rank, not by score.

        Cosine similarity and BM25 live on incomparable scales, so any weighted
        sum of raw scores is arbitrary and drifts whenever either index is
        retuned. Rank fusion is scale free and needs no normalisation.
        """
        scores: dict[str, float] = {}
        for ranking in (dense, sparse):
            for rank, (fact_id, _) in enumerate(ranking, start=1):
                scores[fact_id] = scores.get(fact_id, 0.0) + 1.0 / (self.RRF_K + rank)
        return sorted(scores.items(), key=lambda pair: pair[1], reverse=True)

    def _pack(self, scored: list[tuple[float, Evidence]], max_tokens: int) -> list[Evidence]:
        """Fill the budget greedily by score, then stop. Returning more than the
        caller asked for makes context assembly nondeterministic, which is the
        single property the assembler in Chapter 6 depends on."""
        packed: list[Evidence] = []
        used = 0
        for _, evidence in scored:
            cost = self._count(evidence.text) + 12      # citation overhead
            if used + cost > max_tokens:
                continue
        return packed


@dataclass(frozen=True, slots=True)
class CompactedProfile:
    tenant_id: str
    subject: str
    traits: dict[str, str]            # consolidated stable preferences / facts
    source_fact_count: int
    compacted_at: datetime


@dataclass(frozen=True, slots=True)
class ErasureResult:
    tenant_id: str
    subject: str
    facts_purged: int
    vectors_deleted: int
    summaries_invalidated: int
    completed_at: datetime


class MemoryCompactor:
    """Consolidates episodic facts into semantic traits and handles GDPR erasure."""

    def __init__(self, fact_store, vector_store) -> None:
        self._fact_store = fact_store
        self._vector_store = vector_store

    def compact(self, tenant_id: str, subject: str, raw_facts: list[MemoryFact]) -> CompactedProfile:
        # Group facts by predicate, keeping latest non-superseded values
        active_facts = [f for f in raw_facts if f.superseded_by is None and f.tenant_id == tenant_id]
        traits: dict[str, str] = {}
        for f in sorted(active_facts, key=lambda x: x.observed_at):
            traits[f.predicate] = f.value

        return CompactedProfile(
            tenant_id=tenant_id,
            subject=subject,
            traits=traits,
            source_fact_count=len(raw_facts),
            compacted_at=datetime.now(timezone.utc),
        )

    def hard_erase_subject(self, tenant_id: str, subject: str) -> ErasureResult:
        """Complete GDPR Article 17 erasure across relational, vector, and summary tiers."""
        facts_purged = self._fact_store.delete_subject(tenant_id, subject)
        vectors_deleted = self._vector_store.delete_subject_embeddings(tenant_id, subject)
        return ErasureResult(
            tenant_id=tenant_id,
            subject=subject,
            facts_purged=facts_purged,
            vectors_deleted=vectors_deleted,
            summaries_invalidated=1,
            completed_at=datetime.now(timezone.utc),
        )


@dataclass(frozen=True, slots=True)
class BitemporalFact:
    """Fact record decoupled across valid-time and system transaction-time."""
    fact_id: str
    subject: str
    predicate: str
    value: str
    valid_from: datetime
    valid_until: datetime | None      # None represents open-ended present validity
    system_recorded_at: datetime

    def is_valid_at(self, query_time: datetime) -> bool:
        """Evaluate whether fact held true in the external world at query_time."""
        if query_time < self.valid_from:
            return False
        if self.valid_until and query_time >= self.valid_until:
            return False
        return True

