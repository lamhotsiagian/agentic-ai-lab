from __future__ import annotations

"""Semantic answer cache.

Correct invalidation is the whole problem. The key includes every corpus
version the answer depended on, so a corpus update invalidates exactly the
entries it should and nothing else.
"""


import hashlib
import time
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class CachedAnswer:
    answer: str
    citations: tuple[str, ...]
    corpus_versions: tuple[tuple[str, str], ...]   # (corpus_id, version)
    intent: str
    created_at: float
    cost_saved_usd: float
    embedding: tuple[float, ...]


class SemanticAnswerCache:
    # Freshness tolerance differs by intent, because staleness costs differ.
    MAX_AGE_SECONDS = {
        "policy_lookup": 86_400,
        "documentation": 21_600,
        "account_status": 60,
        "inventory": 15,
    }
    SIMILARITY_FLOOR = 0.94        # deliberately high; a near miss is a wrong answer

    def __init__(self, vector_store, corpus_registry, embedder, metrics) -> None:
        self._store = vector_store
        self._registry = corpus_registry
        self._embed = embedder
        self._metrics = metrics

    def get(self, question: str, tenant_id: str, intent: str) -> CachedAnswer | None:
        vector = self._embed(self._normalise(question))
        hits = self._store.search(
            vector, tenant_id=tenant_id, intent=intent, k=3
        )
        for entry, similarity in hits:
            if similarity < self.SIMILARITY_FLOOR:
                continue
            if not self._fresh(entry):
                self._metrics.increment("cache.miss", reason="stale")
                continue
            if not self._corpora_unchanged(entry):
                self._metrics.increment("cache.miss", reason="corpus_changed")
                continue
            self._metrics.increment("cache.hit", intent=intent)
            self._metrics.observe("cache.cost_saved_usd", entry.cost_saved_usd)
            return entry
        self._metrics.increment("cache.miss", reason="no_match")
        return None

    def put(self, question: str, tenant_id: str, intent: str, answer: str,
            citations: tuple[str, ...], corpus_versions: tuple[tuple[str, str], ...],
            degraded_rung: int, tool_statuses: tuple[str, ...],
            cost_usd: float) -> bool:
        # Never cache a degraded run. Caching one converts a transient outage
        # into a persistent wrong answer, which is strictly worse than a miss.
        if degraded_rung > 0:
            return False
        if any(status in {"EMPTY", "UNAVAILABLE"} for status in tool_statuses):
            return False
        if not citations:
            # An answer with no citations in a grounded mode is not a fact,
            # so it must not become a cached fact.
            return False

        self._store.upsert(
            key=self._key(question, tenant_id, intent),
            entry=CachedAnswer(
                answer=answer,
                citations=citations,
                corpus_versions=corpus_versions,
                intent=intent,
                created_at=time.time(),
                cost_saved_usd=cost_usd,
                embedding=self._embed(self._normalise(question)),
            ),
        )
        return True

    def _fresh(self, entry: CachedAnswer) -> bool:
        limit = self.MAX_AGE_SECONDS.get(entry.intent, 3_600)
        return (time.time() - entry.created_at) < limit

    def _corpora_unchanged(self, entry: CachedAnswer) -> bool:
        return all(
            self._registry.current_version(corpus_id) == version
            for corpus_id, version in entry.corpus_versions
        )

    @staticmethod
    def _normalise(question: str) -> str:
        return " ".join(question.lower().split())

    @staticmethod
    def _key(question: str, tenant_id: str, intent: str) -> str:
        payload = f"{tenant_id}|{intent}|{' '.join(question.lower().split())}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

"""Cost governor.

Three nested ceilings, each with a graduated response. A single hard stop at
one level is a poor design: it converts a cost problem into an availability
problem for every user at once.
"""


from dataclasses import dataclass
from datetime import date
from enum import Enum


class Action(str, Enum):
    ALLOW = "allow"
    DOWNGRADE = "downgrade"        # cheaper tier, tighter step cap
    QUEUE = "queue"                # defer to the batch tier
    DENY = "deny"


@dataclass(frozen=True, slots=True)
class GovernorDecision:
    action: Action
    reason: str
    max_steps: int
    model_tier: str
    user_message: str = ""


class CostGovernor:
    def __init__(self, ledger, per_run_usd: float, per_tenant_daily_usd: dict[str, float],
                 global_daily_usd: float) -> None:
        self._ledger = ledger
        self._per_run = per_run_usd
        self._per_tenant = per_tenant_daily_usd
        self._global = global_daily_usd

    def authorise(self, tenant_id: str, estimated_usd: float,
                  interactive: bool, today: date | None = None) -> GovernorDecision:
        today = today or date.today()
        tenant_spend = self._ledger.tenant_spend(tenant_id, today)
        tenant_cap = self._per_tenant.get(tenant_id, 25.0)
        global_spend = self._ledger.global_spend(today)

        # Global ceiling: protect the business, but degrade rather than deny,
        # because a hard global stop is an outage.
        if global_spend >= self._global:
            return GovernorDecision(
                Action.DENY, "global_daily_ceiling", 0, "none",
                "The service is temporarily limited. Please retry later.",
            )
        if global_spend >= 0.9 * self._global:
            return GovernorDecision(
                Action.DOWNGRADE, "global_soft_ceiling", 4, "small",
                "Running in reduced mode to stay within today's capacity.",
            )

        # Tenant ceiling: this is a commercial boundary, so it is explicit to
        # the tenant rather than presented as a system failure.
        if tenant_spend >= tenant_cap:
            if interactive:
                return GovernorDecision(
                    Action.DENY, "tenant_daily_ceiling", 0, "none",
                    "Your organisation has reached today's usage limit. "
                    "An administrator can raise it in settings.",
                )
            return GovernorDecision(
                Action.QUEUE, "tenant_ceiling_batch", 6, "small",
                "Queued for overnight processing.",
            )
        if tenant_spend >= 0.85 * tenant_cap:
            return GovernorDecision(
                Action.DOWNGRADE, "tenant_soft_ceiling", 5, "small",
                "Approaching today's usage limit, running in reduced mode.",
            )

        # Per-run ceiling: an individual request must not be able to consume a
        # meaningful fraction of the daily budget on its own.
        if estimated_usd > self._per_run:
            return GovernorDecision(
                Action.DOWNGRADE, "expensive_request", 4, "small",
                "This request is unusually large, so it runs in reduced mode.",
            )

        return GovernorDecision(Action.ALLOW, "within_budget", 10, "primary")
