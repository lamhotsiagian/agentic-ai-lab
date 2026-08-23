from __future__ import annotations

"""Exemplar selection for agent prompts.

Selection is similarity-based, ranking is outcome-weighted, and the final
set is diversified. The cluster cache exists because per-request exemplar
churn destroys prompt prefix caching, which is often a larger cost than the
quality gain is worth.
"""


from dataclasses import dataclass
from functools import lru_cache


@dataclass(frozen=True, slots=True)
class Exemplar:
    exemplar_id: str
    request: str
    trajectory: str            # rendered tool calls and observations
    outcome_score: float       # 0..1, from the evaluation harness
    cluster_id: str
    embedding: tuple[float, ...]


class ExemplarSelector:
    def __init__(self, index, max_exemplars: int = 3,
                 min_outcome: float = 0.8, diversity_lambda: float = 0.65) -> None:
        self._index = index
        self._k = max_exemplars
        self._min_outcome = min_outcome
        self._lambda = diversity_lambda

    def select(self, request: str, cluster_id: str | None = None) -> list[Exemplar]:
        if cluster_id is not None:
            # Cluster-level caching keeps the prompt prefix byte-stable across
            # every request in the cluster, which preserves the provider cache.
            return self._for_cluster(cluster_id)
        candidates = [
            item for item in self._index.similar(request, k=20)
            if item.outcome_score >= self._min_outcome
        ]
        return self._diversify(request, candidates)

    @lru_cache(maxsize=512)
    def _for_cluster(self, cluster_id: str) -> tuple[Exemplar, ...]:
        candidates = [
            item for item in self._index.by_cluster(cluster_id, k=20)
            if item.outcome_score >= self._min_outcome
        ]
        centroid = self._index.centroid(cluster_id)
        return tuple(self._diversify_from_vector(centroid, candidates))

    def _diversify(self, request: str, candidates: list[Exemplar]) -> list[Exemplar]:
        return self._diversify_from_vector(self._index.embed(request), candidates)

    def _diversify_from_vector(self, query_vec, candidates: list[Exemplar]) -> list[Exemplar]:
        """Maximal marginal relevance.

        Relevance to the query, penalised by similarity to what is already
        selected. Without this, three exemplars of the same shape crowd out the
        one exemplar covering the case the request actually resembles.
        """
        selected: list[Exemplar] = []
        pool = list(candidates)
        while pool and len(selected) < self._k:
            best, best_score = None, float("-inf")
            for candidate in pool:
                relevance = _cosine(query_vec, candidate.embedding)
                redundancy = max(
                    (_cosine(candidate.embedding, chosen.embedding) for chosen in selected),
                    default=0.0,
                )
                score = (
                    self._lambda * relevance
                    - (1.0 - self._lambda) * redundancy
                    + 0.15 * candidate.outcome_score
                )
                if score > best_score:
                    best, best_score = candidate, score
            selected.append(best)
            pool.remove(best)
        return selected


def _cosine(a, b) -> float:
    numerator = sum(x * y for x, y in zip(a, b))
    left = sum(x * x for x in a) ** 0.5
    right = sum(y * y for y in b) ** 0.5
    return numerator / (left * right + 1e-9)

"""Skill library with a promotion gate.

A skill enters production only after it beats the current baseline on
held-out cases by a stated margin AND survives a shadow period on live
traffic. It leaves production automatically when its measured success rate
regresses. Both directions are automatic on purpose: a library that only
grows becomes a liability within two quarters.
"""


from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum


class SkillStatus(str, Enum):
    CANDIDATE = "candidate"
    SHADOW = "shadow"
    ACTIVE = "active"
    DEMOTED = "demoted"


@dataclass(frozen=True, slots=True)
class ToolStep:
    tool: str
    argument_template: str


@dataclass
class Skill:
    skill_id: str
    version: int
    name: str
    task_signature: str            # normalised intent this skill applies to
    precondition: str              # checked before the skill may be used
    postcondition: str             # checked after, to record success
    steps: tuple[ToolStep, ...]
    status: SkillStatus = SkillStatus.CANDIDATE
    offline_success: float = 0.0
    live_success: float = 0.0
    live_attempts: int = 0
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    source_run_ids: tuple[str, ...] = ()


class SkillLibrary:
    PROMOTION_MARGIN = 0.05        # must beat the generic agent by 5 points
    SHADOW_MINIMUM = 200           # live attempts before promotion
    DEMOTION_FLOOR = 0.04          # demote if it falls 4 points below baseline

    def __init__(self, evaluator, store) -> None:
        self._evaluator = evaluator
        self._store = store

    def consider(self, candidate: Skill, baseline_success: float) -> SkillStatus:
        """Offline gate. Runs the candidate against held-out cases only.

        Held-out means cases that were never used to extract the candidate.
        Evaluating on the traces that produced the skill measures memorisation,
        which is how a library fills up with skills that work exactly once.
        """
        report = self._evaluator.run(
            skill=candidate, suite="held_out", exclude_runs=candidate.source_run_ids
        )
        candidate.offline_success = report.success_rate

        if report.success_rate < baseline_success + self.PROMOTION_MARGIN:
            candidate.status = SkillStatus.DEMOTED
            self._store.record(candidate, reason="failed_offline_gate")
            return candidate.status

        candidate.status = SkillStatus.SHADOW
        self._store.record(candidate, reason="entered_shadow")
        return candidate.status

    def observe_live(self, skill: Skill, succeeded: bool, baseline_success: float) -> SkillStatus:
        """Online gate, called once per live invocation."""
        skill.live_attempts += 1
        # Running mean, so the library needs no separate aggregation job.
        skill.live_success += (
            (1.0 if succeeded else 0.0) - skill.live_success
        ) / skill.live_attempts

        if skill.status is SkillStatus.SHADOW and skill.live_attempts >= self.SHADOW_MINIMUM:
            if skill.live_success >= baseline_success:
                skill.status = SkillStatus.ACTIVE
                self._store.record(skill, reason="promoted")
            else:
                skill.status = SkillStatus.DEMOTED
                self._store.record(skill, reason="failed_shadow")

        elif skill.status is SkillStatus.ACTIVE and skill.live_attempts >= self.SHADOW_MINIMUM:
            if skill.live_success < baseline_success - self.DEMOTION_FLOOR:
                # Environments drift. A skill that encoded last quarter's API
                # shape must leave production without a human noticing first.
                skill.status = SkillStatus.DEMOTED
                self._store.record(skill, reason="live_regression")

        return skill.status

    def applicable(self, task_signature: str) -> list[Skill]:
        return [
            skill for skill in self._store.by_signature(task_signature)
            if skill.status is SkillStatus.ACTIVE
        ]
