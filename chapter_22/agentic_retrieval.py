from __future__ import annotations

"""Adaptive retrieval policy.

Three runtime decisions: whether to retrieve, how, and whether what came back
is good enough. Each decision uses measurable signals, and the correction
loop is bounded so a hard question degrades into a refusal rather than into
an unbounded search.
"""


from dataclasses import dataclass
from enum import Enum


class Strategy(str, Enum):
    NONE = "no_retrieval"
    HYBRID = "hybrid_single_hop"
    DECOMPOSED = "decomposed_multi_hop"
    GRAPH_LOCAL = "graph_local"
    GRAPH_GLOBAL = "graph_global"


class Grade(str, Enum):
    GOOD = "good"
    AMBIGUOUS = "ambiguous"
    POOR = "poor"


@dataclass(frozen=True, slots=True)
class EvidenceGrade:
    grade: Grade
    coverage: float              # fraction of query content terms found
    distinct_sources: int
    agreement: float             # pairwise consistency among sources
    top_relevance: float         # cross-encoder score of the best passage
    rationale: str


@dataclass(frozen=True, slots=True)
class RetrievalOutcome:
    strategy: Strategy
    evidence: tuple[dict, ...]
    grade: EvidenceGrade
    corrections: int
    refused: bool
    disclosure: str = ""


class AdaptiveRetrievalPolicy:
    MAX_CORRECTIONS = 2
    COVERAGE_FLOOR = 0.55
    RELEVANCE_FLOOR = 0.45

    def __init__(self, hybrid, graph, web, reranker, decomposer,
                 classifier, grounded_mode: bool = True) -> None:
        self._hybrid = hybrid
        self._graph = graph
        self._web = web
        self._reranker = reranker
        self._decomposer = decomposer
        self._classifier = classifier
        self._grounded = grounded_mode

    # -- question one ------------------------------------------------------
    def needs_retrieval(self, query: str, state) -> bool:
        if self._classifier.is_conversational(query):
            return False
        if self._classifier.answerable_from_state(query, state):
            return False
        if self._grounded and self._classifier.is_factual(query):
            # In grounded mode the model is not permitted to decide that a
            # factual question needs no evidence.
            return True
        return self._classifier.needs_evidence(query)

    # -- question two ------------------------------------------------------
    def choose_strategy(self, query: str) -> Strategy:
        shape = self._classifier.question_shape(query)
        return {
            "lookup":      Strategy.HYBRID,
            "multi_hop":   Strategy.DECOMPOSED,
            "entity_join": Strategy.GRAPH_LOCAL,
            "thematic":    Strategy.GRAPH_GLOBAL,
        }.get(shape, Strategy.HYBRID)

    # -- question three ----------------------------------------------------
    def grade(self, query: str, evidence: list[dict]) -> EvidenceGrade:
        if not evidence:
            return EvidenceGrade(Grade.POOR, 0.0, 0, 0.0, 0.0, "empty result set")

        coverage = self._classifier.term_coverage(query, evidence)
        sources = len({item["source_id"] for item in evidence})
        agreement = self._classifier.pairwise_agreement(evidence)
        top = max(item.get("rerank_score", 0.0) for item in evidence)

        if coverage >= 0.75 and sources >= 2 and top >= 0.6:
            grade = Grade.GOOD
        elif coverage < self.COVERAGE_FLOOR or top < self.RELEVANCE_FLOOR:
            grade = Grade.POOR
        else:
            grade = Grade.AMBIGUOUS

        return EvidenceGrade(
            grade, coverage, sources, agreement, top,
            f"coverage={coverage:.2f} sources={sources} top={top:.2f}",
        )

    # -- the loop ----------------------------------------------------------
    def retrieve(self, query: str, tenant_id: str, state) -> RetrievalOutcome:
        if not self.needs_retrieval(query, state):
            return RetrievalOutcome(Strategy.NONE, (), 
                                    EvidenceGrade(Grade.GOOD, 1.0, 0, 1.0, 1.0,
                                                  "no retrieval required"),
                                    0, False)

        working_query = query
        corrections = 0
        strategy = self.choose_strategy(query)

        while True:
            evidence = self._execute(strategy, working_query, tenant_id)
            evidence = self._reranker.rerank(working_query, evidence, top_k=8)
            grade = self.grade(working_query, evidence)

            if grade.grade is Grade.GOOD:
                return RetrievalOutcome(strategy, tuple(evidence), grade,
                                        corrections, False)

            if grade.grade is Grade.AMBIGUOUS:
                # Refine rather than re-retrieve: keep only the supporting
                # strips, which raises precision without another round trip.
                refined = self._classifier.extract_supporting_strips(
                    working_query, evidence
                )
                if refined:
                    return RetrievalOutcome(
                        strategy, tuple(refined), grade, corrections, False,
                        "Some sources were only partially relevant.",
                    )

            if corrections >= self.MAX_CORRECTIONS:
                # Honest refusal beats a confident answer from poor evidence.
                return RetrievalOutcome(
                    strategy, tuple(evidence), grade, corrections, True,
                    "I could not find sufficient supporting material for this "
                    "question in the available sources.",
                )

            corrections += 1
            working_query, strategy = self._correct(
                query, working_query, strategy, grade
            )

    def _correct(self, original: str, current: str, strategy: Strategy,
                 grade: EvidenceGrade) -> tuple[str, Strategy]:
        """Escalate the correction: rewrite first, then change retriever,
        then fall back to an external source."""
        if grade.coverage < self.COVERAGE_FLOOR:
            return self._classifier.rewrite_for_coverage(original), strategy
        if strategy is Strategy.HYBRID:
            return current, Strategy.DECOMPOSED
        if strategy is Strategy.DECOMPOSED:
            return current, Strategy.GRAPH_LOCAL
        return current, Strategy.GRAPH_GLOBAL

    def _execute(self, strategy: Strategy, query: str, tenant_id: str) -> list[dict]:
        if strategy is Strategy.HYBRID:
            return self._hybrid.search(query, tenant_id=tenant_id, k=20)
        if strategy is Strategy.DECOMPOSED:
            merged: list[dict] = []
            for sub in self._decomposer.decompose(query, max_hops=3):
                merged.extend(self._hybrid.search(sub, tenant_id=tenant_id, k=10))
            return merged
        if strategy is Strategy.GRAPH_LOCAL:
            return self._graph.local_search(query, tenant_id=tenant_id,
                                            depth=2, fanout=12)
        return self._graph.global_search(query, tenant_id=tenant_id, communities=6)
