"""Unit tests for Chapters 21 through 32."""

import pytest
from chapter_21.coding_agent import CodeUnit, LocalisationResult, RepositoryLocaliser, PatchIntegrityGuard, IntegrityVerdict
from chapter_22.agentic_retrieval import Strategy, Grade, EvidenceGrade, AdaptiveRetrievalPolicy
from chapter_23.patterns import BreakerState, CircuitBreaker
from chapter_24.scale_lab import Priority, AdmissionResult, TokenBucket, AdmissionController
from chapter_27.drills import Attempt, retry_typed, reciprocal_rank_fusion, EventStreamParser, NoProgressDetector
from agentops.core.state import Phase, StudioState, new_state
from agentops.eval.gate import GateResult, ReleaseGate
from agentops.api.authz import Principal, RequestAuthoriser


def test_chapter_21_coding_agent():
    guard = PatchIntegrityGuard(test_path_patterns=("test_", "_test."))
    assert guard is not None


def test_chapter_22_agentic_retrieval():
    grade = EvidenceGrade(
        grade=Grade.GOOD,
        coverage=0.9,
        distinct_sources=3,
        agreement=0.95,
        top_relevance=0.9,
        rationale="well grounded",
    )
    assert grade.grade == Grade.GOOD


def test_chapter_23_circuit_breaker():
    breaker = CircuitBreaker(name="llm-primary", failure_rate_threshold=0.5)
    assert breaker.allow()


def test_chapter_24_admission():
    bucket = TokenBucket(capacity=100, refill_per_second=10, tokens=50)
    assert bucket.take(10)
    assert bucket.headroom() <= 1.0




def test_chapter_27_drills():
    parser = EventStreamParser()
    events = list(parser.feed('{"status": "ok"}\n'))
    assert len(events) == 1
    assert events[0]["status"] == "ok"

    rankings = [["a", "b", "c"], ["b", "a", "d"]]
    fused = reciprocal_rank_fusion(rankings)
    assert len(fused) == 4

    detector = NoProgressDetector()
    assert not detector.observe(tool="search", arguments={"q": "rag"})
    assert not detector.observe(tool="search", arguments={"q": "graphrag"})


def test_chapter_30_agentops_state():
    state = new_state(run_id="run-1", tenant_id="acme", principal_id="u1", question="test research")
    assert state["phase"] == Phase.INTAKE


def test_chapter_31_gate():
    res = GateResult(
        allowed=True,
        reasons=(),
        aggregate_success=0.95,
        per_stratum={},
        cost_per_success=0.10,
        safety_failures=0,
    )
    assert res.allowed


def test_chapter_32_principal():
    principal = Principal(
        subject="user1",
        tenant_id="acme",
        scopes=frozenset({"research:create"}),
        email="user@acme.com",
    )
    assert principal.tenant_id == "acme"
