"""Unit tests for Chapters 11 through 20."""

import pytest
from chapter_12.eval_harness import TraceStep, Trajectory, AssertionResult, must_call, step_ceiling
from chapter_13.telemetry import Pricing
from chapter_14.cost_lab import CachedAnswer
from chapter_15.improvement_loop import Surface, FailureRecord, FailureClass, FailureTriage
from chapter_16.defenses import EgressVerdict, EgressFirewall, TrifectaGuard
from chapter_17.responsible_ai import RunOutcome, DisparityReport, TrajectoryFairnessEvaluator
from chapter_18.governance import Trigger, Queue, EscalationDecision, QUEUES
from chapter_19.computer_use import (
    Grounding, ActionIntent, Expectation, StepOutcome,
    PerceptionTier, ObservationSlice, PerceptionLadder
)
from chapter_20.edge_agent import Route, LocalSignals, DeviceContext, LocalFirstRouter


def test_chapter_12_eval():
    trace = Trajectory(
        case_id="case-1",
        steps=[TraceStep(index=1, tool="doc_search", arguments={"q": "test"}, status="ok", observation={}, tokens=50)],
        answer="Paris",
        stop_reason="answered",
        total_tokens=100,
        elapsed_seconds=0.5,
    )
    assertion = must_call("doc_search")
    res = assertion(trace)
    assert res.passed


def test_chapter_13_telemetry():
    pricing = Pricing(
        input_per_million=1.0,
        output_per_million=2.0,
        reasoning_per_million=3.0,
        cached_input_per_million=0.5,
    )
    assert pricing.input_per_million == 1.0


def test_chapter_14_cached_answer():
    ans = CachedAnswer(
        answer="The capital is Paris.",
        citations=("doc1",),
        corpus_versions=(("corpus_a", "v1"),),
        intent="general",
        created_at=100.0,
        cost_saved_usd=0.05,
        embedding=(0.1, 0.2),
    )
    assert ans.answer == "The capital is Paris."


def test_chapter_15_triage():
    record = FailureRecord(
        run_id="r1",
        stop_reason="step_limit",
        failing_assertion="must_call:search",
        last_tool="search",
        last_tool_status="error",
        verify_score=0.4,
        citations=("doc1",),
        segment="en-US",
        severity="high",
        cost_usd=0.05,
    )
    assert record.run_id == "r1"


def test_chapter_16_defenses():
    firewall = EgressFirewall(allowed_hosts=frozenset({"api.internal.com", "trusted.org"}))
    verdict = firewall.check_outbound("https://api.internal.com/data", "user1")
    assert verdict.allowed


def test_chapter_17_responsible_ai():
    outcome = RunOutcome(
        segment="en-US",
        routed_path="fast",
        steps=3,
        verification_tools_used=1,
        escalated=False,
        budget_terminated=False,
        approved=True,
        correct=True,
        latency_seconds=1.2,
        cost_usd=0.05,
    )
    assert outcome.correct


def test_chapter_18_governance():
    assert "safety" in QUEUES
    assert Trigger.GUARDRAIL_BLOCK == "guardrail_block"


def test_chapter_19_computer_use():
    intent = ActionIntent(verb="click", description="Submit button", value="submit_btn")
    assert intent.verb == "click"

    class MockEnv:
        def get_a11y_tree(self):
            return "<button id='submit'>Submit</button>"
        def get_differential_crop(self, prev_hash):
            return None
        def get_full_screenshot(self):
            return b"image_bytes"

    ladder = PerceptionLadder()
    obs = ladder.perceive(MockEnv())
    assert obs.tier == PerceptionTier.A11Y_TREE
    assert obs.token_cost == 120


def test_chapter_20_edge_agent():
    router = LocalFirstRouter()
    signals = LocalSignals(
        schema_valid=True,
        retrieval_coverage=0.85,
        intent_in_distribution=True,
        output_length_ratio=1.0,
        verifier_score=0.9,
        required_steps_estimate=2,
    )
    device = DeviceContext(
        online=True,
        egress_permitted=True,
        battery_fraction=0.8,
        thermal_throttled=False,
        data_classification="ordinary",
    )
    result = router.decide(signals, device)
    assert result.route == Route.LOCAL
