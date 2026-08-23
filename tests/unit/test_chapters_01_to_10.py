"""Unit tests for Chapters 1 through 10."""

from datetime import datetime, timezone
import pytest

from chapter_01.bounded_loop import RunBudget, StopReason, Step, RunResult, Proposal, AgentState, BoundedAgentLoop
from chapter_02.design_console import (
    RunPhase, Evidence, SubGoal, merge_budget, keep_latest_plan,
    DegradationController, Rung, Capability, RunProfile, new_run_state
)
from chapter_03.reasoning_router import Effort, ComplexityClassifier, PlanStep, RewooPlanner
from chapter_04.durable_graph import (
    ResearchState, StepDecision, ResearchPolicy, make_verify_router,
    WorkflowEvent, DurableWorkflowEngine
)
from chapter_05.tool_gateway import ResultStatus, ToolResult, ToolSpec, PolicyEngine, ToolGateway
from chapter_06.context_assembler import Section, AssembledContext, ContextAssembler, Classification, IntentRouter
from chapter_07.memory_service import (
    Sensitivity, MemoryFact, MemoryWritePolicy, ConflictResolver, MemoryService,
    CompactedProfile, ErasureResult, MemoryCompactor, BitemporalFact
)
from chapter_08.skill_library import Exemplar, ExemplarSelector, Skill, SkillLibrary, SkillStatus, ToolStep
from chapter_09.supervisor import (
    Budget, Handoff, Supervisor, WorkerOutcome, ConflictResolver as SupConflictResolver,
    SagaStatus, SagaStep, CompensationVerdict, SagaCoordinator, DeadlockDetector
)
from chapter_10.protocols import ServerPin, PinMismatch, HardenedMcpClient, TaskState, Task


class MockLLMClient:
    def complete(self, **kwargs):
        class Response:
            text = "0.25"
        return Response()


def test_chapter_01_bounded_loop():
    budget = RunBudget(max_steps=5, max_tokens=1000, wall_clock_seconds=10.0)
    budget.steps_used = 2
    assert budget.exhausted() is None
    budget.steps_used = 6
    assert budget.exhausted() == StopReason.STEP_LIMIT


def test_chapter_02_state_and_degradation():
    cap = Capability(reranker=True, web_search=True, primary_model=True, vector_store=True, lexical_store=True)
    controller = DegradationController()
    profile = controller.select(cap)
    assert profile.rung == Rung.FULL

    now = datetime.now(timezone.utc)
    ev = Evidence(claim="test fact", source_id="doc1", source_uri="http://example.com", confidence=0.95, retrieved_at=now)
    assert ev.as_citation() == "[doc1] test fact"

    st = new_run_state(run_id="r1", tenant_id="acme", goal="find answer")
    assert st["phase"] == RunPhase.PLANNING


def test_chapter_03_reasoning():
    client = MockLLMClient()
    classifier = ComplexityClassifier(client=client, model_id="qwen2.5:3b")
    score = classifier.score("simple question")
    assert isinstance(score, float)

    step = PlanStep(index=1, rationale="Look up facts", tool="search", argument_template={"q": "python"})
    assert step.index == 1


def test_chapter_04_durable_graph():
    st = ResearchState(run_id="run-1", tenant_id="acme", goal="research")
    assert st["run_id"] == "run-1"

    handlers = {"query_db": lambda sql: {"rows": [{"id": 1}]}}
    engine = DurableWorkflowEngine(activity_handlers=handlers)
    res = engine.execute_activity("act_1", "query_db", {"sql": "SELECT 1"})
    assert res == {"rows": [{"id": 1}]}
    assert len(engine.history) == 1


def test_chapter_05_tool_gateway():
    res = ToolResult(status=ResultStatus.SUCCESS, tool="search", content={"items": []})
    assert res.ok
    assert res.status == ResultStatus.SUCCESS

    spec = ToolSpec(
        name="search",
        description="Search tool",
        schema={"type": "object"},
        handler=lambda **kw: {"items": []},
        has_write_effect=False,
        timeout_seconds=5.0,
        max_attempts=3,
        projection=("items",),
        required_scopes=frozenset({"read"}),
    )
    assert spec.name == "search"


def test_chapter_06_context_assembler():
    assembler = ContextAssembler(
        count_tokens=lambda text: len(text.split()),
        window=4000,
        reserve_for_output=500,
    )
    sections = [Section(name="system", content="You are an agent.", budget_tokens=500, evictable=False, priority=1)]
    assembled = assembler.assemble(sections)
    assert "You are an agent." in assembled.text


def test_chapter_07_memory():
    now = datetime.now(timezone.utc)
    fact = MemoryFact(
        tenant_id="acme",
        subject="user",
        predicate="prefers_model",
        value="fast",
        source_run_id="r1",
        source_uri="http://example.com",
        confidence=0.9,
        observed_at=now,
        sensitivity=Sensitivity.ORDINARY,
        superseded_by=None,
        fact_id="f1",
    )
    assert fact.subject == "user"

    class MockFactStore:
        def delete_subject(self, t, s):
            return 3

    class MockVecStore:
        def delete_subject_embeddings(self, t, s):
            return 3

    compactor = MemoryCompactor(MockFactStore(), MockVecStore())
    erasure = compactor.hard_erase_subject("acme", "user")
    assert erasure.facts_purged == 3

    bitemp = BitemporalFact(
        fact_id="bf1",
        subject="company",
        predicate="lead_architect",
        value="Alice",
        valid_from=datetime(2023, 1, 1, tzinfo=timezone.utc),
        valid_until=datetime(2024, 1, 1, tzinfo=timezone.utc),
        system_recorded_at=datetime(2023, 1, 2, tzinfo=timezone.utc),
    )
    assert bitemp.is_valid_at(datetime(2023, 6, 1, tzinfo=timezone.utc))
    assert not bitemp.is_valid_at(datetime(2024, 6, 1, tzinfo=timezone.utc))


def test_chapter_08_skill_library():
    skill = Skill(
        skill_id="sk1",
        version=1,
        name="extract_data",
        task_signature="extract",
        precondition="raw text",
        postcondition="json",
        steps=(),
        status=SkillStatus.CANDIDATE,
        offline_success=0.9,
        live_success=0.85,
        live_attempts=10,
        created_at=datetime.now(timezone.utc),
        source_run_ids=("r1",),
    )
    assert skill.skill_id == "sk1"


def test_chapter_09_supervisor():
    b = Budget(tokens_remaining=10000, usd_remaining=1.0, deadline_utc=datetime.now(timezone.utc))
    assert b.tokens_remaining == 10000

    executors = {
        "hold_funds": lambda amount: True,
        "release_funds": lambda amount: True,
    }
    coord = SagaCoordinator(executors)
    step = SagaStep(
        step_id="step_1",
        agent_id="billing_agent",
        action_name="hold_funds",
        forward_arguments={"amount": 100},
        compensate_action="release_funds",
        compensate_arguments={"amount": 100},
    )
    assert coord.execute_step(step)
    rollback = coord.rollback()
    assert rollback.success
    assert "step_1" in rollback.compensated_steps

    # Test DeadlockDetector
    detector = DeadlockDetector()
    assert detector.add_dependency("AgentA", "AgentB")
    assert detector.add_dependency("AgentB", "AgentC")
    # Adding AgentC -> AgentA should be detected as a cycle (deadlock)
    assert not detector.add_dependency("AgentC", "AgentA")
    detector.remove_dependency("AgentA", "AgentB")
    assert detector.add_dependency("AgentC", "AgentA")


def test_chapter_10_protocols():
    pin = ServerPin(
        server_id="test-server",
        protocol_version="1.0",
        tool_description_hash="abc",
        sampling_allowed=False,
        sampling_max_tokens=1000,
        sampling_calls_per_minute=30,
    )
    assert pin.server_id == "test-server"
