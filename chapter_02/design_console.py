from __future__ import annotations

"""Typed state for a research-and-analysis agent.

Three properties make this schema production ready:
  * Every field declares how two concurrent branches merge. Without that
    rule, a fan-out node silently loses one branch's writes.
  * Budget lives in state, not in the orchestrator, so a checkpoint restore
    resumes with the spend already incurred rather than a fresh allowance.
  * Nothing in the schema holds a live connection or a callable, so the
    whole object serialises for durable checkpointing.
"""


import operator
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Annotated, Any, Literal, TypedDict


class RunPhase(str, Enum):
    PLANNING = "planning"
    EXECUTING = "executing"
    VERIFYING = "verifying"
    ESCALATED = "escalated"
    COMPLETE = "complete"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class Evidence:
    """One retrieved or computed fact, with provenance attached."""
    claim: str
    source_id: str
    source_uri: str | None
    confidence: float
    retrieved_at: datetime

    def as_citation(self) -> str:
        return f"[{self.source_id}] {self.claim}"


@dataclass(frozen=True, slots=True)
class SubGoal:
    identifier: str
    description: str
    status: Literal["pending", "running", "done", "blocked"]
    depends_on: tuple[str, ...] = ()


def merge_budget(left: dict[str, float], right: dict[str, float]) -> dict[str, float]:
    """Budget fields add across branches. Concurrent tool calls both spend."""
    merged = dict(left)
    for key, value in right.items():
        merged[key] = merged.get(key, 0.0) + value
    return merged


def keep_latest_plan(left: list[SubGoal], right: list[SubGoal]) -> list[SubGoal]:
    """A replan replaces the plan wholesale. Appending would duplicate goals."""
    return right or left


class ResearchAgentState(TypedDict, total=False):
    # --- immutable run identity -------------------------------------------
    run_id: str
    tenant_id: str
    goal: str
    submitted_at: datetime

    # --- accumulating fields, safe to merge by concatenation --------------
    messages: Annotated[list[dict[str, Any]], operator.add]
    evidence: Annotated[list[Evidence], operator.add]
    tool_errors: Annotated[list[str], operator.add]

    # --- replaced wholesale on each write ---------------------------------
    phase: RunPhase
    plan: Annotated[list[SubGoal], keep_latest_plan]
    draft_answer: str
    verification_notes: str

    # --- accounting, summed across concurrent branches --------------------
    budget: Annotated[dict[str, float], merge_budget]

    # --- loop control -----------------------------------------------------
    step_count: int
    replan_count: int
    escalation_reason: str


def new_run_state(run_id: str, tenant_id: str, goal: str) -> ResearchAgentState:
    """Factory that guarantees every required field is present from step zero."""
    return ResearchAgentState(
        run_id=run_id,
        tenant_id=tenant_id,
        goal=goal,
        submitted_at=datetime.now(timezone.utc),
        messages=[],
        evidence=[],
        tool_errors=[],
        phase=RunPhase.PLANNING,
        plan=[],
        draft_answer="",
        verification_notes="",
        budget={"input_tokens": 0.0, "output_tokens": 0.0, "usd": 0.0},
        step_count=0,
        replan_count=0,
        escalation_reason="",
    )

from typing import Protocol, runtime_checkable


@runtime_checkable
class Planner(Protocol):
    """Decomposes a goal into an ordered plan.

    Invariants:
      * Returns at least one subgoal, or raises PlanningRefused with a reason.
      * Every dependency identifier referenced by a subgoal exists in the plan,
        so the executor never blocks on a goal that will not arrive.
      * The returned plan is acyclic. The implementation checks this rather
        than trusting the model.
    """

    def plan(self, goal: str, context: str) -> list["SubGoal"]: ...


@runtime_checkable
class ToolGateway(Protocol):
    """The only path from the agent to any external effect.

    Invariants:
      * Rejects any tool not present in the caller principal's allowlist,
        before the call is attempted and before any token is spent.
      * Validates arguments against the declared JSON Schema and returns a
        typed ValidationError rather than raising, so the model can correct.
      * Applies a per-tool timeout and returns TimeoutResult rather than
        blocking the orchestrator past its wall clock deadline.
      * Returns a result object that distinguishes SUCCESS, EMPTY, DENIED,
        UNAVAILABLE, and INVALID. Never returns a bare string.
    """

    def invoke(self, principal: str, name: str, arguments: dict) -> "ToolResult": ...


@runtime_checkable
class MemoryService(Protocol):
    """Working, episodic, and semantic recall behind one interface.

    Invariants:
      * read() never returns more than max_tokens worth of content, so context
        assembly is deterministic and the window cannot overflow.
      * Every returned item carries a source identifier, so any claim the agent
        makes can be traced to the memory that produced it.
      * Writes are scoped to the tenant in the call, enforced in the store and
        not only in the query.
    """

    def read(self, query: str, tenant_id: str, max_tokens: int) -> list["Evidence"]: ...
    def write(self, tenant_id: str, items: list["Evidence"]) -> None: ...


@runtime_checkable
class Verifier(Protocol):
    """Checks a draft answer against the evidence that produced it.

    Invariants:
      * Returns a score in [0, 1] plus the specific unsupported claims, so the
        orchestrator can decide between accept, revise, and escalate.
      * Never modifies the draft. Verification and revision are separate steps
        so that a failed verification is visible in the trace.
    """

    def verify(self, draft: str, evidence: list["Evidence"]) -> "VerificationResult": ...

from dataclasses import dataclass
from enum import IntEnum


class Rung(IntEnum):
    FULL = 0
    NO_RERANK = 1
    INTERNAL_ONLY = 2
    SMALL_MODEL = 3
    LEXICAL_ONLY = 4
    REFUSE = 5


@dataclass(frozen=True)
class Capability:
    """Health snapshot, refreshed by the circuit breakers in Chapter 24."""
    reranker: bool
    web_search: bool
    primary_model: bool
    vector_store: bool
    lexical_store: bool


@dataclass(frozen=True)
class RunProfile:
    rung: Rung
    max_steps: int
    top_k: int
    model_tier: str
    disclosure: str


class DegradationController:
    """Picks the best rung the current health state supports.

    The controller is deliberately a pure function of the health snapshot.
    Pure selection means the chosen rung is reproducible from a trace, which
    is what makes a post-incident review possible.
    """

    def select(self, health: Capability) -> RunProfile:
        if not health.vector_store and not health.lexical_store:
            return RunProfile(
                Rung.REFUSE, 0, 0, "none",
                "Retrieval is unavailable, so this question cannot be answered "
                "from sources right now.",
            )
        if not health.vector_store:
            return RunProfile(
                Rung.LEXICAL_ONLY, 4, 20, "small",
                "Semantic search is degraded. Results use keyword matching and "
                "may miss paraphrased passages.",
            )
        if not health.primary_model:
            return RunProfile(
                Rung.SMALL_MODEL, 4, 8, "small",
                "Running on the standby model. Answers are shorter than usual.",
            )
        if not health.web_search:
            return RunProfile(
                Rung.INTERNAL_ONLY, 8, 10, "primary",
                "Answer is based on internal sources only.",
            )
        if not health.reranker:
            return RunProfile(
                Rung.NO_RERANK, 8, 18, "primary",
                "Ranking is degraded. Review the citations before acting.",
            )
        return RunProfile(Rung.FULL, 10, 8, "primary", "")
