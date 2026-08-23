"""agentops/core/state.py

Nothing in this module imports a framework. It is plain Python, fully unit
testable without a graph runtime, a database, or a network.
"""

from __future__ import annotations

import operator
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Annotated, Any, Literal, TypedDict


class Phase(str, Enum):
    INTAKE = "intake"
    PLANNING = "planning"
    RESEARCHING = "researching"
    GRADING = "grading"
    SYNTHESISING = "synthesising"
    VERIFYING = "verifying"
    AWAITING_APPROVAL = "awaiting_approval"
    COMPLETE = "complete"
    REFUSED = "refused"
    FAILED = "failed"


class Provenance(str, Enum):
    SYSTEM = "system"
    USER = "user"
    INTERNAL = "internal"
    UNTRUSTED = "untrusted"


@dataclass(frozen=True, slots=True)
class Evidence:
    evidence_id: str
    claim: str
    source_id: str
    source_uri: str | None
    content_hash: str
    provenance: Provenance
    retrieved_at: datetime
    score: float

    def citation(self) -> str:
        return f"[{self.source_id}]"


@dataclass(frozen=True, slots=True)
class SubGoal:
    identifier: str
    description: str
    source_scope: Literal["documents", "knowledge_base", "web"]
    status: Literal["pending", "running", "done", "blocked"]
    depends_on: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class StagedAction:
    action_id: str
    kind: Literal["export", "share"]
    payload: dict[str, Any]
    effect_description: str
    reversible: bool


# --- reducers ---------------------------------------------------------------
def merge_budget(left: dict[str, float], right: dict[str, float]) -> dict[str, float]:
    """Spend adds across concurrent sub-agents. This is the reducer that stops
    a four-sub-agent run from silently authorising four budgets."""
    merged = dict(left)
    for key, value in right.items():
        merged[key] = merged.get(key, 0.0) + value
    return merged


def replace_plan(left: list[SubGoal], right: list[SubGoal]) -> list[SubGoal]:
    """A replan replaces wholesale. Appending would duplicate subgoals."""
    return right or left


def union_provenance(left: set[Provenance], right: set[Provenance]) -> set[Provenance]:
    """Provenance is monotonic within a run: once untrusted content has entered,
    it has entered, and the guard must see that for the rest of the run."""
    return set(left) | set(right)


def dedupe_evidence(left: list[Evidence], right: list[Evidence]) -> list[Evidence]:
    """Concatenate, then drop duplicates by content hash. This is the evidence
    ledger behaviour: two sub-agents finding the same source pay once."""
    seen = {item.content_hash for item in left}
    merged = list(left)
    for item in right:
        if item.content_hash not in seen:
            merged.append(item)
            seen.add(item.content_hash)
    return merged


class StudioState(TypedDict, total=False):
    # identity, immutable for the run
    run_id: str
    tenant_id: str
    principal_id: str
    question: str
    submitted_at: datetime

    # accumulating, with explicit merge semantics
    evidence: Annotated[list[Evidence], dedupe_evidence]
    messages: Annotated[list[dict[str, Any]], operator.add]
    tool_errors: Annotated[list[str], operator.add]
    provenance: Annotated[set[Provenance], union_provenance]

    # replaced on write
    phase: Phase
    plan: Annotated[list[SubGoal], replace_plan]
    draft: str
    grade: dict[str, float]          # coverage, sources, agreement, top_relevance
    unsupported_claims: list[str]
    coverage_note: str
    staged_action: StagedAction | None
    approval: Literal["granted", "denied", ""]

    # accounting, summed across branches
    budget: Annotated[dict[str, float], merge_budget]

    # control
    step_count: int
    replan_count: int
    correction_count: int
    degradation_rung: int
    refusal_reason: str


def new_state(run_id: str, tenant_id: str, principal_id: str,
              question: str) -> StudioState:
    return StudioState(
        run_id=run_id,
        tenant_id=tenant_id,
        principal_id=principal_id,
        question=" ".join(question.split())[:4000],
        submitted_at=datetime.now(timezone.utc),
        evidence=[], messages=[], tool_errors=[],
        provenance={Provenance.SYSTEM, Provenance.USER},
        phase=Phase.INTAKE, plan=[], draft="",
        grade={}, unsupported_claims=[], coverage_note="",
        staged_action=None, approval="",
        budget={"input_tokens": 0.0, "output_tokens": 0.0,
                "reasoning_tokens": 0.0, "usd": 0.0},
        step_count=0, replan_count=0, correction_count=0,
        degradation_rung=0, refusal_reason="",
    )
