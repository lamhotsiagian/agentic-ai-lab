from __future__ import annotations

"""Handoff envelope for inter-agent delegation.

Everything the receiving agent needs is explicit. Nothing is implied by
conversational context, because conversational context does not survive a
process boundary, a retry, or a checkpoint restore.
"""


from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable


@dataclass(frozen=True, slots=True)
class Budget:
    """Shared across the whole run, not per agent.

    A per-agent budget lets a five-agent system spend five times the intended
    amount, which is the most common cost surprise in multi-agent designs.
    """
    tokens_remaining: int
    usd_remaining: float
    deadline_utc: datetime

    def spend(self, tokens: int, usd: float) -> "Budget":
        return replace(
            self,
            tokens_remaining=self.tokens_remaining - tokens,
            usd_remaining=self.usd_remaining - usd,
        )

    def exhausted(self) -> bool:
        return (
            self.tokens_remaining <= 0
            or self.usd_remaining <= 0.0
            or datetime.now(timezone.utc) >= self.deadline_utc
        )


@dataclass(frozen=True, slots=True)
class Handoff:
    # --- identity and control ------------------------------------------
    run_id: str
    trace_id: str                  # one trace across every agent in the run
    from_agent: str
    to_agent: str
    hops_remaining: int            # decremented on every handoff, hard floor 0
    budget: Budget

    # --- the actual request --------------------------------------------
    task: str                      # imperative, specific, self contained
    success_criterion: str         # how the receiver knows it is finished
    output_schema: dict[str, Any]  # what shape to return, validated on receipt

    # --- everything the receiver needs to avoid re-deriving -------------
    facts: tuple[dict[str, Any], ...] = ()      # evidence with provenance
    constraints: tuple[str, ...] = ()           # policy the receiver inherits
    already_tried: tuple[str, ...] = ()         # dead ends, so they are not repeated
    principal: str = ""                         # end user identity, propagated

    # --- provenance -----------------------------------------------------
    path: tuple[str, ...] = ()     # every agent this work has passed through
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def forward(self, to_agent: str, task: str, success_criterion: str) -> "Handoff":
        if self.hops_remaining <= 0:
            raise HopLimitExceeded(f"hop limit reached on path {' -> '.join(self.path)}")
        if to_agent in self.path[-2:]:
            # Immediate ping-pong. Catching the pair is cheaper than waiting
            # for the hop limit to drain the budget.
            raise HandoffLoopDetected(f"{self.from_agent} <-> {to_agent}")
        return replace(
            self,
            from_agent=self.to_agent,
            to_agent=to_agent,
            task=task,
            success_criterion=success_criterion,
            hops_remaining=self.hops_remaining - 1,
            path=self.path + (to_agent,),
        )


class HopLimitExceeded(RuntimeError):
    pass


class HandoffLoopDetected(RuntimeError):
    pass

"""Supervisor with shared budget, partial failure tolerance, and explicit
conflict resolution."""


import asyncio
from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True, slots=True)
class WorkerOutcome:
    agent: str
    ok: bool
    conclusion: str
    evidence: tuple[dict, ...]
    tokens_spent: int
    usd_spent: float
    failure: str = ""


class Supervisor:
    def __init__(self, workers: dict[str, "AgentClient"], resolver: "ConflictResolver",
                 min_coverage: float = 0.5) -> None:
        self._workers = workers
        self._resolver = resolver
        self._min_coverage = min_coverage

    async def run(self, handoff: Handoff, assignments: dict[str, str]) -> dict:
        results = await asyncio.gather(
            *(
                self._delegate(handoff, agent, task)
                for agent, task in assignments.items()
            ),
            return_exceptions=False,          # failures are values, not exceptions
        )

        # Reconcile the shared budget from actual spend rather than estimates.
        spent_tokens = sum(r.tokens_spent for r in results)
        spent_usd = sum(r.usd_spent for r in results)
        remaining = handoff.budget.spend(spent_tokens, spent_usd)

        succeeded = [r for r in results if r.ok]
        coverage = len(succeeded) / max(len(results), 1)
        if coverage < self._min_coverage:
            return {
                "status": "insufficient_coverage",
                "coverage": coverage,
                "failures": [(r.agent, r.failure) for r in results if not r.ok],
                "budget": remaining,
            }

        resolution = self._resolver.resolve(succeeded)
        return {
            "status": "ok",
            "coverage": coverage,
            "conclusion": resolution.conclusion,
            "conflicts": resolution.conflicts,
            "policy_used": resolution.policy,
            "missing_agents": [r.agent for r in results if not r.ok],
            "budget": remaining,
        }

    async def _delegate(self, parent: Handoff, agent: str, task: str) -> WorkerOutcome:
        if parent.budget.exhausted():
            return WorkerOutcome(agent, False, "", (), 0, 0.0, "budget_exhausted")
        try:
            envelope = parent.forward(
                to_agent=agent, task=task,
                success_criterion=f"return a conclusion for: {task}",
            )
        except (HopLimitExceeded, HandoffLoopDetected) as exc:
            return WorkerOutcome(agent, False, "", (), 0, 0.0, type(exc).__name__)

        try:
            return await self._workers[agent].invoke(envelope)
        except Exception as exc:                # noqa: BLE001
            # A worker failure degrades coverage. It does not fail the run.
            return WorkerOutcome(agent, False, "", (), 0, 0.0, f"{type(exc).__name__}")


class ConflictResolver:
    """Resolves incompatible worker conclusions with a recorded policy."""

    def __init__(self, domain_owner: dict[str, str]) -> None:
        self._owner = domain_owner       # claim topic -> authoritative agent

    def resolve(self, outcomes: Sequence[WorkerOutcome]) -> "Resolution":
        conflicts = self._detect(outcomes)
        if not conflicts:
            return Resolution(
                conclusion=self._merge([o.conclusion for o in outcomes]),
                conflicts=(), policy="no_conflict",
            )

        # Policy 1: the agent that owns the domain wins for claims in it.
        for conflict in conflicts:
            authoritative = self._owner.get(conflict.topic)
            if authoritative:
                conflict.winner = authoritative
                conflict.policy = "domain_ownership"
                continue
            # Policy 2: more independent sources wins, with a clear margin.
            ranked = sorted(
                conflict.positions, key=lambda p: len({e["source_id"] for e in p.evidence}),
                reverse=True,
            )
            if len(ranked) >= 2 and _source_count(ranked[0]) >= _source_count(ranked[1]) + 2:
                conflict.winner = ranked[0].agent
                conflict.policy = "evidence_weight"
            else:
                # Policy 3: neither policy is decisive, so escalate rather than
                # letting the synthesiser choose arbitrarily.
                conflict.winner = None
                conflict.policy = "escalate_to_human"

        return Resolution(
            conclusion=self._merge_with_conflicts(outcomes, conflicts),
            conflicts=tuple(conflicts),
            policy="mixed",
        )


class SagaStatus(str, Enum):
    PENDING = "pending"
    COMMITTED = "committed"
    FAILED = "failed"
    COMPENSATED = "compensated"


@dataclass(frozen=True, slots=True)
class SagaStep:
    step_id: str
    agent_id: str
    action_name: str
    forward_arguments: dict[str, Any]
    compensate_action: str
    compensate_arguments: dict[str, Any]


@dataclass(frozen=True, slots=True)
class CompensationVerdict:
    success: bool
    compensated_steps: tuple[str, ...]
    unresolved_steps: tuple[str, ...]
    error: str = ""


class SagaCoordinator:
    """Coordinates forward execution and reverse-order compensating transactions."""

    def __init__(self, action_executors: dict[str, Callable[..., bool]]) -> None:
        self._executors = action_executors
        self._completed_steps: list[SagaStep] = []

    def execute_step(self, step: SagaStep) -> bool:
        executor = self._executors.get(step.action_name)
        if not executor:
            raise ValueError(f"Unknown action: {step.action_name}")

        ok = executor(**step.forward_arguments)
        if ok:
            self._completed_steps.append(step)
        return ok

    def rollback(self) -> CompensationVerdict:
        compensated: list[str] = []
        unresolved: list[str] = []

        # Execute inverse actions in reverse chronological order
        for step in reversed(self._completed_steps):
            comp_fn = self._executors.get(step.compensate_action)
            if not comp_fn:
                unresolved.append(step.step_id)
                continue
            try:
                ok = comp_fn(**step.compensate_arguments)
                if ok:
                    compensated.append(step.step_id)
                else:
                    unresolved.append(step.step_id)
            except Exception:
                unresolved.append(step.step_id)

        self._completed_steps.clear()
        return CompensationVerdict(
            success=len(unresolved) == 0,
            compensated_steps=tuple(compensated),
            unresolved_steps=tuple(unresolved),
        )


class DeadlockDetector:
    """Detects cyclic dependencies across concurrent multi-agent delegations using a Wait-For-Graph."""

    def __init__(self) -> None:
        self._wait_for: dict[str, set[str]] = {}

    def add_dependency(self, waiter: str, holder: str) -> bool:
        """Register that `waiter` is waiting on `holder`. Returns True if safe, False if deadlock cycle detected."""
        if waiter not in self._wait_for:
            self._wait_for[waiter] = set()
        self._wait_for[waiter].add(holder)

        visited: set[str] = set()
        stack: list[str] = [waiter]

        while stack:
            curr = stack.pop()
            if curr in visited:
                continue
            visited.add(curr)
            for neighbor in self._wait_for.get(curr, set()):
                if neighbor == waiter:
                    return False  # Cycle detected!
                if neighbor not in visited:
                    stack.append(neighbor)
        return True

    def remove_dependency(self, waiter: str, holder: str) -> None:
        """Clear dependency edge once subtask completes."""
        if waiter in self._wait_for:
            self._wait_for[waiter].discard(holder)
            if not self._wait_for[waiter]:
                del self._wait_for[waiter]

