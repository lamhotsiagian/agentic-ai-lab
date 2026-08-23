from __future__ import annotations

"""Minimal production-shaped agent loop.

Design notes that matter more than the code:
  * The model never executes anything. It returns a proposal, and the
    orchestrator decides whether that proposal is allowed to run.
  * Every termination path emits the same RunResult shape, so callers
    never branch on "did it work" by inspecting exceptions.
  * The budget ledger is updated before the call, not after, so a crash
    mid-call still leaves an accurate spend record.
"""


import hashlib
import time
from dataclasses import dataclass, field
from typing import Any, Protocol


class StopReason:
    ANSWERED = "answered"
    STEP_LIMIT = "step_limit"
    TOKEN_LIMIT = "token_limit"
    DEADLINE = "deadline"
    NO_PROGRESS = "no_progress"
    GUARDRAIL = "guardrail_block"


@dataclass
class RunBudget:
    """Every bound the orchestrator enforces, in one place."""
    max_steps: int = 12
    max_tokens: int = 120_000
    wall_clock_seconds: float = 90.0
    repeat_threshold: int = 2          # identical calls tolerated before halt

    tokens_used: int = 0
    steps_used: int = 0
    started_at: float = field(default_factory=time.monotonic)

    def elapsed(self) -> float:
        return time.monotonic() - self.started_at

    def exhausted(self) -> str | None:
        """Return the StopReason that fired, or None while budget remains."""
        if self.steps_used >= self.max_steps:
            return StopReason.STEP_LIMIT
        if self.tokens_used >= self.max_tokens:
            return StopReason.TOKEN_LIMIT
        if self.elapsed() >= self.wall_clock_seconds:
            return StopReason.DEADLINE
        return None


@dataclass
class Step:
    index: int
    thought: str
    tool_name: str | None
    arguments: dict[str, Any]
    observation: str | None
    latency_ms: float
    tokens: int


@dataclass
class RunResult:
    answer: str | None
    stop_reason: str
    steps: list[Step]
    tokens_used: int
    elapsed_seconds: float

    @property
    def succeeded(self) -> bool:
        return self.stop_reason == StopReason.ANSWERED


class ReasoningModel(Protocol):
    def propose(self, state: "AgentState") -> "Proposal": ...


class ToolGateway(Protocol):
    def schema(self) -> list[dict[str, Any]]: ...
    def invoke(self, name: str, arguments: dict[str, Any]) -> str: ...


@dataclass
class Proposal:
    """What the model returns: either a final answer or a tool call."""
    final_answer: str | None
    tool_name: str | None
    arguments: dict[str, Any]
    thought: str
    tokens: int


@dataclass
class AgentState:
    goal: str
    scratchpad: list[Step] = field(default_factory=list)
    tool_schema: list[dict[str, Any]] = field(default_factory=list)


class BoundedAgentLoop:
    """Orchestrator. Owns control flow, budget, and the termination decision."""

    def __init__(
        self,
        model: ReasoningModel,
        gateway: ToolGateway,
        budget: RunBudget | None = None,
    ) -> None:
        self._model = model
        self._gateway = gateway
        self._budget_template = budget or RunBudget()

    def run(self, goal: str) -> RunResult:
        budget = RunBudget(
            max_steps=self._budget_template.max_steps,
            max_tokens=self._budget_template.max_tokens,
            wall_clock_seconds=self._budget_template.wall_clock_seconds,
            repeat_threshold=self._budget_template.repeat_threshold,
        )
        state = AgentState(goal=goal, tool_schema=self._gateway.schema())
        call_fingerprints: dict[str, int] = {}

        while True:
            fired = budget.exhausted()
            if fired is not None:
                return self._finish(None, fired, state, budget)

            started = time.perf_counter()
            proposal = self._model.propose(state)
            budget.tokens_used += proposal.tokens
            budget.steps_used += 1

            if proposal.final_answer is not None:
                self._record(state, budget, proposal, None, started)
                return self._finish(
                    proposal.final_answer, StopReason.ANSWERED, state, budget
                )

            # No-progress detection. An agent that reissues an identical call
            # has stopped learning from its observations, so halt rather than
            # burn the remaining budget discovering that fact slowly.
            fingerprint = self._fingerprint(proposal)
            call_fingerprints[fingerprint] = call_fingerprints.get(fingerprint, 0) + 1
            if call_fingerprints[fingerprint] > budget.repeat_threshold:
                self._record(state, budget, proposal, "duplicate call", started)
                return self._finish(None, StopReason.NO_PROGRESS, state, budget)

            try:
                observation = self._gateway.invoke(
                    proposal.tool_name or "", proposal.arguments
                )
            except PermissionError as exc:
                # A denied call is a guardrail event, not a crash. It halts the
                # run so the denial is visible in the trace and in metrics.
                self._record(state, budget, proposal, f"denied: {exc}", started)
                return self._finish(None, StopReason.GUARDRAIL, state, budget)
            except Exception as exc:                       # noqa: BLE001
                # Tool errors are observations. The model may recover from them,
                # so feed the error back rather than aborting the run.
                observation = f"tool_error: {type(exc).__name__}: {exc}"

            self._record(state, budget, proposal, observation, started)

    @staticmethod
    def _fingerprint(proposal: Proposal) -> str:
        payload = f"{proposal.tool_name}|{sorted(proposal.arguments.items())}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

    @staticmethod
    def _record(
        state: AgentState,
        budget: RunBudget,
        proposal: Proposal,
        observation: str | None,
        started: float,
    ) -> None:
        state.scratchpad.append(
            Step(
                index=budget.steps_used,
                thought=proposal.thought,
                tool_name=proposal.tool_name,
                arguments=proposal.arguments,
                observation=observation,
                latency_ms=(time.perf_counter() - started) * 1000.0,
                tokens=proposal.tokens,
            )
        )

    @staticmethod
    def _finish(
        answer: str | None, reason: str, state: AgentState, budget: RunBudget
    ) -> RunResult:
        return RunResult(
            answer=answer,
            stop_reason=reason,
            steps=state.scratchpad,
            tokens_used=budget.tokens_used,
            elapsed_seconds=budget.elapsed(),
        )
