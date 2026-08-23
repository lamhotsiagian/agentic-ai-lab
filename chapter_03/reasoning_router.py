from __future__ import annotations

"""Reasoning budget router.

The router answers one question: how much deliberation does this request
deserve? It is deliberately cheap, because a router that costs as much as
the call it is routing has no reason to exist.

Stage 1  deterministic signals   ~0 ms, resolves the majority of traffic
Stage 2  small classifier call   ~120 ms, resolves the ambiguous remainder
Stage 3  escalation on failure   only after a verification miss
"""


import re
from dataclasses import dataclass
from enum import Enum


class Effort(str, Enum):
    MINIMAL = "minimal"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass(frozen=True, slots=True)
class RoutingDecision:
    effort: Effort
    model_id: str
    max_reasoning_tokens: int
    reason: str            # recorded on the span, so routing is auditable
    stage: int             # which stage decided, for router quality metrics


# Deterministic signals. Each one encodes a fact we know without asking a
# model, which is why stage one is nearly free.
_TRIVIAL_INTENTS = frozenset({
    "greeting", "acknowledgement", "cancel", "status_check", "list_options",
})
_HARD_MARKERS = (
    re.compile(r"\breconcile\b", re.I),
    re.compile(r"\bwhy did\b.*\bfail\b", re.I),
    re.compile(r"\bcompare\b.*\band\b.*\bacross\b", re.I),
    re.compile(r"\bprove\b|\bderive\b|\boptimi[sz]e\b", re.I),
)


class ComplexityClassifier:
    """Wraps a small model that returns a 0..1 complexity score.

    Implementations should cache aggressively on a normalised hash of the
    request, because production traffic is far more repetitive than teams
    expect and this call is on the critical path.
    """

    def __init__(self, client, model_id: str = "small-instruct") -> None:
        self._client = client
        self._model_id = model_id

    def score(self, text: str) -> float:
        response = self._client.complete(
            model=self._model_id,
            temperature=0.0,
            max_tokens=8,
            system=(
                "Rate how much multi-step reasoning this request needs. "
                "Reply with a single number from 0.00 to 1.00 and nothing else. "
                "0.0 means a lookup or a restatement. "
                "1.0 means multi-constraint planning or formal derivation."
            ),
            user=text[:2000],
        )
        try:
            return max(0.0, min(1.0, float(response.text.strip())))
        except ValueError:
            # A malformed score must not fail the request. Default to the
            # middle band, which is the safe direction for quality.
            return 0.5


class ReasoningRouter:
    def __init__(
        self,
        classifier: ComplexityClassifier,
        small_model: str = "small-instruct",
        primary_model: str = "primary-instruct",
        reasoning_model: str = "primary-reasoning",
        low_band: float = 0.30,
        high_band: float = 0.70,
    ) -> None:
        self._classifier = classifier
        self._small = small_model
        self._primary = primary_model
        self._reasoning = reasoning_model
        self._low = low_band
        self._high = high_band

    # -- stage 1 -----------------------------------------------------------
    def _deterministic(self, text: str, intent: str | None, step_role: str) -> RoutingDecision | None:
        if intent in _TRIVIAL_INTENTS:
            return RoutingDecision(
                Effort.MINIMAL, self._small, 0,
                f"trivial_intent:{intent}", stage=1,
            )
        if step_role in {"plan", "replan", "verify"}:
            # The plan step's output gates every later step, so it always
            # receives the largest budget the tier allows.
            return RoutingDecision(
                Effort.HIGH, self._reasoning, 8000,
                f"structural_role:{step_role}", stage=1,
            )
        if step_role in {"format", "extract", "summarise"}:
            return RoutingDecision(
                Effort.MINIMAL, self._small, 0,
                f"structural_role:{step_role}", stage=1,
            )
        if any(pattern.search(text) for pattern in _HARD_MARKERS):
            return RoutingDecision(
                Effort.HIGH, self._reasoning, 8000, "hard_marker_match", stage=1,
            )
        return None

    # -- stage 2 -----------------------------------------------------------
    def route(self, text: str, intent: str | None = None, step_role: str = "act") -> RoutingDecision:
        decided = self._deterministic(text, intent, step_role)
        if decided is not None:
            return decided

        score = self._classifier.score(text)
        if score < self._low:
            return RoutingDecision(
                Effort.MINIMAL, self._small, 0, f"score:{score:.2f}", stage=2
            )
        if score < self._high:
            return RoutingDecision(
                Effort.MEDIUM, self._primary, 2000, f"score:{score:.2f}", stage=2
            )
        return RoutingDecision(
            Effort.HIGH, self._reasoning, 8000, f"score:{score:.2f}", stage=2
        )

    # -- stage 3 -----------------------------------------------------------
    def escalate(self, previous: RoutingDecision) -> RoutingDecision | None:
        """One-tier escalation after a failed verification. Returns None at the
        ceiling, which the orchestrator treats as an escalation to a human."""
        if previous.effort is Effort.MINIMAL:
            return RoutingDecision(
                Effort.MEDIUM, self._primary, 2000, "escalated_from_minimal", stage=3
            )
        if previous.effort is Effort.MEDIUM:
            return RoutingDecision(
                Effort.HIGH, self._reasoning, 8000, "escalated_from_medium", stage=3
            )
        return None

"""ReWOO style plan-and-execute.

The plan is a list of steps, each of which may reference the output of an
earlier step through a #E<n> placeholder. The executor resolves the
dependency graph, runs independent steps concurrently, and substitutes
results before invoking each tool.
"""


import asyncio
import re
from dataclasses import dataclass

PLACEHOLDER = re.compile(r"#E(\d+)")


@dataclass(frozen=True, slots=True)
class PlanStep:
    index: int                 # 1-based, matches the #E numbering
    rationale: str
    tool: str
    argument_template: str     # may contain #E1, #E2, ...

    def dependencies(self) -> set[int]:
        return {int(match) for match in PLACEHOLDER.findall(self.argument_template)}


class RewooPlanner:
    """Single model call that emits the whole plan with placeholders."""

    SYSTEM = (
        "Decompose the task into numbered steps. Each step names one tool and "
        "its argument. Reference an earlier step's output as #E<n>. Emit no "
        "prose outside the numbered list. Emit at most 8 steps."
    )

    def __init__(self, client, model_id: str, tool_catalogue: str) -> None:
        self._client = client
        self._model_id = model_id
        self._catalogue = tool_catalogue

    def plan(self, goal: str) -> list[PlanStep]:
        response = self._client.complete(
            model=self._model_id,
            temperature=0.0,
            system=f"{self.SYSTEM}\n\nAvailable tools:\n{self._catalogue}",
            user=goal,
        )
        steps = _parse_plan(response.text)
        _assert_acyclic(steps)          # never trust the model's dependency graph
        return steps


class RewooExecutor:
    """Runs the plan, honouring dependencies and parallelising what it can."""

    def __init__(self, gateway, max_concurrency: int = 4) -> None:
        self._gateway = gateway
        self._semaphore = asyncio.Semaphore(max_concurrency)

    async def execute(self, steps: list[PlanStep], principal: str) -> dict[int, str]:
        results: dict[int, str] = {}
        pending = {step.index: step for step in steps}

        while pending:
            ready = [
                step for step in pending.values()
                if step.dependencies() <= results.keys()
            ]
            if not ready:
                # A step depends on something that never resolved. Fail loudly
                # rather than deadlocking on an impossible plan.
                raise RuntimeError(
                    f"plan deadlock, unresolved steps: {sorted(pending)}"
                )

            outcomes = await asyncio.gather(
                *(self._run_one(step, results, principal) for step in ready),
                return_exceptions=False,
            )
            for step, outcome in zip(ready, outcomes):
                results[step.index] = outcome
                pending.pop(step.index)

        return results

    async def _run_one(self, step: PlanStep, results: dict[int, str], principal: str) -> str:
        argument = PLACEHOLDER.sub(
            lambda m: results.get(int(m.group(1)), ""), step.argument_template
        )
        async with self._semaphore:
            outcome = await self._gateway.invoke_async(principal, step.tool, {"input": argument})
        # An unavailable tool yields a typed marker rather than an exception,
        # so the synthesiser can state honestly what it could not obtain.
        return outcome.value if outcome.ok else f"[unavailable: {outcome.status}]"
