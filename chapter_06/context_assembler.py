from __future__ import annotations

"""Composed orchestration.

The router is a small model. The workflows are code. The loop is the
expensive fallback. Traffic share, not elegance, decides the split, and the
share is measured continuously rather than assumed once.
"""


from dataclasses import dataclass
from typing import Callable, Protocol


@dataclass(frozen=True, slots=True)
class Classification:
    intent: str
    confidence: float
    complexity: float


class IntentRouter(Protocol):
    def classify(self, request: str) -> Classification: ...


class ComposedOrchestrator:
    """Routes to a deterministic workflow when confident, otherwise to a loop."""

    def __init__(
        self,
        router: IntentRouter,
        workflows: dict[str, Callable[[str], "RunResult"]],
        fallback_loop: Callable[[str], "RunResult"],
        confidence_floor: float = 0.82,
        metrics=None,
    ) -> None:
        self._router = router
        self._workflows = workflows
        self._loop = fallback_loop
        self._floor = confidence_floor
        self._metrics = metrics

    def handle(self, request: str) -> "RunResult":
        classification = self._router.classify(request)
        workflow = self._workflows.get(classification.intent)

        route_to_workflow = (
            workflow is not None
            and classification.confidence >= self._floor
            and classification.complexity < 0.6
        )

        # Record the routing decision as a metric dimension, not only as a log
        # line. Router drift is invisible without this series, and router drift
        # is the most common silent quality regression in production agents.
        if self._metrics:
            self._metrics.increment(
                "orchestrator.route",
                path="workflow" if route_to_workflow else "loop",
                intent=classification.intent,
            )

        if route_to_workflow:
            result = workflow(request)
            if result.succeeded:
                return result
            # A workflow that fails structurally escalates to the loop once.
            # Escalation is bounded so a broken workflow cannot double spend
            # on every request indefinitely.
            if self._metrics:
                self._metrics.increment("orchestrator.workflow_escalation",
                                        intent=classification.intent)
        return self._loop(request)

"""Deterministic context assembly.

Guarantees:
  * Section order never varies, so the cacheable prefix stays byte stable.
  * Each section has its own budget, and eviction happens inside a section.
    Global eviction is what causes the "the agent forgot its instructions"
    class of bug.
  * The assembler returns the assembled sections alongside the string, so a
    trace can record exactly what the model saw at every step.
"""


from dataclasses import dataclass, field
from typing import Callable, Sequence


@dataclass(frozen=True, slots=True)
class Section:
    name: str
    content: str
    budget_tokens: int
    evictable: bool          # False for instructions and the goal
    priority: int            # lower evicts first when the total still overflows


@dataclass
class AssembledContext:
    text: str
    sections: list[tuple[str, int]] = field(default_factory=list)
    evicted: list[str] = field(default_factory=list)
    total_tokens: int = 0


class ContextAssembler:
    ORDER = (
        "system", "tools", "durable_facts", "evidence",
        "history_summary", "recent_turns", "goal",
    )

    def __init__(self, count_tokens: Callable[[str], int], window: int = 128_000,
                 reserve_for_output: int = 4_000) -> None:
        self._count = count_tokens
        self._ceiling = window - reserve_for_output

    def assemble(self, sections: Sequence[Section]) -> AssembledContext:
        by_name = {s.name: s for s in sections}
        ordered = [by_name[name] for name in self.ORDER if name in by_name]

        # Pass 1: trim each section to its own budget. Trimming inside a
        # section preserves the section's most valuable content, which a
        # global tail-truncation cannot do.
        trimmed: list[tuple[Section, str, int]] = []
        for section in ordered:
            text, tokens = self._fit(section)
            trimmed.append((section, text, tokens))

        # Pass 2: if the total still overflows, drop whole evictable sections
        # in priority order rather than truncating everything a little.
        result = AssembledContext(text="")
        total = sum(tokens for _, _, tokens in trimmed)
        droppable = sorted(
            (item for item in trimmed if item[0].evictable),
            key=lambda item: item[0].priority,
        )
        for candidate in droppable:
            if total <= self._ceiling:
                break
            trimmed.remove(candidate)
            total -= candidate[2]
            result.evicted.append(candidate[0].name)

        parts = []
        for section, text, tokens in trimmed:
            parts.append(f"<{section.name}>\n{text}\n</{section.name}>")
            result.sections.append((section.name, tokens))
        result.text = "\n\n".join(parts)
        result.total_tokens = total
        return result

    def _fit(self, section: Section) -> tuple[str, int]:
        tokens = self._count(section.content)
        if tokens <= section.budget_tokens:
            return section.content, tokens
        # Proportional character trim, then an explicit marker so the model
        # knows content was removed rather than inferring that it never existed.
        ratio = section.budget_tokens / max(tokens, 1)
        cut = int(len(section.content) * ratio) - 80
        clipped = section.content[: max(cut, 0)]
        marker = f"\n[... {tokens - section.budget_tokens} tokens omitted from {section.name} ...]"
        return clipped + marker, section.budget_tokens
