from __future__ import annotations

"""Trajectory assertions.

Rather than comparing against one reference path, a case declares what must
be true of any acceptable path. Several correct strategies therefore pass,
while a lucky-but-unsafe path fails.
"""


from dataclasses import dataclass, field
from typing import Callable, Sequence


@dataclass(frozen=True, slots=True)
class TraceStep:
    index: int
    tool: str | None
    arguments: dict
    status: str            # SUCCESS | EMPTY | INVALID | DENIED | UNAVAILABLE
    observation: str
    tokens: int


@dataclass(frozen=True, slots=True)
class Trajectory:
    case_id: str
    steps: tuple[TraceStep, ...]
    answer: str
    stop_reason: str
    total_tokens: int
    elapsed_seconds: float

    def tools_used(self) -> list[str]:
        return [s.tool for s in self.steps if s.tool]


Assertion = Callable[[Trajectory], "AssertionResult"]


@dataclass(frozen=True, slots=True)
class AssertionResult:
    name: str
    passed: bool
    detail: str = ""
    severity: str = "error"      # error blocks release, warning is reported


# ---- assertion constructors -------------------------------------------
def must_call(tool: str) -> Assertion:
    def check(trajectory: Trajectory) -> AssertionResult:
        used = tool in trajectory.tools_used()
        return AssertionResult(f"must_call:{tool}", used,
                               "" if used else f"{tool} never invoked")
    return check


def must_not_call(tool: str) -> Assertion:
    def check(trajectory: Trajectory) -> AssertionResult:
        used = tool in trajectory.tools_used()
        return AssertionResult(f"must_not_call:{tool}", not used,
                               f"{tool} was invoked" if used else "")
    return check


def ordering(before: str, after: str) -> Assertion:
    """Safety ordering, for example: check entitlement before issuing a refund."""
    def check(trajectory: Trajectory) -> AssertionResult:
        used = trajectory.tools_used()
        if after not in used:
            return AssertionResult(f"order:{before}<{after}", True, "after not called")
        if before not in used:
            return AssertionResult(f"order:{before}<{after}", False,
                                   f"{after} ran without {before}")
        ok = used.index(before) < used.index(after)
        return AssertionResult(f"order:{before}<{after}", ok,
                               "" if ok else "wrong order")
    return check


def step_ceiling(maximum: int) -> Assertion:
    def check(trajectory: Trajectory) -> AssertionResult:
        count = len(trajectory.steps)
        return AssertionResult("step_ceiling", count <= maximum,
                               f"{count} steps, ceiling {maximum}",
                               severity="warning")
    return check


def no_repeated_call() -> Assertion:
    """Catches no-progress cycles that still produce a correct answer."""
    def check(trajectory: Trajectory) -> AssertionResult:
        seen: set[tuple] = set()
        for step in trajectory.steps:
            key = (step.tool, tuple(sorted(step.arguments.items())))
            if key in seen:
                return AssertionResult("no_repeated_call", False,
                                       f"repeated {step.tool} at step {step.index}")
            seen.add(key)
        return AssertionResult("no_repeated_call", True)
    return check


def grounded_in(minimum_sources: int) -> Assertion:
    def check(trajectory: Trajectory) -> AssertionResult:
        sources = {
            s.observation.split("|")[0] for s in trajectory.steps
            if s.status == "SUCCESS"
        }
        ok = len(sources) >= minimum_sources
        return AssertionResult("grounded_in", ok,
                               f"{len(sources)} distinct sources")
    return check


@dataclass
class EvalCase:
    case_id: str
    goal: str
    segment: str                        # for stratified reporting
    assertions: tuple[Assertion, ...]
    reference_answer: str | None = None
    metadata: dict = field(default_factory=dict)


class TrajectoryEvaluator:
    def evaluate(self, case: EvalCase, trajectory: Trajectory) -> dict:
        results = [assertion(trajectory) for assertion in case.assertions]
        blocking = [r for r in results if not r.passed and r.severity == "error"]
        return {
            "case_id": case.case_id,
            "segment": case.segment,
            "passed": not blocking,
            "failures": [(r.name, r.detail) for r in blocking],
            "warnings": [(r.name, r.detail) for r in results
                         if not r.passed and r.severity == "warning"],
            "steps": len(trajectory.steps),
            "tokens": trajectory.total_tokens,
            "stop_reason": trajectory.stop_reason,
        }

"""Judge calibration.

A judge is a classifier. Ship it only with a measured agreement figure, and
report that figure everywhere the judge's scores are reported. Cohen's kappa
corrects for agreement that would occur by chance, which raw agreement does
not.
"""


from dataclasses import dataclass
from statistics import mean


@dataclass(frozen=True, slots=True)
class Calibration:
    raw_agreement: float
    cohens_kappa: float
    position_bias: float          # 0 means unbiased in pairwise comparisons
    sample_size: int
    usable: bool
    note: str


class JudgeCalibrator:
    KAPPA_FLOOR = 0.60            # below this the judge is not usable alone
    BIAS_CEILING = 0.10           # more than 10 points of position bias is fatal

    def calibrate(
        self,
        human_labels: list[int],
        judge_labels: list[int],
        pairwise_first_wins: int,
        pairwise_second_wins: int,
    ) -> Calibration:
        n = len(human_labels)
        agreement = mean(1.0 if h == j else 0.0
                         for h, j in zip(human_labels, judge_labels))
        kappa = self._cohens_kappa(human_labels, judge_labels)

        total_pairs = pairwise_first_wins + pairwise_second_wins
        bias = abs(pairwise_first_wins / max(total_pairs, 1) - 0.5) * 2.0

        usable = kappa >= self.KAPPA_FLOOR and bias <= self.BIAS_CEILING and n >= 100
        note = ""
        if kappa < self.KAPPA_FLOOR:
            note = ("kappa below floor: the judge disagrees with humans more than "
                    "a coin weighted to the label distribution would")
        elif bias > self.BIAS_CEILING:
            note = ("position bias: the judge favours one slot regardless of "
                    "content, so randomise order and average both orders")
        elif n < 100:
            note = "sample too small to trust the agreement estimate"

        return Calibration(agreement, kappa, bias, n, usable, note)

    @staticmethod
    def _cohens_kappa(a: list[int], b: list[int]) -> float:
        labels = sorted(set(a) | set(b))
        n = len(a)
        observed = sum(1 for x, y in zip(a, b) if x == y) / n
        expected = sum(
            (a.count(label) / n) * (b.count(label) / n) for label in labels
        )
        return (observed - expected) / (1.0 - expected + 1e-12)


class PairwiseJudge:
    """Pairwise comparison with order randomisation, which removes most of the
    position bias that pointwise scoring hides rather than avoids."""

    def __init__(self, client, model_id: str, rubric: str) -> None:
        self._client = client
        self._model_id = model_id
        self._rubric = rubric

    def compare(self, question: str, answer_a: str, answer_b: str) -> str:
        # Run both orders and require consistency. An inconsistent verdict is
        # a tie, which is more honest than reporting a coin flip as a result.
        first = self._ask(question, answer_a, answer_b)
        second = self._ask(question, answer_b, answer_a)
        if first == "A" and second == "B":
            return "a_wins"
        if first == "B" and second == "A":
            return "b_wins"
        return "tie"

    def _ask(self, question: str, left: str, right: str) -> str:
        response = self._client.complete(
            model=self._model_id,
            temperature=0.0,
            system=(
                f"{self._rubric}\n\n"
                "Quote the specific span that justifies your verdict, then reply "
                "with exactly one character on the final line: A or B."
            ),
            user=f"Question:\n{question}\n\nAnswer A:\n{left}\n\nAnswer B:\n{right}",
        )
        return response.text.strip()[-1].upper()
