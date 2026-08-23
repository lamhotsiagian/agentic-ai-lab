"""agentops/eval/gate.py

The gate answers one question: may this commit be promoted? It answers it
with four independent checks, and a failure in any one blocks.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class GateResult:
    allowed: bool
    reasons: tuple[str, ...]
    aggregate_success: float
    per_stratum: dict[str, float]
    cost_per_success: float
    safety_failures: tuple[str, ...]


class ReleaseGate:
    STRATUM_FLOOR_DROP = 0.03      # no stratum may fall more than 3 points
    AGGREGATE_FLOOR = 0.85
    COST_CEILING_USD = 0.25

    def __init__(self, harness, baseline_report) -> None:
        self._harness = harness
        self._baseline = baseline_report

    def evaluate(self, build_id: str) -> GateResult:
        # Frozen tool fixtures make the suite deterministic. A flaky safety
        # test is deleted within a quarter, so determinism is a safety property
        # rather than a convenience.
        report = self._harness.run(
            build_id=build_id, fixtures="frozen", paired_with=self._baseline
        )
        reasons: list[str] = []

        # Check 1: safety assertions block absolutely, regardless of anything
        # else in the report.
        safety = tuple(
            f"{case}: {name}" for case, name in report.failed_safety_assertions
        )
        if safety:
            reasons.append(f"{len(safety)} safety assertion failures")

        # Check 2: aggregate floor.
        if report.success_rate < self.AGGREGATE_FLOOR:
            reasons.append(
                f"aggregate {report.success_rate:.3f} below floor "
                f"{self.AGGREGATE_FLOOR}"
            )

        # Check 3: per-stratum floor. An aggregate gain is compatible with a
        # stratum collapse, and the stratum collapse is what escalates.
        for stratum, value in report.per_stratum.items():
            baseline = self._baseline.per_stratum.get(stratum)
            if baseline is None:
                continue
            if value < baseline - self.STRATUM_FLOOR_DROP:
                reasons.append(
                    f"stratum {stratum} fell {baseline - value:.3f} "
                    f"({baseline:.3f} -> {value:.3f})"
                )

        # Check 4: cost is reported, and a ceiling breach blocks. A quality gain
        # that triples spend must be a visible decision, not a silent one.
        if report.cost_per_success > self.COST_CEILING_USD:
            reasons.append(
                f"cost per success ${report.cost_per_success:.3f} above "
                f"ceiling ${self.COST_CEILING_USD:.2f}"
            )

        return GateResult(
            allowed=not reasons,
            reasons=tuple(reasons),
            aggregate_success=report.success_rate,
            per_stratum=report.per_stratum,
            cost_per_success=report.cost_per_success,
            safety_failures=safety,
        )
