from __future__ import annotations

"""Trajectory fairness evaluation.

Reports disparity across every stage of the trajectory, not only the final
outcome, and refuses to report a ratio when a segment is too small for the
estimate to mean anything.
"""


import math
from collections import defaultdict
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RunOutcome:
    segment: str                 # protected or proxy segment label
    routed_path: str
    steps: int
    verification_tools_used: int
    escalated: bool
    budget_terminated: bool
    approved: bool
    correct: bool                # against adjudicated ground truth
    latency_seconds: float
    cost_usd: float


@dataclass(frozen=True, slots=True)
class DisparityReport:
    metric: str
    reference_segment: str
    ratios: dict[str, float]
    intervals: dict[str, tuple[float, float]]
    flagged: tuple[str, ...]
    suppressed: tuple[str, ...]   # segments too small to estimate


class TrajectoryFairnessEvaluator:
    MIN_SEGMENT_N = 100
    # The four-fifths convention: a selection ratio below 0.8 is a common
    # regulatory screening threshold. It is a screen, not a legal conclusion.
    LOW_RATIO = 0.80
    HIGH_RATIO = 1.25

    def evaluate(self, outcomes: list[RunOutcome], reference: str) -> list[DisparityReport]:
        by_segment: dict[str, list[RunOutcome]] = defaultdict(list)
        for outcome in outcomes:
            by_segment[outcome.segment].append(outcome)

        metrics = {
            "approval_rate":        lambda rs: _rate(rs, lambda r: r.approved),
            "accuracy":             lambda rs: _rate(rs, lambda r: r.correct),
            "escalation_rate":      lambda rs: _rate(rs, lambda r: r.escalated),
            "budget_termination":   lambda rs: _rate(rs, lambda r: r.budget_terminated),
            "verification_intensity": lambda rs: _mean(rs, lambda r: r.verification_tools_used),
            "steps":                lambda rs: _mean(rs, lambda r: float(r.steps)),
            "latency_seconds":      lambda rs: _mean(rs, lambda r: r.latency_seconds),
        }

        reports: list[DisparityReport] = []
        for name, compute in metrics.items():
            base = compute(by_segment[reference])
            ratios, intervals, flagged, suppressed = {}, {}, [], []
            for segment, runs in by_segment.items():
                if segment == reference:
                    continue
                if len(runs) < self.MIN_SEGMENT_N:
                    # Reporting a ratio from thirty cases invites a decision
                    # that the data cannot support, in either direction.
                    suppressed.append(segment)
                    continue
                value = compute(runs)
                ratio = value / base if base else float("nan")
                ratios[segment] = ratio
                intervals[segment] = _ratio_interval(runs, value, base, len(by_segment[reference]))
                if ratio < self.LOW_RATIO or ratio > self.HIGH_RATIO:
                    flagged.append(segment)
            reports.append(DisparityReport(
                name, reference, ratios, intervals,
                tuple(flagged), tuple(suppressed),
            ))
        return reports


def _rate(runs, predicate) -> float:
    return sum(1 for r in runs if predicate(r)) / max(len(runs), 1)


def _mean(runs, extract) -> float:
    return sum(extract(r) for r in runs) / max(len(runs), 1)


def _ratio_interval(runs, value, base, base_n) -> tuple[float, float]:
    """Normal approximation on the log ratio. Adequate for screening, and the
    interval matters more than the point estimate when segments are uneven."""
    n = len(runs)
    se = math.sqrt(max(value * (1 - value), 1e-6) / n
                   + max(base * (1 - base), 1e-6) / max(base_n, 1))
    ratio = value / base if base else float("nan")
    delta = 1.96 * se / max(base, 1e-6)
    return (max(0.0, ratio - delta), ratio + delta)

"""Decision record.

Written once per consequential decision, immutable, retained per policy, and
sufficient to answer four questions: what was decided, on what evidence, under
which rules and versions, and how to contest it.
"""


from dataclasses import dataclass, field
from datetime import datetime, timezone


@dataclass(frozen=True, slots=True)
class EvidenceRef:
    source_id: str
    source_uri: str | None
    content_hash: str            # proves what the document said at the time
    retrieved_at: datetime


@dataclass(frozen=True, slots=True)
class DecisionRecord:
    # --- identity -------------------------------------------------------
    decision_id: str
    run_id: str
    trace_id: str
    tenant_id: str
    subject_id: str              # the person the decision is about
    decided_at: datetime

    # --- the decision ---------------------------------------------------
    decision_type: str           # e.g. refund_denied, ticket_deprioritised
    outcome: str
    reason_codes: tuple[str, ...]   # enumerated, stable, human readable
    natural_language_reason: str

    # --- how it was reached ---------------------------------------------
    evidence: tuple[EvidenceRef, ...]
    policy_version: str
    prompt_version: str
    model_id: str
    agent_version: str
    routing_reason: str
    degradation_rung: int

    # --- human involvement ----------------------------------------------
    autonomy_level: int
    approver_id: str | None
    approved_at: datetime | None
    approval_latency_seconds: float | None

    # --- contestability -------------------------------------------------
    contest_channel: str
    contest_deadline_utc: datetime
    superseded_by: str | None = None
    metadata: dict = field(default_factory=dict)

    def explanation_for_subject(self) -> str:
        """Plain-language explanation the affected person receives.

        Reason codes are enumerated rather than generated, so two people with
        the same situation receive the same explanation, and so the set of
        possible explanations can be reviewed in advance by a compliance team.
        """
        reasons = "\n".join(f"  - {REASON_TEXT[code]}" for code in self.reason_codes)
        review = (
            "A person reviewed and approved this decision."
            if self.approver_id
            else "This decision was made automatically."
        )
        return (
            f"Decision: {self.outcome}\n"
            f"Reference: {self.decision_id}\n"
            f"Date: {self.decided_at.date().isoformat()}\n\n"
            f"Reasons:\n{reasons}\n\n"
            f"{review}\n"
            f"If you disagree, you can request a review at {self.contest_channel} "
            f"before {self.contest_deadline_utc.date().isoformat()}. "
            f"A human will review your case."
        )


REASON_TEXT = {
    "outside_policy_window": "The request was received after the policy window closed.",
    "no_matching_transaction": "We could not find a transaction matching the details provided.",
    "duplicate_request": "An earlier request for the same item has already been processed.",
    "requires_documentation": "Supporting documentation is required and was not attached.",
    "amount_above_threshold": "The amount requires review by a specialist team.",
}
