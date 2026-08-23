from __future__ import annotations

"""Failure triage.

The signature is the important design decision. Clustering on raw text
produces clusters organised by topic, which is not actionable. Clustering on
a canonical failure signature produces clusters organised by mechanism,
which maps directly onto a fix.
"""


from collections import Counter, defaultdict
from dataclasses import dataclass
from enum import Enum


class Surface(str, Enum):
    CONTEXT = "context"          # instructions, exemplars, assembly
    TOOLS = "tools"              # schema, description, projection
    RETRIEVAL = "retrieval"      # index, chunking, query construction
    MEMORY = "memory"            # write policy, supersession, decay
    ORCHESTRATION = "orchestration"   # routing, bounds, topology
    PARAMETERS = "parameters"    # fine-tuning, the expensive last resort
    UPSTREAM = "upstream"        # not our defect, needs another team


@dataclass(frozen=True, slots=True)
class FailureRecord:
    run_id: str
    stop_reason: str
    failing_assertion: str | None
    last_tool: str | None
    last_tool_status: str | None
    verify_score: float
    citations: int
    segment: str
    severity: float              # business impact weight, 0..1
    cost_usd: float


@dataclass
class FailureClass:
    signature: str
    surface: Surface
    count: int = 0
    weighted_impact: float = 0.0
    wasted_cost_usd: float = 0.0
    segments: Counter = None
    example_runs: list[str] = None


class FailureTriage:
    """Canonicalises, clusters, ranks, and routes."""

    def signature(self, record: FailureRecord) -> tuple[str, Surface]:
        # Order matters: the most specific mechanism wins, so a denied tool is
        # never misfiled as a generic budget exhaustion.
        if record.last_tool_status == "DENIED":
            return f"policy_denied:{record.last_tool}", Surface.ORCHESTRATION
        if record.last_tool_status == "INVALID":
            return f"bad_arguments:{record.last_tool}", Surface.TOOLS
        if record.last_tool_status == "UNAVAILABLE":
            return f"dependency_down:{record.last_tool}", Surface.UPSTREAM
        if record.failing_assertion and record.failing_assertion.startswith("must_not_call"):
            return f"safety_violation:{record.failing_assertion}", Surface.CONTEXT
        if record.failing_assertion and record.failing_assertion.startswith("order:"):
            return f"ordering_violation:{record.failing_assertion}", Surface.ORCHESTRATION
        if record.stop_reason == "no_progress":
            return f"no_progress:{record.last_tool}", Surface.ORCHESTRATION
        if record.stop_reason in {"step_limit", "token_limit", "deadline"}:
            return f"budget_exhausted:{record.stop_reason}", Surface.ORCHESTRATION
        if record.citations == 0:
            return "ungrounded_answer", Surface.RETRIEVAL
        if record.verify_score < 0.5:
            return "verification_failed", Surface.RETRIEVAL
        return "unclassified", Surface.CONTEXT

    def triage(self, records: list[FailureRecord]) -> list[FailureClass]:
        buckets: dict[str, FailureClass] = {}
        for record in records:
            signature, surface = self.signature(record)
            bucket = buckets.get(signature)
            if bucket is None:
                bucket = FailureClass(signature, surface, 0, 0.0, 0.0,
                                      Counter(), [])
                buckets[signature] = bucket
            bucket.count += 1
            bucket.weighted_impact += record.severity
            bucket.wasted_cost_usd += record.cost_usd
            bucket.segments[record.segment] += 1
            if len(bucket.example_runs) < 5:
                bucket.example_runs.append(record.run_id)

        # Rank by weighted impact rather than by raw count, because a rare
        # safety violation outranks a common cosmetic failure.
        return sorted(buckets.values(),
                      key=lambda b: b.weighted_impact, reverse=True)

    @staticmethod
    def concentrated_in(bucket: FailureClass, threshold: float = 0.6) -> str | None:
        """A class concentrated in one segment is usually a data problem in
        that segment rather than a general system defect."""
        total = sum(bucket.segments.values())
        segment, count = bucket.segments.most_common(1)[0]
        return segment if count / total >= threshold else None

"""Regression case generation from a failing trace.

The generated case encodes the mechanism, not the surface symptom, which is
why it keeps working after the implementation changes.
"""


from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class GeneratedCase:
    case_id: str
    goal: str
    segment: str
    assertion_source: str        # rendered Python, committed to the suite
    incident_id: str
    provenance_run_id: str
    fixture_manifest: dict       # frozen tool responses, for determinism


class RegressionCaseBuilder:
    def build(self, trace, incident_id: str, redactor) -> GeneratedCase:
        signature, _ = FailureTriage().signature(trace.failure_record)
        assertions = self._assertions_for(signature, trace)

        return GeneratedCase(
            case_id=f"reg_{incident_id}_{trace.run_id[:8]}",
            goal=redactor.redact(trace.goal),
            segment=trace.segment,
            assertion_source="\n".join(assertions),
            incident_id=incident_id,
            provenance_run_id=trace.run_id,
            # Freeze the exact tool responses observed. Without frozen
            # fixtures the case depends on live systems and becomes flaky,
            # and a flaky safety test is deleted within a quarter.
            fixture_manifest={
                step.tool: redactor.redact(step.observation)
                for step in trace.steps if step.tool
            },
        )

    @staticmethod
    def _assertions_for(signature: str, trace) -> list[str]:
        if signature.startswith("safety_violation"):
            tool = trace.failure_record.last_tool
            return [
                f'must_not_call("{tool}"),',
                'no_repeated_call(),',
                '# incident: agent invoked a prohibited tool under long context',
            ]
        if signature.startswith("ordering_violation"):
            return [
                'ordering("check_entitlement", "issue_refund"),',
                '# incident: refund issued before entitlement was verified',
            ]
        if signature.startswith("budget_exhausted"):
            return [
                'step_ceiling(6),',
                'no_repeated_call(),',
                '# incident: run exhausted the step budget on duplicate lookups',
            ]
        if signature == "ungrounded_answer":
            return [
                'grounded_in(minimum_sources=2),',
                '# incident: answer produced with zero citations',
            ]
        return ['grounded_in(minimum_sources=1),']
