from __future__ import annotations

"""Agent telemetry.

Design rules encoded here:
  * cost is computed at emission, not derived later from token counts and a
    price table that has since changed
  * prompts and completions are redacted before export, and a content hash
    is recorded so the exact context can be matched to a stored copy in the
    secure trace store
  * a failed step still emits a complete span; there is no path where an
    exception loses the telemetry for the work already paid for
"""


import contextlib
import hashlib
import time
from dataclasses import dataclass
from typing import Iterator

from opentelemetry import metrics, trace
from opentelemetry.trace import Status, StatusCode

tracer = trace.get_tracer("agentic.runtime")
meter = metrics.get_meter("agentic.runtime")

step_duration = meter.create_histogram(
    "agent.step.duration", unit="s", description="Wall clock per agent step"
)
run_cost = meter.create_histogram(
    "agent.run.cost", unit="USD", description="Total cost per completed run"
)
step_counter = meter.create_counter(
    "agent.step.count", description="Steps executed, by role and outcome"
)


@dataclass(frozen=True, slots=True)
class Pricing:
    """Prices are configuration, refreshed independently of code."""
    input_per_million: float
    output_per_million: float
    reasoning_per_million: float
    cached_input_per_million: float

    def cost(self, usage: "Usage") -> float:
        return (
            usage.input_tokens * self.input_per_million
            + usage.output_tokens * self.output_per_million
            + usage.reasoning_tokens * self.reasoning_per_million
            + usage.cached_input_tokens * self.cached_input_per_million
        ) / 1_000_000.0


@contextlib.contextmanager
def step_span(
    *,
    run_id: str,
    tenant_id: str,
    principal_id: str,
    step_index: int,
    role: str,
    rendered_context: str,
    redactor,
    secure_store,
) -> Iterator["StepRecorder"]:
    started = time.perf_counter()
    context_hash = hashlib.sha256(rendered_context.encode("utf-8")).hexdigest()

    with tracer.start_as_current_span(f"agent.step.{role}") as span:
        span.set_attribute("agent.run_id", run_id)
        span.set_attribute("tenant.id", tenant_id)
        span.set_attribute("principal.id", principal_id)
        span.set_attribute("agent.step.index", step_index)
        span.set_attribute("agent.step.role", role)
        span.set_attribute("agent.context.hash", context_hash)
        span.set_attribute("agent.context.tokens", len(rendered_context) // 4)

        # The full context goes to a restricted store keyed by hash, never to
        # the general telemetry pipeline. Engineers can reconstruct any step by
        # hash, and the trace itself carries no user content.
        secure_store.put(context_hash, redactor.redact(rendered_context))

        recorder = StepRecorder(span)
        try:
            yield recorder
        except Exception as exc:                     # noqa: BLE001
            span.set_status(Status(StatusCode.ERROR, type(exc).__name__))
            span.set_attribute("agent.step.outcome", "exception")
            raise
        finally:
            elapsed = time.perf_counter() - started
            span.set_attribute("cost.usd", recorder.cost_usd)
            attrs = {"role": role, "tenant": tenant_id,
                     "outcome": recorder.outcome or "unknown"}
            step_duration.record(elapsed, attrs)
            step_counter.add(1, attrs)


class StepRecorder:
    """Accumulates what the step spent and decided."""

    def __init__(self, span) -> None:
        self._span = span
        self.cost_usd = 0.0
        self.outcome: str | None = None

    def model_call(self, model_id: str, provider: str, usage: "Usage",
                   pricing: Pricing, routing_reason: str) -> None:
        cost = pricing.cost(usage)
        self.cost_usd += cost
        self._span.set_attribute("gen_ai.system", provider)
        self._span.set_attribute("gen_ai.request.model", model_id)
        self._span.set_attribute("gen_ai.usage.input_tokens", usage.input_tokens)
        self._span.set_attribute("gen_ai.usage.output_tokens", usage.output_tokens)
        self._span.set_attribute("gen_ai.usage.reasoning_tokens", usage.reasoning_tokens)
        self._span.set_attribute("gen_ai.usage.cached_input_tokens",
                                 usage.cached_input_tokens)
        # Recording the routing reason turns "why was this expensive" into a
        # filter rather than an archaeology exercise.
        self._span.set_attribute("agent.routing.reason", routing_reason)

    def tool_call(self, name: str, status: str, payload_bytes: int,
                  latency_ms: float, cost_usd: float = 0.0) -> None:
        self.cost_usd += cost_usd
        with tracer.start_as_current_span("tool.invoke") as child:
            child.set_attribute("tool.name", name)
            child.set_attribute("tool.status", status)
            child.set_attribute("tool.payload_bytes", payload_bytes)
            child.set_attribute("tool.latency_ms", latency_ms)
            child.set_attribute("cost.usd", cost_usd)
            if status not in {"SUCCESS", "EMPTY"}:
                child.set_status(Status(StatusCode.ERROR, status))

    def finish(self, outcome: str) -> None:
        self.outcome = outcome
        self._span.set_attribute("agent.step.outcome", outcome)

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RunSummary:
    stop_reason: str
    cost_usd: float
    steps: int
    guardrail_fired: bool
    degradation_rung: int
    verify_score: float
    approval_rejected: bool
    tenant_id: str


class TailSampler:
    """Decides retention after the run, when the interesting facts are known."""

    def __init__(self, cost_threshold_usd: float, baseline_rate: float = 0.02,
                 step_threshold: int = 8, verify_floor: float = 0.72) -> None:
        self._cost_threshold = cost_threshold_usd
        self._baseline_rate = baseline_rate
        self._step_threshold = step_threshold
        self._verify_floor = verify_floor

    def decide(self, summary: RunSummary, random_value: float) -> tuple[bool, str]:
        if summary.stop_reason != "answered":
            return True, f"stop_reason:{summary.stop_reason}"
        if summary.guardrail_fired:
            return True, "guardrail"
        if summary.approval_rejected:
            return True, "approval_rejected"
        if summary.degradation_rung > 0:
            return True, f"degraded:{summary.degradation_rung}"
        if summary.verify_score < self._verify_floor:
            return True, "low_verification"
        if summary.cost_usd > self._cost_threshold:
            return True, "expensive"
        if summary.steps > self._step_threshold:
            return True, "long_run"
        # Healthy runs are sampled thinly, because a baseline is required to
        # tell whether the interesting population has shifted.
        if random_value < self._baseline_rate:
            return True, "baseline"
        return False, "dropped"
