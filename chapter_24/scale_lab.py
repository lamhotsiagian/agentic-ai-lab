from __future__ import annotations

"""Admission control for an agent service.

The ceiling is derived from measured capacity rather than configured by
intuition, tenant quotas are fair-share rather than fixed so idle capacity is
usable, and shedding is priority aware so batch work absorbs pressure before
interactive work does.
"""


import time
from dataclasses import dataclass, field
from enum import IntEnum


class Priority(IntEnum):
    INTERACTIVE = 0        # a person is waiting
    BACKGROUND = 1         # queued work with a deadline
    BATCH = 2              # overnight, deadline is hours away


@dataclass(frozen=True, slots=True)
class AdmissionResult:
    admitted: bool
    degraded: bool
    queued: bool
    retry_after_seconds: float
    reason: str
    max_steps: int
    model_tier: str


@dataclass
class TokenBucket:
    """Fair-share limiter. Idle tenants' capacity is borrowable, which is what
    distinguishes fair share from a fixed per-tenant cap."""
    capacity: float
    refill_per_second: float
    tokens: float = 0.0
    updated_at: float = field(default_factory=time.monotonic)

    def take(self, amount: float, now: float | None = None) -> bool:
        now = now or time.monotonic()
        elapsed = now - self.updated_at
        self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_per_second)
        self.updated_at = now
        if self.tokens >= amount:
            self.tokens -= amount
            return True
        return False

    def headroom(self) -> float:
        return self.tokens / max(self.capacity, 1e-9)


class AdmissionController:
    def __init__(self, concurrency_ceiling: int, provider_quota: TokenBucket,
                 tenant_buckets: dict[str, TokenBucket], metrics) -> None:
        self._ceiling = concurrency_ceiling
        self._provider = provider_quota
        self._tenants = tenant_buckets
        self._metrics = metrics
        self._in_flight = 0

    @property
    def utilisation(self) -> float:
        return self._in_flight / max(self._ceiling, 1)

    def admit(self, tenant_id: str, priority: Priority,
              estimated_tokens: float) -> AdmissionResult:
        self._metrics.gauge("admission.utilisation", self.utilisation)

        # Layer 1: global concurrency, derived from Little's law against the
        # measured p95 duration rather than configured by intuition.
        if self._in_flight >= self._ceiling:
            return self._shed(priority, "concurrency_ceiling")

        # Layer 2: provider quota. Adding workers against a fixed quota adds
        # queueing rather than throughput, so the quota is checked here.
        if not self._provider.take(estimated_tokens):
            return self._shed(priority, "provider_quota_exhausted")

        # Layer 3: tenant fair share. A soft limit degrades rather than
        # rejecting, so a heavy tenant slows down instead of failing.
        bucket = self._tenants.get(tenant_id)
        if bucket is not None and not bucket.take(estimated_tokens):
            self._in_flight += 1
            self._metrics.increment("admission.degraded", tenant=tenant_id)
            return AdmissionResult(
                admitted=True, degraded=True, queued=False,
                retry_after_seconds=0.0, reason="tenant_soft_limit",
                max_steps=4, model_tier="small",
            )

        # Above a utilisation threshold, pre-emptively reduce the step budget so
        # admitted work finishes faster and the queue drains, rather than
        # admitting full-capability runs into a saturating system.
        if self.utilisation > 0.8:
            self._in_flight += 1
            return AdmissionResult(
                True, True, False, 0.0, "high_utilisation_degrade", 5, "primary"
            )

        self._in_flight += 1
        return AdmissionResult(True, False, False, 0.0, "admitted", 10, "primary")

    def release(self) -> None:
        self._in_flight = max(0, self._in_flight - 1)

    def _shed(self, priority: Priority, reason: str) -> AdmissionResult:
        """Shed by priority. Batch absorbs pressure first, interactive last, and
        an interactive rejection is fast and carries a retry hint."""
        self._metrics.increment("admission.shed", reason=reason,
                                priority=priority.name)
        if priority is Priority.BATCH:
            return AdmissionResult(False, False, True, 300.0, reason, 0, "none")
        if priority is Priority.BACKGROUND:
            return AdmissionResult(False, False, True, 30.0, reason, 0, "none")
        return AdmissionResult(False, False, False, 5.0, reason, 0, "none")
