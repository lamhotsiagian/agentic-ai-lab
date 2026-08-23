from __future__ import annotations

"""Circuit breaker for model and tool dependencies.

Agent traffic is bursty and long-tailed, so a consecutive-failure trigger
either trips constantly or never trips. A windowed failure rate with a
minimum sample is the shape that works here.
"""


import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum


class BreakerState(str, Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass
class CircuitBreaker:
    name: str
    failure_rate_threshold: float = 0.5
    minimum_samples: int = 20
    window_seconds: float = 30.0
    open_seconds: float = 15.0
    half_open_probes: int = 3

    state: BreakerState = BreakerState.CLOSED
    _events: deque = field(default_factory=deque)   # (timestamp, ok)
    _opened_at: float = 0.0
    _probe_successes: int = 0
    _probe_attempts: int = 0

    def allow(self, now: float | None = None) -> bool:
        now = now or time.monotonic()
        if self.state is BreakerState.OPEN:
            if now - self._opened_at >= self.open_seconds:
                self.state = BreakerState.HALF_OPEN
                self._probe_successes = 0
                self._probe_attempts = 0
                return True
            return False
        if self.state is BreakerState.HALF_OPEN:
            # Admit a small number of probes, not the full load, otherwise
            # recovery is indistinguishable from a second outage.
            if self._probe_attempts >= self.half_open_probes:
                return False
            self._probe_attempts += 1
            return True
        return True

    def record(self, ok: bool, now: float | None = None) -> None:
        now = now or time.monotonic()
        self._events.append((now, ok))
        while self._events and now - self._events[0][0] > self.window_seconds:
            self._events.popleft()

        if self.state is BreakerState.HALF_OPEN:
            if ok:
                self._probe_successes += 1
                if self._probe_successes >= self.half_open_probes:
                    self.state = BreakerState.CLOSED
                    self._events.clear()
            else:
                self.state = BreakerState.OPEN
                self._opened_at = now
            return

        if len(self._events) < self.minimum_samples:
            return
        failures = sum(1 for _, succeeded in self._events if not succeeded)
        if failures / len(self._events) >= self.failure_rate_threshold:
            self.state = BreakerState.OPEN
            self._opened_at = now
