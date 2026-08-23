"""Shared telemetry, metrics, and tracing utilities."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable


@dataclass
class Span:
    name: str
    trace_id: str
    span_id: str = field(default_factory=lambda: uuid.uuid4().hex[:16])
    parent_id: str | None = None
    start_time: float = field(default_factory=time.time)
    end_time: float | None = None
    attributes: dict[str, Any] = field(default_factory=dict)
    events: list[dict[str, Any]] = field(default_factory=list)

    def set_attribute(self, key: str, value: Any) -> None:
        self.attributes[key] = value

    def add_event(self, name: str, attributes: dict[str, Any] | None = None) -> None:
        self.events.append({
            "name": name,
            "timestamp": time.time(),
            "attributes": attributes or {},
        })

    def end(self) -> None:
        self.end_time = time.time()

    @property
    def duration_ms(self) -> float:
        end = self.end_time or time.time()
        return (end - self.start_time) * 1000.0


class Tracer:
    """Lightweight in-memory OpenTelemetry-compatible tracer."""

    def __init__(self, service_name: str = "agentic-ai-lab"):
        self.service_name = service_name
        self.spans: list[Span] = []

    def start_span(
        self,
        name: str,
        trace_id: str | None = None,
        parent_id: str | None = None,
    ) -> Span:
        t_id = trace_id or uuid.uuid4().hex
        span = Span(name=name, trace_id=t_id, parent_id=parent_id)
        span.set_attribute("service.name", self.service_name)
        self.spans.append(span)
        return span

    def clear(self) -> None:
        self.spans.clear()


default_tracer = Tracer()
