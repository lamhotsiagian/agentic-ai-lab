from __future__ import annotations

"""Typed tool results.

Distinguishing EMPTY from UNAVAILABLE is the difference between an agent
that says "no orders match" and an agent that spends its entire budget
rewording the same query during a partial outage.
"""


from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ResultStatus(str, Enum):
    SUCCESS = "success"          # the call worked and returned content
    EMPTY = "empty"              # the call worked, and the answer is "nothing"
    INVALID = "invalid"          # arguments were wrong, the model may correct
    DENIED = "denied"            # policy refused, the model may NOT retry
    UNAVAILABLE = "unavailable"  # the dependency failed, retry may help later


@dataclass(frozen=True, slots=True)
class ToolResult:
    status: ResultStatus
    tool: str
    content: Any = None
    message: str = ""
    retryable: bool = False
    latency_ms: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.status is ResultStatus.SUCCESS

    def render_for_model(self) -> str:
        """Text the model sees. Each status renders differently on purpose.

        DENIED deliberately gives no detail about the policy, because a model
        that learns why it was refused will search for a phrasing that is not.
        """
        if self.status is ResultStatus.SUCCESS:
            return f"{self.tool} returned:\n{self.content}"
        if self.status is ResultStatus.EMPTY:
            return (
                f"{self.tool} completed successfully and found no matching "
                f"records. This is a definitive answer, not an error. Do not "
                f"retry with reworded arguments."
            )
        if self.status is ResultStatus.INVALID:
            return f"{self.tool} rejected the arguments: {self.message} Correct them and retry once."
        if self.status is ResultStatus.DENIED:
            return (
                f"{self.tool} is not permitted for this request. Do not attempt "
                f"this action or an equivalent one. Continue without it or stop."
            )
        return (
            f"{self.tool} is temporarily unavailable ({self.message}). "
            f"Proceed using other sources and state clearly in your answer that "
            f"this source was unavailable."
        )

"""Tool gateway.

Responsibilities, in execution order:
  1. authorise   principal may call this tool at all
  2. validate    arguments conform to the declared schema
  3. bound       per-tool timeout and a bounded retry ladder
  4. execute     with an idempotency key for any write
  5. classify    map the outcome onto a ResultStatus
  6. project     shrink the payload to the declared fields
  7. observe     emit one span with cost, latency, and outcome
"""


import hashlib
import json
import random
import time
from dataclasses import dataclass
from typing import Any, Callable

import jsonschema


@dataclass(frozen=True, slots=True)
class ToolSpec:
    name: str
    description: str
    schema: dict[str, Any]
    handler: Callable[..., Any]
    has_write_effect: bool = False
    timeout_seconds: float = 8.0
    max_attempts: int = 2
    projection: tuple[str, ...] = ()      # fields returned to the model
    required_scopes: frozenset[str] = frozenset()


class PolicyEngine:
    """Authorisation for tool access, evaluated per principal per call."""

    def __init__(self, scopes_by_principal: dict[str, frozenset[str]]) -> None:
        self._scopes = scopes_by_principal

    def authorised(self, principal: str, spec: ToolSpec) -> bool:
        held = self._scopes.get(principal, frozenset())
        return spec.required_scopes <= held


class ToolGateway:
    def __init__(self, policy: PolicyEngine, tracer, clock=time.monotonic) -> None:
        self._specs: dict[str, ToolSpec] = {}
        self._policy = policy
        self._tracer = tracer
        self._clock = clock

    def register(self, spec: ToolSpec) -> None:
        if spec.has_write_effect and "idempotency_key" not in spec.schema.get("properties", {}):
            # Refuse at registration time rather than discovering the duplicate
            # side effect during an incident.
            raise ValueError(f"write tool {spec.name} must accept idempotency_key")
        self._specs[spec.name] = spec

    def schema_for(self, principal: str) -> list[dict[str, Any]]:
        """Only advertise tools the principal may actually call.

        Filtering the advertised schema, rather than denying at call time, both
        improves selection accuracy and avoids leaking the existence of
        privileged tools.
        """
        return [
            {"name": s.name, "description": s.description, "parameters": s.schema}
            for s in self._specs.values()
            if self._policy.authorised(principal, s)
        ]

    def invoke(self, principal: str, name: str, arguments: dict[str, Any],
               run_id: str = "") -> ToolResult:
        spec = self._specs.get(name)
        if spec is None:
            return ToolResult(ResultStatus.INVALID, name,
                              message=f"Unknown tool '{name}'.")

        with self._tracer.span("tool.invoke", tool=name, principal=principal) as span:
            if not self._policy.authorised(principal, spec):
                span.set("outcome", "denied")
                return ToolResult(ResultStatus.DENIED, name, message="not permitted")

            try:
                jsonschema.validate(arguments, spec.schema)
            except jsonschema.ValidationError as exc:
                span.set("outcome", "invalid")
                return ToolResult(
                    ResultStatus.INVALID, name,
                    message=self._explain(exc), retryable=True,
                )

            if spec.has_write_effect:
                arguments = {
                    **arguments,
                    "idempotency_key": self._idempotency_key(run_id, name, arguments),
                }

            return self._execute_with_retries(spec, arguments, span)

    # ------------------------------------------------------------------
    def _execute_with_retries(self, spec: ToolSpec, arguments: dict, span) -> ToolResult:
        last_message = ""
        for attempt in range(1, spec.max_attempts + 1):
            started = self._clock()
            try:
                payload = spec.handler(**arguments, _timeout=spec.timeout_seconds)
            except TimeoutError:
                last_message = f"timeout after {spec.timeout_seconds}s"
            except Exception as exc:                       # noqa: BLE001
                last_message = f"{type(exc).__name__}: {exc}"
            else:
                elapsed = (self._clock() - started) * 1000.0
                span.set("outcome", "success")
                span.set("latency_ms", elapsed)
                if payload in (None, [], {}, ""):
                    return ToolResult(ResultStatus.EMPTY, spec.name, latency_ms=elapsed)
                return ToolResult(
                    ResultStatus.SUCCESS, spec.name,
                    content=self._project(payload, spec.projection),
                    latency_ms=elapsed,
                )

            # A write must never be retried automatically unless the handler
            # honours the idempotency key, which register() enforced.
            if attempt < spec.max_attempts:
                time.sleep(min(2.0, 0.25 * (2 ** attempt)) * (0.5 + random.random()))

        span.set("outcome", "unavailable")
        return ToolResult(
            ResultStatus.UNAVAILABLE, spec.name, message=last_message, retryable=True
        )

    @staticmethod
    def _project(payload: Any, projection: tuple[str, ...]) -> Any:
        """Return only declared fields, so a schema change upstream cannot
        silently double the context cost of every downstream step."""
        if not projection:
            return payload
        if isinstance(payload, dict):
            return {k: payload.get(k) for k in projection}
        if isinstance(payload, list):
            return [{k: row.get(k) for k in projection} for row in payload]
        return payload

    @staticmethod
    def _idempotency_key(run_id: str, name: str, arguments: dict) -> str:
        canonical = json.dumps(arguments, sort_keys=True, default=str)
        digest = hashlib.sha256(f"{run_id}|{name}|{canonical}".encode()).hexdigest()
        return digest[:32]

    @staticmethod
    def _explain(exc: jsonschema.ValidationError) -> str:
        location = ".".join(str(part) for part in exc.absolute_path) or "(root)"
        return f"Field '{location}' is invalid: {exc.message}"

"""An MCP server for an internal incident database.

Two design points worth copying:
  * The tool advertises tight enums and bounds, so the client's constrained
    decoder cannot emit an invalid call.
  * Resource access is confined to an allowlisted root, checked after path
    resolution, which is what defeats traversal through symlinks.
"""


import pathlib
from datetime import date
from typing import Literal

try:
    from mcp.server.fastmcp import FastMCP
except ImportError:
    class FastMCP:  # type: ignore
        def __init__(self, name: str, **kwargs):
            self.name = name
        def tool(self):
            def decorator(fn):
                return fn
            return decorator
        def resource(self, path: str):
            def decorator(fn):
                return fn
            return decorator


server = FastMCP("incident-db")
RUNBOOK_ROOT = pathlib.Path("/srv/runbooks").resolve()


@server.tool()
def search_incidents(
    service: str,
    severity: Literal["sev1", "sev2", "sev3"],
    since_date_utc: date,
    limit: int = 10,
) -> list[dict]:
    """Find past incidents for a service, newest first.

    Use this to find precedent for a live incident, or to summarise reliability
    history for one service. Do not use this to create or update an incident;
    use create_incident for that.

    Args:
        service: exact service name as registered in the service catalogue.
        severity: incident severity band.
        since_date_utc: only incidents opened on or after this UTC date.
        limit: maximum incidents to return, between 1 and 50.
    """
    limit = max(1, min(50, limit))
    rows = incident_store.query(
        service=service, severity=severity, since=since_date_utc, limit=limit
    )
    # Projection. The full row carries thirty fields and several kilobytes of
    # postmortem text, which would enter context on every subsequent step.
    return [
        {
            "incident_id": row.incident_id,
            "opened_at_utc": row.opened_at.isoformat(),
            "severity": row.severity,
            "one_line_cause": row.one_line_cause,
            "duration_minutes": row.duration_minutes,
        }
        for row in rows
    ]


@server.resource("runbook://{service}/{name}")
def runbook(service: str, name: str) -> str:
    """Return a runbook document for a service.

    Resources are application controlled, so the host decides when a runbook
    enters context. That keeps a large corpus out of the model's tool loop.
    """
    candidate = (RUNBOOK_ROOT / service / f"{name}.md").resolve()
    # Resolve first, then check containment. Checking before resolution is the
    # classic traversal bug, since '../' and symlinks both survive it.
    if RUNBOOK_ROOT not in candidate.parents:
        raise PermissionError("runbook path outside the allowed root")
    if not candidate.is_file():
        raise FileNotFoundError(f"no runbook {service}/{name}")
    return candidate.read_text(encoding="utf-8")
