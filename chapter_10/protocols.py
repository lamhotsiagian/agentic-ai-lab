from __future__ import annotations

"""Hardened MCP client wrapper.

Three controls that a default client does not give you:
  * sampling is gated by an allowlist, a rate limit, and a token cap
  * tool descriptions are hashed and pinned, so a silent change is detected
  * capability assumptions fail at connect time rather than at call time
"""


import hashlib
import time
from dataclasses import dataclass, field


@dataclass
class ServerPin:
    """What we agreed to when this server was reviewed."""
    server_id: str
    protocol_version: str
    tool_description_hash: str
    sampling_allowed: bool = False
    sampling_max_tokens: int = 512
    sampling_calls_per_minute: int = 6


class PinMismatch(RuntimeError):
    pass


class HardenedMcpClient:
    def __init__(self, transport, pin: ServerPin, audit) -> None:
        self._transport = transport
        self._pin = pin
        self._audit = audit
        self._sampling_calls: list[float] = []

    async def connect(self) -> None:
        result = await self._transport.initialize(
            client_capabilities={"sampling": self._pin.sampling_allowed, "roots": True}
        )
        if result.protocol_version != self._pin.protocol_version:
            # Fail at connect time. A version drift discovered during a call is
            # an incident; discovered at connect it is a deployment error.
            raise PinMismatch(
                f"{self._pin.server_id} speaks {result.protocol_version}, "
                f"pinned to {self._pin.protocol_version}"
            )

        tools = await self._transport.list_tools()
        observed = self._hash_descriptions(tools)
        if observed != self._pin.tool_description_hash:
            # Tool descriptions enter the model's context verbatim, so a change
            # is a prompt change. Treat it as a security-relevant diff and
            # refuse until it has been reviewed.
            self._audit.security_event(
                "mcp.tool_description_changed",
                server=self._pin.server_id,
                expected=self._pin.tool_description_hash,
                observed=observed,
            )
            raise PinMismatch(f"{self._pin.server_id} tool descriptions changed")

    async def handle_sampling_request(self, request) -> "SamplingResult":
        """Server asked us to run a completion on its behalf."""
        if not self._pin.sampling_allowed:
            return SamplingResult.refused("sampling not permitted for this server")

        now = time.monotonic()
        self._sampling_calls = [t for t in self._sampling_calls if now - t < 60.0]
        if len(self._sampling_calls) >= self._pin.sampling_calls_per_minute:
            return SamplingResult.refused("sampling rate limit exceeded")
        self._sampling_calls.append(now)

        capped = min(request.max_tokens, self._pin.sampling_max_tokens)
        self._audit.record(
            "mcp.sampling", server=self._pin.server_id, max_tokens=capped
        )
        return await self._run_completion(request, max_tokens=capped)

    @staticmethod
    def _hash_descriptions(tools) -> str:
        canonical = "\n".join(
            f"{tool.name}\x1f{tool.description}\x1f{sorted(tool.input_schema.items())}"
            for tool in sorted(tools, key=lambda t: t.name)
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

"""A2A peer agent: capability declaration and task lifecycle.

The card is the contract. Everything the client needs to decide whether to
delegate, and how to authenticate, is declared here rather than documented
elsewhere.
"""


from dataclasses import dataclass, field
from enum import Enum
from typing import AsyncIterator


class TaskState(str, Enum):
    SUBMITTED = "submitted"
    WORKING = "working"
    INPUT_REQUIRED = "input-required"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


AGENT_CARD = {
    "name": "policy-review-agent",
    "description": (
        "Reviews contract clauses against a jurisdiction's regulatory corpus "
        "and returns findings with citations. Does not provide legal advice."
    ),
    "url": "https://agents.internal.example/policy-review",
    "version": "2.3.0",
    "capabilities": {
        "streaming": True,
        "pushNotifications": True,
        "stateTransitionHistory": True,
    },
    "authentication": {"schemes": ["oauth2-client-credentials"]},
    "defaultInputModes": ["text/plain", "application/json"],
    "defaultOutputModes": ["application/json", "text/markdown"],
    "skills": [
        {
            "id": "clause_review",
            "name": "Contract clause review",
            "description": (
                "Given clause text and a jurisdiction, return regulatory findings "
                "with severity and citations. Use for pre-signature review. "
                "Do not use for litigation strategy."
            ),
            "inputModes": ["application/json"],
            "outputModes": ["application/json"],
            "examples": ["Review a data processing clause under EU jurisdiction"],
        }
    ],
}


@dataclass
class Task:
    task_id: str
    state: TaskState
    context_id: str
    principal: str
    artifacts: list[dict] = field(default_factory=list)
    history: list[tuple[TaskState, str]] = field(default_factory=list)

    def transition(self, state: TaskState, note: str = "") -> None:
        # History is part of the contract when stateTransitionHistory is
        # declared, and it is what makes a delegated failure explainable.
        self.history.append((state, note))
        self.state = state


class PolicyReviewAgent:
    """Handles A2A tasks. Internals stay opaque to the caller by design."""

    def __init__(self, orchestrator, store, max_runtime_seconds: float = 240.0) -> None:
        self._orchestrator = orchestrator
        self._store = store
        self._max_runtime = max_runtime_seconds

    async def handle(self, task: Task, message: dict) -> AsyncIterator[dict]:
        task.transition(TaskState.WORKING, "started")
        yield {"state": task.state, "progress": 0.0}

        jurisdiction = message.get("jurisdiction")
        if not jurisdiction:
            # Ask rather than guess. An input request is cheaper than a wrong
            # answer, and the client can route the question to a human.
            task.transition(TaskState.INPUT_REQUIRED, "jurisdiction missing")
            yield {
                "state": task.state,
                "required_input": {
                    "name": "jurisdiction",
                    "schema": {"type": "string", "enum": ["EU", "UK", "US", "SG"]},
                    "prompt": "Which jurisdiction governs this contract?",
                },
            }
            return

        try:
            async for update in self._orchestrator.run_streaming(
                goal=message["clause_text"],
                jurisdiction=jurisdiction,
                principal=task.principal,
                deadline_seconds=self._max_runtime,
            ):
                if update.kind == "progress":
                    yield {"state": TaskState.WORKING, "progress": update.fraction}
                elif update.kind == "artifact":
                    task.artifacts.append(update.artifact)
                    yield {"state": TaskState.WORKING, "artifact": update.artifact}
        except TimeoutError:
            task.transition(TaskState.FAILED, "exceeded max runtime")
            yield {"state": task.state, "error": "deadline_exceeded"}
            return
        except PermissionError as exc:
            task.transition(TaskState.FAILED, f"authorisation: {exc}")
            yield {"state": task.state, "error": "not_authorised"}
            return

        task.transition(TaskState.COMPLETED, "review complete")
        yield {"state": task.state, "artifacts": task.artifacts}

    async def cancel(self, task: Task) -> None:
        """Cancellation must be honoured promptly, because the caller may be
        holding a user-facing deadline that this task no longer fits inside."""
        await self._orchestrator.cancel(task.task_id)
        task.transition(TaskState.CANCELLED, "cancelled by caller")
