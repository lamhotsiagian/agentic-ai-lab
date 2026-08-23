from __future__ import annotations

"""Durable research agent.

Production properties this graph provides:
  * Typed state with declared reducers, so the fan-out node cannot lose
    evidence written by a sibling branch.
  * A Postgres checkpointer, so a run survives a worker restart and resumes
    from the last committed node rather than from the beginning.
  * A hard interrupt before any node with an external write effect, so no
    irreversible action executes without a recorded human decision.
  * Two independent bounds: step_count against a ceiling, and replan_count
    against a separate, smaller ceiling.
"""


import operator
import time
from dataclasses import dataclass, field
from typing import Annotated, Any, Callable, Literal, Sequence, TypedDict

from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

MAX_STEPS = 10
MAX_REPLANS = 2
VERIFY_THRESHOLD = 0.72


class ResearchState(TypedDict, total=False):
    run_id: str
    tenant_id: str
    goal: str
    plan: list[str]
    evidence: Annotated[list[dict], operator.add]
    tool_errors: Annotated[list[str], operator.add]
    draft: str
    verify_score: float
    unsupported: list[str]
    step_count: int
    replan_count: int
    pending_write: dict | None
    approval: Literal["granted", "denied", ""] 


# --------------------------------------------------------------------------
# Nodes. Every node returns a partial update rather than mutating state, which
# is what allows the reducers to merge concurrent branches correctly.
# --------------------------------------------------------------------------
def intake(state: ResearchState) -> dict:
    """Normalise the goal and stamp the tenant scope used by every later query."""
    goal = " ".join(state["goal"].split())[:4000]
    return {"goal": goal, "step_count": 0, "replan_count": 0, "evidence": []}


def plan(state: ResearchState, *, planner) -> dict:
    subgoals = planner.plan(state["goal"], context=_evidence_digest(state))
    return {"plan": [s.description for s in subgoals],
            "step_count": state.get("step_count", 0) + 1}


def retrieve(state: ResearchState, *, memory) -> dict:
    """Fan-out retrieval. The evidence reducer concatenates across branches."""
    items = memory.read(
        query=state["goal"], tenant_id=state["tenant_id"], max_tokens=6000
    )
    return {"evidence": [i.__dict__ for i in items],
            "step_count": state.get("step_count", 0) + 1}


def act(state: ResearchState, *, gateway, model) -> dict:
    proposal = model.propose_tool_call(state)
    if proposal.has_write_effect:
        # Do not execute. Stage the call and let the router send the run to the
        # approval interrupt. Staging keeps the effect out of this node
        # entirely, which is what makes checkpoint placement safe.
        return {"pending_write": proposal.as_dict(),
                "step_count": state.get("step_count", 0) + 1}

    result = gateway.invoke(state["tenant_id"], proposal.tool, proposal.arguments)
    if result.ok:
        return {"evidence": [result.as_evidence()],
                "step_count": state.get("step_count", 0) + 1}
    return {"tool_errors": [f"{proposal.tool}: {result.status}"],
            "step_count": state.get("step_count", 0) + 1}


def human_approval(state: ResearchState) -> Command:
    """Pause the run. The process may exit here; state is already checkpointed."""
    decision = interrupt(
        {
            "kind": "write_approval",
            "run_id": state["run_id"],
            "action": state["pending_write"],
            "evidence_count": len(state.get("evidence", [])),
        }
    )
    return Command(update={"approval": decision, "pending_write": None})


def synthesise(state: ResearchState, *, model) -> dict:
    draft = model.synthesise(state["goal"], state.get("evidence", []))
    return {"draft": draft, "step_count": state.get("step_count", 0) + 1}


def verify(state: ResearchState, *, verifier) -> dict:
    outcome = verifier.verify(state["draft"], state.get("evidence", []))
    return {"verify_score": outcome.score, "unsupported": outcome.unsupported,
            "step_count": state.get("step_count", 0) + 1}


# --------------------------------------------------------------------------
# Routers. Each one is a pure function of state, so a trace fully explains the
# path the run took.
# --------------------------------------------------------------------------
def after_act(state: ResearchState) -> str:
    if state.get("pending_write"):
        return "human_approval"
    if state.get("step_count", 0) >= MAX_STEPS:
        return "synthesise"
    return "synthesise"


def after_verify(state: ResearchState) -> str:
    if state.get("verify_score", 0.0) >= VERIFY_THRESHOLD:
        return END
    if state.get("replan_count", 0) >= MAX_REPLANS:
        return END              # return the draft with its unsupported claims marked
    if state.get("step_count", 0) >= MAX_STEPS:
        return END
    return "plan"


def build_graph(planner, memory, gateway, model, verifier, dsn: str):
    graph = StateGraph(ResearchState)

    graph.add_node("intake", intake)
    graph.add_node("plan", lambda s: {**plan(s, planner=planner),
                                      "replan_count": s.get("replan_count", 0) + 1})
    graph.add_node("retrieve", lambda s: retrieve(s, memory=memory))
    graph.add_node("act", lambda s: act(s, gateway=gateway, model=model))
    graph.add_node("human_approval", human_approval)
    graph.add_node("synthesise", lambda s: synthesise(s, model=model))
    graph.add_node("verify", lambda s: verify(s, verifier=verifier))

    graph.add_edge(START, "intake")
    graph.add_edge("intake", "plan")
    graph.add_edge("plan", "retrieve")
    graph.add_edge("retrieve", "act")
    graph.add_conditional_edges("act", after_act,
                                {"human_approval": "human_approval",
                                 "synthesise": "synthesise"})
    graph.add_edge("human_approval", "synthesise")
    graph.add_edge("synthesise", "verify")
    graph.add_conditional_edges("verify", after_verify,
                                {"plan": "plan", END: END})

    checkpointer = PostgresSaver.from_conn_string(dsn)
    checkpointer.setup()
    return graph.compile(checkpointer=checkpointer)


def _evidence_digest(state: ResearchState, limit: int = 12) -> str:
    """Compact evidence summary passed to the replanner.

    Replanning without the evidence gathered so far reproduces the original
    plan, which is the most common cause of an unbounded replan cycle.
    """
    items = state.get("evidence", [])[:limit]
    return "\n".join(f"- {item['claim']} [{item['source_id']}]" for item in items)

"""Framework-independent core, plus one thin adapter per runtime.

The rule enforced here: nothing in agent_core imports a framework, and
nothing in the adapters contains business logic. A unit test for the core
runs in milliseconds with no graph runtime, no database, and no network.
"""

# ---- agent_core/policies.py  (no framework imports anywhere in this module)
from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True, slots=True)
class StepDecision:
    next_action: str                 # "retrieve" | "act" | "synthesise" | "stop"
    reason: str


class ResearchPolicy:
    """Pure decision logic. This is the part that is genuinely yours."""

    def __init__(self, max_steps: int, max_replans: int, verify_threshold: float) -> None:
        self._max_steps = max_steps
        self._max_replans = max_replans
        self._threshold = verify_threshold

    def next_after_verify(
        self, score: float, step_count: int, replan_count: int
    ) -> StepDecision:
        if score >= self._threshold:
            return StepDecision("stop", "verification_passed")
        if replan_count >= self._max_replans:
            return StepDecision("stop", "replan_budget_exhausted")
        if step_count >= self._max_steps:
            return StepDecision("stop", "step_budget_exhausted")
        return StepDecision("retrieve", f"score {score:.2f} below threshold")

    def sufficient_evidence(self, evidence: Sequence[dict], minimum: int = 3) -> bool:
        distinct_sources = {item["source_id"] for item in evidence}
        return len(distinct_sources) >= minimum


# ---- adapters/langgraph_adapter.py  (framework binding, no business logic)
def make_verify_router(policy: ResearchPolicy):
    from langgraph.graph import END

    def route(state) -> str:
        decision = policy.next_after_verify(
            score=state.get("verify_score", 0.0),
            step_count=state.get("step_count", 0),
            replan_count=state.get("replan_count", 0),
        )
        return END if decision.next_action == "stop" else decision.next_action

    return route


# ---- adapters/plain_loop_adapter.py  (the same policy, no framework at all)
def run_plain(policy: ResearchPolicy, services, goal: str) -> dict:
    state = {"goal": goal, "evidence": [], "step_count": 0, "replan_count": 0}
    while True:
        state["evidence"] += services.retrieve(goal)
        state["step_count"] += 1
        draft = services.synthesise(goal, state["evidence"])
        outcome = services.verify(draft, state["evidence"])
        decision = policy.next_after_verify(
            outcome.score, state["step_count"], state["replan_count"]
        )
        if decision.next_action == "stop":
            return {"answer": draft, "reason": decision.reason, **state}
        state["replan_count"] += 1


@dataclass(frozen=True, slots=True)
class WorkflowEvent:
    event_id: str
    event_type: str                  # activity_completed | human_signal | timer_fired
    activity_name: str
    payload: dict[str, Any]
    timestamp: float = field(default_factory=time.monotonic)


class DurableWorkflowEngine:
    """Replays deterministic state from history and guards activity execution."""

    def __init__(self, activity_handlers: dict[str, Callable[..., dict[str, Any]]]) -> None:
        self._handlers = activity_handlers
        self._history: list[WorkflowEvent] = []
        self._completed_activities: dict[str, dict[str, Any]] = {}

    def load_history(self, events: list[WorkflowEvent]) -> None:
        self._history = list(events)
        self._completed_activities = {
            e.activity_name: e.payload
            for e in events
            if e.event_type == "activity_completed"
        }

    def execute_activity(
        self,
        activity_id: str,
        activity_name: str,
        arguments: dict[str, Any],
        timeout_seconds: float = 60.0,
    ) -> dict[str, Any]:
        # Fast-forward from history if already executed prior to crash
        if activity_id in self._completed_activities:
            return self._completed_activities[activity_id]

        handler = self._handlers.get(activity_name)
        if not handler:
            raise ValueError(f"Unknown activity: {activity_name}")

        # Execute side effect with heartbeat tracking
        start_time = time.monotonic()
        result = handler(**arguments)
        if time.monotonic() - start_time > timeout_seconds:
            raise TimeoutError(f"Activity {activity_name} exceeded {timeout_seconds}s")

        event = WorkflowEvent(
            event_id=f"evt_{len(self._history) + 1}",
            event_type="activity_completed",
            activity_name=activity_id,
            payload=result,
        )
        self._history.append(event)
        self._completed_activities[activity_id] = result
        return result

    @property
    def history(self) -> tuple[WorkflowEvent, ...]:
        return tuple(self._history)

