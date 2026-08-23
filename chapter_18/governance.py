from __future__ import annotations

"""Escalation router.

Reviewer attention is a finite resource with a queue, a service level, and a
saturation point. Modelling it explicitly is what prevents the failure where
an overloaded queue silently becomes a source of stale approvals.
"""


from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum


class Trigger(str, Enum):
    GUARDRAIL_BLOCK = "guardrail_block"
    INJECTION_SUSPECTED = "injection_suspected"
    IRREVERSIBLE_ACTION = "irreversible_action"
    COSTLY_ACTION = "costly_action"
    LOW_CONFIDENCE = "low_confidence"
    USER_REQUESTED = "user_requested"
    POLICY_ASSERTION = "policy_assertion"
    REPEATED_FAILURE = "repeated_failure"


@dataclass(frozen=True, slots=True)
class Queue:
    name: str
    sla: timedelta
    reviewer_pool: str
    expiry: timedelta
    on_expiry: str                    # deny | proceed_degraded | cancel
    max_depth: int                    # beyond this the router sheds load


ROUTING: dict[Trigger, str] = {
    Trigger.GUARDRAIL_BLOCK:     "safety",
    Trigger.INJECTION_SUSPECTED: "safety",
    Trigger.IRREVERSIBLE_ACTION: "approval",
    Trigger.COSTLY_ACTION:       "approval",
    Trigger.POLICY_ASSERTION:    "approval",
    Trigger.LOW_CONFIDENCE:      "expert",
    Trigger.REPEATED_FAILURE:    "expert",
    Trigger.USER_REQUESTED:      "expert",
}

QUEUES = {
    "safety":   Queue("safety", timedelta(minutes=5), "security_oncall",
                      timedelta(minutes=30), "deny", max_depth=50),
    "approval": Queue("approval", timedelta(minutes=30), "operations",
                      timedelta(hours=4), "deny", max_depth=400),
    "expert":   Queue("expert", timedelta(hours=4), "domain_specialists",
                      timedelta(hours=48), "cancel", max_depth=1200),
}


@dataclass(frozen=True, slots=True)
class EscalationDecision:
    queue: str
    due_at: datetime
    expires_at: datetime
    shed: bool
    shed_action: str = ""


class EscalationRouter:
    def __init__(self, store, metrics) -> None:
        self._store = store
        self._metrics = metrics

    def route(self, trigger: Trigger, run_id: str, payload: dict,
              now: datetime | None = None) -> EscalationDecision:
        now = now or datetime.now(timezone.utc)
        queue = QUEUES[ROUTING[trigger]]
        depth = self._store.depth(queue.name)

        self._metrics.gauge("escalation.queue_depth", depth, queue=queue.name)

        if depth >= queue.max_depth:
            # Saturation. Shedding explicitly is better than accepting work the
            # queue cannot service within its SLA, because a stale approval is
            # indistinguishable from an unreviewed one.
            self._metrics.increment("escalation.shed", queue=queue.name)
            action = "deny" if queue.name != "expert" else "cancel"
            return EscalationDecision(queue.name, now, now, shed=True,
                                      shed_action=action)

        item_id = self._store.enqueue(
            queue=queue.name, run_id=run_id, trigger=trigger.value,
            payload=payload, due_at=now + queue.sla,
            expires_at=now + queue.expiry,
        )
        self._metrics.increment("escalation.enqueued",
                                queue=queue.name, trigger=trigger.value)
        return EscalationDecision(
            queue.name, now + queue.sla, now + queue.expiry, shed=False,
        )

    def sweep_expired(self, now: datetime | None = None) -> int:
        """A paused run holds partial state indefinitely without this.

        Expiry produces a distinct terminal status so that "nobody reviewed it"
        is never confused with "it was reviewed and denied".
        """
        now = now or datetime.now(timezone.utc)
        swept = 0
        for item in self._store.expired(now):
            queue = QUEUES[item.queue]
            self._store.terminate(
                item.item_id,
                status=f"expired_{queue.on_expiry}",
                note=f"no reviewer decision within {queue.expiry}",
            )
            self._metrics.increment("escalation.expired", queue=item.queue,
                                    trigger=item.trigger)
            swept += 1
        return swept

"""Agent registry.

Every deployed agent has an entry. The gateway reads authority from here at
call time, so the registry is the enforcement point rather than a record of
intent. An expired or unowned agent cannot act.
"""


from dataclasses import dataclass, field
from datetime import date, datetime, timezone


@dataclass(frozen=True, slots=True)
class AgentAuthority:
    allowed_tools: frozenset[str]
    max_spend_per_run_usd: float
    max_spend_per_day_usd: float
    data_scopes: frozenset[str]           # which datasets, which tenants
    autonomy_by_action_class: dict[str, int]
    gated_action_classes: frozenset[str]
    egress_allowlist: frozenset[str]


@dataclass(frozen=True, slots=True)
class AgentRecord:
    agent_id: str
    name: str
    purpose: str                          # one sentence, in scope
    out_of_scope: tuple[str, ...]         # explicit, and enforced by refusal

    # --- accountability -------------------------------------------------
    business_owner: str                   # accountable for outcomes
    engineering_owner: str                # accountable for behaviour
    reviewer_group: str                   # approves its escalations
    risk_classification: str              # minimal | limited | high

    # --- authority ------------------------------------------------------
    authority: AgentAuthority
    authority_granted_by: str
    authority_granted_at: datetime
    authority_review_due: date            # re-authorisation deadline

    # --- lifecycle ------------------------------------------------------
    version: str
    deployed_at: datetime
    system_card_uri: str
    kill_switch_scope: str                # agent | action_class | global
    metadata: dict = field(default_factory=dict)

    def is_authorised(self, today: date | None = None) -> tuple[bool, str]:
        today = today or datetime.now(timezone.utc).date()
        if today > self.authority_review_due:
            # Authority expires. Without expiry, an agent deployed by someone
            # who left the company two years ago still holds its credentials.
            return False, "authority_expired"
        if not self.business_owner or not self.engineering_owner:
            return False, "no_named_owner"
        return True, "authorised"

    def may_call(self, tool: str, action_class: str, today: date | None = None
                 ) -> tuple[bool, bool, str]:
        """Returns (allowed, requires_approval, reason)."""
        ok, reason = self.is_authorised(today)
        if not ok:
            return False, False, reason
        if tool not in self.authority.allowed_tools:
            return False, False, "tool_not_in_authority"
        if action_class in self.authority.gated_action_classes:
            return True, True, "gated_action_class"
        return True, False, "within_authority"
