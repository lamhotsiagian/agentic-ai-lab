from __future__ import annotations

"""Local-first execution with honest escalation.

Escalation is decided from measured signals, never from the small model's
stated confidence, which is not calibrated. When the device is offline or the
user has declined egress, the local path must produce a defined outcome
rather than a quietly worse answer.
"""


from dataclasses import dataclass
from enum import Enum


class Route(str, Enum):
    LOCAL = "local"
    ESCALATE = "escalate"
    LOCAL_DEGRADED = "local_degraded"
    REFUSE = "refuse"


@dataclass(frozen=True, slots=True)
class LocalSignals:
    schema_valid: bool               # did structured output validate
    retrieval_coverage: float        # fraction of the query covered by local index
    intent_in_distribution: bool     # router saw this intent during evaluation
    output_length_ratio: float       # observed / expected, anomaly detector
    verifier_score: float
    required_steps_estimate: int


@dataclass(frozen=True, slots=True)
class DeviceContext:
    online: bool
    egress_permitted: bool           # user or policy consent for this data class
    battery_fraction: float
    thermal_throttled: bool
    data_classification: str         # ordinary | personal | restricted


@dataclass(frozen=True, slots=True)
class RoutingResult:
    route: Route
    reason: str
    disclosure: str = ""


class LocalFirstRouter:
    COVERAGE_FLOOR = 0.55
    VERIFIER_FLOOR = 0.70
    LOCAL_STEP_CEILING = 3

    def decide(self, signals: LocalSignals, device: DeviceContext) -> RoutingResult:
        needs_help = (
            not signals.schema_valid
            or signals.retrieval_coverage < self.COVERAGE_FLOOR
            or not signals.intent_in_distribution
            or signals.verifier_score < self.VERIFIER_FLOOR
            or signals.required_steps_estimate > self.LOCAL_STEP_CEILING
            or not (0.4 <= signals.output_length_ratio <= 2.5)
        )

        if not needs_help:
            return RoutingResult(Route.LOCAL, "local_signals_healthy")

        # Restricted data never leaves the device, regardless of consent state
        # or how much better the cloud answer would be.
        if device.data_classification == "restricted":
            return RoutingResult(
                Route.LOCAL_DEGRADED, "restricted_data_no_egress",
                "This answer was produced entirely on your device and may be "
                "less complete than usual.",
            )

        if not device.online:
            return RoutingResult(
                Route.LOCAL_DEGRADED, "offline",
                "You are offline, so this answer used on-device knowledge only. "
                "Reconnect for a fuller answer.",
            )

        if not device.egress_permitted:
            return RoutingResult(
                Route.LOCAL_DEGRADED, "egress_not_permitted",
                "This answer stayed on your device because cloud assistance is "
                "turned off for this data.",
            )

        if device.thermal_throttled or device.battery_fraction < 0.15:
            # Escalating is actually the power-efficient choice here, since a
            # long local generation costs more energy than one network call.
            return RoutingResult(
                Route.ESCALATE, "power_constrained",
                "Using cloud assistance to save battery.",
            )

        return RoutingResult(
            Route.ESCALATE, "local_signals_insufficient",
            "Some of this request was processed in the cloud to give a more "
            "complete answer.",
        )
