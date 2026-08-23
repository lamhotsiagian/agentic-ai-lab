from __future__ import annotations

"""Computer use step executor.

Three properties matter more than the action vocabulary:
  * grounding is layered, cheapest and most robust representation first
  * every action is followed by a verification of the expected end state
  * recovery is bounded and escalates, rather than retrying indefinitely
"""


from dataclasses import dataclass
from enum import Enum


class Grounding(str, Enum):
    ACCESSIBILITY = "accessibility_tree"
    MARKS = "set_of_marks"
    PIXELS = "pixel_coordinates"


@dataclass(frozen=True, slots=True)
class ActionIntent:
    verb: str                    # click | type | scroll | select | key
    description: str             # "the Submit button in the payment dialog"
    value: str | None = None     # text to type, option to select


@dataclass(frozen=True, slots=True)
class Expectation:
    """What must be true after the action, checked without asking the model."""
    description: str
    predicate_kind: str          # element_present | text_present | url_matches
    predicate_argument: str


@dataclass(frozen=True, slots=True)
class StepOutcome:
    ok: bool
    grounding_used: Grounding | None
    attempts: int
    reason: str
    screenshot_tokens: int


class ComputerUseExecutor:
    MAX_ATTEMPTS = 3

    def __init__(self, environment, model, strategy_memory, budget) -> None:
        self._env = environment
        self._model = model
        self._memory = strategy_memory      # per-application grounding history
        self._budget = budget

    def execute(self, intent: ActionIntent, expect: Expectation,
                app_id: str) -> StepOutcome:
        order = self._memory.preferred_order(app_id) or [
            Grounding.ACCESSIBILITY, Grounding.MARKS, Grounding.PIXELS
        ]
        tokens = 0

        for attempt in range(1, self.MAX_ATTEMPTS + 1):
            for grounding in order:
                if self._budget.exhausted():
                    return StepOutcome(False, None, attempt, "budget_exhausted", tokens)

                target, used_tokens = self._resolve(intent, grounding, app_id)
                tokens += used_tokens
                if target is None:
                    continue

                self._env.perform(intent.verb, target, intent.value)
                self._env.settle(timeout_seconds=2.0)

                # Verification is deterministic and does not consult the model.
                # A model asked whether its own action worked is a poor judge,
                # and a screenshot round trip is the most expensive thing here.
                if self._satisfied(expect):
                    self._memory.record_success(app_id, grounding)
                    return StepOutcome(True, grounding, attempt, "verified", tokens)

                self._memory.record_failure(app_id, grounding)
                # Undo where possible before trying another representation, so
                # a partially applied action does not corrupt the next attempt.
                self._env.revert_if_possible()

        return StepOutcome(False, None, self.MAX_ATTEMPTS, "unverified_after_retries",
                           tokens)

    def _resolve(self, intent: ActionIntent, grounding: Grounding,
                 app_id: str) -> tuple[object | None, int]:
        if grounding is Grounding.ACCESSIBILITY:
            tree = self._env.accessibility_tree()
            if not tree.usable:
                return None, 0
            # A compact textual tree costs a small fraction of a screenshot.
            choice = self._model.select_element(intent.description, tree.render())
            return tree.resolve(choice.element_id), choice.tokens

        if grounding is Grounding.MARKS:
            shot, marks = self._env.screenshot_with_marks()
            choice = self._model.select_mark(intent.description, shot, marks)
            return marks.resolve(choice.mark_id), choice.tokens

        shot = self._env.screenshot()
        point = self._model.locate(intent.description, shot)
        if point.confidence < 0.6:
            return None, point.tokens
        return point.as_target(), point.tokens

    def _satisfied(self, expect: Expectation) -> bool:
        if expect.predicate_kind == "element_present":
            return self._env.has_element(expect.predicate_argument)
        if expect.predicate_kind == "text_present":
            return self._env.has_text(expect.predicate_argument)
        if expect.predicate_kind == "url_matches":
            return self._env.url_matches(expect.predicate_argument)
        return False


class PerceptionTier(str, Enum):
    A11Y_TREE = "a11y_tree"
    DIFF_SLICE = "diff_slice"
    FULL_VISION = "full_vision"


@dataclass(frozen=True, slots=True)
class ObservationSlice:
    tier: PerceptionTier
    payload: str | bytes
    token_cost: int
    bounding_box: tuple[int, int, int, int] | None = None   # (x1, y1, x2, y2)


class PerceptionLadder:
    """Selects the cheapest perception representation that satisfies grounding."""

    def __init__(self, a11y_token_cost: int = 120, diff_token_cost: int = 240,
                 full_vision_token_cost: int = 1200) -> None:
        self._costs = {
            PerceptionTier.A11Y_TREE: a11y_token_cost,
            PerceptionTier.DIFF_SLICE: diff_token_cost,
            PerceptionTier.FULL_VISION: full_vision_token_cost,
        }

    def perceive(self, environment, previous_hash: str | None = None) -> ObservationSlice:
        # Tier 1: Try accessibility tree first (cheapest & semantic)
        a11y_data = environment.get_a11y_tree()
        if a11y_data:
            return ObservationSlice(
                tier=PerceptionTier.A11Y_TREE,
                payload=a11y_data,
                token_cost=self._costs[PerceptionTier.A11Y_TREE],
            )

        # Tier 2: Check for differential screenshot region if layout changed locally
        diff_crop = environment.get_differential_crop(previous_hash)
        if diff_crop:
            return ObservationSlice(
                tier=PerceptionTier.DIFF_SLICE,
                payload=diff_crop["bytes"],
                token_cost=self._costs[PerceptionTier.DIFF_SLICE],
                bounding_box=diff_crop["bbox"],
            )

        # Tier 3: Full viewport screenshot fallback
        full_frame = environment.get_full_screenshot()
        return ObservationSlice(
            tier=PerceptionTier.FULL_VISION,
            payload=full_frame,
            token_cost=self._costs[PerceptionTier.FULL_VISION],
        )

