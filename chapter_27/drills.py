from __future__ import annotations

"""Retry that respects typed outcomes.

The interview point: never retry a DENIED or INVALID result. Retrying a
denial probes your policy, and retrying invalid arguments without correcting
them wastes budget deterministically.
"""


import random
import time
from dataclasses import dataclass
from typing import Callable, TypeVar

T = TypeVar("T")

RETRYABLE = {"UNAVAILABLE", "TIMEOUT", "RATE_LIMITED"}


@dataclass(frozen=True, slots=True)
class Attempt:
    index: int
    status: str
    delay_before: float


def retry_typed(
    call: Callable[[], "ToolResult"],
    max_attempts: int = 3,
    base_delay: float = 0.25,
    max_delay: float = 8.0,
    deadline_seconds: float | None = None,
    sleep: Callable[[float], None] = time.sleep,
    rng: random.Random | None = None,
) -> tuple["ToolResult", list[Attempt]]:
    rng = rng or random.Random()
    started = time.monotonic()
    history: list[Attempt] = []
    delay = 0.0

    for index in range(1, max_attempts + 1):
        if delay:
            # Full jitter. Equal jitter and no jitter both leave retries
            # synchronised across concurrent runs, which is how a degraded
            # dependency becomes an outage.
            actual = rng.uniform(0.0, delay)
            if deadline_seconds is not None:
                remaining = deadline_seconds - (time.monotonic() - started)
                if actual >= remaining:
                    history.append(Attempt(index, "DEADLINE", actual))
                    return _deadline_result(), history
            sleep(actual)
        else:
            actual = 0.0

        result = call()
        history.append(Attempt(index, result.status, actual))

        if result.status not in RETRYABLE:
            return result, history          # success, empty, invalid, denied
        if index == max_attempts:
            return result, history

        delay = min(max_delay, base_delay * (2 ** (index - 1)))
        # Honour a server-provided hint over our own schedule when present.
        hint = result.metadata.get("retry_after_seconds")
        if hint is not None:
            delay = min(max_delay, float(hint))

    raise AssertionError("unreachable")

from collections import defaultdict
from typing import Iterable, Sequence


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[str]],
    k: int = 60,
    weights: Sequence[float] | None = None,
) -> list[tuple[str, float]]:
    """Fuse ranked identifier lists by ordinal position.

    Scale free, so no normalisation is needed and the result is stable when
    either index is retuned. Weights are optional and multiply a whole
    ranking's contribution, which is a legitimate use of a constant because it
    expresses trust in a retriever rather than comparing incomparable scores.
    """
    if weights is None:
        weights = [1.0] * len(rankings)
    if len(weights) != len(rankings):
        raise ValueError("weights must match rankings")

    scores: dict[str, float] = defaultdict(float)
    first_seen: dict[str, tuple[int, int]] = {}

    for source_index, (ranking, weight) in enumerate(zip(rankings, weights)):
        for rank, identifier in enumerate(ranking, start=1):
            scores[identifier] += weight / (k + rank)
            first_seen.setdefault(identifier, (source_index, rank))

    # Deterministic tie-breaking. Without it, equal scores order by dictionary
    # insertion, which makes results irreproducible across runs and makes any
    # regression test flaky.
    return sorted(
        scores.items(),
        key=lambda pair: (-pair[1], first_seen[pair[0]]),
    )

"""Incremental stream parsing.

The bug interviewers look for: assuming each network chunk contains whole
events. It does not. A robust parser buffers, splits on complete delimiters,
and keeps the remainder.
"""


import json
from dataclasses import dataclass, field
from typing import Iterator


@dataclass
class EventStreamParser:
    last_seq: int = 0
    _buffer: str = ""
    malformed: int = 0
    duplicates: int = 0
    _pending: list[dict] = field(default_factory=list)

    def feed(self, chunk: str) -> Iterator[dict]:
        """Yield complete events from an arbitrary byte chunk boundary."""
        self._buffer += chunk
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                # A malformed line must not kill the stream. Count it, emit a
                # metric, and continue; the sequence gap is detectable later.
                self.malformed += 1
                continue

            seq = event.get("seq")
            if seq is None:
                yield event
                continue
            if seq <= self.last_seq:
                self.duplicates += 1        # redelivery after a reconnect
                continue
            self.last_seq = seq
            yield event

    def resume_header(self) -> dict[str, str]:
        """What to send on reconnect so the server replays from the gap."""
        return {"Last-Event-Sequence": str(self.last_seq)}

    def finish(self) -> Iterator[dict]:
        """Flush a trailing event with no terminating newline."""
        if self._buffer.strip():
            yield from self.feed("\n")

"""No-progress detection.

Exact-argument fingerprinting misses the common failure where the agent
rewords the same query. Normalising before hashing catches it, and the
normaliser is where the interview conversation goes.
"""


import hashlib
import re
import unicodedata
from collections import Counter

_WHITESPACE = re.compile(r"\s+")
_STOPWORDS = frozenset({"the", "a", "an", "of", "for", "in", "on", "to", "and"})


def normalise_value(value: object) -> str:
    if isinstance(value, str):
        text = unicodedata.normalize("NFKC", value).casefold()
        text = _WHITESPACE.sub(" ", text).strip()
        tokens = [t for t in text.split(" ") if t not in _STOPWORDS]
        # Sorting the token multiset makes reordered phrasings collide, which
        # is the behaviour we want for a progress check even though it would be
        # wrong for a cache key.
        return " ".join(sorted(tokens))
    if isinstance(value, float):
        return f"{value:.6g}"
    if isinstance(value, dict):
        return "{" + ",".join(
            f"{k}={normalise_value(v)}" for k, v in sorted(value.items())
        ) + "}"
    if isinstance(value, (list, tuple)):
        return "[" + ",".join(sorted(normalise_value(v) for v in value)) + "]"
    return str(value)


def fingerprint(tool: str, arguments: dict) -> str:
    payload = tool + "|" + normalise_value(arguments)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


class NoProgressDetector:
    def __init__(self, threshold: int = 2) -> None:
        self._counts: Counter[str] = Counter()
        self._threshold = threshold

    def observe(self, tool: str, arguments: dict) -> bool:
        """Return True when the run should halt for lack of progress."""
        key = fingerprint(tool, arguments)
        self._counts[key] += 1
        return self._counts[key] > self._threshold
