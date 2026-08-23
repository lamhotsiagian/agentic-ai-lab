from __future__ import annotations

"""Localisation for a coding agent.

Similarity search alone performs poorly on code, because two files can be
lexically similar and causally unrelated. The symbol graph supplies the
causal signal, and the co-change history supplies a cheap prior over what
tends to move together.
"""


from collections import defaultdict
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class CodeUnit:
    path: str
    symbol: str
    kind: str                  # function | class | method | module
    start_line: int
    end_line: int
    text: str
    tokens: int


@dataclass(frozen=True, slots=True)
class LocalisationResult:
    units: tuple[CodeUnit, ...]
    tests: tuple[CodeUnit, ...]
    rationale: dict[str, str]      # path -> why it was included, for the trace
    tokens_used: int


class RepositoryLocaliser:
    def __init__(self, lexical_index, symbol_graph, test_map, vcs_history,
                 count_tokens) -> None:
        self._lexical = lexical_index
        self._graph = symbol_graph
        self._tests = test_map
        self._history = vcs_history
        self._count = count_tokens

    def localise(self, issue_text: str, budget_tokens: int = 40_000
                 ) -> LocalisationResult:
        scores: dict[str, float] = defaultdict(float)
        rationale: dict[str, str] = {}

        # Signal 1: exact identifier matches. Code names are precise, so a
        # lexical hit is a much stronger relevance signal here than in prose.
        for unit, score in self._lexical.search_identifiers(issue_text, k=30):
            scores[unit.symbol] += 3.0 * score
            rationale.setdefault(unit.path, "identifier mentioned in the issue")

        # Signal 2: symbol graph expansion, one hop in each direction.
        seeds = list(scores.keys())
        for symbol in seeds:
            for caller in self._graph.callers(symbol, limit=8):
                scores[caller] += 1.4
                rationale.setdefault(self._graph.path_of(caller), "calls a located symbol")
            for callee in self._graph.callees(symbol, limit=8):
                scores[callee] += 1.0
                rationale.setdefault(self._graph.path_of(callee), "called by a located symbol")
            for dependency in self._graph.type_dependencies(symbol, limit=6):
                scores[dependency] += 0.8
                rationale.setdefault(self._graph.path_of(dependency), "type dependency")

        # Signal 3: co-change prior from version control.
        for symbol in seeds:
            for path, strength in self._history.co_changed(self._graph.path_of(symbol), top=5):
                for unit_symbol in self._graph.symbols_in(path):
                    scores[unit_symbol] += 0.5 * strength
                rationale.setdefault(path, "historically changes with a located file")

        ranked = sorted(scores.items(), key=lambda pair: pair[1], reverse=True)
        units, used = [], 0
        for symbol, _ in ranked:
            unit = self._graph.unit(symbol)
            if used + unit.tokens > budget_tokens * 0.7:   # reserve for tests
                continue
            units.append(unit)
            used += unit.tokens

        # Signal 4: tests for the selected symbols. These are the specification
        # and the oracle at once, so they are reserved budget rather than
        # competing for it.
        tests, test_used = [], 0
        for unit in units:
            for test in self._tests.tests_for(unit.symbol):
                if test in tests:
                    continue
                if used + test_used + test.tokens > budget_tokens:
                    continue
                tests.append(test)
                test_used += test.tokens
                rationale.setdefault(test.path, f"exercises {unit.symbol}")

        return LocalisationResult(tuple(units), tuple(tests), rationale,
                                  used + test_used)

"""Integrity guard for coding agent patches.

Every check here exists because an agent found the corresponding shortcut.
The guard runs after the ladder passes, and a violation fails the run rather
than merely warning, because a warning on a green build is not read.
"""


import ast
import re
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class IntegrityVerdict:
    ok: bool
    violations: tuple[str, ...]


SKIP_MARKERS = re.compile(
    r"@(pytest\.mark\.(skip|xfail)|unittest\.skip)", re.I
)


class PatchIntegrityGuard:
    def __init__(self, test_path_patterns: tuple[str, ...] = ("test_", "_test.", "/tests/")
                 ) -> None:
        self._test_patterns = test_path_patterns

    def check(self, patch, baseline_tests: dict[str, str],
              allow_test_changes: bool = False) -> IntegrityVerdict:
        violations: list[str] = []

        for path, new_text in patch.changed_files.items():
            if not self._is_test(path):
                violations.extend(self._check_source(path, new_text))
                continue

            if not allow_test_changes:
                # The default is that an agent fixing a defect does not touch
                # tests. Adding a NEW test is permitted; editing or deleting an
                # existing one is not, unless the task explicitly authorised it.
                old = baseline_tests.get(path)
                if old is None:
                    continue                      # brand new test file, fine
                removed = self._removed_assertions(old, new_text)
                if removed:
                    violations.append(
                        f"{path}: removed or altered {len(removed)} existing "
                        f"assertions: {', '.join(sorted(removed)[:3])}"
                    )
                if SKIP_MARKERS.search(new_text) and not SKIP_MARKERS.search(old):
                    violations.append(f"{path}: added a skip or xfail marker")
                if self._test_count(new_text) < self._test_count(old):
                    violations.append(f"{path}: test count decreased")

        return IntegrityVerdict(not violations, tuple(violations))

    def _is_test(self, path: str) -> bool:
        return any(pattern in path for pattern in self._test_patterns)

    @staticmethod
    def _check_source(path: str, text: str) -> list[str]:
        """Detect special-casing: a branch whose condition compares against a
        literal that looks like a fixture value, returning a constant."""
        problems: list[str] = []
        try:
            tree = ast.parse(text)
        except SyntaxError:
            return [f"{path}: does not parse"]

        for node in ast.walk(tree):
            if not isinstance(node, ast.If):
                continue
            compares_literal = any(
                isinstance(comparator, ast.Constant)
                for test in ast.walk(node.test)
                if isinstance(test, ast.Compare)
                for comparator in test.comparators
            )
            body_returns_constant = (
                len(node.body) == 1
                and isinstance(node.body[0], ast.Return)
                and isinstance(node.body[0].value, ast.Constant)
            )
            if compares_literal and body_returns_constant:
                problems.append(
                    f"{path}:{node.lineno}: literal comparison returning a "
                    f"constant, likely special-casing a test input"
                )

        # A bare except that swallows everything frequently appears when an
        # agent is making a failure disappear rather than fixing it.
        for node in ast.walk(tree):
            if isinstance(node, ast.ExceptHandler) and node.type is None:
                problems.append(f"{path}:{node.lineno}: bare except added")
        return problems

    @staticmethod
    def _removed_assertions(old: str, new: str) -> set[str]:
        def assertions(text: str) -> set[str]:
            return {
                line.strip() for line in text.splitlines()
                if line.strip().startswith(("assert ", "self.assert"))
            }
        return assertions(old) - assertions(new)

    @staticmethod
    def _test_count(text: str) -> int:
        return len(re.findall(r"^\s*def\s+test_\w+", text, re.M))
