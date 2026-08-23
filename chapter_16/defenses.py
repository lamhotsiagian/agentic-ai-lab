from __future__ import annotations

"""Egress control for agent outputs and tool arguments.

The threat: an injection instructs the agent to append retrieved data to a
URL the attacker controls, or to embed it in an image reference that the
rendering client will fetch. Both are ordinary-looking outputs.
"""


import base64
import re
from dataclasses import dataclass
from urllib.parse import urlparse, parse_qs


@dataclass(frozen=True, slots=True)
class EgressVerdict:
    allowed: bool
    reason: str
    redacted_output: str | None = None


URL_PATTERN = re.compile(r"https?://[^\s<>\"')\]]+", re.I)
MARKDOWN_IMAGE = re.compile(r"!\[[^\]]*\]\((?P<url>[^)]+)\)")
HTML_FETCHER = re.compile(r"<\s*(img|iframe|script|link|object)\b[^>]*>", re.I)


class EgressFirewall:
    """Applied to every outbound tool argument and to every rendered answer."""

    def __init__(self, allowed_hosts: frozenset[str], max_query_bytes: int = 128,
                 audit=None) -> None:
        self._allowed = allowed_hosts
        self._max_query_bytes = max_query_bytes
        self._audit = audit

    def check_outbound(self, url: str, principal: str) -> EgressVerdict:
        parsed = urlparse(url)
        host = (parsed.hostname or "").lower()

        if parsed.scheme not in {"https"}:
            return self._deny("non_https_scheme", principal, url)
        if host not in self._allowed:
            # Allowlist, never blocklist. An attacker owns infinite domains and
            # you cannot enumerate them.
            return self._deny("host_not_allowlisted", principal, url)

        # A permitted host can still be an exfiltration channel if the agent
        # packs data into the query string or the path.
        query_bytes = len(parsed.query.encode("utf-8"))
        if query_bytes > self._max_query_bytes:
            return self._deny("oversized_query_string", principal, url)
        for values in parse_qs(parsed.query).values():
            for value in values:
                if self._looks_encoded(value):
                    return self._deny("encoded_payload_in_query", principal, url)
        if len(parsed.path) > 256:
            return self._deny("oversized_path", principal, url)

        return EgressVerdict(True, "allowed")

    def check_rendered_answer(self, answer: str, principal: str) -> EgressVerdict:
        """Markdown images and HTML tags cause the CLIENT to make a request,
        which bypasses every server-side control. Strip them unless the target
        is allowlisted."""
        redacted = answer
        for match in MARKDOWN_IMAGE.finditer(answer):
            verdict = self.check_outbound(match.group("url"), principal)
            if not verdict.allowed:
                redacted = redacted.replace(match.group(0), "[image removed by policy]")

        if HTML_FETCHER.search(redacted):
            redacted = HTML_FETCHER.sub("[markup removed by policy]", redacted)

        for url in URL_PATTERN.findall(redacted):
            verdict = self.check_outbound(url, principal)
            if not verdict.allowed:
                redacted = redacted.replace(url, "[link removed by policy]")

        changed = redacted != answer
        if changed and self._audit:
            self._audit.security_event("egress.answer_redacted", principal=principal)
        return EgressVerdict(True, "redacted" if changed else "clean", redacted)

    @staticmethod
    def _looks_encoded(value: str) -> bool:
        if len(value) < 40:
            return False
        candidate = value.strip("=")
        if re.fullmatch(r"[A-Za-z0-9+/_-]+", candidate) is None:
            return False
        try:
            decoded = base64.urlsafe_b64decode(value + "==")
        except Exception:                    # noqa: BLE001
            return False
        # Decodes to mostly printable text: almost certainly packed content.
        printable = sum(1 for b in decoded if 32 <= b < 127)
        return printable / max(len(decoded), 1) > 0.85

    def _deny(self, reason: str, principal: str, url: str) -> EgressVerdict:
        if self._audit:
            self._audit.security_event("egress.denied", reason=reason,
                                       principal=principal, url=url[:200])
        return EgressVerdict(False, reason)

"""Guarded execution.

Content provenance is the piece most implementations omit. Once you tag
where each piece of context came from, you can enforce rules such as: a run
whose context contains untrusted content may not invoke a tool in the
external-communication class.
"""


from dataclasses import dataclass
from enum import Enum


class Provenance(str, Enum):
    SYSTEM = "system"            # written by us, fully trusted
    USER = "user"                # the authenticated end user's own words
    INTERNAL = "internal"        # our own systems, trusted content
    UNTRUSTED = "untrusted"      # web, uploaded docs, email, peer artifacts


class ToolClass(str, Enum):
    READ_PRIVATE = "read_private"
    READ_PUBLIC = "read_public"
    WRITE_INTERNAL = "write_internal"
    EXTERNAL_COMM = "external_comm"     # the trifecta's third leg
    CODE_EXEC = "code_exec"


@dataclass(frozen=True, slots=True)
class GuardDecision:
    allowed: bool
    reason: str
    requires_approval: bool = False


class TrifectaGuard:
    """Refuses capability combinations rather than trusting the model."""

    def evaluate(
        self,
        tool_class: ToolClass,
        context_provenance: set[Provenance],
        private_data_read: bool,
        principal_scopes: frozenset[str],
        required_scope: str,
    ) -> GuardDecision:
        # Layer: identity. Checked first, and never derived from anything the
        # model or a peer asserted.
        if required_scope not in principal_scopes:
            return GuardDecision(False, "principal_lacks_scope")

        untrusted_present = Provenance.UNTRUSTED in context_provenance

        # Layer: trifecta. All three legs present means refuse, regardless of
        # how reasonable the individual request looks.
        if (
            tool_class is ToolClass.EXTERNAL_COMM
            and untrusted_present
            and private_data_read
        ):
            return GuardDecision(
                False,
                "trifecta: untrusted content plus private data plus external "
                "communication in one run",
            )

        # Code execution never runs in a context that has seen untrusted
        # content, because the sandbox bounds damage but does not prevent the
        # attacker from choosing what runs inside it.
        if tool_class is ToolClass.CODE_EXEC and untrusted_present:
            return GuardDecision(False, "code_exec_with_untrusted_context")

        # Layer: approval. Irreversible classes always route to a human.
        if tool_class in {ToolClass.EXTERNAL_COMM, ToolClass.WRITE_INTERNAL}:
            return GuardDecision(True, "allowed_with_approval", requires_approval=True)

        return GuardDecision(True, "allowed")
