"""agentops/api/authz.py

Three refusals happen before any work is admitted, and each of them has a
corresponding incident in this book's case studies.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date


@dataclass(frozen=True, slots=True)
class Principal:
    subject: str
    tenant_id: str
    scopes: frozenset[str]
    email: str | None


class AuthorisationError(Exception):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


class RequestAuthoriser:
    def __init__(self, verifier, registry, manifest) -> None:
        self._verifier = verifier          # OIDC token verification
        self._registry = registry          # agent authority records
        self._manifest = manifest          # prompt, model, schema, policy versions

    def authorise(self, bearer: str, agent_id: str, action_class: str,
                  today: date | None = None) -> tuple[Principal, bool]:
        # 1. Identity. An unverified or expired token never reaches the run.
        claims = self._verifier.verify(bearer)
        principal = Principal(
            subject=claims["sub"],
            tenant_id=claims["tenant"],
            scopes=frozenset(claims.get("scopes", [])),
            email=claims.get("email"),
        )

        # 2. Agent authority, read from the registry at call time rather than
        # documented in a wiki. An agent whose authority review date has passed
        # cannot act, which is what stops permission sprawl outliving teams.
        record = self._registry.get(agent_id)
        if record is None:
            raise AuthorisationError("unknown_agent", f"no registry entry for {agent_id}")
        allowed, requires_approval, reason = record.may_call(
            tool="", action_class=action_class, today=today
        )
        if not allowed:
            raise AuthorisationError("agent_not_authorised", reason)

        # 3. Version manifest. A worker or an agent definition holding a
        # different prompt, model, schema, or policy version than the manifest
        # expects refuses rather than serving with a mismatch. Partial
        # deployment is a failure mode you design for, not one you hope against.
        skew = self._manifest.skew(agent_id)
        if skew:
            raise AuthorisationError(
                "version_skew",
                f"expected {skew.expected}, holding {skew.actual}",
            )

        return principal, requires_approval
