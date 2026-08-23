"""Local model client with Ollama support and deterministic offline fallback."""

from __future__ import annotations

import json
import urllib.request
import urllib.error
from typing import Any, Sequence
from shared.config import settings


class LocalModelClient:
    """Client for local LLMs via Ollama, falling back to deterministic mock outputs."""

    def __init__(self, host: str | None = None, model: str | None = None):
        self.host = (host or settings.ollama_host).rstrip("/")
        self.model = model or settings.default_model

    def is_available(self) -> bool:
        """Check if Ollama service is reachable."""
        try:
            req = urllib.request.Request(f"{self.host}/api/tags", method="GET")
            with urllib.request.urlopen(req, timeout=1.0) as resp:
                return resp.status == 200
        except Exception:
            return False

    def generate(
        self,
        prompt: str,
        system: str | None = None,
        temperature: float = 0.0,
        stop: Sequence[str] | None = None,
    ) -> str:
        """Generate a completion from the model."""
        if self.is_available():
            try:
                payload = {
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": temperature},
                }
                if system:
                    payload["system"] = system
                if stop:
                    payload["options"]["stop"] = list(stop)

                data = json.dumps(payload).encode("utf-8")
                req = urllib.request.Request(
                    f"{self.host}/api/generate",
                    data=data,
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=30.0) as resp:
                    res = json.loads(resp.read().decode("utf-8"))
                    return res.get("response", "")
            except Exception:
                pass  # Fall back to deterministic mock

        # Deterministic offline mock response generator
        prompt_lower = prompt.lower()
        if "plan" in prompt_lower:
            return "1. Search documentation\n2. Extract relevant metrics\n3. Synthesise conclusion"
        if "classify" in prompt_lower or "router" in prompt_lower:
            return "GENERAL_QUERY"
        if "grade" in prompt_lower or "verify" in prompt_lower:
            return "SUPPORTED"
        if "sql" in prompt_lower:
            return "SELECT tenant_id, COUNT(*) FROM runs GROUP BY tenant_id;"
        return f"Deterministic response for: {prompt[:40]}..."

    def embed(self, text: str) -> list[float]:
        """Generate embeddings for text."""
        if self.is_available():
            try:
                payload = {"model": settings.embedding_model, "prompt": text}
                data = json.dumps(payload).encode("utf-8")
                req = urllib.request.Request(
                    f"{self.host}/api/embeddings",
                    data=data,
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=10.0) as resp:
                    res = json.loads(resp.read().decode("utf-8"))
                    return res.get("embedding", [0.0] * 384)
            except Exception:
                pass

        # Deterministic pseudo-embedding from text hash
        import hashlib
        h = hashlib.sha256(text.encode("utf-8")).digest()
        return [(b / 255.0) * 2 - 1 for b in h] + [0.0] * (384 - len(h))
