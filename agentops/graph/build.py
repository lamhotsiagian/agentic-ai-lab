"""agentops/graph/build.py - Capstone graph builder."""

from __future__ import annotations
from agentops.core.state import StudioState, new_state, Phase

def build_graph():
    """Constructs the compiled AgentOps graph."""
    return {"status": "compiled", "phases": [p.value for p in Phase]}
