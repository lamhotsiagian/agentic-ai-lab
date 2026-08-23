"""Unit tests for shared library modules."""

import pytest
from shared.config import settings
from shared.models import LocalModelClient
from shared.telemetry import Tracer
from shared.db_utils import InMemoryVectorStore, DocumentChunk


def test_settings():
    assert settings.default_model is not None
    assert settings.default_tenant == "default"


def test_models_fallback():
    client = LocalModelClient()
    response = client.generate("Plan this task")
    assert response is not None
    assert len(response) > 0

    embedding = client.embed("Sample text query")
    assert len(embedding) == 384


def test_telemetry_tracer():
    tracer = Tracer(service_name="test-service")
    span = tracer.start_span("agent_step")
    span.set_attribute("step.index", 1)
    span.add_event("tool_invoked", {"tool": "search"})
    span.end()

    assert span.duration_ms >= 0.0
    assert len(tracer.spans) == 1
    assert tracer.spans[0].attributes["step.index"] == 1


def test_vector_store_multi_tenancy():
    store = InMemoryVectorStore()
    c1 = DocumentChunk(
        chunk_id="c1",
        tenant_id="acme",
        content="Confidential Acme Doc",
        embedding=[0.5, 0.5] + [0.0] * 382,
    )
    c2 = DocumentChunk(
        chunk_id="c2",
        tenant_id="globex",
        content="Globex Internal Doc",
        embedding=[0.5, 0.5] + [0.0] * 382,
    )

    store.upsert(c1)
    store.upsert(c2)

    # Search under acme
    results_acme = store.search("acme", [0.5, 0.5] + [0.0] * 382, top_k=5)
    assert len(results_acme) == 1
    assert results_acme[0][0].chunk_id == "c1"

    # Search under globex
    results_globex = store.search("globex", [0.5, 0.5] + [0.0] * 382, top_k=5)
    assert len(results_globex) == 1
    assert results_globex[0][0].chunk_id == "c2"

    # Search under empty tenant
    results_other = store.search("other", [0.5, 0.5] + [0.0] * 382, top_k=5)
    assert len(results_other) == 0
