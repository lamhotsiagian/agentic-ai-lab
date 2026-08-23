"""Shared configuration for agentic-ai-lab."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Settings:
    # Model defaults
    ollama_host: str = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    default_model: str = os.getenv("DEFAULT_MODEL", "qwen2.5:3b")
    small_model: str = os.getenv("SMALL_MODEL", "qwen2.5:0.5b")
    reasoning_model: str = os.getenv("REASONING_MODEL", "deepseek-r1:8b")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")

    # Database defaults
    postgres_host: str = os.getenv("POSTGRES_HOST", "localhost")
    postgres_port: int = int(os.getenv("POSTGRES_PORT", "5432"))
    postgres_db: str = os.getenv("POSTGRES_DB", "agentic_lab")
    postgres_user: str = os.getenv("POSTGRES_USER", "postgres")
    postgres_password: str = os.getenv("POSTGRES_PASSWORD", "postgres")

    # Telemetry
    otlp_endpoint: str = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317")
    environment: str = os.getenv("ENVIRONMENT", "development")

    # Multi-tenancy
    default_tenant: str = "default"


settings = Settings()
