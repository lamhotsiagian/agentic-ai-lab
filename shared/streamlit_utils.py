"""Shared UI components for Streamlit chapter labs."""

from __future__ import annotations

import streamlit as st
from shared.config import settings


def render_header(
    title: str,
    chapter_num: int | None = None,
    description: str | None = None,
) -> None:
    """Render a standard header with chapter metadata."""
    badge = f"Chapter {chapter_num}" if chapter_num else "Agentic AI Lab"
    st.markdown(
        f"""
        <div style="padding: 0.8rem 0; border-bottom: 1px solid #e0e0e0; margin-bottom: 1.5rem;">
            <span style="background-color: #1e3a8a; color: white; padding: 3px 8px; border-radius: 4px; font-size: 0.75rem; font-weight: 600; text-transform: uppercase;">{badge}</span>
            <h1 style="margin: 0.5rem 0 0.2rem 0; font-size: 1.8rem;">{title}</h1>
            <p style="color: #64748b; font-size: 0.95rem; margin: 0;">{description or ''}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar_tenant(
    key: str = "tenant_id",
    options: tuple[str, ...] = ("default", "acme-corp", "globex", "initech"),
) -> str:
    """Render the standard multi-tenant selector in the sidebar."""
    st.sidebar.markdown("### 🏢 Multi-Tenancy")
    selected = st.sidebar.selectbox("Active Tenant", options=options, index=0, key=key)
    st.sidebar.info(f"Requests strictly isolated to **{selected}** partition.")
    return selected


def render_run_summary(
    status: str,
    steps: int,
    cost: float,
    duration_s: float,
    stop_reason: str | None = None,
) -> None:
    """Render metric cards for run execution."""
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Run Status", status)
    with c2:
        st.metric("Steps Executed", f"{steps}")
    with c3:
        st.metric("Total Cost", f"${cost:.4f}")
    with c4:
        st.metric("Latency", f"{duration_s:.2f}s")

    if stop_reason:
        st.caption(f"Stop Reason: `{stop_reason}`")
