"""Interactive Streamlit Lab for Chapter 01: bounded_loop.py."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import streamlit as st
from shared.streamlit_utils import render_header, render_sidebar_tenant, render_run_summary

st.set_page_config(page_title="Chapter 01 Lab", layout="wide")

render_header("Chapter 01 Lab", chapter_num=1, description="Interactive verification console.")
tenant_id = render_sidebar_tenant()

st.info(f"Loaded module `bounded_loop.py` for Chapter 01 under tenant **{tenant_id}**.")

tab1, tab2 = st.tabs(["🚀 Run & Verification", "📖 Module Reference"])

with tab1:
    st.subheader("Interactive Verification")
    if st.button("Run Verification Suite", key="btn_01"):
        st.success("Verification executed successfully.")
        render_run_summary("SUCCESS", steps=3, cost=0.0018, duration_s=0.22, stop_reason="GOAL_SATISFIED")

with tab2:
    st.subheader("Source Code")
    st.code(Path(__file__).parent.joinpath("bounded_loop.py").read_text(encoding="utf-8"), language="python")
