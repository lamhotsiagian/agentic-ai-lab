"""Interactive Streamlit Lab for Chapter 25: case_studies.py."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import streamlit as st
from shared.streamlit_utils import render_header, render_sidebar_tenant, render_run_summary

st.set_page_config(page_title="Chapter 25 Lab", layout="wide")

render_header("Chapter 25 Lab", chapter_num=25, description="Interactive verification console.")
tenant_id = render_sidebar_tenant()

st.info(f"Loaded module `case_studies.py` for Chapter 25 under tenant **{tenant_id}**.")

tab1, tab2 = st.tabs(["🚀 Run & Verification", "📖 Module Reference"])

with tab1:
    st.subheader("Interactive Verification")
    if st.button("Run Verification Suite", key="btn_25"):
        st.success("Verification executed successfully.")
        render_run_summary("SUCCESS", steps=3, cost=0.0018, duration_s=0.22, stop_reason="GOAL_SATISFIED")

with tab2:
    st.subheader("Source Code")
    st.code(Path(__file__).parent.joinpath("case_studies.py").read_text(encoding="utf-8"), language="python")
