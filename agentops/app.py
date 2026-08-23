"""AgentOps Studio Capstone Application."""
import streamlit as st
from shared.streamlit_utils import render_header, render_sidebar_tenant, render_run_summary
render_header("AgentOps Studio", chapter_num=30, description="Capstone Multi-Agent Architecture")
tenant = render_sidebar_tenant()
st.info(f"Studio active for tenant: {tenant}")
