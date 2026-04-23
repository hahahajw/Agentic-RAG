"""Main entry point for the Agentic RAG frontend."""

import sys
from pathlib import Path

# Ensure project root is on sys.path so frontend package imports work
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st

from frontend.styles import inject_custom_css

st.set_page_config(
    page_title="Agentic RAG",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_custom_css()

# Page navigation
FRONTEND_DIR = Path(__file__).parent

pg = st.navigation(
    [
        st.Page(str(FRONTEND_DIR / "pages" / "1_query.py"), title="Live Query", icon="💬"),
        st.Page(str(FRONTEND_DIR / "pages" / "5_agentic_results.py"), title="Results", icon="🔎"),
        st.Page(str(FRONTEND_DIR / "pages" / "6_fingerprint_heatmap.py"), title="F1 Heatmap", icon="🟩"),
    ]
)

# Sidebar footer — runs after navigation renders sidebar
with st.sidebar:
    st.divider()
    st.caption("多跳问答系统可视化")

pg.run()
