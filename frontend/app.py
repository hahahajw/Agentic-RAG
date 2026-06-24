"""Main entry point for the Agentic RAG frontend."""

import os
import sys
from pathlib import Path

# ═══════════════════════════════════════════════════════════════════
# 加载 .env 文件到 os.environ（若无 .env 则静默跳过）
# ═══════════════════════════════════════════════════════════════════
_env_path = Path(__file__).parent.parent / ".env"
if _env_path.exists():
    with open(_env_path, encoding="utf-8") as _f:
        for _line in _f:
            _line = _line.strip()
            if not _line or _line.startswith("#") or "=" not in _line:
                continue
            _key, _sep, _val = _line.partition("=")
            _key = _key.strip()
            _val = _val.strip().strip('"').strip("'")
            if _key and _key not in os.environ:
                os.environ[_key] = _val

# Ensure project root is on sys.path so frontend package imports work
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st

from frontend.styles import inject_custom_css

st.set_page_config(
    page_title="RAG 系统 — 多跳问答",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_custom_css()

# Page navigation
FRONTEND_DIR = Path(__file__).parent

pg = st.navigation(
    [
        st.Page(str(FRONTEND_DIR / "pages" / "1_在线问答.py"), title="在线问答", icon="💬"),
        st.Page(str(FRONTEND_DIR / "pages" / "2_实验结果.py"), title="实验结果", icon="🔎"),
    ]
)

# Sidebar footer — runs after navigation renders sidebar
with st.sidebar:
    st.divider()
    st.caption("多跳问答系统 — 模块化 RAG | 递归检索 RAG | 规划-执行-反馈闭环 RAG")

pg.run()
