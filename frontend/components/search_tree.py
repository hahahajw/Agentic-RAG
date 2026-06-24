"""搜索树可视化组件——用于 RAG with Judge 的递归探索树。

提供两种视图:
  1. render_search_tree() — Streamlit expander 递归嵌套 (适合逐节点细读)
  2. render_search_tree_graph() — 交互式 SVG 图 (可点击节点查看详情, 适合总览)

Usage:
    from frontend.components.search_tree import render_search_tree, render_search_tree_graph
    render_search_tree(search_path, max_depth=3)
    clicked = render_search_tree_graph(search_path)
"""

import streamlit as st

from frontend.components.graph_viewer import render_search_tree as _graph_render
from frontend.components.graph_viewer import render_tree_node_detail
from frontend.components.graph_viewer import _tree_to_graph


def render_search_tree(search_path: dict, max_depth: int = 3) -> None:
    """渲染 RAG with Judge 的递归探索树。

    Args:
        search_path: SEARCH_PATH dict（树根节点）
        max_depth: 最大递归深度
    """
    if not search_path or not isinstance(search_path, dict):
        st.warning("无搜索树数据")
        return

    if "question" not in search_path:
        st.warning("搜索树数据格式异常——缺少根节点 question")
        return

    st.markdown("""
    <div style="background:#f8f9fa;border-radius:12px;padding:16px;margin:8px 0;">
        <h3 style="margin:0 0 4px 0;">递归探索树 (SEARCH_PATH)</h3>
        <p style="color:#666;font-size:13px;margin:0;">
            绿色 = Judge 认为知识足够 | 橙色 = 需要更多知识 | 灰色 = 达到最大深度
        </p>
    </div>
    """, unsafe_allow_html=True)

    _render_node(search_path, depth=0, max_depth=max_depth)


def _render_node(node: dict, depth: int, max_depth: int) -> None:
    """递归渲染单个搜索树节点。"""
    question = node.get("question", "Unknown")
    answerable = node.get("answerable", False)
    chunks = node.get("chunks", []) if isinstance(node.get("chunks"), list) else []
    next_queries = node.get("next_queries", [])
    answer = node.get("answer", "")
    reason = node.get("judgement_reason", "")
    children = [c for c in next_queries if isinstance(c, dict)]

    # 状态
    if answerable:
        badge, color = "可回答", "#22c55e"
    elif depth >= max_depth:
        badge, color = "达到最大深度", "#94a3b8"
    else:
        badge, color = "需要更多知识", "#f59e0b"

    indent = "&nbsp;&nbsp;" * depth
    arrow = "└─" if depth > 0 else ""

    with st.container(border=True):
        col1, col2 = st.columns([4, 1])
        with col1:
            st.markdown(f"{indent}{arrow} **Q:** {question}", unsafe_allow_html=True)
            st.caption(
                f"状态: {badge} | Depth {depth}/{max_depth} | "
                f"Chunks: {len(chunks)} | Follow-ups: {len(children)}"
            )
            if reason:
                st.caption(f"Judge 理由: {reason[:150]}{'…' if len(reason) > 150 else ''}")
            if answer:
                st.caption(f"中间答案: {answer[:120]}{'…' if len(answer) > 120 else ''}")
        with col2:
            if chunks:
                with st.expander(f"{len(chunks)} 来源", expanded=False):
                    for i, c in enumerate(chunks, 1):
                        title = c.get("chunk_title", "Unknown")
                        content = c.get("page_content", "")[:200]
                        st.markdown(f"**{i}. {title}**\n\n{content}")

    for child in children:
        _render_node(child, depth=depth + 1, max_depth=max_depth)


def render_search_tree_graph(search_path: dict, selected: str = "",
                            key_suffix: str = "") -> str | None:
    """渲染交互式搜索树图 (可点击节点)。

    Args:
        search_path: SEARCH_PATH dict
        selected: 当前选中的节点 ID (t0, t1, ...)
        key_suffix: 同页多个图时用于区分 widget key

    Returns:
        被点击的节点 ID, 或 None。
        调用方应使用 _tree_to_graph() 查找对应节点数据并调用 render_node_detail()。
    """
    return _graph_render(search_path, selected=selected, key_suffix=key_suffix)


# ── 旧 Mermaid 渲染已移除, 由交互式 SVG 图替代 ──