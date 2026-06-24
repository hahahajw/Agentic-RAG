"""可复用的检索来源卡片组件。

所有算法使用统一格式的 chunk dict，包含:
  - chunk_id, chunk_title, chunk_summary, context_title, page_content
  - score (可选), source_url (可选)

Usage:
    from frontend.components.chunk_card import render_chunk_card, render_chunks_list
    render_chunk_card(chunk, rank=1)
    render_chunks_list(chunks, max_display=8)
"""

import streamlit as st

# 深度颜色（与 styles.py 保持一致）
DEPTH_COLORS = {
    0: "#2563EB", 1: "#0891B2", 2: "#059669",
    3: "#D97706", 4: "#DC2626", 5: "#9333EA",
}


def render_chunk_card(chunk: dict, rank: int, depth: int = 0) -> None:
    """渲染单个检索来源卡片。

    Args:
        chunk: chunk dict (含 chunk_title, page_content, score 等)
        rank: 排名序号
        depth: 搜索树深度（用于颜色区分）
    """
    score = chunk.get("score", 0)
    title = chunk.get("chunk_title", "Unknown")
    context = chunk.get("context_title", "")
    content = chunk.get("page_content", chunk.get("content", ""))
    summary = chunk.get("chunk_summary", "")
    source_url = chunk.get("source_url", "")
    color = DEPTH_COLORS.get(depth % len(DEPTH_COLORS), DEPTH_COLORS[0])

    header_parts = [f"#{rank}"]
    if score:
        header_parts.append(f"`{score:.4f}`")
    header_parts.append(f"**{title}**")
    if context and context != title:
        header_parts.append(f"— {context}")
    header = " ".join(header_parts)

    with st.expander(header, expanded=(rank <= 3)):
        if score:
            bar_pct = min(100, (score / 0.09) * 100)
            st.markdown(
                f'<div style="background:#e5e7eb;border-radius:4px;height:6px;margin:4px 0;">'
                f'<div style="background:{color};border-radius:4px;height:6px;width:{bar_pct:.0f}%;"></div>'
                f'</div>',
                unsafe_allow_html=True,
            )
        if summary:
            st.caption(f"摘要: {summary}")
        if source_url:
            st.caption(f"来源: [{source_url}]({source_url})")
        st.markdown(content[:500])


def render_chunks_list(chunks: list[dict], max_display: int = 8) -> None:
    """渲染检索来源列表（带摘要信息）。

    Args:
        chunks: chunk dict 列表
        max_display: 最多展示数量
    """
    if not chunks:
        st.caption("无检索来源")
        return

    with st.expander(f"检索到 {len(chunks)} 个来源", expanded=False):
        for i, chunk in enumerate(chunks[:max_display], 1):
            with st.container(border=True):
                st.write(f"**{i}. {chunk.get('chunk_title', 'Unknown')}**")
                source = chunk.get("source_url", "")
                if source:
                    st.caption(f"[{source}]({source})")
                st.write(chunk.get("page_content", "")[:500])