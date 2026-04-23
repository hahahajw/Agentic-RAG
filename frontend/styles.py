"""CSS injection for the Agentic RAG frontend."""

import streamlit as st

DEPTH_COLORS = {
    0: "#2563EB",  # Deep blue (root)
    1: "#0891B2",  # Cyan
    2: "#059669",  # Emerald
    3: "#D97706",  # Amber
    4: "#DC2626",  # Red
    5: "#9333EA",  # Purple
}


def inject_custom_css():
    """Inject custom CSS for tree visualization and card styling."""
    st.markdown("""
<style>
    /* ===== Tree Node Containers ===== */
    .tree-node {
        border-left: 4px solid var(--depth-color);
        padding: 12px 16px;
        margin: 8px 0 8px calc(var(--depth) * 24px);
        background: #f8f9fa;
        border-radius: 0 8px 8px 0;
        transition: box-shadow 0.2s;
    }
    .tree-node:hover {
        box-shadow: 0 2px 12px rgba(0,0,0,0.08);
    }

    /* ===== Depth Badges ===== */
    .depth-badge {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 24px;
        height: 24px;
        border-radius: 12px;
        background: var(--depth-color);
        color: white;
        font-size: 12px;
        font-weight: bold;
        margin-right: 8px;
        flex-shrink: 0;
    }

    /* ===== Node Header ===== */
    .node-header {
        display: flex;
        align-items: flex-start;
        gap: 8px;
        flex-wrap: wrap;
    }
    .question-text {
        font-weight: 600;
        color: #1a1a2e;
        flex: 1;
        min-width: 200px;
    }

    /* ===== Judge Verdict Badges ===== */
    .judge-badge {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 12px;
        font-size: 11px;
        font-weight: 600;
        white-space: nowrap;
    }
    .judge-badge.answerable {
        background: #10B981;
        color: white;
    }
    .judge-badge.unanswerable {
        background: #F59E0B;
        color: white;
    }

    /* ===== Children Container (tree nesting) ===== */
    .children-container {
        position: relative;
        margin-top: 4px;
    }
    .children-container::before {
        content: '';
        position: absolute;
        left: calc(var(--depth) * 24px + 10px);
        top: 0;
        bottom: 0;
        width: 2px;
        background: var(--depth-color);
        opacity: 0.25;
    }

    /* ===== Chunk Cards ===== */
    .chunk-card {
        border: 1px solid #e5e7eb;
        border-radius: 6px;
        padding: 10px 14px;
        margin: 6px 0;
        background: white;
        border-left: 3px solid #e5e7eb;
    }
    .chunk-card:hover {
        border-left-color: var(--depth-color, #2563EB);
        background: #f9fafb;
    }
    .chunk-rank {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 20px;
        height: 20px;
        border-radius: 4px;
        background: #e5e7eb;
        color: #374151;
        font-size: 11px;
        font-weight: 600;
        margin-right: 6px;
    }
    .chunk-score {
        font-family: 'SF Mono', 'Fira Code', monospace;
        font-size: 12px;
        color: #6b7280;
        margin-left: 8px;
    }

    /* ===== Score Bar ===== */
    .score-bar-bg {
        width: 100%;
        height: 4px;
        background: #e5e7eb;
        border-radius: 2px;
        margin: 4px 0;
    }
    .score-bar-fill {
        height: 100%;
        border-radius: 2px;
        background: var(--depth-color, #2563EB);
        transition: width 0.3s ease;
    }

    /* ===== Correctness Badges ===== */
    .correct-badge {
        background: #10B981;
        color: white;
        padding: 2px 8px;
        border-radius: 10px;
        font-size: 11px;
        font-weight: 600;
    }
    .incorrect-badge {
        background: #EF4444;
        color: white;
        padding: 2px 8px;
        border-radius: 10px;
        font-size: 11px;
        font-weight: 600;
    }

    /* ===== Metric Cards ===== */
    [data-testid="stMetricValue"] {
        font-size: 1.5rem !important;
    }

    /* ===== Override Streamlit Expander ===== */
    .streamlit-expander-header {
        font-weight: 500;
    }

    /* ===== Tree Node Streamlit Overrides ===== */
    div[data-testid="stVerticalBlock"] > div:has(> .tree-node) {
        padding: 0;
    }

    /* ===== Round Cards (Agentic RAG timeline) ===== */
    .round-card {
        border-left: 4px solid var(--round-color, #6b7280);
        padding: 14px 18px;
        margin: 10px 0;
        background: #f8f9fa;
        border-radius: 0 8px 8px 0;
        transition: box-shadow 0.2s;
    }
    .round-card:hover {
        box-shadow: 0 2px 12px rgba(0,0,0,0.08);
    }
    .round-card.status-answered {
        border-left-color: #10B981;
    }
    .round-card.status-progressing {
        border-left-color: #F59E0B;
    }
    .round-card.status-stuck {
        border-left-color: #EF4444;
    }

    /* ===== Round Number Badge ===== */
    .round-badge {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        min-width: 28px;
        height: 28px;
        border-radius: 14px;
        background: #374151;
        color: white;
        font-size: 13px;
        font-weight: 700;
        padding: 0 8px;
        margin-right: 8px;
        flex-shrink: 0;
    }

    /* ===== Status Badges (Agentic RAG) ===== */
    .status-badge {
        display: inline-block;
        padding: 3px 12px;
        border-radius: 12px;
        font-size: 12px;
        font-weight: 600;
        white-space: nowrap;
        text-transform: uppercase;
        letter-spacing: 0.3px;
    }
    .status-badge.answered {
        background: #10B981;
        color: white;
    }
    .status-badge.progressing {
        background: #F59E0B;
        color: white;
    }
    .status-badge.stuck {
        background: #EF4444;
        color: white;
    }

    /* ===== Target Tags ===== */
    .target-tag {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 6px;
        font-size: 11px;
        font-weight: 500;
        background: #EEF2FF;
        color: #4338CA;
        border: 1px solid #C7D2FE;
        margin: 2px 4px 2px 0;
        font-family: 'SF Mono', 'Fira Code', monospace;
    }

    /* ===== Timeline Connector ===== */
    .timeline-connector {
        position: relative;
        margin-left: 14px;
        padding-left: 0;
        height: 20px;
    }
    .timeline-connector::before {
        content: '';
        position: absolute;
        left: 14px;
        top: -10px;
        bottom: -10px;
        width: 2px;
        background: #d1d5db;
        opacity: 0.5;
    }

    /* ===== Evaluation Section ===== */
    .eval-section {
        margin-top: 8px;
        padding-top: 8px;
        border-top: 1px solid #e5e7eb;
    }
    .gap-tag {
        display: inline-block;
        padding: 1px 8px;
        border-radius: 4px;
        font-size: 11px;
        background: #FEF3C7;
        color: #92400E;
        margin: 1px 4px 1px 0;
    }

    /* ===== Search Tree SVG ===== */
    .tree-svg-container {
        padding: 16px 0;
        overflow-x: auto;
    }

    /* ===== Tree Node Buttons ===== */
    .tree-node-btn-row {
        display: flex;
        flex-wrap: wrap;
        gap: 6px;
        margin: 8px 0;
    }

    /* ===== Tree Detail Section ===== */
    .tree-detail-section {
        margin-top: 12px;
        padding-top: 12px;
        border-top: 1px solid #e5e7eb;
    }

    /* ===== Round Header & Targets ===== */
    .round-header {
        display: flex;
        align-items: center;
        gap: 8px;
        flex-wrap: wrap;
        margin-bottom: 8px;
    }
    .round-targets {
        display: flex;
        flex-wrap: wrap;
        gap: 4px;
        margin: 6px 0;
    }
</style>
    """, unsafe_allow_html=True)


def depth_color_style(depth: int) -> str:
    """Return CSS custom properties string for a given depth level."""
    color = DEPTH_COLORS.get(depth % len(DEPTH_COLORS), DEPTH_COLORS[0])
    return f'--depth: {depth}; --depth-color: {color};'
