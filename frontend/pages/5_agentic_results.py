"""Unified result browser for all RAG pipelines."""

import sys
from pathlib import Path
from typing import Optional

import pandas as pd

# Ensure project root is on sys.path so frontend package imports work
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import streamlit as st

from frontend.data_loader import list_available_datasets, load_results
from frontend.styles import DEPTH_COLORS

# System comparison imports
try:
    from analyse.system_comparison.viz.data_loader import (
        load_analysis_data, build_behavior_profiles, compute_behavior_patterns,
        MODES, DATASETS
    )
    from analyse.system_comparison.viz.styles import MODE_COLORS, MODE_LABELS, MODE_ORDER, DATASET_LABELS
    _SC_AVAILABLE = True
except ImportError:
    _SC_AVAILABLE = False
    MODES = []
    DATASETS = []
    MODE_ORDER = []
    MODE_COLORS = {}
    MODE_LABELS = {}
    DATASET_LABELS = {}


def _format_ms(ms: float) -> str:
    """Format milliseconds to human-readable string."""
    if ms < 1000:
        return f"{ms:.0f}ms"
    return f"{ms / 1000:.1f}s"


def _truncate(text: str, max_len: int = 300) -> str:
    """Truncate text with ellipsis."""
    if not text:
        return ""
    return text if len(text) <= max_len else text[:max_len] + "..."


def _chunk_card(chunk: dict, rank: int, depth: int = 0) -> None:
    """Render a single chunk as a styled card."""
    score = chunk.get("score", 0)
    title = chunk.get("chunk_title", "Unknown")
    context = chunk.get("context_title", "")
    content = chunk.get("content", chunk.get("page_content", ""))
    summary = chunk.get("chunk_summary", "")
    color = DEPTH_COLORS.get(depth % len(DEPTH_COLORS), DEPTH_COLORS[0])

    header = f"#{rank} `{score:.4f}` **{title}**"
    if context:
        header += f"  \u2014 {context}"

    with st.expander(header, expanded=(rank <= 3)):
        bar_pct = min(100, (score / 0.09) * 100)
        st.markdown(
            f'<div class="score-bar-bg"><div class="score-bar-fill" style="width:{bar_pct:.0f}%;background:{color}"></div></div>',
            unsafe_allow_html=True,
        )
        if summary:
            st.caption(f"**Summary:** {summary}")
        n_props = chunk.get("aggregated_propositions")
        if n_props:
            st.caption(f"**Propositions:** {n_props}")
        st.markdown(f"**Content:**\n{_truncate(content)}")


# ─── 角色配置 ────────────────────────────────────────────────────

ROLE_CONFIG = {
    "START": {"icon": "\u25cf", "label": "START", "color": "#6b7280"},
    "END": {"icon": "\u25cf", "label": "END", "color": "#6b7280"},
    "planner": {"icon": "\U0001f4cb", "label": "Planner", "color": "#2563EB"},
    "executor": {"icon": "\U0001f50d", "label": "Executor", "color": "#EA580C"},
    "synthesizer": {"icon": "\u2696\ufe0f", "label": "Synthesizer", "color": "#CA8A04"},
    "reflect": {"icon": "\U0001fa9e", "label": "Reflector", "color": "#9333EA"},
    "answer": {"icon": "\u2705", "label": "Answer", "color": "#059669"},
}

STATUS_LABELS = {"complete": "Answered", "incomplete": "Progressing", "stuck": "Stuck"}
STATUS_COLORS = {"complete": "#10B981", "incomplete": "#F59E0B", "stuck": "#EF4444"}
SQ_STATUS_ICONS = {"solved": "\u2713", "unsolved": "\u2192", "stuck": "\u2717"}
SQ_STATUS_COLORS = {"solved": "#10B981", "unsolved": "#F59E0B", "stuck": "#EF4444"}

# Pipeline config: mode string for data_loader, display name, icon
PIPELINE_CONFIG = {
    "naive-rag": {"label": "Naive RAG Results", "icon": "📊"},
    "rag-with-judge": {"label": "RAG with Judge Results", "icon": "⚖️"},
    "agentic-rag": {"label": "Agentic RAG Results", "icon": "🤖"},
    "system-comparison": {"label": "System Comparison", "icon": "📈"},
}

SC_FIGURES_DIR = Path(__file__).resolve().parent.parent.parent / "analyse" / "system_comparison" / "figures"

FIGURE_CATALOG = {
    "A1_em_comparison.png": "EM Comparison",
    "A2_radar_f1_precision_recall.png": "F1/Precision/Recall Radar",
    "A3_retrieval_quality.png": "Retrieval Quality",
    "A4_improvement_delta.png": "Improvement Delta",
    "A5_heatmap.png": "Metrics Heatmap",
    "B1_latency_comparison.png": "Latency Comparison",
    "B2_pareto.png": "Accuracy vs Latency (Pareto)",
    "B3_retrieval_resources.png": "Resource Consumption",
    "B4_latency_distribution.png": "Latency Distribution",
    "C1_error_types.png": "Error Types Distribution",
    "C2_correct_overlap.png": "Correct Answer Overlap",
    "C3_behavior_patterns.png": "Behavior Patterns",
    "C4_v3_only_deepdive.png": "V3-Only Correct Deep Dive",
    "D1_accuracy_by_type.png": "Accuracy by Question Type",
    "D2_type_distribution.png": "Question Type Distribution",
    "D3_improvement_by_type.png": "Improvement by Type",
    "E1_causal_ablation.png": "Causal Ablation",
    "E2_known_unknown.png": "Known-Unknown Analysis",
    "E3_dose_response.png": "Dose-Response Curve",
    "E4_faithfulness_cross.png": "Faithfulness Cross-Analysis",
    "F1_retrieval_precision_vs_f1.png": "Retrieval Precision vs F1",
    "F2_latency_vs_em.png": "Latency vs EM",
    "F3_chunk_count.png": "Chunk Count Distribution",
    "F4_search_depth.png": "Search Depth Distribution",
}


# ─── Trace 重建（v3 schema） ─────────────────────────────────────

def _reconstruct_trace(history: list[dict], answer: str) -> list[dict]:
    """Rebuild round-grouped trace from v3 exploration_history.

    Returns: [{role, round, data}, ...]
    Flow per round: [reflect?] → planner → executor → synthesizer
    Final: answer → END
    """
    trace = [{"role": "START", "round": None, "data": {}}]

    for rd in history:
        plan = rd.get("plan", {})
        round_num = rd.get("round", 1)

        # Reflector (only if reflection key exists, i.e. stuck >= 2)
        if "reflection" in plan:
            refl = plan["reflection"]
            trace.append({
                "role": "reflect",
                "round": round_num,
                "data": {
                    "diagnosis": refl.get("diagnosis", ""),
                    "strategy": refl.get("pivot_strategy", ""),
                    "pivot_questions": refl.get("pivot_sub_questions", []),
                },
            })

        # Planner
        trace.append({
            "role": "planner",
            "round": round_num,
            "data": {
                "hypotheses": plan.get("hypotheses", []),
                "sub_questions": plan.get("sub_questions", []),
            },
        })

        # Executor
        subs = rd.get("sub_questions", [])
        trace.append({
            "role": "executor",
            "round": round_num,
            "data": {
                "sub_questions": subs,
                "solved_count": sum(1 for s in subs if s.get("status") == "solved"),
                "total_count": len(subs),
                "total_chunks": sum(len(s.get("retrieved_chunks", [])) for s in subs),
            },
        })

        # Synthesizer
        ev = rd.get("evaluation", {})
        trace.append({
            "role": "synthesizer",
            "round": round_num,
            "data": {
                "status": ev.get("status", "unknown"),
                "reasoning_chain": ev.get("reasoning_chain", ""),
                "missing": ev.get("missing", []),
            },
        })

    # Final answer
    trace.append({"role": "answer", "round": None, "data": {"answer": answer}})
    trace.append({"role": "END", "round": None, "data": {}})
    return trace


def _count_stuck_rounds(history: list[dict]) -> int:
    """Count consecutive stuck rounds at the end of history."""
    count = 0
    for rd in reversed(history):
        ev = rd.get("evaluation", {})
        if ev.get("status") == "stuck":
            count += 1
        else:
            break
    return count


# ─── 路由标签 ────────────────────────────────────────────────────

_ROUTING_MAP = {
    "complete": {"label": "\u2192 \u751f\u6210\u7b54\u6848", "color": "#10B981", "bg": "#ECFDF5"},
    "incomplete": {"label": "\u2192 \u8865\u5145\u89c4\u5212", "color": "#CA8A04", "bg": "#FEFCE8"},
    "stuck": {"label": "\u2192 \u91cd\u65b0\u89c4\u5212", "color": "#EA580C", "bg": "#FFF7ED"},
}


def _render_routing_label(status: str, stuck_count: int = 0) -> None:
    """Render colored routing label after Synthesizer card."""
    cfg = _ROUTING_MAP.get(status, {"label": "\u2192 \u672a\u77e5", "color": "#6b7280", "bg": "#f3f4f6"})
    label = cfg["label"]
    if status == "stuck":
        if stuck_count >= 2:
            label = "\u2192 \u53cd\u601d\u5207\u6362\u7b56\u7565"
            cfg = {"label": label, "color": "#9333EA", "bg": "#FAF5FF"}
        else:
            label = f"\u2192 \u91cd\u65b0\u89c4\u5212 (stuck {stuck_count}/2)"
    st.markdown(
        f'<div style="text-align:center;padding:4px 16px;margin:4px 0 8px;'
        f'border-radius:6px;font-size:12px;font-weight:600;color:{cfg["color"]};'
        f'background:{cfg["bg"]};">{label}</div>',
        unsafe_allow_html=True,
    )


# ─── 角色卡片渲染 ────────────────────────────────────────────────

def _render_planner_card(data: dict, round_num: int) -> None:
    cfg = ROLE_CONFIG["planner"]
    with st.expander(f'{cfg["icon"]} {cfg["label"]} (Round {round_num})', expanded=False):
        hypotheses = data.get("hypotheses", [])
        subs = data.get("sub_questions", [])
        if hypotheses:
            with st.expander(f"**Hypotheses** ({len(hypotheses)})", expanded=False):
                for h in hypotheses:
                    st.markdown(f"- {h}")
        if subs:
            with st.expander(f"**Sub-Questions** ({len(subs)})", expanded=False):
                for sq in subs:
                    p = sq.get("priority", "?")
                    q = sq.get("question", "")
                    st.markdown(f"  - **[P{p}]** {q}")
        if not hypotheses and not subs:
            st.caption("\u672c\u8f6e\u672a\u751f\u6210\u63a2\u7d22\u8ba1\u5212")


def _render_executor_card(data: dict, round_num: int) -> None:
    cfg = ROLE_CONFIG["executor"]
    with st.expander(f'{cfg["icon"]} {cfg["label"]} (Round {round_num})', expanded=False):
        subs = data.get("sub_questions", [])
        solved = data.get("solved_count", 0)
        total = data.get("total_count", 0)
        chunks = data.get("total_chunks", 0)

        # Summary badge
        if total == 0:
            st.warning("\u672c\u8f6e\u672a\u751f\u6210\u5b50\u95ee\u9898")
        else:
            badge_text = f"**{solved}/{total} resolved** \u2022 {chunks} chunks retrieved"
            if solved == 0:
                st.warning(badge_text)
            else:
                st.info(badge_text)

            # Sub-question details
            for sq in subs:
                status = sq.get("status", "unsolved")
                icon = SQ_STATUS_ICONS.get(status, "?")
                color = SQ_STATUS_COLORS.get(status, "#6b7280")
                q = sq.get("question", "")
                a = sq.get("answer", "")
                st.markdown(
                    f'<span style="color:{color};font-weight:700;">{icon}</span> '
                    f'<b>{q}</b>',
                    unsafe_allow_html=True,
                )
                if a and status == "solved":
                    with st.expander("Answer", expanded=False):
                        st.markdown(a)

                # Show retrieved chunks count
                sq_chunks = sq.get("retrieved_chunks", [])
                if sq_chunks:
                    st.caption(f"{len(sq_chunks)} chunks retrieved")


def _render_synthesizer_card(data: dict, round_num: int) -> None:
    cfg = ROLE_CONFIG["synthesizer"]
    with st.expander(f'{cfg["icon"]} {cfg["label"]} (Round {round_num})', expanded=False):
        status = data.get("status", "unknown")
        missing = data.get("missing", [])
        reasoning = data.get("reasoning_chain", "")

        # Status badge
        color = STATUS_COLORS.get(status, "#6b7280")
        label = STATUS_LABELS.get(status, status)
        st.markdown(
            f'<span style="display:inline-block;padding:2px 12px;border-radius:10px;'
            f'font-size:12px;font-weight:600;color:white;background:{color};">{label}</span>',
            unsafe_allow_html=True,
        )

        if missing:
            with st.expander(f"**Missing Facts** ({len(missing)})", expanded=False):
                for m in missing:
                    st.markdown(f"- {m}")

        if reasoning and status == "complete":
            with st.expander("**Reasoning Chain**", expanded=False):
                st.markdown(f"> {reasoning}")


def _render_reflect_card(data: dict, round_num: int) -> None:
    cfg = ROLE_CONFIG["reflect"]
    with st.expander(f'{cfg["icon"]} {cfg["label"]} (Round {round_num})', expanded=False):
        diagnosis = data.get("diagnosis", "")
        strategy = data.get("strategy", "")
        pivots = data.get("pivot_questions", [])

        if diagnosis:
            with st.expander("**Diagnosis**", expanded=False):
                st.markdown(diagnosis)
        if strategy:
            with st.expander("**Pivot Strategy**", expanded=False):
                st.markdown(strategy)
        if pivots:
            with st.expander(f"**Pivot Sub-Questions** ({len(pivots)})", expanded=False):
                for pq in pivots:
                    st.markdown(f"- {pq}")


def _render_round_section(trace: list[dict], round_num: int, default_expanded: bool = False) -> None:
    """Render all steps for a single round inside a collapsible expander."""
    round_steps = [s for s in trace if s.get("round") == round_num]
    if not round_steps:
        return

    # Find synthesizer status for label
    synth_step = next((s for s in round_steps if s["role"] == "synthesizer"), None)
    status = "unknown"
    if synth_step:
        status = synth_step.get("data", {}).get("status", "unknown")
    color = STATUS_COLORS.get(status, "#6b7280")
    label = STATUS_LABELS.get(status, status)

    status_emoji = {"Answered": "✅", "Progressing": "🔶", "Stuck": "🔴"}.get(label, "⚪")
    expander_label = f"**Round {round_num}** {status_emoji} {label}"

    with st.expander(expander_label, expanded=default_expanded):
        for step in round_steps:
            role = step["role"]
            data = step.get("data", {})

            if role == "reflect":
                _render_reflect_card(data, round_num)
            elif role == "planner":
                _render_planner_card(data, round_num)
            elif role == "executor":
                _render_executor_card(data, round_num)
            elif role == "synthesizer":
                _render_synthesizer_card(data, round_num)
                # Routing label after synthesizer
                _render_routing_label(data.get("status", "unknown"))


# ─── 通用结果头部 ────────────────────────────────────────────────

def _render_result_header(item: dict) -> None:
    """Render question / ground truth / prediction / correctness."""
    col_q, col_gt, col_pred = st.columns(3)
    col_q.markdown(f"**Question:** {item.get('question', 'N/A')}")
    col_gt.markdown(f"**Ground Truth:** {item.get('answer', 'N/A')}")
    col_pred.markdown(f"**Prediction:** {item.get('prediction', 'N/A')}")

    is_correct = item.get("answer", "").strip().lower() == item.get("prediction", "").strip().lower()
    if is_correct:
        st.success("Exact Match: Correct")
    else:
        st.error("Exact Match: Incorrect")

    error = item.get("error")
    if error:
        st.error(f"Error: {error}")


def _render_result_metrics(item: dict, mode: str) -> None:
    """Render metadata metrics based on pipeline mode."""
    latency = item.get("latency_ms")

    if mode == "agentic-rag":
        meta_col1, meta_col2, meta_col3, meta_col4 = st.columns(4)
        if latency:
            meta_col1.caption(f"Latency: {_format_ms(latency)}")
        meta_col2.caption(f"Rounds: {item.get('search_depth', '?')}")
        meta_col3.caption(f"Unique Chunks: {item.get('total_chunks', '?')}")
        meta_col4.caption(f"Distinct Titles: {item.get('total_distinct_titles', '?')}")
    elif mode == "rag-with-judge":
        meta_col1, meta_col2, meta_col3 = st.columns(3)
        if latency:
            meta_col1.caption(f"Latency: {_format_ms(latency)}")
        meta_col2.caption(f"Retrieval Rounds: {item.get('retrieval_count', '?')}")
        meta_col3.caption(f"Unique Chunks: {item.get('total_chunks', '?')}")
    else:  # naive-rag
        meta_col1 = st.columns(1)[0]
        if latency:
            meta_col1.caption(f"Latency: {_format_ms(latency)}")


# ─── Agentic RAG 时间线 ──────────────────────────────────────────

def _render_agentic_timeline(item: dict) -> None:
    """Render the full Agentic RAG exploration timeline."""
    history = item.get("search_path", {})
    if isinstance(history, dict):
        history = history.get("exploration_history", [])

    if history:
        answer = item.get("prediction", "")
        trace = _reconstruct_trace(history, answer)

        round_nums = sorted(set(s["round"] for s in trace if s.get("round") is not None))
        for i, rn in enumerate(round_nums):
            _render_round_section(trace, rn, default_expanded=(i == 0))

        answer_steps = [s for s in trace if s["role"] == "answer"]
        if answer_steps:
            st.markdown("### Final Answer")
            ans = answer_steps[0]["data"].get("answer", "")
            st.markdown(
                f'<div style="font-size:16px;font-weight:700;color:#059669;'
                f'padding:8px 16px;border-radius:8px;border-left:4px solid #059669;'
                f'background:#ECFDF5;">{ans}</div>',
                unsafe_allow_html=True,
            )
    else:
        st.info("No exploration history available.")

    chunks = item.get("chunks", [])
    if chunks:
        with st.expander(f"All Unique Chunks ({len(chunks)})"):
            for i, chunk in enumerate(chunks):
                _chunk_card(chunk, rank=i + 1)


# ─── Naive RAG 简化视图 ──────────────────────────────────────────

def _render_naive_rag_detail(item: dict) -> None:
    """Render Naive RAG result detail."""
    # Eval results use "chunks", live query uses "fused_chunks"
    chunks = item.get("chunks", item.get("fused_chunks", []))
    if chunks:
        with st.expander(f"**Retrieved Chunks ({len(chunks)})**", expanded=False):
            for i, chunk in enumerate(chunks):
                _chunk_card(chunk, rank=i + 1)

    answer = item.get("answer", "")
    if answer:
        st.markdown("### Answer")
        st.success(answer)

    followups = item.get("suggested_followups", [])
    if followups:
        with st.expander(f"Follow-up Suggestions ({len(followups)})", expanded=False):
            for i, q in enumerate(followups):
                st.markdown(f"{i + 1}. {q}")

    rewritten = item.get("rewritten_queries", [])
    if rewritten:
        with st.expander(f"Rewritten Queries ({len(rewritten)})", expanded=False):
            for i, q in enumerate(rewritten):
                st.markdown(f"{i + 1}. {q}")


# ─── RAG with Judge 简化视图 ─────────────────────────────────────

# ─── RAG with Judge 搜索树 ────────────────────────────────────────

def _build_judge_tree_html(path: dict) -> tuple[str, list[dict]]:
    """Build CSS flexbox tree HTML and node_id_map for RAG with Judge."""
    node_id_map: list[dict] = []

    def _build_node(node: dict, node_id: str) -> str:
        question = node.get("question", "")
        answerable = node.get("answerable", False)
        depth = node_id.count(".")
        color = DEPTH_COLORS.get(depth % len(DEPTH_COLORS), "#2563EB")

        node_id_map.append({
            "node_id": node_id,
            "question": question,
            "answerable": answerable,
            "depth": depth,
        })

        idx = len(node_id_map) - 1
        q_short = question[:55] + "\u2026" if len(question) > 55 else question
        verdict_char = "\u2713" if answerable else "\u2192"
        verdict_bg = "#10B981" if answerable else "#F59E0B"

        html = '<div class="tw">'
        html += (
            f'<div class="tn" data-idx="{idx}" '
            f'style="border-color:{color}" onclick="sel({idx})">'
            f'<span class="tb" style="background:{color}">{depth}</span>'
            f'<span class="tq">{q_short}</span>'
            f'<span class="tv" style="background:{verdict_bg}">{verdict_char}</span>'
            '</div>'
        )

        children = node.get("next_queries", [])
        if children:
            html += '<div class="tc">'
            for i, child in enumerate(children):
                html += _build_node(child, f"{node_id}.{i}")
            html += '</div>'

        html += '</div>'
        return html

    body = _build_node(path, "0")
    n = len(node_id_map)

    js = f"""<script>
    function sel(i) {{
        var inputs = window.parent.document.querySelectorAll(
            'input[type="radio"][name="{n}"]'
        );
        if (inputs[i]) {{ inputs[i].click(); }}
    }}
    function connectNodes() {{
        document.querySelectorAll('.tc svg').forEach(function(s){{ s.remove(); }});
        var roots = document.querySelectorAll('.tw > .tn');
        roots.forEach(function(root) {{
            var container = root.parentElement;
            var childRow = container.querySelector(':scope > .tc');
            if (!childRow) return;
            var children = childRow.querySelectorAll(':scope > .tw > .tn');
            if (!children.length) return;
            var svg = document.createElementNS('http://www.w3.org/2000/svg','svg');
            svg.setAttribute('style','position:absolute;top:0;left:0;width:100%;height:100%;pointer-events:none;z-index:1;');
            childRow.appendChild(svg);
            var rr = root.getBoundingClientRect();
            var rc = childRow.getBoundingClientRect();
            var px = rr.left + rr.width/2 - rc.left;
            var py = rr.top + rr.height - rc.top;
            var first = children[0].getBoundingClientRect();
            var last = children[children.length-1].getBoundingClientRect();
            var lx = first.left + first.width/2 - rc.left;
            var rx = last.left + last.width/2 - rc.left;
            var midY = 10;
            function mkLine(x1,y1,x2,y2) {{
                var l = document.createElementNS('http://www.w3.org/2000/svg','line');
                l.setAttribute('x1',x1);l.setAttribute('y1',y1);
                l.setAttribute('x2',x2);l.setAttribute('y2',y2);
                l.setAttribute('stroke','#9CA3AF');l.setAttribute('stroke-width','3');
                return l;
            }}
            svg.appendChild(mkLine(px,py,px,midY));
            svg.appendChild(mkLine(lx,midY,rx,midY));
            children.forEach(function(ch) {{
                var cr2 = ch.getBoundingClientRect();
                var cx = cr2.left + cr2.width/2 - rc.left;
                svg.appendChild(mkLine(cx,midY,cx,18));
            }});
        }});
    }}
    if (document.readyState === 'loading') {{
        document.addEventListener('DOMContentLoaded', function() {{
            requestAnimationFrame(function() {{ requestAnimationFrame(connectNodes); }});
        }});
    }} else {{
        requestAnimationFrame(function() {{ requestAnimationFrame(connectNodes); }});
    }}
    var _rt;
    window.addEventListener('resize', function() {{
        clearTimeout(_rt);
        _rt = setTimeout(connectNodes, 100);
    }});
    </script>"""

    css = """<style>
    * { margin: 0; padding: 0; box-sizing: border-box; }
    body {
        margin: 0; padding: 10px 8px;
        background: #fafbfc;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    }
    .tw {
        display: flex;
        flex-direction: column;
        align-items: center;
    }
    .tn {
        display: flex;
        align-items: center;
        gap: 6px;
        padding: 5px 10px;
        background: white;
        border: 2px solid #2563EB;
        border-radius: 8px;
        cursor: pointer;
        max-width: 220px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06);
        transition: all 0.15s ease;
        white-space: normal;
        overflow: visible;
        height: auto;
    }
    .tw > .tn {
        padding: 7px 14px;
        border-width: 3px;
        max-width: 300px;
        font-size: 13px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.12);
    }
    .tn:hover {
        box-shadow: 0 3px 10px rgba(0,0,0,0.12);
        transform: translateY(-1px);
    }
    .tn:active {
        transform: translateY(0);
        box-shadow: 0 1px 2px rgba(0,0,0,0.1);
    }
    .tb {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        min-width: 20px;
        height: 20px;
        border-radius: 10px;
        color: #fff;
        font-size: 11px;
        font-weight: 700;
        flex-shrink: 0;
        padding: 0 4px;
    }
    .tw > .tn > .tb {
        min-width: 24px;
        height: 24px;
        font-size: 13px;
    }
    .tq {
        font-size: 11px;
        color: #1a1a2e;
        overflow: hidden;
        text-overflow: ellipsis;
        line-height: 1.3;
        max-height: 2.6em;
        display: -webkit-box;
        -webkit-line-clamp: 2;
        -webkit-box-orient: vertical;
    }
    .tw > .tn > .tq {
        font-size: 12px;
        font-weight: 600;
    }
    .tv {
        font-size: 11px;
        font-weight: 700;
        color: #fff;
        border-radius: 8px;
        padding: 1px 5px;
        flex-shrink: 0;
    }
    .tc {
        display: flex;
        flex-wrap: nowrap;
        gap: 16px;
        justify-content: center;
        position: relative;
        padding-top: 20px;
        margin-top: 4px;
        max-width: 100%;
    }
    .tc > .tw {
        position: relative;
        margin-top: 2px;
    }
    </style>"""

    return f"<html><head>{css}</head><body>{body}{js}</body></html>", node_id_map


def _find_node_by_id(path: dict, node_id: str, current_id: str = "0") -> Optional[dict]:
    """Find a node by its dotted ID in the SEARCH_PATH tree."""
    if current_id == node_id:
        return path
    for i, child in enumerate(path.get("next_queries", [])):
        child_id = f"{current_id}.{i}"
        if child_id == node_id:
            return child
        found = _find_node_by_id(child, node_id, child_id)
        if found:
            return found
    return None


def _count_nodes_in_tree(path: dict) -> int:
    """Count total nodes in the search tree."""
    count = 1
    for child in path.get("next_queries", []):
        count += _count_nodes_in_tree(child)
    return count


def _render_judge_node_detail(node: dict) -> None:
    """Render detail panel for a selected tree node."""
    question = node.get("question", "")
    answerable = node.get("answerable", False)
    chunks = node.get("chunks", [])
    children = node.get("next_queries", [])
    reason = node.get("judgement_reason", "")
    answer = node.get("answer")

    st.markdown(f"### {question}")

    if answerable:
        st.success("**Judge:** Answerable")
    else:
        st.warning("**Judge:** Needs Follow-up")

    if answer:
        st.success(f"**Answer:** {answer}")

    if reason:
        st.info(f"**Reasoning:** {reason}")

    if chunks:
        st.markdown(f"#### Retrieved Chunks ({len(chunks)})")
        for i, chunk in enumerate(chunks, 1):
            _chunk_card(chunk, rank=i)

    if children:
        st.markdown(f"#### Follow-up Questions ({len(children)})")
        for i, child in enumerate(children, 1):
            child_q = child.get("question", "")
            child_answerable = child.get("answerable", False)
            icon = "\u2713" if child_answerable else "\u2192"
            st.markdown(f"{i}. {icon} {child_q}")


def _render_judge_detail(item: dict, item_idx: int = 0) -> None:
    """Render RAG with Judge result detail with tree visualization."""
    # Final answer
    answer = item.get("answer", "")
    if answer:
        st.markdown("### Answer")
        st.success(answer)

    # Search path tree
    search_path = item.get("search_path", {})
    if search_path and "question" in search_path:
        st.markdown("---")
        st.markdown("#### Search Path")
        node_count = _count_nodes_in_tree(search_path)
        st.caption(f"{node_count} nodes in search tree")

        html_str, node_id_map = _build_judge_tree_html(search_path)

        col_tree, col_sel = st.columns([3, 2])

        with col_tree:
            if node_id_map:
                st.components.v1.html(html_str, height=450, scrolling=True)
            else:
                st.info("Empty search path.")

        selected_node = None
        with col_sel:
            if node_id_map:
                labels = [
                    f"{'  ' * info['depth']}{info['node_id']}: {info['question'][:70]}"
                    for info in node_id_map
                ]

                radio_key = f"judge_tree_node_{item_idx}"
                selected_label = st.radio(
                    "Select a node",
                    options=labels,
                    index=0,
                    key=radio_key,
                    label_visibility="collapsed",
                )
                try:
                    sel_idx = labels.index(selected_label)
                    sel_info = node_id_map[sel_idx]
                    selected_node = _find_node_by_id(search_path, sel_info["node_id"])
                except (ValueError, IndexError):
                    pass

        # Node detail panel
        if selected_node:
            st.markdown("---")
            _render_judge_node_detail(selected_node)

    # All unique chunks
    chunks = item.get("chunks", [])
    if chunks:
        with st.expander(f"All Unique Chunks ({len(chunks)})"):
            for i, chunk in enumerate(chunks):
                _chunk_card(chunk, rank=i + 1)


# ─── 结果详情分派 ────────────────────────────────────────────────

def _render_result_detail(item: dict, mode: str, item_idx: int) -> None:
    """Render the full detail view of a single result."""
    st.markdown("---")
    _render_result_header(item)
    _render_result_metrics(item, mode)
    st.markdown("---")

    if mode == "agentic-rag":
        st.markdown("#### Role Conversation Timeline")
        _render_agentic_timeline(item)
    elif mode == "naive-rag":
        _render_naive_rag_detail(item)
    else:  # rag-with-judge
        _render_judge_detail(item, item_idx)


# ─── System Comparison 可视化 ─────────────────────────────────────

@st.cache_data
def _load_sc_analysis_data():
    """Load system comparison analysis data with graceful fallback."""
    if not _SC_AVAILABLE:
        return None
    try:
        return load_analysis_data()
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def _render_figure_grid(filenames: list[str], cols_count: int = 2) -> None:
    """Render a 2-column grid of pre-generated figure PNGs."""
    if not SC_FIGURES_DIR.exists():
        st.warning(f"Figures directory not found: {SC_FIGURES_DIR}")
        return
    for i in range(0, len(filenames), cols_count):
        row_files = filenames[i:i + cols_count]
        cols = st.columns(len(row_files))
        for col, fname in zip(cols, row_files):
            fpath = SC_FIGURES_DIR / fname
            if fpath.exists():
                caption = FIGURE_CATALOG.get(fname, fname.replace(".png", "").replace("_", " ").title())
                col.image(str(fpath), caption=caption, use_container_width=True)
            else:
                col.warning(f"Missing: {fname}")


def _render_efficiency_table(sc_data: dict, dataset: str) -> None:
    """Render the efficiency metrics table."""
    rows = []
    for row in sc_data.get("efficiency", []):
        if row["dataset"] != dataset:
            continue
        rows.append({
            "System": MODE_LABELS.get(row["mode"], row["mode"]),
            "Avg Latency": f"{row.get('avg_latency_ms', 0)/1000:.1f}s",
            "P50 Latency": f"{row.get('p50_latency_ms', 0)/1000:.1f}s",
            "P95 Latency": f"{row.get('p95_latency_ms', 0)/1000:.1f}s",
            "Retrieval Rounds": row.get("retrieval_count", "-"),
            "Chunk Count": row.get("total_chunks", "-"),
            "Search Depth": row.get("search_depth", "-"),
        })
    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def _render_behavior_patterns(sc_data: dict, dataset: str) -> None:
    """Render top behavior patterns as a table (2Wiki only)."""
    if dataset != "2wikimultihopqa":
        return
    try:
        raw = {}
        profiles = build_behavior_profiles(sc_data, raw, dataset)
        if not profiles:
            return
        patterns = compute_behavior_patterns(profiles)
        st.write(f"**{len(patterns)} unique patterns**")
        table = [
            {"Pattern": p, "Count": c, "Percentage": f"{c/len(profiles)*100:.1f}%"}
            for p, c in patterns[:10]
        ]
        st.dataframe(pd.DataFrame(table), use_container_width=True, hide_index=True)
    except Exception:
        pass


def _render_system_comparison(dataset: str) -> None:
    """Render the System Comparison view with figures and data tables."""
    sc_data = _load_sc_analysis_data()
    if sc_data is None:
        st.warning("System comparison data not available. Run the analysis pipeline first.")
        return

    st.markdown(f"### Dataset: {DATASET_LABELS.get(dataset, dataset)}")
    st.caption("Charts are static (generated from 2WikiMultiHopQA). Tables update with selected dataset.")

    tab_macro, tab_eff, tab_err, tab_qtype, tab_faith, tab_retr = st.tabs([
        "Macro Metrics", "Efficiency", "Error Analysis",
        "Question Types", "Faithfulness", "Retrieval Quality",
    ])

    # ── Tab 1: Macro Metrics ──
    with tab_macro:
        # Per-method metrics table
        metrics_filter = ["em", "f1", "precision", "recall"]
        metric_labels_map = {"em": "EM", "f1": "F1", "precision": "Precision", "recall": "Recall"}

        table_rows = []
        for mode in MODE_ORDER:
            row = {"System": MODE_LABELS.get(mode, mode)}
            for m in metrics_filter:
                for r in sc_data.get("macro_metrics", []):
                    if r["mode"] == mode and r["dataset"] == dataset:
                        val = r.get(m)
                        row[metric_labels_map[m]] = f"{val*100:.1f}%" if val is not None else "N/A"
                        break
            table_rows.append(row)
        st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)

        st.divider()
        _render_figure_grid(["A1_em_comparison.png", "A2_radar_f1_precision_recall.png"])
        _render_figure_grid(["A3_retrieval_quality.png", "A4_improvement_delta.png"])
        _render_figure_grid(["A5_heatmap.png"])

    # ── Tab 2: Efficiency ──
    with tab_eff:
        _render_efficiency_table(sc_data, dataset)
        st.divider()
        _render_figure_grid(["B1_latency_comparison.png", "B2_pareto.png"])
        _render_figure_grid(["B3_retrieval_resources.png", "B4_latency_distribution.png"])

    # ── Tab 3: Error Analysis ──
    with tab_err:
        error_types = sc_data.get("error_types", {})
        if error_types:
            error_labels = {
                "all_correct": "All Correct",
                "llm_only_wrong": "Only LLM Wrong",
                "llm_correct_others_wrong": "Only LLM Correct",
                "all_wrong": "All Wrong",
                "judge_only_correct": "Only Judge Correct",
                "v3_only_correct": "Only V3 Correct",
            }
            total_err = sum(error_types.values())
            err_cols = st.columns(min(6, len(error_types)))
            for i, (k, v) in enumerate(error_types.items()):
                err_cols[i % 6].metric(error_labels.get(k, k), f"{v} ({v/total_err*100:.1f}%)")
            st.divider()
        _render_figure_grid(["C1_error_types.png", "C2_correct_overlap.png"])
        if dataset == "2wikimultihopqa":
            _render_figure_grid(["C3_behavior_patterns.png"])
            _render_behavior_patterns(sc_data, dataset)
        _render_figure_grid(["C4_v3_only_deepdive.png"])

    # ── Tab 4: Question Types ──
    with tab_qtype:
        qta = sc_data.get("question_type_analysis", {})
        if qta:
            table_rows = []
            for qtype, mode_data in qta.items():
                row = {"Question Type": qtype.replace("_", " ").title()}
                for mode, vals in mode_data.items():
                    acc = vals.get("accuracy", 0)
                    row[MODE_LABELS.get(mode, mode)] = f"{acc*100:.1f}%" if acc else "N/A"
                table_rows.append(row)
            st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)
        st.divider()
        _render_figure_grid(["D1_accuracy_by_type.png", "D2_type_distribution.png"])
        _render_figure_grid(["D3_improvement_by_type.png"])

    # ── Tab 5: Faithfulness ──
    with tab_faith:
        if dataset != "2wikimultihopqa":
            st.info("Faithfulness experiments are 2WikiMultiHopQA-specific.")
        _render_figure_grid(["E1_causal_ablation.png", "E2_known_unknown.png"])
        _render_figure_grid(["E3_dose_response.png", "E4_faithfulness_cross.png"])

    # ── Tab 6: Retrieval Quality ──
    with tab_retr:
        _render_figure_grid(["F1_retrieval_precision_vs_f1.png", "F2_latency_vs_em.png"])
        _render_figure_grid(["F3_chunk_count.png", "F4_search_depth.png"])


# ─── 指标汇总 ────────────────────────────────────────────────────

def _render_metrics_summary(results: dict, mode: str) -> None:
    """Render aggregate metrics based on pipeline mode."""
    summary = results.get("summary", {})
    aggregate = summary.get("aggregate", {})

    st.markdown("### Metrics Summary")

    if mode == "agentic-rag":
        # Top row: Total / Answered / Unanswered
        stat_cols = st.columns(3)
        stat_cols[0].metric("Total", summary.get("total", "?"))
        stat_cols[1].metric("Answered", summary.get("answered", "?"))
        stat_cols[2].metric("Unanswered", summary.get("unanswered", "?"))
        st.divider()

        # Bottom row: 6 aggregate metrics
        metric_cols = st.columns(6)
        metrics = [
            ("em", "EM"),
            ("f1", "F1"),
            ("hit", "Hit"),
            ("mrr", "MRR"),
            ("search_depth", "Avg Rounds"),
            ("total_chunks", "Avg Chunks"),
        ]
        for i, (key, label) in enumerate(metrics):
            val = aggregate.get(key)
            display = f"{val:.3f}" if val is not None else "N/A"
            metric_cols[i].metric(label, display)
    else:
        # Naive RAG / RAG with Judge: standard metrics
        metric_cols = st.columns(5)
        metrics = [
            ("em", "EM"),
            ("f1", "F1"),
            ("hit", "Hit"),
            ("mrr", "MRR"),
            ("precision", "Precision"),
        ]
        for i, (key, label) in enumerate(metrics):
            val = aggregate.get(key)
            display = f"{val:.3f}" if val is not None else "N/A"
            metric_cols[i].metric(label, display)


# ─── Main ─────────────────────────────────────────────────────────

def main():
    st.title("Results")
    st.caption("Browse evaluation results for all RAG pipelines")

    # Sidebar
    with st.sidebar:
        st.subheader("Pipeline")
        pipeline = st.radio(
            "Select pipeline",
            options=list(PIPELINE_CONFIG.keys()),
            format_func=lambda k: f"{PIPELINE_CONFIG[k]['icon']} {PIPELINE_CONFIG[k]['label']}",
            index=3,  # default to system-comparison
            key="result_pipeline",
        )

        if pipeline == "system-comparison":
            st.subheader("Dataset")
            if not _SC_AVAILABLE or not DATASETS:
                st.warning("System comparison module not available.")
                return
            sc_dataset = st.selectbox(
                "Select dataset",
                DATASETS,
                format_func=lambda d: DATASET_LABELS.get(d, d),
                key="sc_dataset",
            )
        else:
            st.subheader("Filters")
            datasets = list_available_datasets(pipeline)
            if not datasets:
                st.warning(f"No {PIPELINE_CONFIG[pipeline]['label']} results found.")
                return

            dataset = st.selectbox("Dataset", options=datasets, index=0, key="result_dataset")
            search_text = st.text_input("Search questions", key="result_search")

    # System Comparison view
    if pipeline == "system-comparison":
        _render_system_comparison(sc_dataset)
        return

    # Load results
    results = load_results(pipeline, dataset)
    if not results or "results" not in results:
        st.warning("No results found.")
        return

    items = results["results"]

    # Filter
    if search_text:
        items = [
            item for item in items
            if search_text.lower() in item.get("question", "").lower()
        ]

    # Metrics summary
    _render_metrics_summary(results, pipeline)
    st.divider()

    # Result list with pagination
    st.markdown(f"### Results ({len(items)} questions)")

    page_size = 20
    total_pages = max(1, (len(items) + page_size - 1) // page_size)
    page = st.slider("Page", 1, total_pages, 1, key="result_page")

    start = (page - 1) * page_size
    end = min(start + page_size, len(items))
    page_items = items[start:end]

    for idx, item in enumerate(page_items, start=start):
        question = item.get("question", "")
        prediction = item.get("prediction", "")
        answer = item.get("answer", "")
        error = item.get("error")

        is_correct = answer.strip().lower() == prediction.strip().lower() if answer and prediction else False
        badge = ":green[\u2713]" if is_correct else ":red[\u2717]"

        if pipeline == "agentic-rag":
            depth = item.get("search_depth", "?")
            total_chunks = item.get("total_chunks", "?")
            status_text = f"[{depth} rounds]"
        elif pipeline == "rag-with-judge":
            depth = item.get("search_depth", "?")
            total_chunks = item.get("total_chunks", "?")
            status_text = f"[depth={depth}]"
        else:
            status_text = ""

        if error:
            status_text = ":red[Error]"

        label = f"{badge} {status_text} \u2014 {question[:120]}{'...' if len(question) > 120 else ''}"
        with st.expander(label, expanded=False):
            _render_result_detail(item, pipeline, item_idx=idx)

    st.caption(f"Showing {start + 1}\u2013{end} of {len(items)} results")


main()
