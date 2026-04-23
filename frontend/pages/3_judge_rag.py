"""RAG with Judge result browser with tree graph visualization."""

import sys
from pathlib import Path
from typing import Optional

# Ensure project root is on sys.path so frontend package imports work
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import streamlit as st

from frontend.data_loader import list_available_datasets, load_results
from frontend.styles import DEPTH_COLORS, inject_custom_css

# ─── CSS tree graph visualization ─────────────────────────────────

def _build_tree_html(path: dict) -> tuple[str, list[dict]]:
    """Build CSS flexbox tree HTML and node_id_map.

    Returns (full_html_string, list_of_node_info_dicts).
    Each node_info: {node_id, question, answerable, depth}.
    """
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
        q_short = question[:55] + "…" if len(question) > 55 else question
        verdict_char = "\u2713" if answerable else "\u2192"
        verdict_bg = "#10B981" if answerable else "#F59E0B"

        html = f'<div class="tw">'
        html += (
            f'<div class="tn" data-idx="{idx}" '
            f'style="border-color:{color}" onclick="sel({idx})">'
            f'<span class="tb" style="background:{color}">{depth}</span>'
            f'<span class="tq">{q_short}</span>'
            f'<span class="tv" style="background:{verdict_bg}">{verdict_char}</span>'
            f'</div>'
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
            requestAnimationFrame(function() {{
                requestAnimationFrame(connectNodes);
            }});
        }});
    }} else {{
        requestAnimationFrame(function() {{
            requestAnimationFrame(connectNodes);
        }});
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
    /* Tree wrapper: parent + children stacked vertically */
    .tw {
        display: flex;
        flex-direction: column;
        align-items: center;
    }
    /* Tree node: clickable card — default size for children */
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
    /* Root node (depth 0): larger and more prominent */
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
    /* Depth badge */
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
    /* Root node depth badge: larger */
    .tw > .tn > .tb {
        min-width: 24px;
        height: 24px;
        font-size: 13px;
    }
    /* Question text */
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
    /* Root node question text */
    .tw > .tn > .tq {
        font-size: 12px;
        font-weight: 600;
    }
    /* Verdict icon */
    .tv {
        font-size: 11px;
        font-weight: 700;
        color: #fff;
        border-radius: 8px;
        padding: 1px 5px;
        flex-shrink: 0;
    }
    /* Children container: horizontal row, SVG draws lines */
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
    /* Child wrappers */
    .tc > .tw {
        position: relative;
        margin-top: 2px;
    }
    </style>"""

    return f"<html><head>{css}</head><body>{body}{js}</body></html>", node_id_map


# ─── Tree widget (iframe + hidden radio bridge) ────────────────────

# ─── Node selector widget ──────────────────────────────────────────

def _render_node_selector(labels: list[str], item_idx: int = 0) -> str | None:
    """Render the radio widget for node selection. Returns selected node_id or None."""
    if not labels:
        return None

    hidden_key = f"tree_node_{item_idx}"

    selected_label = st.radio(
        "Select a node",
        options=labels,
        index=0,
        key=hidden_key,
        label_visibility="collapsed",
    )

    # The radio label format is "indent{node_id}: {question}" — extract node_id
    # Match against the labels list to get the index, then caller maps to node
    try:
        selected_idx = labels.index(selected_label)
        return str(selected_idx)
    except ValueError:
        return None


# ─── Utility helpers ───────────────────────────────────────────────

def _count_nodes_in_tree(path: dict) -> int:
    """Count total nodes in the search tree."""
    count = 1
    for child in path.get("next_queries", []):
        count += _count_nodes_in_tree(child)
    return count


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


def _render_node_detail(node: dict) -> None:
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
        st.markdown(f"#### Generated Follow-up Questions ({len(children)})")
        for i, child in enumerate(children, 1):
            child_q = child.get("question", "")
            child_answerable = child.get("answerable", False)
            icon = "\u2713" if child_answerable else "\u2192"
            st.markdown(f"{i}. {icon} {child_q}")


def _format_ms(ms: float) -> str:
    """Format milliseconds to human-readable string."""
    if ms < 1000:
        return f"{ms:.0f}ms"
    return f"{ms / 1000:.1f}s"


def _truncate(text: str, max_len: int = 300) -> str:
    """Truncate text for preview."""
    if len(text) <= max_len:
        return text
    return text[:max_len] + "..."


def _chunk_card(chunk: dict, rank: int, depth: int = 0) -> None:
    """Render a single chunk card with score bar."""
    score = chunk.get("score", 0)
    title = chunk.get("chunk_title", "Unknown")
    context = chunk.get("context_title", "")
    content = chunk.get("page_content", "")
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
        st.markdown(f"**Content:**\n{_truncate(content)}")


# ─── Main ──────────────────────────────────────────────────────────

def main():
    st.title("RAG with Judge Results")
    st.caption("Browse evaluation results with tree graph visualization")

    # Inject custom CSS
    inject_custom_css()

    # Sidebar
    with st.sidebar:
        st.subheader("Filters")
        datasets = list_available_datasets("rag-with-judge")
        if not datasets:
            st.warning("No RAG with Judge results found.")
            return

        dataset = st.selectbox("Dataset", options=datasets, index=0, key="judge_dataset")
        search_text = st.text_input("Search questions", key="judge_search")

    # Load results
    results = load_results("rag-with-judge", dataset)
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
    summary = results.get("summary", {})
    aggregate = summary.get("aggregate", {})

    st.markdown("### Metrics Summary")
    cols = st.columns(6)
    metrics = [
        ("em", "EM"),
        ("f1", "F1"),
        ("hit", "Hit"),
        ("mrr", "MRR"),
        ("search_depth", "Avg Depth"),
        ("retrieval_count", "Avg Retriev."),
    ]
    for i, (key, label) in enumerate(metrics):
        val = aggregate.get(key)
        display = f"{val:.3f}" if val is not None else "N/A"
        cols[i].metric(label, display)

    st.divider()

    # Result list with pagination
    st.markdown(f"### Results ({len(items)} questions)")

    page_size = 20
    total_pages = max(1, (len(items) + page_size - 1) // page_size)
    page = st.slider("Page", 1, total_pages, 1, key="judge_page")

    start = (page - 1) * page_size
    end = min(start + page_size, len(items))
    page_items = items[start:end]

    for idx, item in enumerate(page_items, start=start + 1):
        question = item.get("question", "")
        prediction = item.get("prediction", "")
        answer = item.get("answer", "")
        latency = item.get("latency_ms", 0)
        search_depth = item.get("search_depth", "?")
        retrieval_count = item.get("retrieval_count", "?")
        total_chunks = item.get("total_chunks", "?")
        search_path = item.get("search_path", {})
        error = item.get("error")

        is_correct = answer.strip().lower() == prediction.strip().lower() if answer and prediction else False
        badge = ":green[\u2713]" if is_correct else ":red[\u2717]"

        status_text = f"[depth={search_depth}]"
        if error:
            status_text = ":red[Error]"

        with st.expander(
            f"{badge} {status_text} \u2014 {question[:150]}{'...' if len(question) > 150 else ''}",
            expanded=False,
        ):
            # Basic info
            col_q, col_a, col_p = st.columns(3)
            col_q.markdown(f"**Question:** {question}")
            col_a.markdown(f"**Ground Truth:** {answer}")
            col_p.markdown(f"**Prediction:** {prediction}")

            if error:
                st.error(f"Error: {error}")

            meta_col1, meta_col2, meta_col3 = st.columns(3)
            meta_col1.caption(f"Latency: {_format_ms(latency)}")
            meta_col2.caption(f"Retrieval rounds: {retrieval_count}")
            meta_col3.caption(f"Unique chunks: {total_chunks}")

            # Search Path Graph — tree | selector row, detail below
            if search_path:
                st.markdown("---")
                st.markdown("#### Search Path")
                node_count = _count_nodes_in_tree(search_path)
                st.caption(f"{node_count} nodes in search tree")

                # Row 1: tree visualization | node selector (side by side)
                col_tree, col_selector = st.columns([3, 2])

                html_str, node_id_map = _build_tree_html(search_path)

                labels = [
                    f"{'  ' * info['depth']}{info['node_id']}: {info['question'][:70]}"
                    for info in node_id_map
                ]

                with col_tree:
                    if node_id_map:
                        st.components.v1.html(html_str, height=450, scrolling=True)
                    else:
                        st.info("Empty search path.")

                selected_idx_str = None
                with col_selector:
                    if labels:
                        selected_idx_str = _render_node_selector(labels, item_idx=idx)

                # Row 2: detail panel (full width, conditional)
                if selected_idx_str is not None and node_id_map:
                    try:
                        selected_idx = int(selected_idx_str)
                        selected_info = node_id_map[selected_idx]
                        selected_node = _find_node_by_id(search_path, selected_info["node_id"])
                        if selected_node:
                            st.markdown("---")
                            _render_node_detail(selected_node)
                    except (ValueError, IndexError):
                        pass
            else:
                st.info("No search path available for this question.")

    st.caption(f"Showing {start + 1}\u2013{end} of {len(items)} results")


main()
