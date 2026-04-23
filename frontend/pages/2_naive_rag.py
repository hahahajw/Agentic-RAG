"""Naive RAG result browser page."""

import sys
from pathlib import Path

# Ensure project root is on sys.path so frontend package imports work
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import streamlit as st

from frontend.data_loader import list_available_datasets, load_results
from frontend.styles import DEPTH_COLORS


def _format_ms(ms: float) -> str:
    """Format milliseconds to human-readable string."""
    if ms < 1000:
        return f"{ms:.0f}ms"
    return f"{ms / 1000:.1f}s"


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
        header += f"  — {context}"

    with st.expander(header, expanded=(rank <= 3)):
        # Score bar
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

        st.markdown(f"**Content:**\n{content}")


def _render_result_detail(item: dict, scheme: str) -> None:
    """Render the full detail view of a single result."""
    st.markdown("---")

    # Question + Ground Truth + Prediction
    col_q, col_gt, col_pred = st.columns(3)
    col_q.markdown(f"**Question:** {item.get('question', 'N/A')}")
    col_gt.markdown(f"**Ground Truth:** {item.get('answer', 'N/A')}")
    col_pred.markdown(f"**Prediction:** {item.get('prediction', 'N/A')}")

    # Correctness
    is_correct = item.get("answer", "").strip().lower() == item.get("prediction", "").strip().lower()
    if is_correct:
        st.success("Exact Match: Correct")
    else:
        st.error("Exact Match: Incorrect")

    # Latency
    latency = item.get("latency_ms")
    if latency:
        st.caption(f"Latency: {_format_ms(latency)}")

    st.markdown("---")

    # Chunks
    chunks = item.get("chunks", [])
    if chunks:
        st.markdown(f"**Retrieved Chunks ({len(chunks)})**")
        for i, chunk in enumerate(chunks):
            _chunk_card(chunk, rank=i + 1)


def main():
    st.title("Naive RAG Results")
    st.caption("Browse evaluation results for Naive RAG (Scheme A/B)")

    # Sidebar filters
    with st.sidebar:
        st.subheader("Filters")
        datasets = list_available_datasets("naive-rag")
        if not datasets:
            st.warning("No Naive RAG results found.")
            return

        dataset = st.selectbox("Dataset", options=datasets, index=0, key="naive_dataset")

        scheme = st.radio("Scheme", options=["All", "A", "B"], index=0, key="naive_scheme")

        search_text = st.text_input("Search questions", key="naive_search")

    # Determine which schemas to show
    schemas_to_show = []
    if scheme == "All":
        schemas_to_show = ["a", "b"]
    else:
        schemas_to_show = [scheme.lower()]

    # Load results for each schema
    all_items = []
    for schema in schemas_to_show:
        results = load_results("naive-rag", dataset, schema)
        if results and "results" in results:
            for item in results["results"]:
                item["_schema"] = schema
                all_items.append(item)

    if not all_items:
        st.warning("No results found.")
        return

    # Filter by search
    if search_text:
        all_items = [
            item for item in all_items
            if search_text.lower() in item.get("question", "").lower()
        ]

    # Metrics summary — render both schemas side by side
    st.markdown("### Metrics Summary")

    results_a = load_results("naive-rag", dataset, "a")
    results_b = load_results("naive-rag", dataset, "b")

    agg_a = (results_a or {}).get("summary", {}).get("aggregate", {})
    agg_b = (results_b or {}).get("summary", {}).get("aggregate", {})

    metric_keys = ["em", "f1", "hit", "mrr", "context_recall", "retrieval_precision"]
    metric_labels = ["EM", "F1", "Hit", "MRR", "Ctx Recall", "Ret Prec"]

    # Build a compact table: row = metric, cols = A / B
    header_cols = st.columns(3)
    header_cols[0].markdown("**Metric**")
    header_cols[1].markdown("**Scheme A**")
    header_cols[2].markdown("**Scheme B**")
    for key, label in zip(metric_keys, metric_labels):
        row = st.columns(3)
        row[0].text(label)
        row[1].text(f"{agg_a.get(key):.3f}" if agg_a.get(key) is not None else "N/A")
        row[2].text(f"{agg_b.get(key):.3f}" if agg_b.get(key) is not None else "N/A")

    st.markdown("---")

    # Result list with pagination
    st.markdown(f"### Results ({len(all_items)} questions)")

    page_size = 20
    total_pages = max(1, (len(all_items) + page_size - 1) // page_size)
    page = st.slider("Page", 1, total_pages, 1, key="naive_page")

    start = (page - 1) * page_size
    end = min(start + page_size, len(all_items))
    page_items = all_items[start:end]

    for item in page_items:
        question = item.get("question", "")
        prediction = item.get("prediction", "")
        answer = item.get("answer", "")
        schema = item.get("_schema", "?")
        n_chunks = len(item.get("chunks", []))
        latency = item.get("latency_ms", 0)

        is_correct = answer.strip().lower() == prediction.strip().lower() if answer and prediction else False
        badge = ":green[✓ Correct]" if is_correct else ":red[✗ Incorrect]"

        with st.expander(
            f"[{schema.upper()}] {badge} — {question[:120]}{'...' if len(question) > 120 else ''}",
            expanded=False,
        ):
            _render_result_detail(item, schema)

    st.caption(f"Showing {start + 1}–{end} of {len(all_items)} results")


main()
