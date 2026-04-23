"""Agentic RAG live query interface with exploration chain rendering."""

import os
import sys
import time
from pathlib import Path

# Ensure project root is on sys.path so frontend package imports work
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import streamlit as st

from frontend.styles import DEPTH_COLORS


def _get_llm(model: str, temperature: float = 0.0):
    """Create an LLM instance."""
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(
        api_key=os.getenv("BL_API_KEY"),
        base_url=os.getenv("BL_BASE_URL"),
        model=model,
        temperature=temperature,
    )


def _chunk_card(chunk: dict, rank: int, depth: int = 0) -> None:
    """Render a chunk card."""
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
        bar_pct = min(100, (score / 0.09) * 100)
        st.markdown(
            f'<div class="score-bar-bg"><div class="score-bar-fill" style="width:{bar_pct:.0f}%;background:{color}"></div></div>',
            unsafe_allow_html=True,
        )
        if summary:
            st.caption(f"**Summary:** {summary}")
        st.markdown(f"**Content:**\n{content}")


def _run_agentic_rag_live(query: str, dataset: str, model: str, rerank: bool,
                          rewrite_model: str | None, max_rounds: int):
    """Run an Agentic RAG query by directly importing backend functions."""
    from agentic_rag.nodes import run_agentic_rag
    from agentic_rag.workflow import build_agentic_rag_graph
    from Retrieval.milvus_retriever import MilvusRetriever

    llm = _get_llm(model)
    rewrite_llm = _get_llm(rewrite_model) if rewrite_model else llm

    retriever = MilvusRetriever(
        dataset_type=dataset,
        topk_propositions=50,
        max_chunks=8,
        use_reranker=rerank,
        reranker_params={"model_name": "qwen3-rerank"} if rerank else {},
    )

    config = {
        "configurable": {
            "llm": llm,
            "planner_llm": llm,
            "evaluator_llm": llm,
            "reflector_llm": llm,
            "answer_llm": llm,
            "rewrite_llm": rewrite_llm,
            "retriever": retriever,
            "max_chunks": 8,
            "max_rounds": max_rounds,
        }
    }

    app = build_agentic_rag_graph()
    return run_agentic_rag(query, app, config, max_rounds)


def _render_exploration_chain(history: list[dict]) -> None:
    """Render the linear exploration chain as a timeline."""
    for i, rd in enumerate(history):
        status = rd.get("evaluation", {}).get("status", "unknown")
        status_class = f"status-{status}"
        rnd = rd.get("round", i + 1)

        # Timeline connector (skip first)
        if i > 0:
            st.markdown('<div class="timeline-connector"></div>', unsafe_allow_html=True)

        # Round card
        st.markdown(f'<div class="round-card {status_class}">', unsafe_allow_html=True)

        # Header: Round badge + Status badge + Confidence
        ev = rd.get("evaluation", {})
        conf = ev.get("confidence", 0)
        st.markdown(
            f'<div class="round-header">'
            f'<span class="round-badge">R{rnd}</span>'
            f'<span class="status-badge {status_class}">{status.capitalize()}</span>'
            f'<span style="font-size:12px;color:#6b7280">Confidence: {conf:.2f}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

        # Plan targets as tags
        plan = rd.get("plan", {})
        targets = plan.get("targets", [])
        if targets:
            tags = "".join(f'<span class="target-tag">{t}</span>' for t in targets)
            st.markdown(f'<div class="round-targets">{tags}</div>', unsafe_allow_html=True)

        # Hypotheses
        hyps = plan.get("hypotheses", [])
        if hyps:
            with st.expander(f"Hypotheses ({len(hyps)})", expanded=False):
                for j, h in enumerate(hyps, 1):
                    st.markdown(f"{j}. {h}")

        # Reflection (if Reflector triggered)
        reflection = plan.get("reflection")
        if reflection:
            with st.expander("Reflector Analysis", expanded=False):
                st.markdown(f"**Diagnosis:** {reflection.get('diagnosis', '')}")
                pivots = reflection.get("pivot_queries", [])
                if pivots:
                    st.markdown("**Pivot Queries:**")
                    for pq in pivots:
                        st.markdown(f"- `{pq}`")

        # Queries
        queries = rd.get("rewritten_queries", [])
        if queries:
            with st.expander(f"Queries ({len(queries)})", expanded=False):
                for j, q in enumerate(queries, 1):
                    st.markdown(f"{j}. `{q}`")

        # Chunks
        chunks = rd.get("chunks", [])
        if chunks:
            with st.expander(f"Retrieved Chunks ({len(chunks)})", expanded=(rnd == 1)):
                for j, chunk in enumerate(chunks):
                    _chunk_card(chunk, rank=j + 1, depth=rnd)

        # Evaluation section
        st.markdown('<div class="eval-section">', unsafe_allow_html=True)
        feedback = ev.get("feedback", "")
        if feedback:
            if status == "answered":
                st.success(f"**Feedback:** {feedback}")
            elif status == "progressing":
                st.info(f"**Feedback:** {feedback}")
            else:
                st.warning(f"**Feedback:** {feedback}")

        gaps = ev.get("knowledge_gaps", [])
        if gaps:
            st.markdown("**Knowledge Gaps:**")
            gaps_html = "".join(f'<span class="gap-tag">{g}</span>' for g in gaps)
            st.markdown(f'<div>{gaps_html}</div>', unsafe_allow_html=True)

        actions = ev.get("suggested_actions", [])
        if actions:
            with st.expander(f"Suggested Actions ({len(actions)})", expanded=False):
                for j, a in enumerate(actions, 1):
                    st.markdown(f"{j}. {a}")

        st.markdown('</div>', unsafe_allow_html=True)  # close eval-section
        st.markdown('</div>', unsafe_allow_html=True)  # close round-card


def main():
    st.title("Agentic RAG")
    st.caption("Plan-Execute-Evaluate closed-loop multi-hop Q&A")

    # Sidebar configuration
    with st.sidebar:
        st.subheader("Configuration")

        dataset = st.selectbox(
            "Dataset",
            options=["hotpotqa", "2wikimultihopqa", "musique"],
            index=0,
            key="agentic_live_dataset",
        )

        model = st.selectbox(
            "Model",
            options=["qwen3.5-plus", "qwen-plus", "qwen-max", "qwen3-max"],
            index=0,
            key="agentic_live_model",
        )

        rewrite_model = st.selectbox(
            "Rewrite Model (optional)",
            options=["(same as Model)", "qwen3.5-plus", "qwen-plus", "qwen-max"],
            index=0,
            key="agentic_live_rewrite_model",
        )

        rerank = st.toggle("Enable Reranker", value=False, key="agentic_live_rerank")
        max_rounds = st.slider("Max Rounds", 1, 10, 5, key="agentic_live_max_rounds")

    # Query input
    query = st.text_area(
        "Enter your question",
        placeholder="e.g., In what year was the university where Tokarev was a professor founded?",
        height=80,
        key="agentic_live_query",
    )

    col_run, col_clear = st.columns([1, 4])
    run_clicked = col_run.button("Run Query", type="primary", disabled=not query.strip())
    col_clear.button("Clear")

    # Run query
    if run_clicked and query.strip():
        start_time = time.time()

        with st.status("Running...", expanded=True) as status:
            try:
                status.write("Initializing models and connecting to Milvus...")

                rm = rewrite_model if rewrite_model != "(same as Model)" else None

                status.write(f"Running Agentic RAG (max_rounds={max_rounds})...")
                result = _run_agentic_rag_live(
                    query=query.strip(),
                    dataset=dataset,
                    model=model,
                    rerank=rerank,
                    rewrite_model=rm,
                    max_rounds=max_rounds,
                )
                st.session_state["agentic_live_result"] = result

                elapsed = time.time() - start_time
                status.update(label=f"Done ({elapsed:.1f}s)", state="complete")

            except Exception as e:
                status.update(label=f"Error: {e}", state="error")
                st.error(f"Query failed: {e}")
                st.exception(e)
                return

    # Display results
    live_result = st.session_state.get("agentic_live_result")

    if live_result:
        st.markdown("---")
        st.markdown("### Agentic RAG Result")

        # Final answer
        answer = live_result.get("answer", "")
        if answer:
            st.markdown("### Answer")
            st.success(answer)

        # Metadata
        total_rounds = live_result.get("total_rounds", 0)
        all_chunks = live_result.get("all_chunks", [])
        meta_col1, meta_col2, meta_col3 = st.columns(3)
        meta_col1.metric("Rounds", total_rounds)
        meta_col2.metric("Unique Chunks", len(all_chunks))
        elapsed = time.time() - start_time if 'start_time' in dir() else None

        st.markdown("### Exploration Chain")
        history = live_result.get("exploration_history", [])
        if history:
            _render_exploration_chain(history)
        else:
            st.info("No exploration history available.")


main()
