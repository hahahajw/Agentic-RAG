"""Live query interface for Agentic RAG."""

import os
import sys
import time
from pathlib import Path

# Ensure project root is on sys.path so frontend package imports work
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import streamlit as st

from frontend.styles import DEPTH_COLORS, depth_color_style, inject_custom_css


def _get_llm(model: str, temperature: float = 0.0):
    """创建 LLM 实例"""
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(
        api_key=os.getenv("BL_API_KEY"),
        base_url=os.getenv("BL_BASE_URL"),
        model=model,
        temperature=temperature,
    )


def _doc_to_dict(doc, score: float) -> dict:
    """将 (Document, score) 转换为渲染用的 dict"""
    return {
        "chunk_id": doc.metadata.get("chunk_id", ""),
        "chunk_title": doc.metadata.get("chunk_title", ""),
        "chunk_summary": doc.metadata.get("chunk_summary", ""),
        "context_title": doc.metadata.get("context_title", ""),
        "content": doc.page_content,
        "score": round(score, 6),
        "aggregated_propositions": doc.metadata.get("aggregated_propositions"),
    }


def run_naive_rag(query: str, scheme: str, dataset: str, model: str, rerank: bool, rewrite_model: str | None = None):
    """运行 Naive RAG 查询"""
    from naive_rag.workflow import get_workflow

    llm = _get_llm(model)
    rewrite_llm = _get_llm(rewrite_model or model) if rewrite_model else llm

    app = get_workflow(scheme=scheme)

    config = {
        "configurable": {
            "llm": llm,
            "rewrite_llm": rewrite_llm,
            "answer_llm": llm,
            "suggest_llm": llm,
            "dataset_type": dataset,
            "topk_propositions": 50,
            "max_chunks": 8,
            "use_reranker": rerank,
            "reranker_params": {"model_name": "qwen3-rerank"} if rerank else {},
        }
    }

    result = app.invoke(
        {"original_query": query, "messages": []},
        config=config,
    )

    fused = result.get("fused_chunks", [])
    result["fused_chunks"] = [_doc_to_dict(doc, score) for doc, score in fused]

    return result


def run_rag_with_judge(query: str, dataset: str, model: str, max_depth: int, rerank: bool, rewrite_model: str | None = None):
    """运行 RAG with Judge 查询"""
    from rag_with_judge.nodes import rag_with_judge
    from rag_with_judge.workflow import build_judge_rag_graph

    llm = _get_llm(model)
    rewrite_llm = _get_llm(rewrite_model or model) if rewrite_model else llm

    from Retrieval.milvus_retriever import MilvusRetriever
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
            "rewrite_llm": rewrite_llm,
            "judge_llm": llm,
            "answer_llm": llm,
            "retriever": retriever,
            "max_chunks": 8,
            "judge_variant": "B",
        }
    }

    app = build_judge_rag_graph()
    search_path = {}
    visited = set()

    rag_with_judge(
        query=query,
        path=search_path,
        visited=visited,
        depth=0,
        max_depth=max_depth,
        app=app,
        config=config,
    )

    return search_path


def _chunk_card(chunk: dict, rank: int, depth: int = 0) -> None:
    """渲染单个片段卡片"""
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


def render_search_path_tree(search_path: dict, depth: int = 0) -> None:
    """递归渲染 SEARCH_PATH 搜索树"""
    if not search_path or "question" not in search_path:
        return

    question = search_path.get("question", "")
    answerable = search_path.get("answerable", False)
    answer = search_path.get("answer")
    chunks = search_path.get("chunks", [])
    children = search_path.get("next_queries", [])
    reason = search_path.get("judgement_reason", "")
    color = DEPTH_COLORS.get(depth % len(DEPTH_COLORS), DEPTH_COLORS[0])

    with st.container():
        st.markdown(
            f'<div class="tree-node" style="{depth_color_style(depth)}">',
            unsafe_allow_html=True,
        )

        badge_text = '<span class="judge-badge answerable">Answerable</span>' if answerable else '<span class="judge-badge unanswerable">Needs Follow-up</span>'
        st.markdown(
            f'<div class="node-header">'
            f'<span class="depth-badge" style="background:{color}">{depth}</span>'
            f'<span class="question-text">{question}</span>'
            f'{badge_text}'
            f'</div>',
            unsafe_allow_html=True,
        )

        if chunks:
            with st.expander(f"Retrieved Chunks ({len(chunks)})", expanded=(depth == 0)):
                for i, chunk in enumerate(chunks):
                    _chunk_card(chunk, rank=i + 1, depth=depth)

        if reason:
            st.info(f"**Judge Reason:** {reason}")

        if answer:
            st.success(f"**Answer:** {answer}")

        st.markdown('</div>', unsafe_allow_html=True)

        if children:
            with st.container():
                st.markdown(
                    f'<div class="children-container" style="{depth_color_style(depth)}">',
                    unsafe_allow_html=True,
                )
                for child in children:
                    render_search_path_tree(child, depth + 1)
                st.markdown('</div>', unsafe_allow_html=True)


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

        if i > 0:
            st.markdown('<div class="timeline-connector"></div>', unsafe_allow_html=True)

        st.markdown(f'<div class="round-card {status_class}">', unsafe_allow_html=True)

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

        plan = rd.get("plan", {})
        targets = plan.get("targets", [])
        if targets:
            tags = "".join(f'<span class="target-tag">{t}</span>' for t in targets)
            st.markdown(f'<div class="round-targets">{tags}</div>', unsafe_allow_html=True)

        hyps = plan.get("hypotheses", [])
        if hyps:
            with st.expander(f"Hypotheses ({len(hyps)})", expanded=False):
                for j, h in enumerate(hyps, 1):
                    st.markdown(f"{j}. {h}")

        reflection = plan.get("reflection")
        if reflection:
            with st.expander("Reflector Analysis", expanded=False):
                st.markdown(f"**Diagnosis:** {reflection.get('diagnosis', '')}")
                pivots = reflection.get("pivot_queries", [])
                if pivots:
                    st.markdown("**Pivot Queries:**")
                    for pq in pivots:
                        st.markdown(f"- `{pq}`")

        queries = rd.get("rewritten_queries", [])
        if queries:
            with st.expander(f"Queries ({len(queries)})", expanded=False):
                for j, q in enumerate(queries, 1):
                    st.markdown(f"{j}. `{q}`")

        chunks = rd.get("chunks", [])
        if chunks:
            with st.expander(f"Retrieved Chunks ({len(chunks)})", expanded=(rnd == 1)):
                for j, chunk in enumerate(chunks):
                    _chunk_card(chunk, rank=j + 1, depth=rnd)

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

        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)


def main():
    st.title("Live Query")
    st.caption("Run a live query against the Agentic RAG system")

    # Sidebar configuration
    with st.sidebar:
        st.subheader("Configuration")
        pipeline = st.radio(
            "Pipeline",
            options=["Naive RAG", "RAG with Judge", "Agentic RAG"],
            index=0,
            key="live_pipeline",
        )

        dataset = st.selectbox(
            "Dataset",
            options=["hotpotqa", "2wikimultihopqa", "musique"],
            index=0,
            key="live_dataset",
        )

        model = st.selectbox(
            "Model",
            options=["qwen3.5-plus", "qwen-plus", "qwen-max", "qwen3-max"],
            index=0,
            key="live_model",
        )

        rewrite_model = st.selectbox(
            "Rewrite Model (optional)",
            options=["(same as Model)", "qwen3.5-plus", "qwen-plus", "qwen-max"],
            index=0,
            key="live_rewrite_model",
        )

        rerank = st.toggle("Enable Reranker", value=False, key="live_rerank")

        if pipeline == "Naive RAG":
            scheme = st.radio("Scheme", options=["a", "b"], index=0, key="live_scheme")
            max_depth = None
            max_rounds = None
        elif pipeline == "RAG with Judge":
            scheme = None
            max_depth = st.slider("Max Depth", 1, 5, 3, key="live_max_depth")
            max_rounds = None
        else:  # Agentic RAG
            scheme = None
            max_depth = None
            max_rounds = st.slider("Max Rounds", 1, 10, 5, key="live_max_rounds")

    # Query input
    query = st.text_area(
        "Enter your question",
        placeholder="e.g., In what year was the university where Tokarev was a professor founded?",
        height=80,
        key="live_query",
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

                if pipeline == "Naive RAG":
                    status.write(f"Running Naive RAG (Scheme {scheme.upper()})...")
                    result = run_naive_rag(
                        query=query.strip(),
                        scheme=scheme,
                        dataset=dataset,
                        model=model,
                        rerank=rerank,
                        rewrite_model=rm,
                    )
                    st.session_state["live_result"] = result
                    st.session_state["live_type"] = "naive_rag"
                elif pipeline == "RAG with Judge":
                    status.write(f"Running RAG with Judge (max_depth={max_depth})...")
                    search_path = run_rag_with_judge(
                        query=query.strip(),
                        dataset=dataset,
                        model=model,
                        max_depth=max_depth,
                        rerank=rerank,
                        rewrite_model=rm,
                    )
                    st.session_state["live_result"] = search_path
                    st.session_state["live_type"] = "rag_with_judge"
                else:  # Agentic RAG
                    status.write(f"Running Agentic RAG (max_rounds={max_rounds})...")
                    result = _run_agentic_rag_live(
                        query=query.strip(),
                        dataset=dataset,
                        model=model,
                        rerank=rerank,
                        rewrite_model=rm,
                        max_rounds=max_rounds,
                    )
                    st.session_state["live_result"] = result
                    st.session_state["live_type"] = "agentic_rag"

                elapsed = time.time() - start_time
                status.update(label=f"Done ({elapsed:.1f}s)", state="complete")

            except Exception as e:
                status.update(label=f"Error: {e}", state="error")
                st.error(f"Query failed: {e}")
                st.exception(e)
                return

    # Display results
    live_result = st.session_state.get("live_result")
    live_type = st.session_state.get("live_type")

    if live_result and live_type == "naive_rag":
        st.markdown("---")
        st.markdown("### Naive RAG Result")

        rewritten = live_result.get("rewritten_queries", [])
        if rewritten:
            with st.expander(f"Rewritten Queries ({len(rewritten)})", expanded=False):
                for i, q in enumerate(rewritten):
                    st.markdown(f"{i + 1}. {q}")

        chunks = live_result.get("fused_chunks", [])
        if chunks:
            st.markdown(f"**Retrieved Chunks ({len(chunks)})**")
            for i, chunk in enumerate(chunks):
                _chunk_card(chunk, rank=i + 1)

        answer = live_result.get("answer", "")
        if answer:
            st.markdown("### Answer")
            st.success(answer)

        followups = live_result.get("suggested_followups", [])
        if followups:
            with st.expander(f"Follow-up Suggestions ({len(followups)})", expanded=False):
                for i, q in enumerate(followups):
                    st.markdown(f"{i + 1}. {q}")

    elif live_result and live_type == "rag_with_judge":
        st.markdown("---")
        st.markdown("### RAG with Judge Result")

        answer = live_result.get("answer", "")
        if answer:
            st.markdown("### Answer")
            st.success(answer)

        st.markdown("### Search Path")
        render_search_path_tree(live_result, depth=0)

    elif live_result and live_type == "agentic_rag":
        st.markdown("---")
        st.markdown("### Agentic RAG Result")

        answer = live_result.get("answer", "")
        if answer:
            st.markdown("### Answer")
            st.success(answer)

        total_rounds = live_result.get("total_rounds", 0)
        all_chunks = live_result.get("all_chunks", [])
        meta_col1, meta_col2 = st.columns(2)
        meta_col1.metric("Rounds", total_rounds)
        meta_col2.metric("Unique Chunks", len(all_chunks))

        st.markdown("### Exploration Chain")
        history = live_result.get("exploration_history", [])
        if history:
            _render_exploration_chain(history)
        else:
            st.info("No exploration history available.")


main()
