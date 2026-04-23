"""LangGraph StateGraph 构建 — Scheme A (Client-Side RRF) 和 Scheme B (AnnSearchRequest-Level Fusion)"""

from langgraph.graph import END, START, StateGraph

from .nodes import (
    fan_out_retrieval,
    fuse_results,
    generate_answer,
    rewrite_query,
    retrieve_for_query,
    batch_retrieve,
    suggest_followups,
)
from .state import NaiveRAGState


def build_scheme_a_graph(skip_suggest: bool = False):
    """
    Scheme A: Client-Side RRF Fusion
    START → rewrite → [fan-out] retrieve_for_query (parallel) → fuse → generate → [suggest] → END
    """
    graph = StateGraph(NaiveRAGState)

    # Nodes
    graph.add_node("rewrite", rewrite_query)
    graph.add_node("retrieve_for_query", retrieve_for_query)
    graph.add_node("fuse_results", fuse_results)
    graph.add_node("generate_answer", generate_answer)
    if not skip_suggest:
        graph.add_node("suggest_followups", suggest_followups)

    # Edges
    graph.add_edge(START, "rewrite")
    graph.add_conditional_edges(
        "rewrite",
        fan_out_retrieval,
        ["retrieve_for_query"],
    )
    graph.add_edge("retrieve_for_query", "fuse_results")
    graph.add_edge("fuse_results", "generate_answer")
    if skip_suggest:
        graph.add_edge("generate_answer", END)
    else:
        graph.add_edge("generate_answer", "suggest_followups")
        graph.add_edge("suggest_followups", END)

    return graph.compile()


def build_scheme_b_graph(skip_suggest: bool = False):
    """
    Scheme B: AnnSearchRequest-Level Fusion
    START → rewrite → batch_retrieve (single hybrid_search) → generate → [suggest] → END
    """
    graph = StateGraph(NaiveRAGState)

    # Nodes
    graph.add_node("rewrite", rewrite_query)
    graph.add_node("batch_retrieve", batch_retrieve)
    graph.add_node("generate_answer", generate_answer)
    if not skip_suggest:
        graph.add_node("suggest_followups", suggest_followups)

    # Edges
    graph.add_edge(START, "rewrite")
    graph.add_edge("rewrite", "batch_retrieve")
    graph.add_edge("batch_retrieve", "generate_answer")
    if skip_suggest:
        graph.add_edge("generate_answer", END)
    else:
        graph.add_edge("generate_answer", "suggest_followups")
        graph.add_edge("suggest_followups", END)

    return graph.compile()


def get_workflow(scheme: str = "a", skip_suggest: bool = False):
    """返回编译好的 workflow"""
    if scheme.lower() == "a":
        return build_scheme_a_graph(skip_suggest=skip_suggest)
    elif scheme.lower() == "b":
        return build_scheme_b_graph(skip_suggest=skip_suggest)
    else:
        raise ValueError(f"Unknown scheme: {scheme}. Use 'a' or 'b'.")
