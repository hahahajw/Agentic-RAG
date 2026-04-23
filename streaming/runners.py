"""流式 Runner：调用各 RAG 模块的 generate_answer_stream 方法。

设计：runner 先执行完整的 graph（获取检索结果、chunks 等中间状态），
然后手动调用 generate_answer_stream 替代 invoke 返回的 answer，
通过 on_token 回调逐 token 推送。

原 generate_answer 节点仍会执行，其答案被丢弃（仅用于非流式兼容）。
"""

import logging
from typing import Any, Callable, Generator

logger = logging.getLogger(__name__)


# ── Naive RAG ──────────────────────────────────────────────────

def run_naive_rag_streaming(
    query: str,
    app: Any,
    config: dict,
    on_token: Callable[[str], None],
) -> dict:
    """流式运行 Naive RAG。

    先执行 graph 获取 fused_chunks，再流式生成答案。
    on_token 接收每个 token 片段。

    Returns:
        完整 result dict（含 answer 字段）
    """
    from naive_rag.nodes import generate_answer_stream
    from naive_rag.state import NaiveRAGState

    result = app.invoke(
        {"original_query": query, "messages": []},
        config=config,
    )

    # 流式生成答案（丢弃 invoke 返回的 answer，用流式版本替代）
    tokens = []
    for token in generate_answer_stream(result, config):
        tokens.append(token)
        on_token(token)

    result["answer"] = "".join(tokens)
    return result


# ── RAG with Judge ─────────────────────────────────────────────

def run_rag_with_judge_streaming(
    query: str,
    app: Any,
    config: dict,
    max_depth: int = 3,
    on_token: Callable[[str], None] = lambda _: None,
) -> tuple[str, dict]:
    """流式运行 RAG with Judge。

    先执行递归探索获取 SEARCH_PATH，再流式生成答案。

    Returns:
        (answer, search_path) 元组
    """
    from rag_with_judge.nodes import rag_with_judge, generate_answer_stream

    search_path: dict = {}
    visited: set = set()

    # 执行递归探索（answer 会被同步生成）
    _ = rag_with_judge(
        query=query,
        path=search_path,
        visited=visited,
        depth=0,
        max_depth=max_depth,
        app=app,
        config=config,
    )

    # 流式重新生成答案（如果 search_path 有 answerable 节点）
    if search_path and search_path.get("answer"):
        tokens = []
        for token in generate_answer_stream(search_path, config):
            tokens.append(token)
            on_token(token)
        return "".join(tokens), search_path

    return search_path.get("answer", "") or "", search_path


# ── Agentic RAG v1 ─────────────────────────────────────────────

def run_agentic_rag_streaming(
    query: str,
    app: Any,
    config: dict,
    max_rounds: int = 5,
    on_token: Callable[[str], None] = lambda _: None,
) -> dict:
    """流式运行 Agentic RAG v1。

    先执行 graph 获取 exploration_history，再流式生成答案。

    Returns:
        result dict（含 answer, exploration_history, total_rounds, all_chunks）
    """
    from agentic_rag.nodes import run_agentic_rag, generate_answer_stream
    from agentic_rag.state import AgenticRAGState

    # 执行完整闭环（同步生成 answer）
    result = run_agentic_rag(
        query=query,
        app=app,
        config=config,
        max_rounds=max_rounds,
    )

    # 构建 state 用于流式答案生成
    state: AgenticRAGState = {
        "query": query,
        "exploration_history": result["exploration_history"],
        "done": True,
        "messages": [],
        "plan": result.get("plan", {}),
    }

    # 流式重新生成答案
    tokens = []
    for token in generate_answer_stream(state, config):
        tokens.append(token)
        on_token(token)

    result["answer"] = "".join(tokens)
    return result


# ── Agentic RAG v2 ─────────────────────────────────────────────

def run_agentic_rag_v2_streaming(
    query: str,
    app: Any,
    config: dict,
    max_rounds: int = 5,
    on_token: Callable[[str], None] = lambda _: None,
) -> dict:
    """流式运行 Agentic RAG v2。

    先执行 graph 获取 sub_questions 和 exploration_history，
    再流式生成答案。

    Returns:
        result dict（含 answer, exploration_history, sub_questions, total_rounds, all_chunks）
    """
    from agentic_rag_v2.nodes import run_agentic_rag_v2, generate_answer_stream
    from agentic_rag_v2.state import AgenticRAGV2State

    # 执行完整闭环
    result = run_agentic_rag_v2(
        query=query,
        app=app,
        config=config,
        max_rounds=max_rounds,
    )

    # 构建 state 用于流式答案生成
    state: AgenticRAGV2State = {
        "query": query,
        "exploration_history": result["exploration_history"],
        "sub_questions": result.get("sub_questions", []),
        "done": True,
        "messages": [],
        "plan": result.get("plan", {}),
    }

    # 流式重新生成答案
    tokens = []
    for token in generate_answer_stream(state, config):
        tokens.append(token)
        on_token(token)

    result["answer"] = "".join(tokens)
    return result
