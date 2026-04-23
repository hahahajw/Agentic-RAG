"""RAG with Judge — 节点函数与递归入口

节点函数（供 LangGraph StateGraph 调用）：
- rewrite_query_node: 生成多个查询变体（复用 Naive RAG prompt）
- batch_retrieve_node: 并行多路检索 + RRF 融合（等效 Schema A）
- judge_node: 判断 chunks 是否足以回答问题

递归入口：
- rag_with_judge(): Python 递归函数，控制整棵搜索树的探索
"""

import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from langchain_core.documents import Document
from langchain_core.runnables import RunnableConfig

from rag_with_judge.state import JudgeRAGState
from rag_with_judge.prompts import get_judge_prompt, get_answer_prompt
from naive_rag.prompts import QUERY_REWRITE_PROMPT

logger = logging.getLogger(__name__)


# ─── 辅助函数：从 config 获取 LLM 实例 ─────────────────────────

def _get_rewrite_llm(config: RunnableConfig):
    """获取查询重写 LLM，优先使用 rewrite_llm，否则使用默认 llm。"""
    return config["configurable"].get("rewrite_llm") or _get_llm(config)


def _get_judge_llm(config: RunnableConfig):
    """获取 Judge LLM，优先使用 judge_llm，否则使用默认 llm。"""
    return config["configurable"].get("judge_llm") or _get_llm(config)


def _get_answer_llm(config: RunnableConfig):
    """获取回答 LLM，优先使用 answer_llm，否则使用默认 llm。"""
    return config["configurable"].get("answer_llm") or _get_llm(config)


def _get_llm(config: RunnableConfig):
    """获取默认 LLM。"""
    return config["configurable"]["llm"]


def _get_retriever(config: RunnableConfig):
    """获取 MilvusRetriever 实例。"""
    return config["configurable"]["retriever"]


def _get_topk_propositions(config: RunnableConfig) -> int:
    return config["configurable"].get("topk_propositions", 50)


def _get_max_chunks(config: RunnableConfig) -> int:
    return config["configurable"].get("max_chunks", 8)


def _get_use_reranker(config: RunnableConfig) -> bool:
    return config["configurable"].get("use_reranker", False)


# ─── LangGraph 节点函数 ────────────────────────────────────────

def rewrite_query_node(state: JudgeRAGState, config: RunnableConfig) -> dict:
    """节点：生成多个查询变体（复用 Naive RAG prompt，JSON 数组输出）。

    返回: {"rewritten_queries": [str, ...]}
    """
    query = state["query"]
    llm = _get_rewrite_llm(config)

    prompt = QUERY_REWRITE_PROMPT.format(query=query)
    response = llm.invoke(prompt)
    content = response.content.strip()

    # 解析 JSON 输出（含 markdown code block fallback）
    try:
        variants = json.loads(content)
    except json.JSONDecodeError:
        if "```" in content:
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
        variants = json.loads(content.strip())

    assert isinstance(variants, list), f"Expected list, got {type(variants)}"

    logger.info("查询重写完成，原始='%s'，变体=%d", query, len(variants))
    return {"rewritten_queries": variants}


def batch_retrieve_node(state: JudgeRAGState, config: RunnableConfig) -> dict:
    """节点：并行多路检索 + RRF 融合 + 截断到 max_chunks。

    等效于 Naive RAG Schema A 的 fan_out → retrieve → fuse_results。

    返回: {"chunks": [dict, ...]}（按 RRF 分数降序，截断到 max_chunks）
    """
    query = state["query"]
    rewritten = state.get("rewritten_queries", [])
    retriever = _get_retriever(config)
    max_chunks = _get_max_chunks(config)
    RRF_K = 60

    # 去重查询列表（原始 + 变体）
    all_queries = [query] + rewritten
    seen = set()
    unique_queries = [q for q in all_queries if q not in seen and not seen.add(q)]

    # 并行检索：每个 query 独立检索
    with ThreadPoolExecutor(max_workers=len(unique_queries)) as executor:
        futures = {executor.submit(retriever.get_similar_chunk_with_score, q): q for q in unique_queries}
        results_by_query = {}
        for f in as_completed(futures):
            results_by_query[futures[f]] = f.result()

    # RRF 融合：score = sum(1/(RRF_K + rank))，按 chunk_id 合并
    chunk_scores: dict[str, float] = {}
    chunk_docs: dict[str, Document] = {}
    for q, results in results_by_query.items():
        for rank, (doc, _) in enumerate(results, 1):
            cid = doc.metadata.get("chunk_id")
            if not cid:
                continue
            rrf_score = 1.0 / (RRF_K + rank)
            chunk_scores[cid] = chunk_scores.get(cid, 0) + rrf_score
            if cid not in chunk_docs:
                chunk_docs[cid] = doc

    # 按 RRF 分数排序 + 截断
    sorted_chunks = sorted(chunk_scores.items(), key=lambda x: x[1], reverse=True)
    top_chunks = sorted_chunks[:max_chunks]

    chunks_out = []
    for cid, rrf_score in top_chunks:
        doc = chunk_docs[cid]
        chunks_out.append({
            "chunk_id": cid,
            "chunk_title": doc.metadata.get("chunk_title", ""),
            "chunk_summary": doc.metadata.get("chunk_summary", ""),
            "context_title": doc.metadata.get("context_title", ""),
            "page_content": doc.page_content,
            "score": round(rrf_score, 6),
        })

    logger.info(
        "并行检索完成，query 数=%d，RRF 融合后 chunks=%d（截断到 %d）",
        len(unique_queries), len(chunks_out), max_chunks
    )
    return {"chunks": chunks_out}


def judge_node(state: JudgeRAGState, config: RunnableConfig) -> dict:
    """节点：全局 Judge，判断 chunks 是否足以回答问题。

    返回: {"answerable": bool, "next_queries": list[str], "judgement_reason": str}
    """
    from pydantic import BaseModel, Field
    from typing import List
    from langchain_core.messages import SystemMessage
    from rag_with_judge.prompts import get_judge_prompt

    query = state["query"]
    chunks = state["chunks"]
    llm = _get_judge_llm(config)
    variant = config["configurable"].get("judge_variant", "B")

    # 格式化 chunks 文本
    chunks_text = _format_chunks_for_judge(chunks)

    # 优先使用 config 中的自定义 prompt
    custom_prompt = config["configurable"].get("custom_judge_prompt")
    if custom_prompt:
        prompt = SystemMessage(content=custom_prompt.format(
            question=query,
            chunks_text=chunks_text,
        ))
    else:
        prompt = get_judge_prompt(question=query, chunks_text=chunks_text, variant=variant)

    # 结构化输出
    class JudgeResult(BaseModel):
        answerable: bool = Field(description="当前知识是否足以完整回答问题")
        reason: str = Field(description="判断理由")
        next_queries: List[str] = Field(description="需要补充知识的 follow-up 查询")

    structured_llm = llm.with_structured_output(JudgeResult)
    judgement = structured_llm.invoke([prompt])

    logger.info(
        "Judge 判断 (variant=%s): query='%s', answerable=%s, next_queries=%d, reason='%s'",
        variant, query, judgement.answerable, len(judgement.next_queries), judgement.reason
    )

    return {
        "answerable": judgement.answerable,
        "next_queries": judgement.next_queries,
        "judgement_reason": judgement.reason,
    }


def _format_chunks_for_judge(chunks: list[dict]) -> str:
    """将 chunks 格式化为 Judge 可读的文本。"""
    if not chunks:
        return "(no knowledge retrieved)"

    parts = []
    for i, c in enumerate(chunks, 1):
        title = c.get("chunk_title", "Unknown")
        summary = c.get("chunk_summary", "")
        content = c.get("page_content", "")
        context = c.get("context_title", "")

        part = f"--- Chunk {i} ---\n"
        part += f"Title: {title}\n"
        if context:
            part += f"Context: {context}\n"
        if summary:
            part += f"Summary: {summary}\n"
        part += f"Content: {content}\n"
        parts.append(part)

    return "\n".join(parts)


# ─── 递归入口 ──────────────────────────────────────────────────

def rag_with_judge(
    query: str,
    path: dict,
    visited: set[str],
    depth: int,
    max_depth: int,
    app,               # CompiledStateGraph
    config: RunnableConfig,
) -> str:
    """递归探索搜索树。

    Args:
        query: 当前问题
        path: SEARCH_PATH 的当前节点（通过引用传递，子节点修改自动反映在父节点）
        visited: 已探索过的问题集合（归一化后），用于防止循环
        depth: 当前递归深度（0 表示根节点）
        max_depth: 最大允许深度
        app: 编译好的 LangGraph StateGraph
        config: LangGraph 运行时配置

    Returns:
        最终答案字符串
    """
    # 1. 初始化 path
    path["question"] = query

    # 2. 去重
    normalized = _normalize_query(query)
    if normalized in visited:
        logger.info("跳过重复问题 (depth=%d): '%s'", depth, query[:80])
        path["answerable"] = False
        path["chunks"] = []
        path["next_queries"] = []
        return ""
    visited.add(normalized)

    # 3. 调用单层 LangGraph: rewrite → retrieve → judge
    logger.info("[depth=%d] 开始探索: '%s'", depth, query[:80])
    start = time.perf_counter()

    result = app.invoke(
        {"query": query, "messages": []},
        config=config,
    )

    latency = (time.perf_counter() - start) * 1000
    logger.info("[depth=%d] 单层执行耗时: %.0fms", depth, latency)

    # 4. 写入 path
    path["chunks"] = result.get("chunks", [])
    path["answerable"] = result.get("answerable", False)

    # 5. 如果不能回答，并行探索 follow-up
    if not path["answerable"] and depth < max_depth:
        follow_ups = result.get("next_queries", [])
        logger.info(
            "[depth=%d] answerable=False，生成 %d 个 follow-up: %s",
            depth, len(follow_ups), follow_ups
        )
        children = [None] * len(follow_ups)

        def _run_follow(idx: int, next_q: str, child: dict):
            if not next_q or _normalize_query(next_q) in visited:
                logger.info("[depth=%d] 跳过空/重复 follow-up: '%s'", depth, next_q)
                return
            rag_with_judge(
                next_q, child, visited, depth + 1, max_depth, app, config
            )
            if child:
                children[idx] = child

        # 深度感知 worker 限制：越深层 worker 越少，防止 API 洪水
        # depth+1 是子节点深度，depth=0 时子节点最多 5 worker，depth=4 时退化为 1
        max_workers = max(1, 6 - (depth + 1))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for i, next_q in enumerate(follow_ups):
                child = {}
                futures.append(executor.submit(_run_follow, i, next_q, child))
            for f in as_completed(futures):
                f.result()  # 传播子节点异常

        path["next_queries"] = [c for c in children if c is not None]

    # 6. 生成答案（叶子节点或根节点）
    if path["answerable"] or depth == 0:
        path["answer"] = generate_answer(path, config)
    else:
        path["answer"] = None

    logger.info(
        "[depth=%d] 探索完成: answerable=%s, children=%d, answer=%s",
        depth, path["answerable"],
        len(path.get("next_queries", [])),
        bool(path.get("answer"))
    )
    return path.get("answer", "")


def generate_answer(path: dict, config: RunnableConfig) -> str:
    """基于 SEARCH_PATH 综合生成最终答案。

    Args:
        path: SEARCH_PATH 节点（包含 question, chunks, answerable, next_queries）
        config: 运行时配置

    Returns:
        答案字符串
    """
    llm = _get_answer_llm(config)
    search_path_text = _format_search_path_for_answer(path)
    question = path.get("question", "")
    cot = config["configurable"].get("cot", False)
    messages = get_answer_prompt(search_path_text, question, cot=cot)

    response = llm.invoke(messages)
    answer = response.content.strip()
    logger.info("答案生成完成 (question='%s...', len=%d)", question[:50], len(answer))
    return answer


def generate_answer_stream(path: dict, config: RunnableConfig):
    """流式版本：基于 SEARCH_PATH 逐 token 生成答案。

    返回: generator，逐个产出 token 内容。
    """
    llm = _get_answer_llm(config)
    search_path_text = _format_search_path_for_answer(path)
    question = path.get("question", "")
    cot = config["configurable"].get("cot", False)
    messages = get_answer_prompt(search_path_text, question, cot=cot)

    for chunk in llm.stream(messages):
        content = chunk.content if hasattr(chunk, 'content') else str(chunk)
        if content:
            yield content


def _format_search_path_for_answer(path: dict, indent: str = "") -> str:
    """将 SEARCH_PATH 递归格式化为可读文本，供 LLM 生成答案。"""
    lines = []
    q = path.get("question", "")
    answerable = path.get("answerable", False)
    answer = path.get("answer")
    chunks = path.get("chunks", [])
    next_queries = path.get("next_queries", [])

    lines.append(f"{indent}## Question: {q}")
    lines.append(f"{indent}  Answerable: {answerable}")

    # 如果有子节点，先递归子节点（让 LLM 看到探索路径）
    if next_queries:
        lines.append(f"{indent}  Follow-up Exploration:")
        for i, child in enumerate(next_queries, 1):
            lines.append(f"{indent}  ### Sub-question {i}:")
            lines.append(_format_search_path_for_answer(child, indent + "  "))

    # 展示 chunks（title + summary + content）
    if chunks:
        lines.append(f"{indent}  Retrieved Knowledge:")
        for i, c in enumerate(chunks, 1):
            title = c.get("chunk_title", "Unknown")
            summary = c.get("chunk_summary", "")
            content = c.get("page_content", "")
            context = c.get("context_title", "")
            entry = f"{indent}    [{i}] {title}"
            if context:
                entry += f" (Context: {context})"
            lines.append(entry)
            if summary:
                lines.append(f"{indent}        Summary: {summary}")
            lines.append(f"{indent}        Content: {content[:500]}")
            if len(content) > 500:
                lines.append(f"{indent}        ...(truncated)")

    # 如果有子节点生成的中间答案
    if next_queries:
        lines.append(f"{indent}  Sub-question Answers:")
        for i, child in enumerate(next_queries, 1):
            child_ans = child.get("answer", "")
            lines.append(f"{indent}    [{i}] {child_ans or '(no answer)'}")

    # 叶子节点的中间答案
    if answerable and answer:
        lines.append(f"{indent}  Intermediate Answer: {answer}")

    return "\n".join(lines)


def _normalize_query(query: str) -> str:
    """归一化查询字符串，用于去重。"""
    return query.strip().lower().rstrip("?")
