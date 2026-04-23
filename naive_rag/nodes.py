"""LangGraph node 函数 — 问题重写、检索、融合、回答、后续问题"""

import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Tuple

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.types import Send

from .prompts import QUERY_REWRITE_PROMPT, FOLLOWUP_PROMPT
from .state import NaiveRAGState

logger = logging.getLogger(__name__)

# --- Scheme A 配置 ---
RRF_K = 60

# --- Scheme B 配置 (AnnSearchRequest-Level Fusion) ---
_OUTPUT_FIELDS = [
    "id", "chunk_id", "question_id", "context_index", "context_title",
    "chunk_title", "chunk_summary", "proposition_text",
]


def _get_llm(config: RunnableConfig):
    """从 config 获取 LLM（通用 fallback）"""
    return config["configurable"]["llm"]


def _get_rewrite_llm(config: RunnableConfig):
    """问题重写 LLM，优先取专用配置，回退到通用 llm"""
    return config["configurable"].get("rewrite_llm") or _get_llm(config)


def _get_answer_llm(config: RunnableConfig):
    """回答生成 LLM，优先取专用配置，回退到通用 llm"""
    return config["configurable"].get("answer_llm") or _get_llm(config)


def _get_suggest_llm(config: RunnableConfig):
    """后续问题生成 LLM，优先取专用配置，回退到通用 llm"""
    return config["configurable"].get("suggest_llm") or _get_llm(config)


def _get_dataset_type(config: RunnableConfig) -> str:
    return config["configurable"].get("dataset_type", "")


def _get_topk_propositions(config: RunnableConfig) -> int:
    return config["configurable"].get("topk_propositions", 50)


def _get_max_chunks(config: RunnableConfig) -> int:
    return config["configurable"].get("max_chunks", 8)


def _get_use_reranker(config: RunnableConfig) -> bool:
    return config["configurable"].get("use_reranker", False)


def _get_reranker_params(config: RunnableConfig) -> dict:
    return config["configurable"].get("reranker_params", {"model_name": "qwen3-rerank"})


# ─── Node: rewrite_query ───────────────────────────────────────

def rewrite_query(state: NaiveRAGState, config: RunnableConfig) -> dict:
    """LLM 生成 4 个重写变体"""
    llm = _get_rewrite_llm(config)
    custom_prompt = config["configurable"].get("custom_rewrite_prompt")
    if custom_prompt:
        prompt = custom_prompt.format(query=state["original_query"])
    else:
        prompt = QUERY_REWRITE_PROMPT.format(query=state["original_query"])
    response = llm.invoke(prompt)
    content = response.content.strip()

    # 解析 JSON 输出
    try:
        # 尝试直接解析
        rewritten = json.loads(content)
    except json.JSONDecodeError:
        # 尝试从 markdown code block 中提取
        if "```" in content:
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
        rewritten = json.loads(content.strip())

    assert isinstance(rewritten, list), f"Expected list, got {type(rewritten)}"
    all_queries = [state["original_query"]] + rewritten

    logger.info("Original: %s", state["original_query"])
    logger.info("Rewritten: %s", rewritten)

    return {
        "rewritten_queries": rewritten,
        "all_queries": all_queries,
    }


# ─── Node: fan_out_retrieval (Scheme A 路由) ───────────────────

def fan_out_retrieval(state: NaiveRAGState) -> list[Send]:
    """动态 fan-out: 为每个 query 创建一个检索任务"""
    return [
        Send("retrieve_for_query", {"query": q})
        for q in state["all_queries"]
    ]


# ─── Node: retrieve_for_query (Scheme A) ───────────────────────

def retrieve_for_query(state: dict, config: RunnableConfig) -> dict:
    """
    通过 Send 接收 {"query": str}。
    调用检索器（config 注入或 MilvusRetriever）检索单个 query 的结果。
    """
    from Retrieval.milvus_retriever import MilvusRetriever

    query = state["query"]
    dataset_type = _get_dataset_type(config)
    topk_props = _get_topk_propositions(config)
    max_chunks = _get_max_chunks(config)
    use_reranker = _get_use_reranker(config)
    reranker_params = _get_reranker_params(config)

    # 优先使用 config 注入的 retriever（支持 WebRetriever 等替代实现）
    retriever = config["configurable"].get("retriever")
    if retriever is None:
        retriever = MilvusRetriever(
            dataset_type=dataset_type,
            topk_propositions=topk_props,
            max_chunks=max_chunks,
            use_reranker=use_reranker,
            reranker_params=reranker_params,
        )
    results = retriever.get_similar_chunk_with_score(query=query)
    logger.info("Query '%s' retrieved %d chunks", query, len(results))

    return {"retrieval_results": {query: results}}


# ─── Node: fuse_results (Scheme A) ─────────────────────────────

def fuse_results(state: NaiveRAGState, config: RunnableConfig) -> dict:
    """Client-Side RRF 融合: 遍历所有 query 结果，计算 RRF 分数"""
    chunk_scores: Dict[str, float] = {}
    chunk_docs: Dict[str, Document] = {}

    for query, results in state["retrieval_results"].items():
        for rank, (doc, _score) in enumerate(results, 1):
            cid = doc.metadata.get("chunk_id", "")
            if not cid:
                continue
            rrf_score = 1.0 / (RRF_K + rank)
            chunk_scores[cid] = chunk_scores.get(cid, 0) + rrf_score
            if cid not in chunk_docs:
                chunk_docs[cid] = doc

    sorted_chunks = sorted(chunk_scores.items(), key=lambda x: x[1], reverse=True)
    max_chunks = _get_max_chunks(config)
    fused = [(chunk_docs[cid], score) for cid, score in sorted_chunks[:max_chunks]]

    logger.info("Fused %d unique chunks via RRF (limited to %d)", len(fused), max_chunks)
    return {"fused_chunks": fused}


# ─── Node: batch_retrieve (Scheme B) ───────────────────────────

def batch_retrieve(state: NaiveRAGState, config: RunnableConfig) -> dict:
    """
    AnnSearchRequest-Level Fusion: 单次 hybrid_search 调用。
    为每个 query 创建 dense + sparse AnnSearchRequest，合并到一次调用中。
    注：当 config 注入 retriever（如 WebRetriever）时，退化为多路检索 + RRF 融合。
    """
    # 优先使用 config 注入的 retriever（支持 WebRetriever 等替代实现）
    retriever = config["configurable"].get("retriever")
    if retriever is not None:
        # WebRetriever 路径：多路检索 + RRF 融合
        from Retrieval.milvus_retriever import MilvusRetriever

        RRF_K = 60
        max_chunks = _get_max_chunks(config)
        with ThreadPoolExecutor(max_workers=len(state["all_queries"])) as executor:
            futures = {
                executor.submit(retriever.get_similar_chunk_with_score, q): q
                for q in state["all_queries"]
            }
            results_by_query = {}
            for f in as_completed(futures):
                results_by_query[futures[f]] = f.result()

        chunk_scores: dict[str, float] = {}
        chunk_docs: dict[str, Document] = {}
        for q, chunk_results in results_by_query.items():
            for rank, (doc, _) in enumerate(chunk_results, 1):
                cid = doc.metadata.get("chunk_id")
                if not cid:
                    continue
                rrf_score = 1.0 / (RRF_K + rank)
                chunk_scores[cid] = chunk_scores.get(cid, 0) + rrf_score
                if cid not in chunk_docs:
                    chunk_docs[cid] = doc

        sorted_chunks = sorted(chunk_scores.items(), key=lambda x: x[1], reverse=True)
        fused = [(chunk_docs[cid], score) for cid, score in sorted_chunks[:max_chunks]]
        logger.info("Scheme B (web): fused %d chunks", len(fused))
        return {"fused_chunks": fused}

    # 原有 Milvus 路径
    from pymilvus import MilvusClient, AnnSearchRequest, RRFRanker

    from Index.milvus_config import MILVUS_URI, MILVUS_TOKEN, get_collection_name, get_embedding_function

    dataset_type = _get_dataset_type(config)
    collection_name = get_collection_name(dataset_type)
    topk_props = _get_topk_propositions(config)
    max_chunks = _get_max_chunks(config)
    embedding_func = get_embedding_function()

    # 连接 Milvus
    connect_kwargs = {"uri": MILVUS_URI}
    if MILVUS_TOKEN:
        connect_kwargs["token"] = MILVUS_TOKEN
    client = MilvusClient(**connect_kwargs)

    # 为每个 query 创建 dense + sparse AnnSearchRequest
    all_requests: list[AnnSearchRequest] = []
    for query in state["all_queries"]:
        query_embedding = embedding_func.embed_query(query)
        dense_req = AnnSearchRequest(
            data=[query_embedding],
            anns_field="embedding",
            param={"metric_type": "COSINE", "params": {"ef": 64}},
            limit=topk_props,
        )
        sparse_req = AnnSearchRequest(
            data=[query],
            anns_field="sparse_embedding",
            param={"metric_type": "BM25"},
            limit=topk_props,
        )
        all_requests.extend([dense_req, sparse_req])

    logger.info("Scheme B: hybrid_search with %d AnnSearchRequests (%d queries)",
                len(all_requests), len(state["all_queries"]))

    # 单次 hybrid_search, RRF 融合所有请求
    results = client.hybrid_search(
        collection_name=collection_name,
        reqs=all_requests,
        ranker=RRFRanker(RRF_K),
        limit=topk_props,
        output_fields=_OUTPUT_FIELDS,
    )

    if not results or len(results) == 0:
        logger.warning("No propositions retrieved in Scheme B")
        return {"fused_chunks": []}

    # 解析 proposition 级结果
    prop_results: List[Tuple[Document, float]] = []
    for hit in results[0]:
        meta = dict(hit.fields)
        meta["id"] = hit.id
        doc = Document(
            page_content=hit.fields.get("proposition_text", ""),
            metadata=meta,
        )
        prop_results.append((doc, hit.distance))

    # 按 chunk_id 聚合 + enrich + assemble（复用 MilvusRetriever）
    from Retrieval.milvus_retriever import MilvusRetriever

    use_reranker = _get_use_reranker(config)
    reranker_params = _get_reranker_params(config)

    retriever = MilvusRetriever(
        dataset_type=dataset_type,
        topk_propositions=topk_props,
        max_chunks=max_chunks,
        use_reranker=use_reranker,
        reranker_params=reranker_params,
    )
    all_chunks = retriever._aggregate_by_chunk(prop_results)
    fused = all_chunks[:max_chunks]

    # 测试是否正确使用了 Reranker
    if use_reranker:
        logger.info(f"Rerank params: {reranker_params}")

    logger.info("Scheme B: fused %d chunks", len(fused))
    return {"fused_chunks": fused}


# ─── Node: generate_answer ─────────────────────────────────────
# Do NOT use any external knowledge or world knowledge
RAG_SYS_PROMPT = """\
You are a RAG Q&A assistant. Answer questions using ONLY the knowledge snippets provided below. Do NOT use any external knowledge.

RULES:
1. **Grounding**: Use only the provided knowledge snippets. If the snippets contain no information that can answer the question, respond with exactly: "I cannot answer this question."
2. **Directness**: Output ONLY the final answer — no explanations, no reasoning steps, no references, no polite language, no introductory phrases like "The answer is" or "Based on the information."
3. **Format**: Match the expected answer format precisely:
   - Years: output just the number (e.g., "1755")
   - Names: output the name as given (e.g., "John André")
   - Yes/No: output "yes" or "no" (lowercase)
   - Locations: output the place name (e.g., "Nairobi, Kenya")
   - Organizations: output the name (e.g., "Royal Air Force")
4. **Irrelevance**: If the knowledge snippets are unrelated to the question, respond with exactly: "I cannot answer this question."

Now answer the following question using ONLY the provided knowledge snippets.
"""


def generate_answer(state: NaiveRAGState, config: RunnableConfig) -> dict:
    """LLM 基于融合的 chunks 生成回答"""
    llm = _get_answer_llm(config)
    docs = [doc.page_content for doc, _ in state["fused_chunks"]]
    context = "\n\n".join(f"Knowledge {i}: {d}" for i, d in enumerate(docs, 1))

    custom_prompt = config["configurable"].get("custom_answer_prompt")
    if custom_prompt:
        system_msg = SystemMessage(content=custom_prompt)
    else:
        system_msg = SystemMessage(content=RAG_SYS_PROMPT)
    human_msg = HumanMessage(content=f"[Knowledge]\n{context}\n[question]\n{state['original_query']}")

    response = llm.invoke([system_msg, human_msg])
    return {"messages": [response], "answer": response.content}


def generate_answer_stream(state: NaiveRAGState, config: RunnableConfig):
    """流式版本：LLM 基于融合的 chunks 生成回答，逐 token yield。"""
    llm = _get_answer_llm(config)
    docs = [doc.page_content for doc, _ in state["fused_chunks"]]
    context = "\n\n".join(f"Knowledge {i}: {d}" for i, d in enumerate(docs, 1))

    custom_prompt = config["configurable"].get("custom_answer_prompt")
    if custom_prompt:
        system_msg = SystemMessage(content=custom_prompt)
    else:
        system_msg = SystemMessage(content=RAG_SYS_PROMPT)
    human_msg = HumanMessage(content=f"[Knowledge]\n{context}\n[question]\n{state['original_query']}")

    for chunk in llm.stream([system_msg, human_msg]):
        content = chunk.content if hasattr(chunk, 'content') else str(chunk)
        if content:
            yield content


# ─── Node: suggest_followups ───────────────────────────────────

def suggest_followups(state: NaiveRAGState, config: RunnableConfig) -> dict:
    """LLM 生成 3 个后续问题建议"""
    # 感觉传递整个 messages state 更好些
    llm = _get_suggest_llm(config)
    prompt = FOLLOWUP_PROMPT.format(
        original_query=state["original_query"],
        answer=state["answer"],
    )
    response = llm.invoke(prompt)
    content = response.content.strip()

    try:
        followups = json.loads(content)
    except json.JSONDecodeError:
        if "```" in content:
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
        followups = json.loads(content.strip())

    assert isinstance(followups, list), f"Expected list, got {type(followups)}"
    return {"suggested_followups": followups}
