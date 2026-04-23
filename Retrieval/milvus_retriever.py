"""MilvusRetriever：Hybrid Search + Chunk 级聚合 + 可选 Reranker

检索流程（per-dataset collection）：
  Stage 1: Hybrid Search(dense COSINE + BM25 sparse, RRF 融合) → top K propositions
  Stage 2: 按 chunk_id 分组 → 每组取最高分 → 返回所有候选 chunk（不截断）
  Stage 3 (可选): Reranker(gte-rerank-v2) 对全部候选 chunk 重排
  截断: 最终取 top N chunks
"""

import json
import logging
import os
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple

import dashscope
from langchain_core.documents import Document
from pymilvus import MilvusClient, AnnSearchRequest, RRFRanker, WeightedRanker

from Index.milvus_config import (
    MILVUS_URI,
    MILVUS_TOKEN,
    get_collection_name,
    get_embedding_function,
    RETRIEVER_TOPK_PROPOSITIONS,
    RETRIEVER_MAX_CHUNKS,
)
from naive_rag.prompts import QUERY_REWRITE_PROMPT, QUERY_REWRITE_WITH_CONTEXT_PROMPT

logger = logging.getLogger(__name__)

_output_fields = [
    "id", "chunk_id", "question_id", "context_index", "context_title",
    "chunk_title", "chunk_summary", "proposition_text",
]


class MilvusRetriever:
    """
    两阶段 proposition 检索 + chunk 聚合。

    Args:
        dataset_type: 目标数据集（必填），决定查询哪个 collection
        topk_propositions: Stage 1 检索的 proposition 数量
        max_chunks: Stage 2 返回的唯一 chunk 数量
        ranker_type: hybrid search 的融合方式 ('rrf' | 'weighted')
        ranker_params: ranker 参数
    """

    def __init__(
        self,
        dataset_type: str,
        topk_propositions: int = RETRIEVER_TOPK_PROPOSITIONS,
        max_chunks: int = RETRIEVER_MAX_CHUNKS,
        ranker_type: str = "rrf",
        ranker_params: Optional[Dict] = None,
        use_reranker: bool = False,
        reranker_params: Optional[Dict] = None,
    ):
        self.dataset_type = dataset_type.lower()
        self.collection_name = get_collection_name(self.dataset_type)
        self.topk_propositions = topk_propositions
        self.max_chunks = max_chunks

        connect_kwargs = {"uri": MILVUS_URI}
        if MILVUS_TOKEN:
            connect_kwargs["token"] = MILVUS_TOKEN
        self._client = MilvusClient(**connect_kwargs)
        self._embedding_func = get_embedding_function()

        self._ranker_type = ranker_type
        self._ranker_params = ranker_params or {"k": 60}

        self.use_reranker = use_reranker
        self.reranker_params = reranker_params or {"model_name": "qwen3-rerank"}

    def get_similar_chunk_with_score(self, query: str) -> List[Tuple[Document, float]]:
        """
        主检索入口。返回 chunk 级结果（非 proposition 级）。

        Returns:
            [(Document, score), ...]
        """
        # Stage 1: Hybrid Search → top K propositions
        prop_results = self._hybrid_search(query)

        if not prop_results:
            logger.warning("No propositions retrieved for query: %s", query)
            return []

        # Stage 2: 按 chunk_id 分组 → 聚合为 chunk 级结果（不截断）
        all_chunks = self._aggregate_by_chunk(prop_results)

        # Stage 3 (可选): Reranker 重排全部候选 chunk
        if self.use_reranker:
            all_chunks = self._rerank_chunks(query, all_chunks)

        # Stage 4: 截断到 max_chunks
        return all_chunks[: self.max_chunks]

    def get_similar_chunks_with_rewrite(
        self,
        query: str,
        rewrite_llm,
        num_variants: int = 4,
        rewrite_context: str = None,
    ) -> List[Tuple[Document, float]]:
        """单 query 完整检索链：重写 → 独立检索 → RRF 融合。

        遵循 Naive RAG Schema A 模式：
        1. LLM 生成 num_variants 个重写变体
        2. 每个变体独立调用 get_similar_chunk_with_score()
        3. 对返回的 chunk 级结果做 RRF 融合
        4. 截断到 max_chunks

        Args:
            query: 原始查询
            rewrite_llm: 用于问题重写的 LLM 实例
            num_variants: 重写变体数量
            rewrite_context: 可选的已知事实上下文，用于生成更精确的 query

        Returns:
            [(Document, rrf_score), ...] — chunk 级结果，按 RRF 分数降序
        """
        RRF_K = 60

        # 1. LLM 问题重写（可选注入上下文）
        if rewrite_context:
            prompt = QUERY_REWRITE_WITH_CONTEXT_PROMPT.format(
                context=rewrite_context, query=query
            )
        else:
            prompt = QUERY_REWRITE_PROMPT.format(query=query)
        response = rewrite_llm.invoke(prompt)
        content = response.content.strip()
        try:
            variants = json.loads(content)
        except json.JSONDecodeError:
            if "```" in content:
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
            variants = json.loads(content.strip())
        assert isinstance(variants, list)

        all_queries = [query] + variants[:num_variants]
        logger.info("Query rewrite: '%s' → %d variants", query[:60], len(variants))

        # 2. 每个 query 独立调用 get_similar_chunk_with_score()
        with ThreadPoolExecutor(max_workers=len(all_queries)) as executor:
            futures = {
                executor.submit(self.get_similar_chunk_with_score, q): q
                for q in all_queries
            }
            results_by_query = {}
            for f in as_completed(futures):
                results_by_query[futures[f]] = f.result()

        # 3. RRF 融合（chunk 级，与 Schema A 一致）
        chunk_rrf: dict[str, float] = {}
        chunk_docs: dict[str, Document] = {}
        for q, chunk_results in results_by_query.items():
            for rank, (doc, _) in enumerate(chunk_results, 1):
                cid = doc.metadata.get("chunk_id")
                if not cid:
                    continue
                rrf = 1.0 / (RRF_K + rank)
                chunk_rrf[cid] = chunk_rrf.get(cid, 0) + rrf
                if cid not in chunk_docs:
                    chunk_docs[cid] = doc

        # 4. 按 RRF 分数排序 + 截断
        sorted_chunks = sorted(chunk_rrf.items(), key=lambda x: x[1], reverse=True)
        result = []
        for cid, rrf_score in sorted_chunks[: self.max_chunks]:
            result.append((chunk_docs[cid], round(rrf_score, 6)))

        return result

    def _hybrid_search(self, query: str) -> List[Tuple[Document, float]]:
        """Hybrid search: dense embedding + BM25 sparse, RRF fusion"""
        query_embedding = self._embedding_func.embed_query(query)

        dense_req = AnnSearchRequest(
            data=[query_embedding],
            anns_field="embedding",
            param={"metric_type": "COSINE", "params": {"ef": 64}},
            limit=self.topk_propositions,
        )
        sparse_req = AnnSearchRequest(
            data=[query],
            anns_field="sparse_embedding",
            param={"metric_type": "BM25"},
            limit=self.topk_propositions,
        )

        if self._ranker_type == "weighted":
            weights = self._ranker_params.get("weights", [1.0, 1.0])
            ranker = WeightedRanker(*weights)
        else:
            ranker = RRFRanker(self._ranker_params.get("k", 60))

        results = self._client.hybrid_search(
            collection_name=self.collection_name,
            reqs=[dense_req, sparse_req],
            ranker=ranker,
            limit=self.topk_propositions,
            output_fields=_output_fields,
        )

        # Parse results: hybrid_search returns list of lists
        docs_with_scores = []
        if results and len(results) > 0:
            for hit in results[0]:
                meta = dict(hit.fields)
                meta["id"] = hit.id
                doc = Document(
                    page_content=hit.fields.get("proposition_text", ""),
                    metadata=meta,
                )
                docs_with_scores.append((doc, hit.distance))

        return docs_with_scores

    def _enrich_chunk_props(self, chunk_ids: List[str]) -> Dict[str, List[Tuple[Document, float]]]:
        """从 Milvus 查询候选 chunk 的完整 propositions"""
        if not chunk_ids:
            return {}
        filter_expr = " or ".join(f'chunk_id == "{cid}"' for cid in chunk_ids)
        results = self._client.query(
            collection_name=self.collection_name,
            filter=filter_expr,
            output_fields=_output_fields,
            limit=5000,
        )
        groups: Dict[str, List[Tuple[Document, float]]] = defaultdict(list)
        for hit in results:
            cid = hit.get("chunk_id", "")
            meta = dict(hit)
            meta["id"] = hit.get("id")
            doc = Document(page_content=hit.get("proposition_text", ""), metadata=meta)
            groups[cid].append((doc, 0.0))  # query 无 score，分数已在 chunk_scores 中保留
        return dict(groups)

    def _aggregate_by_chunk(
        self,
        prop_results: List[Tuple[Document, float]],
    ) -> List[Tuple[Document, float]]:
        """
        按 chunk_id 分组，每组取最高分作为 chunk 代表分，
        按分数降序返回所有 chunk（不截断）。
        截断由调用方在 rerank 之后执行。
        """
        groups: Dict[str, List[Tuple[Document, float]]] = defaultdict(list)
        for doc, score in prop_results:
            cid = doc.metadata.get("chunk_id", "")
            if cid:
                groups[cid].append((doc, score))

        chunk_scores: Dict[str, float] = {}
        for cid, props in groups.items():
            chunk_scores[cid] = max(s for _, s in props)

        sorted_chunk_ids = sorted(chunk_scores, key=chunk_scores.get, reverse=True)

        # 从 Milvus 获取候选 chunk 的完整 propositions
        full_props_by_chunk = self._enrich_chunk_props(sorted_chunk_ids)

        result = []
        for cid in sorted_chunk_ids:
            if cid in full_props_by_chunk:
                chunk_doc = self._assemble_chunk_document(full_props_by_chunk[cid])
            else:
                logger.warning("Chunk %s not found in enrich query, using subset", cid[:12])
                chunk_doc = self._assemble_chunk_document(groups[cid])
            result.append((chunk_doc, chunk_scores[cid]))

        return result

    def _assemble_chunk_document(
        self,
        props_with_scores: List[Tuple[Document, float]],
    ) -> Document:
        """将一个 chunk 的所有 propositions 组装为单个 Document"""
        if not props_with_scores:
            return Document(page_content="", metadata={})

        first_meta = props_with_scores[0][0].metadata
        page_content = " ".join(doc.page_content for doc, _ in props_with_scores)

        metadata = dict(first_meta)
        metadata["aggregated_propositions"] = len(props_with_scores)

        return Document(page_content=page_content, metadata=metadata)

    def _rerank_chunks(
        self, query: str, chunks_with_scores: List[Tuple[Document, float]]
    ) -> List[Tuple[Document, float]]:
        """使用 DashScope Rerank 模型对 chunk 级结果重排"""
        if not chunks_with_scores:
            return []

        model_name = self.reranker_params.get("model_name", "qwen3-rerank")
        dashscope.api_key = os.getenv("BL_API_KEY")

        texts = [doc.page_content for doc, _ in chunks_with_scores]
        resp = dashscope.TextReRank.call(
            model=model_name,
            query=query,
            documents=texts,
            top_n=len(texts),
            return_documents=False,
        )

        if resp.status_code != 200:
            logger.warning(
                "Rerank API failed (code %s), fallback to original ranking: %s",
                resp.code,
                resp.message if hasattr(resp, "message") else resp,
            )
            return chunks_with_scores

        return [
            (chunks_with_scores[r["index"]][0], r["relevance_score"])
            for r in resp["output"]["results"]
        ]
