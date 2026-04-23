"""CustomMilvusRetriever：参数化的 Milvus 检索器

支持任意 collection 名称、任意字段名的 dense/sparse 检索。
适用于用户自定义 Milvus 数据集接入 RAG Pipeline。

检索流程：
  Stage 1: Dense-only 或 Hybrid Search → top K propositions
  Stage 2: 按 chunk_id 分组 → chunk 聚合 → 不截断
  Stage 3 (可选): Reranker 重排
  Stage 4: 截断到 max_chunks
"""

import json
import logging
import os
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple

import dashscope
from langchain_core.documents import Document
from pymilvus import MilvusClient, AnnSearchRequest, RRFRanker, WeightedRanker

from Index.milvus_config import get_embedding_function

logger = logging.getLogger(__name__)


class CustomMilvusRetriever:
    """参数化的 Milvus 检索器，适用于自定义 collection。

    Args:
        collection_name: Milvus collection 名称（必填）
        dense_field: dense 向量字段名（必填）
        text_field: 文本内容字段名（必填）
        sparse_field: sparse/BM25 向量字段名（None 表示纯 dense 检索）
        uri: Milvus 服务地址
        token: Milvus 认证 token
        embedding_func: 嵌入函数（None 时使用项目默认的 text-embedding-v4）
        topk: Stage 1 检索的 proposition 数量
        max_chunks: 最终返回的 chunk 数量
        use_reranker: 是否启用 Reranker
        reranker_model: Reranker 模型名称
        chunk_id_field: chunk 分组字段名
        output_fields: 需要从 Milvus 返回的字段列表（None 时自动推断）
    """

    def __init__(
        self,
        collection_name: str,
        dense_field: str,
        text_field: str,
        sparse_field: Optional[str] = None,
        uri: str = "http://localhost:19530",
        token: str = "",
        embedding_func: Any = None,
        topk: int = 50,
        max_chunks: int = 8,
        use_reranker: bool = False,
        reranker_model: str = "qwen3-rerank",
        chunk_id_field: str = "chunk_id",
        output_fields: Optional[List[str]] = None,
    ):
        self.collection_name = collection_name
        self.dense_field = dense_field
        self.text_field = text_field
        self.sparse_field = sparse_field
        self.chunk_id_field = chunk_id_field
        self.topk = topk
        self.max_chunks = max_chunks
        self.use_reranker = use_reranker
        self.reranker_model = reranker_model

        connect_kwargs: Dict[str, str] = {"uri": uri}
        if token:
            connect_kwargs["token"] = token
        self._client = MilvusClient(**connect_kwargs)
        self._embedding_func = embedding_func or get_embedding_function()

        self._output_fields = output_fields or [
            "id", chunk_id_field, text_field,
        ]

    # ── 公共接口 ──────────────────────────────────────────────

    def get_similar_chunk_with_score(self, query: str) -> List[Tuple[Document, float]]:
        """主检索入口。返回 chunk 级结果。

        Returns:
            [(Document, score), ...] 按分数降序
        """
        # Stage 1: Dense / Hybrid Search → top K propositions
        prop_results = self._search(query)

        if not prop_results:
            logger.warning("No propositions retrieved for query: %s", query)
            return []

        # Stage 2: 按 chunk_id 分组 → 聚合（不截断）
        all_chunks = self._aggregate_by_chunk(prop_results)

        # Stage 3 (可选): Reranker 重排
        if self.use_reranker:
            all_chunks = self._rerank_chunks(query, all_chunks)

        # Stage 4: 截断到 max_chunks
        return all_chunks[: self.max_chunks]

    def get_similar_chunks_with_rewrite(
        self,
        query: str,
        rewrite_llm: Any,
        num_variants: int = 4,
    ) -> List[Tuple[Document, float]]:
        """单 query 完整检索链：重写 → 独立检索 → RRF 融合。

        1. LLM 生成 num_variants 个重写变体
        2. 每个变体独立调用 get_similar_chunk_with_score()
        3. 对返回的 chunk 级结果做 RRF 融合
        4. 截断到 max_chunks

        Returns:
            [(Document, rrf_score), ...] 按 RRF 分数降序
        """
        from naive_rag.prompts import QUERY_REWRITE_PROMPT

        RRF_K = 60

        # 1. LLM 问题重写
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

        # 3. RRF 融合（chunk 级）
        chunk_rrf: Dict[str, float] = {}
        chunk_docs: Dict[str, Document] = {}
        for q, chunk_results in results_by_query.items():
            for rank, (doc, _) in enumerate(chunk_results, 1):
                cid = doc.metadata.get(self.chunk_id_field)
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

    # ── 内部方法 ──────────────────────────────────────────────

    def _search(self, query: str) -> List[Tuple[Document, float]]:
        """Dense-only 或 Hybrid Search。"""
        query_embedding = self._embedding_func.embed_query(query)

        requests = []

        # Dense request
        dense_req = AnnSearchRequest(
            data=[query_embedding],
            anns_field=self.dense_field,
            param={"metric_type": "COSINE", "params": {"ef": 64}},
            limit=self.topk,
        )
        requests.append(dense_req)

        # Sparse request (if configured)
        if self.sparse_field:
            sparse_req = AnnSearchRequest(
                data=[query],
                anns_field=self.sparse_field,
                param={"metric_type": "BM25"},
                limit=self.topk,
            )
            requests.append(sparse_req)

        # Ranker
        ranker = RRFRanker(60)

        results = self._client.hybrid_search(
            collection_name=self.collection_name,
            reqs=requests,
            ranker=ranker,
            limit=self.topk,
            output_fields=self._output_fields,
        )

        # Parse results
        docs_with_scores: List[Tuple[Document, float]] = []
        if results and len(results) > 0:
            for hit in results[0]:
                meta = dict(hit.fields)
                meta["id"] = hit.id
                doc = Document(
                    page_content=hit.fields.get(self.text_field, ""),
                    metadata=meta,
                )
                docs_with_scores.append((doc, hit.distance))

        return docs_with_scores

    def _enrich_chunk_props(self, chunk_ids: List[str]) -> Dict[str, List[Tuple[Document, float]]]:
        """从 Milvus 查询候选 chunk 的完整 propositions。"""
        if not chunk_ids:
            return {}

        filter_expr = " or ".join(f'{self.chunk_id_field} == "{cid}"' for cid in chunk_ids)
        results = self._client.query(
            collection_name=self.collection_name,
            filter=filter_expr,
            output_fields=self._output_fields,
            limit=5000,
        )

        groups: Dict[str, List[Tuple[Document, float]]] = defaultdict(list)
        for hit in results:
            cid = hit.get(self.chunk_id_field, "")
            meta = dict(hit)
            meta["id"] = hit.get("id")
            doc = Document(page_content=hit.get(self.text_field, ""), metadata=meta)
            groups[cid].append((doc, 0.0))

        return dict(groups)

    def _aggregate_by_chunk(
        self,
        prop_results: List[Tuple[Document, float]],
    ) -> List[Tuple[Document, float]]:
        """按 chunk_id_field 分组，每组取最高分，按分数降序返回所有 chunk（不截断）。"""
        groups: Dict[str, List[Tuple[Document, float]]] = defaultdict(list)
        for doc, score in prop_results:
            cid = doc.metadata.get(self.chunk_id_field, "")
            if cid:
                groups[cid].append((doc, score))

        chunk_scores: Dict[str, float] = {}
        for cid, props in groups.items():
            chunk_scores[cid] = max(s for _, s in props)

        sorted_chunk_ids = sorted(chunk_scores, key=chunk_scores.get, reverse=True)

        # 获取完整 propositions
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
        """将一个 chunk 的所有 propositions 组装为单个 Document。"""
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
        """使用 DashScope Rerank 模型对 chunk 级结果重排。"""
        if not chunks_with_scores:
            return []

        dashscope.api_key = os.getenv("BL_API_KEY")

        texts = [doc.page_content for doc, _ in chunks_with_scores]
        resp = dashscope.TextReRank.call(
            model=self.reranker_model,
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
