"""WebRetriever：使用 DuckDuckGo 进行网络搜索的检索器

接口与 MilvusRetriever 兼容，可作为 drop-in replacement：
- get_similar_chunk_with_score(query) → List[Tuple[Document, float]]
- get_similar_chunks_with_rewrite(query, rewrite_llm) → List[Tuple[Document, float]]

每条 DuckDuckGo 搜索结果封装为一个 Document（即一个 chunk），
metadata 包含 chunk_id（URL hash）、chunk_title（标题）、source_url 等。
"""

import hashlib
import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional, Tuple

from langchain_core.documents import Document

from naive_rag.prompts import QUERY_REWRITE_PROMPT, QUERY_REWRITE_WITH_CONTEXT_PROMPT

logger = logging.getLogger(__name__)


class WebRetriever:
    """使用 DuckDuckGo 搜索的检索器，接口与 MilvusRetriever 兼容。

    Args:
        max_chunks: 返回的最大搜索结果数
    """

    def __init__(self, max_chunks: int = 8):
        self.max_chunks = max_chunks

    def get_similar_chunk_with_score(self, query: str) -> List[Tuple[Document, float]]:
        """执行网络搜索，返回 chunk 级结果。

        Args:
            query: 搜索查询

        Returns:
            [(Document, score), ...] — 按相关性降序
        """
        results = self._search(query)
        return results[: self.max_chunks]

    def get_similar_chunks_with_rewrite(
        self,
        query: str,
        rewrite_llm,
        num_variants: int = 4,
        rewrite_context: str = None,
    ) -> List[Tuple[Document, float]]:
        """完整检索链：重写 → 独立搜索 → RRF 融合。

        与 MilvusRetriever 的 get_similar_chunks_with_rewrite 保持一致的模式。

        Args:
            query: 原始查询
            rewrite_llm: 用于问题重写的 LLM 实例
            num_variants: 重写变体数量
            rewrite_context: 可选的已知事实上下文

        Returns:
            [(Document, rrf_score), ...] — 按 RRF 分数降序
        """
        RRF_K = 60

        # 1. LLM 问题重写
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
        logger.info("Web Query rewrite: '%s' → %d variants", query[:60], len(variants))

        # 2. 每个 query 独立搜索
        with ThreadPoolExecutor(max_workers=len(all_queries)) as executor:
            futures = {
                executor.submit(self.get_similar_chunk_with_score, q): q
                for q in all_queries
            }
            results_by_query = {}
            for f in as_completed(futures):
                results_by_query[futures[f]] = f.result()

        # 3. RRF 融合（与 Schema A 一致）
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

    def _search(self, query: str) -> List[Tuple[Document, float]]:
        """调用 DuckDuckGo 搜索，返回 Document 列表。

        Returns:
            [(Document, score), ...] — 按搜索排名降序
        """
        try:
            from ddgs import DDGS
        except ImportError:
            from duckduckgo_search import DDGS

        try:
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=self.max_chunks * 2))
        except Exception as e:
            logger.warning("DuckDuckGo search failed for query '%s': %s", query[:60], e)
            return []

        if not results:
            logger.info("No web results for query: %s", query[:60])
            return []

        docs_with_scores = []
        for i, r in enumerate(results):
            title = r.get("title", "Unknown")
            snippet = r.get("body", "")
            url = r.get("href", "")

            # 用 URL hash 作为 chunk_id，保证去重
            chunk_id = hashlib.md5(url.encode()).hexdigest()[:12]

            doc = Document(
                page_content=snippet,
                metadata={
                    "chunk_id": chunk_id,
                    "chunk_title": title,
                    "chunk_summary": "",
                    "context_title": title,
                    "source_url": url,
                    "dataset_type": "web",
                    "aggregated_propositions": 1,
                },
            )
            # DuckDuckGo 不提供 score，用排名倒推（排名越高分数越高）
            score = 1.0 / (i + 1)
            docs_with_scores.append((doc, score))

        logger.info("Web search returned %d results for: %s", len(docs_with_scores), query[:60])
        return docs_with_scores
