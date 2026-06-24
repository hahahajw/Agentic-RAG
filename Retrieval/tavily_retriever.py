"""TavilyRetriever：使用 Tavily Search API 进行网络搜索的检索器

接口与 MilvusRetriever / WebRetriever 兼容，可作为 drop-in replacement：
- get_similar_chunk_with_score(query) → List[Tuple[Document, float]]
- get_similar_chunks_with_rewrite(query, rewrite_llm) → List[Tuple[Document, float]]

每条 Tavily 搜索结果封装为一个 Document（即一个 chunk），
metadata 包含 chunk_id（URL hash）、chunk_title（标题）、source_url 等。
Tavily 提供的 relevance score 直接用作 chunk score。
"""

import hashlib
import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional, Tuple

from langchain_core.documents import Document

from naive_rag.prompts import QUERY_REWRITE_PROMPT, QUERY_REWRITE_WITH_CONTEXT_PROMPT

logger = logging.getLogger(__name__)


class TavilyRetriever:
    """使用 Tavily Search API 的检索器，接口与 MilvusRetriever 兼容。

    Tavily 专为 AI Agent 设计，返回结构化结果（title/url/content/score），
    相比 DuckDuckGo 提供更干净的摘要和真实的相关性分数。

    Args:
        max_chunks: 返回的最大搜索结果数
        api_key: Tavily API key（默认从环境变量 TAVILY_KEY 读取）
        search_depth: "basic" 或 "advanced"（advanced 提供更高质量结果但延迟更高）
        include_raw_content: 是否包含页面全文（会增加延迟和 token 消耗）
    """

    def __init__(
        self,
        max_chunks: int = 8,
        api_key: str | None = None,
        search_depth: str = "advanced",
        include_raw_content: bool = False,
    ):
        self.max_chunks = max_chunks
        self._api_key = api_key or os.getenv("TAVILY_KEY", "")
        self._search_depth = search_depth
        self._include_raw_content = include_raw_content
        self._client = None  # Lazy init

    def _get_client(self):
        """延迟初始化 TavilyClient。"""
        if self._client is None:
            from tavily import TavilyClient
            if not self._api_key:
                raise ValueError(
                    "Tavily API key 未配置。请设置环境变量 TAVILY_KEY 或传入 api_key 参数。"
                )
            self._client = TavilyClient(api_key=self._api_key)
        return self._client

    def get_similar_chunk_with_score(self, query: str) -> List[Tuple[Document, float]]:
        """执行 Tavily 搜索，返回 chunk 级结果。

        Args:
            query: 搜索查询

        Returns:
            [(Document, score), ...] — 按 relevance score 降序
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

        与 MilvusRetriever / WebRetriever 的 get_similar_chunks_with_rewrite 保持一致。

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
        logger.info("Tavily Query rewrite: '%s' → %d variants", query[:60], len(variants))

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
        """调用 Tavily Search API，返回 Document 列表。

        Tavily 返回的 results 按 relevance score 降序排列，
        每条结果包含 title、url、content、score。

        Returns:
            [(Document, score), ...] — 按 relevance score 降序
        """
        try:
            client = self._get_client()
        except ValueError as e:
            logger.warning("Tavily client init failed: %s", e)
            return []

        try:
            response = client.search(
                query=query,
                search_depth=self._search_depth,
                max_results=self.max_chunks * 2,
                include_raw_content=self._include_raw_content,
            )
        except Exception as e:
            logger.warning("Tavily search failed for query '%s': %s", query[:60], e)
            return []

        results = response.get("results", [])
        if not results:
            logger.info("No Tavily results for query: %s", query[:60])
            return []

        docs_with_scores = []
        for r in results:
            title = r.get("title", "Unknown")
            content = r.get("content", "")
            url = r.get("url", "")
            score = r.get("score", 0.0)

            # 用 URL hash 作为 chunk_id，保证去重
            chunk_id = hashlib.md5(url.encode()).hexdigest()[:12]

            # 优先使用 raw_content（完整正文），回退到 content（摘要）
            page_content = r.get("raw_content") or content

            doc = Document(
                page_content=page_content,
                metadata={
                    "chunk_id": chunk_id,
                    "chunk_title": title,
                    "chunk_summary": content[:200] if content else "",
                    "context_title": title,
                    "source_url": url,
                    "dataset_type": "web",
                    "aggregated_propositions": 1,
                },
            )
            docs_with_scores.append((doc, float(score)))

        logger.info("Tavily search returned %d results for: %s", len(docs_with_scores), query[:60])
        return docs_with_scores