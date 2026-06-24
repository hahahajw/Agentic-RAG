# ═══════════════════════════════════════════════════════════════════
# 外部适配器
# ═══════════════════════════════════════════════════════════════════
# MilvusAdapter: 将现有 Retrieval/milvus_retriever.py 适配到 SearchFn Protocol。
# 这是算法接触真实 KB 的唯一实现——测试时用 mock lambda 替代。

import sys
import os
import logging

from .interfaces import SearchFn
from .models import ChunkInfo

logger = logging.getLogger(__name__)


class MilvusAdapter:
    """将现有 Milvus 检索流水线适配到 SearchFn Protocol。

    内部封装 MilvusRetriever 的初始化和字段映射，
    对外暴露单一 __call__(query) -> list[ChunkInfo] 接口。

    FRAMEWORK.md 模块 1: 算法只能通过 search(query) 与 KB 交互。
    此适配器是该交互的具体实现。

    Usage:
        adapter = MilvusAdapter(dataset_type="musique")
        chunks = adapter("Moscow State University founding year")
    """

    def __init__(
        self,
        dataset_type: str,
        topk_propositions: int = 50,
        max_chunks: int = 8,
        use_reranker: bool = False,
        # 自定义 collection 支持
        custom_collection: str | None = None,
        custom_dense_field: str = "embedding",
        custom_text_field: str = "proposition_text",
        custom_sparse_field: str | None = None,
        # 多查询重写模型（None = 单查询模式）
        rewrite_model = None,
    ):
        self._dataset_type = dataset_type.lower()
        self._topk_propositions = topk_propositions
        self._max_chunks = max_chunks
        self._use_reranker = use_reranker
        self._custom_collection = custom_collection
        self._custom_dense_field = custom_dense_field
        self._custom_text_field = custom_text_field
        self._custom_sparse_field = custom_sparse_field
        self._rewrite_model = rewrite_model
        self._retriever = None  # Lazy init

    def __call__(self, query: str) -> list[ChunkInfo]:
        """执行检索并返回 ChunkInfo 列表。

        当 rewrite_model 可用时，使用完整检索流水线：
        LLM 生成 4 变体 → 5 查询并行 → RRF 融合 → Reranker。
        否则使用单查询模式。

        失败时返回空列表（不抛异常）——符合 SearchFn 契约。
        """
        try:
            retriever = self._get_retriever()
            if self._rewrite_model is not None:
                results = retriever.get_similar_chunks_with_rewrite(
                    query, self._rewrite_model, num_variants=4
                )
            else:
                results = retriever.get_similar_chunk_with_score(query)
            return [_doc_to_chunkinfo(doc) for doc, _score in results]
        except Exception as e:
            logger.warning("MilvusAdapter search failed for query '%s': %s", query[:80], e)
            return []

    def _get_retriever(self):
        """延迟初始化 retriever——避免 import 时的 Milvus 连接开销。

        支持两种模式：
        - 标准模式：使用 MilvusRetriever + dataset_type
        - 自定义模式：使用 CustomMilvusRetriever + 自定义 collection 参数
        """
        if self._retriever is None:
            project_root = os.path.abspath(
                os.path.join(os.path.dirname(__file__), "..")
            )
            if project_root not in sys.path:
                sys.path.insert(0, project_root)

            if self._custom_collection:
                from custom.retriever import CustomMilvusRetriever
                self._retriever = CustomMilvusRetriever(
                    collection_name=self._custom_collection,
                    dense_field=self._custom_dense_field,
                    text_field=self._custom_text_field,
                    sparse_field=self._custom_sparse_field,
                    topk=self._topk_propositions,
                    max_chunks=self._max_chunks,
                    use_reranker=self._use_reranker,
                )
            else:
                from Retrieval.milvus_retriever import MilvusRetriever
                self._retriever = MilvusRetriever(
                    dataset_type=self._dataset_type,
                    topk_propositions=self._topk_propositions,
                    max_chunks=self._max_chunks,
                    use_reranker=self._use_reranker,
                )
        return self._retriever


class WebAdapter:
    """将 WebRetriever 适配到 SearchFn Protocol。

    内部封装 WebRetriever（DuckDuckGo 网络搜索），
    对外暴露单一 __call__(query) -> list[ChunkInfo] 接口。

    当 rewrite_model 可用时，使用多查询重写 + RRF 融合模式
    （与 MilvusAdapter 行为一致）；否则使用单查询模式。

    Usage:
        adapter = WebAdapter(max_chunks=8)
        chunks = adapter("Moscow State University founding year")
    """

    def __init__(self, max_chunks: int = 8, rewrite_model=None):
        self._max_chunks = max_chunks
        self._rewrite_model = rewrite_model
        self._retriever = None  # Lazy init

    def __call__(self, query: str) -> list[ChunkInfo]:
        """执行网络搜索并返回 ChunkInfo 列表。

        当 rewrite_model 可用时: LLM 生成 4 变体 → 5 查询并行 → RRF 融合。
        否则使用单查询模式。

        失败时返回空列表（不抛异常）——符合 SearchFn 契约。
        """
        try:
            retriever = self._get_retriever()
            if self._rewrite_model is not None:
                results = retriever.get_similar_chunks_with_rewrite(
                    query, self._rewrite_model, num_variants=4
                )
            else:
                results = retriever.get_similar_chunk_with_score(query)
            return [_doc_to_chunkinfo(doc) for doc, _score in results]
        except Exception as e:
            logger.warning("WebAdapter search failed for query '%s': %s", query[:80], e)
            return []

    def _get_retriever(self):
        """延迟初始化 WebRetriever——避免 import 时的连接开销。"""
        if self._retriever is None:
            import sys
            import os
            project_root = os.path.abspath(
                os.path.join(os.path.dirname(__file__), "..")
            )
            if project_root not in sys.path:
                sys.path.insert(0, project_root)
            from Retrieval.web_retriever import WebRetriever
            self._retriever = WebRetriever(max_chunks=self._max_chunks)
        return self._retriever


class TavilyAdapter:
    """将 TavilyRetriever 适配到 SearchFn Protocol。

    内部封装 TavilyRetriever（Tavily Search API 网络搜索），
    对外暴露单一 __call__(query) -> list[ChunkInfo] 接口。

    当 rewrite_model 可用时，使用多查询重写 + RRF 融合模式
    （与 MilvusAdapter 行为一致）；否则使用单查询模式。

    Tavily 专为 AI Agent 设计，提供结构化搜索结果
    （title/url/content/score），相比 DuckDuckGo 摘要更干净、
    相关性分数更准确。

    Usage:
        adapter = TavilyAdapter(max_chunks=8)
        chunks = adapter("Moscow State University founding year")
    """

    def __init__(self, max_chunks: int = 8, rewrite_model=None):
        self._max_chunks = max_chunks
        self._rewrite_model = rewrite_model
        self._retriever = None  # Lazy init

    def __call__(self, query: str) -> list[ChunkInfo]:
        """执行 Tavily 搜索并返回 ChunkInfo 列表。

        当 rewrite_model 可用时: LLM 生成 4 变体 → 5 查询并行 → RRF 融合。
        否则使用单查询模式。

        失败时返回空列表（不抛异常）——符合 SearchFn 契约。
        """
        try:
            retriever = self._get_retriever()
            if self._rewrite_model is not None:
                results = retriever.get_similar_chunks_with_rewrite(
                    query, self._rewrite_model, num_variants=4
                )
            else:
                results = retriever.get_similar_chunk_with_score(query)
            return [_doc_to_chunkinfo(doc) for doc, _score in results]
        except Exception as e:
            logger.warning("TavilyAdapter search failed for query '%s': %s", query[:80], e)
            return []

    def _get_retriever(self):
        """延迟初始化 TavilyRetriever——避免 import 时的连接开销。"""
        if self._retriever is None:
            import sys
            import os
            project_root = os.path.abspath(
                os.path.join(os.path.dirname(__file__), "..")
            )
            if project_root not in sys.path:
                sys.path.insert(0, project_root)
            from Retrieval.tavily_retriever import TavilyRetriever
            self._retriever = TavilyRetriever(max_chunks=self._max_chunks)
        return self._retriever


def _doc_to_chunkinfo(doc) -> ChunkInfo:
    """将 LangChain Document 映射为 ChunkInfo dataclass。

    MilvusRetriever / WebRetriever 返回的 Document:
    - page_content: 该 chunk 所有 proposition_text 的拼接
    - metadata: chunk_id, chunk_title, chunk_summary, context_title, ...
    """
    meta = doc.metadata
    return ChunkInfo(
        chunk_id=meta.get("chunk_id", ""),
        chunk_title=meta.get("chunk_title", ""),
        chunk_summary=meta.get("chunk_summary", ""),
        context_title=meta.get("context_title", ""),
        page_content=doc.page_content,
    )