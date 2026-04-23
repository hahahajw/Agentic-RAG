"""RAG with Judge — 评估器

接入 Eval 的 checkpoint 系统，继承 BaseEvaluator。
每个问题独立运行 rag_with_judge 递归探索，收集检索指标和效率指标。
"""

import logging
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI

from Eval.base import BaseEvaluator, NormalizedQuestion
from Eval.metrics import (
    extract_supporting_titles,
    compute_context_recall,
    compute_hit,
    compute_mrr,
    compute_retrieval_precision,
)
from Retrieval.milvus_retriever import MilvusRetriever

logger = logging.getLogger(__name__)


class JudgeRAGEvaluator(BaseEvaluator):
    """RAG with Judge 评估器。

    每个问题独立创建 rag_with_judge 实例，避免状态污染。
    支持 checkpoint 断点续跑。
    """

    def __init__(
        self,
        llm: ChatOpenAI,
        dataset_type: str,
        batch_size: int = 20,
        max_workers: int = 5,
        max_retries: int = 2,
        topk: int = 50,
        max_chunks: int = 8,
        max_depth: int = 3,
        use_reranker: bool = False,
        rewrite_model: str = None,
        judge_model: str = None,
        answer_model: str = None,
        judge_variant: str = "B",
        cot: bool = False,
        model_params: dict | None = None,
        role_params: dict | None = None,
        # 自定义 Milvus collection 支持
        custom_collection: str = None,
        custom_dense_field: str = None,
        custom_text_field: str = None,
        custom_sparse_field: str = None,
    ):
        super().__init__(
            eval_mode="rag_with_judge",
            llm=llm,
            dataset_type=dataset_type,
            batch_size=batch_size,
            max_workers=max_workers,
            max_retries=max_retries,
        )

        self.topk = topk
        self.max_chunks = max_chunks
        self.max_depth = max_depth
        self.use_reranker = use_reranker

        # 独立模型配置
        self.rewrite_model = rewrite_model
        self.judge_model = judge_model
        self.answer_model = answer_model
        self.judge_variant = judge_variant
        self.cot = cot
        self.model_params = model_params or {}
        self.role_params = role_params or {}

        # 自定义 collection 配置
        self.custom_collection = custom_collection
        self.custom_dense_field = custom_dense_field
        self.custom_text_field = custom_text_field
        self.custom_sparse_field = custom_sparse_field

    def _make_retriever(self):
        """为当前评估创建 Retriever 实例。

        如果配置了自定义 collection，返回 CustomMilvusRetriever；
        否则返回标准的 MilvusRetriever。
        """
        if self.custom_collection:
            from custom.retriever import CustomMilvusRetriever
            return CustomMilvusRetriever(
                collection_name=self.custom_collection,
                dense_field=self.custom_dense_field or "embedding",
                text_field=self.custom_text_field or "proposition_text",
                sparse_field=self.custom_sparse_field,
                topk=self.topk,
                max_chunks=self.max_chunks,
                use_reranker=self.use_reranker,
            )
        return MilvusRetriever(
            dataset_type=self.dataset_type,
            topk_propositions=self.topk,
            max_chunks=self.max_chunks,
            use_reranker=self.use_reranker,
        )

    def _make_config(self, retriever: MilvusRetriever) -> dict:
        """构建 LangGraph 运行时配置。"""
        import os
        from langchain_openai import ChatOpenAI

        _BASE_DEFAULTS = {
            "temperature": 0.0,
            "extra_body": {"enable_thinking": False, "enable_search": False},
        }

        def _create_llm(model_name: str | None, role: str):
            if model_name is None:
                return self.llm
            # 四层合并：base_defaults → model_params → role_params → model=name
            kwargs = {**_BASE_DEFAULTS, **self.model_params,
                      **self.role_params.get(role, {}), "model": model_name}
            kwargs["api_key"] = os.getenv("BL_API_KEY")
            kwargs["base_url"] = os.getenv("BL_BASE_URL")
            return ChatOpenAI(**kwargs)

        return {
            "configurable": {
                "llm": self.llm,
                "rewrite_llm": _create_llm(self.rewrite_model, "rewrite"),
                "judge_llm": _create_llm(self.judge_model, "judge"),
                "answer_llm": _create_llm(self.answer_model, "answer"),
                "retriever": retriever,
                "topk_propositions": self.topk,
                "max_chunks": self.max_chunks,
                "use_reranker": self.use_reranker,
                "max_depth": self.max_depth,
                "judge_variant": self.judge_variant,
                "cot": self.cot,
            }
        }

    def evaluate_single(self, question: NormalizedQuestion) -> dict:
        """处理单个问题。

        Returns:
            {
                "prediction": str,
                "error": None | str,
                "chunks": [...],
                "context_recall": float,
                "hit": int,
                "mrr": float,
                "retrieval_precision": float,
                "retrieval_count": int,
                "total_chunks": int,
                "total_distinct_titles": int,
                "search_depth": int,
                "search_path": dict,
            }
        """
        from rag_with_judge.workflow import build_judge_rag_graph
        from rag_with_judge.nodes import rag_with_judge

        retriever = self._make_retriever()
        config = self._make_config(retriever)
        app = build_judge_rag_graph()

        search_path = {}
        visited = set()

        # 运行递归探索
        answer = rag_with_judge(
            query=question.question,
            path=search_path,
            visited=visited,
            depth=0,
            max_depth=self.max_depth,
            app=app,
            config=config,
        )

        # 从 SEARCH_PATH 收集所有 unique chunks（BFS 顺序）
        all_chunks = _collect_all_chunks(search_path)
        retrieved_titles = [c.get("context_title", "") for c in all_chunks]

        # 计算检索指标
        supporting = extract_supporting_titles(question.raw, self.dataset_type)

        return {
            "prediction": answer.strip() if answer else "",
            "error": None,
            "chunks": all_chunks,
            "context_recall": compute_context_recall(retrieved_titles, supporting),
            "hit": compute_hit(retrieved_titles, supporting),
            "mrr": compute_mrr(retrieved_titles, supporting),
            "retrieval_precision": compute_retrieval_precision(retrieved_titles, supporting),
            # 效率指标
            "retrieval_count": _count_retrievals(search_path),
            "total_chunks": len(all_chunks),
            "total_distinct_titles": len(set(retrieved_titles)),
            "search_depth": _get_max_depth(search_path),
            "search_path": search_path,
        }


# ─── SEARCH_PATH 辅助函数 ─────────────────────────────────────

def _collect_all_chunks(path: dict) -> list[dict]:
    """BFS 遍历 SEARCH_PATH，收集所有 unique chunks（按 chunk_id 去重）。"""
    if not path:
        return []

    all_chunks = []
    seen_ids = set()
    queue = [path]

    while queue:
        node = queue.pop(0)
        for chunk in node.get("chunks", []):
            cid = chunk.get("chunk_id")
            if cid and cid not in seen_ids:
                seen_ids.add(cid)
                all_chunks.append(chunk)
        for child in node.get("next_queries", []):
            if child:
                queue.append(child)

    return all_chunks


def _count_retrievals(path: dict) -> int:
    """统计 SEARCH_PATH 中总共进行了几轮检索（节点总数）。"""
    if not path:
        return 0
    count = 1  # 当前节点
    for child in path.get("next_queries", []):
        if child:
            count += _count_retrievals(child)
    return count


def _get_max_depth(path: dict) -> int:
    """统计搜索树的最大深度（根节点为 1）。"""
    # 搜索树的最大深度 = 最大递归深度 + 1
    if not path:
        return 0
    children = path.get("next_queries", [])
    if not children:
        return 1
    return 1 + max(_get_max_depth(c) for c in children if c)
