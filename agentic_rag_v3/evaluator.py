"""Agentic RAG v3 — 评估器

接入 Eval 的 checkpoint 系统，继承 BaseEvaluator。
每个问题独立运行 agentic_rag_v3 闭环探索（子问题解决链），收集检索指标和效率指标。
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


class AgenticRAGV3Evaluator(BaseEvaluator):
    """Agentic RAG v3 评估器。

    每个问题独立创建 agentic_rag_v3 实例，避免状态污染。
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
        max_rounds: int = 5,
        use_reranker: bool = False,
        planner_model: str = None,
        evaluator_model: str = None,
        answer_model: str = None,
        reflector_model: str = None,
        rewrite_model: str = None,
        synthesizer_model: str = None,
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
            eval_mode="agentic_rag_v3",
            llm=llm,
            dataset_type=dataset_type,
            batch_size=batch_size,
            max_workers=max_workers,
            max_retries=max_retries,
        )

        self.topk = topk
        self.max_chunks = max_chunks
        self.max_rounds = max_rounds
        self.use_reranker = use_reranker

        # 独立模型配置
        self.planner_model = planner_model
        self.evaluator_model = evaluator_model
        self.answer_model = answer_model
        self.reflector_model = reflector_model
        self.rewrite_model = rewrite_model
        self.synthesizer_model = synthesizer_model
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
                "planner_llm": _create_llm(self.planner_model, "planner"),
                "evaluator_llm": _create_llm(self.evaluator_model, "evaluator"),
                "answer_llm": _create_llm(self.answer_model, "answer"),
                "reflector_llm": _create_llm(self.reflector_model, "reflector"),
                "rewrite_llm": _create_llm(self.rewrite_model, "rewrite"),
                "synthesizer_llm": _create_llm(self.synthesizer_model, "synthesizer"),
                "retriever": retriever,
                "max_chunks": self.max_chunks,
                "use_reranker": self.use_reranker,
                "max_rounds": self.max_rounds,
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
                "sub_questions": list,  # v2 新增：子问题解决状态
            }
        """
        from agentic_rag_v3.workflow import build_agentic_rag_v3_graph
        from agentic_rag_v3.nodes import run_agentic_rag_v3

        retriever = self._make_retriever()
        config = self._make_config(retriever)
        app = build_agentic_rag_v3_graph()

        result = run_agentic_rag_v3(
            query=question.question,
            app=app,
            config=config,
            max_rounds=self.max_rounds,
        )

        # 收集检索指标
        all_chunks = result["all_chunks"]
        retrieved_titles = [c.get("context_title", "") for c in all_chunks]

        supporting = extract_supporting_titles(question.raw, self.dataset_type)

        # 统计检索轮次数：等于探索链的总轮次数（与 RAG with Judge 的搜索树节点数对齐）
        # 注意：每轮内部可能求解多个子问题，但本文将一轮视为一个逻辑检索单元
        retrieval_count = result["total_rounds"]

        return {
            "prediction": result["answer"].strip() if result["answer"] else "",
            "error": None,
            "chunks": all_chunks,
            "context_recall": compute_context_recall(retrieved_titles, supporting),
            "hit": compute_hit(retrieved_titles, supporting),
            "mrr": compute_mrr(retrieved_titles, supporting),
            "retrieval_precision": compute_retrieval_precision(retrieved_titles, supporting),
            # 效率指标
            "retrieval_count": retrieval_count,
            "total_chunks": len(all_chunks),
            "total_distinct_titles": len(set(retrieved_titles)),
            "search_depth": result["total_rounds"],
            "chunks_per_round": len(all_chunks) / max(result["total_rounds"], 1),
            "search_path": {"exploration_history": result["exploration_history"]},
            # v2 特有：子问题解决状态
            "sub_questions": result.get("sub_questions", []),
        }
