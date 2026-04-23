"""Naive RAG 评估器 — 复用 naive_rag/ 的 Scheme A/B 工作流。

Scheme A（Client-Side RRF）流程：
  START → rewrite → fan-out retrieve → fuse_results → generate → END

Scheme B（AnnSearchRequest 级融合）流程：
  START → rewrite → batch_retrieve（单次 hybrid_search）→ generate → END
"""

import logging

from Eval.base import BaseEvaluator, NormalizedQuestion

logger = logging.getLogger(__name__)


class NaiveRAGEvaluator(BaseEvaluator):
    """使用 Naive RAG（Scheme A 或 B 检索）评估问题。

    复用 naive_rag/workflow.py 中的工作流。
    每个问题创建独立的工作流实例，避免跨问题状态污染。
    """

    def __init__(self, scheme: str = "b", topk: int = 50, max_chunks: int = 8,
                 use_reranker: bool = False, reranker_params: dict = None,
                 rewrite_model: str = None, suggest_model: str = None,
                 model_params: dict | None = None,
                 role_params: dict | None = None,
                 # 自定义 Milvus collection 支持
                 custom_collection: str = None,
                 custom_dense_field: str = None,
                 custom_text_field: str = None,
                 custom_sparse_field: str = None,
                 **kwargs):
        self.scheme = scheme.lower()
        # dataset_name 编码 schema 信息到文件名中：hotpotqa → hotpotqa_schema_a
        dataset_name = kwargs.pop("dataset_type", "")
        full_dataset = f"{dataset_name}_schema_{self.scheme}"

        super().__init__(eval_mode="naive_rag", dataset_type=full_dataset, **kwargs)
        self.dataset_type = dataset_name  # 保留原始值供 evaluate_single 使用
        self.topk = topk
        self.max_chunks = max_chunks
        self.use_reranker = use_reranker
        self.reranker_params = reranker_params or {"model_name": "qwen3-rerank"}
        self.rewrite_model = rewrite_model
        self.suggest_model = suggest_model
        self.model_params = model_params or {}
        self.role_params = role_params or {}

        # 自定义 collection 配置
        self.custom_collection = custom_collection
        self.custom_dense_field = custom_dense_field
        self.custom_text_field = custom_text_field
        self.custom_sparse_field = custom_sparse_field

    def evaluate_single(self, question: NormalizedQuestion) -> dict:
        """为单个问题运行完整的 Naive RAG（Scheme A/B）工作流。"""
        import os

        from langchain_openai import ChatOpenAI

        from naive_rag.workflow import get_workflow

        # 每个问题新建实例，跳过 suggest 以节省 API 调用
        app = get_workflow(scheme=self.scheme, skip_suggest=True)

        def _make_llm(model_name: str, role: str) -> ChatOpenAI:
            _BASE_DEFAULTS = {
                "temperature": 0.0,
                "extra_body": {"enable_thinking": False, "enable_search": False},
            }
            kwargs = {**_BASE_DEFAULTS, **self.model_params,
                      **self.role_params.get(role, {}), "model": model_name}
            kwargs["api_key"] = os.getenv("BL_API_KEY")
            kwargs["base_url"] = os.getenv("BL_BASE_URL")
            return ChatOpenAI(**kwargs)

        config_dict = {
            "llm": self.llm,
            "dataset_type": self.dataset_type,
            "topk_propositions": self.topk,
            "max_chunks": self.max_chunks,
            "use_reranker": self.use_reranker,
            "reranker_params": self.reranker_params,
        }
        if self.rewrite_model:
            config_dict["rewrite_llm"] = _make_llm(self.rewrite_model, "rewrite")
        if self.suggest_model:
            config_dict["suggest_llm"] = _make_llm(self.suggest_model, "suggest")

        # 自定义 collection：注入 retriever 实例
        if self.custom_collection:
            from custom.retriever import CustomMilvusRetriever
            config_dict["retriever"] = CustomMilvusRetriever(
                collection_name=self.custom_collection,
                dense_field=self.custom_dense_field or "embedding",
                text_field=self.custom_text_field or "proposition_text",
                sparse_field=self.custom_sparse_field,
                topk=self.topk,
                max_chunks=self.max_chunks,
                use_reranker=self.use_reranker,
            )

        result = app.invoke(
            {"original_query": question.question, "messages": []},
            config={"configurable": config_dict},
        )

        answer = result.get("answer", "")
        if not answer:
            return {
                "prediction": None, "error": "工作流返回了空答案", "chunks": [],
                "context_recall": None, "hit": None, "mrr": None, "retrieval_precision": None,
            }

        # 提取 chunk 元信息（避免序列化 Document 对象）
        chunks = []
        for doc, score in result.get("fused_chunks", []):
            chunks.append({
                "chunk_id": doc.metadata.get("chunk_id", ""),
                "chunk_title": doc.metadata.get("chunk_title", ""),
                "chunk_summary": doc.metadata.get("chunk_summary", ""),
                "context_title": doc.metadata.get("context_title", ""),
                "context_index": doc.metadata.get("context_index", ""),
                "content": doc.page_content,
                "score": round(score, 6),
                "aggregated_propositions": doc.metadata.get("aggregated_propositions", 0),
            })

        # 计算检索质量指标
        from Eval.metrics import (
            compute_context_recall, compute_hit, compute_mrr, compute_retrieval_precision,
            extract_supporting_titles,
        )

        retrieved_titles = [c["context_title"] for c in chunks]
        supporting = extract_supporting_titles(question.raw, self.dataset_type)

        return {
            "prediction": answer.strip(), "error": None, "chunks": chunks,
            "context_recall": compute_context_recall(retrieved_titles, supporting),
            "hit": compute_hit(retrieved_titles, supporting),
            "mrr": compute_mrr(retrieved_titles, supporting),
            "retrieval_precision": compute_retrieval_precision(retrieved_titles, supporting),
            "retrieval_count": 1,  # 单次工作流入口调用
        }
