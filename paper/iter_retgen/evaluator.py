"""ITER-RETGEN Evaluator — inherits BaseEvaluator, integrates with checkpoint system."""

import logging

from Eval.base import BaseEvaluator, NormalizedQuestion
from Eval.metrics import (
    compute_context_recall,
    compute_hit,
    compute_mrr,
    compute_retrieval_precision,
    extract_supporting_titles,
)
from Retrieval.milvus_retriever import MilvusRetriever
from paper.iter_retgen.iter_retgen import IterRetGenPipeline

logger = logging.getLogger(__name__)


class IterRetGenEvaluator(BaseEvaluator):
    def __init__(
        self,
        llm,
        dataset_type,
        batch_size=20,
        max_workers=5,
        max_retries=2,
        topk=50,
        max_chunks=8,
        use_reranker=False,
        k_per_iteration=5,
        max_iterations=3,
    ):
        super().__init__(
            eval_mode="iter_retgen",
            llm=llm,
            dataset_type=dataset_type,
            batch_size=batch_size,
            max_workers=max_workers,
            max_retries=max_retries,
        )
        self.topk = topk
        self.max_chunks = max_chunks
        self.use_reranker = use_reranker
        self.k_per_iteration = k_per_iteration
        self.max_iterations = max_iterations

    def evaluate_single(self, question: NormalizedQuestion) -> dict:
        retriever = MilvusRetriever(
            dataset_type=self.dataset_type,
            topk_propositions=self.topk,
            max_chunks=self.max_chunks,
            use_reranker=self.use_reranker,
        )

        pipeline = IterRetGenPipeline(
            llm=self.llm,
            retriever=retriever,
            k_per_iteration=self.k_per_iteration,
            max_iterations=self.max_iterations,
        )

        result = pipeline.run(question.question)

        if not result["answer"]:
            return {
                "prediction": None,
                "error": "ITER-RETGEN returned empty answer",
                "chunks": [],
                "context_recall": None,
                "hit": None,
                "mrr": None,
                "retrieval_precision": None,
            }

        chunks = []
        for para in result["all_paragraphs"]:
            chunks.append(
                {
                    "chunk_id": para.get("chunk_id", ""),
                    "context_title": para.get("context_title", ""),
                    "content": para.get("page_content", ""),
                    "score": para.get("score", 0),
                }
            )

        retrieved_titles = [c["context_title"] for c in chunks]
        supporting = extract_supporting_titles(question.raw, self.dataset_type)

        return {
            "prediction": result["answer"].strip(),
            "error": None,
            "chunks": chunks,
            "context_recall": compute_context_recall(retrieved_titles, supporting),
            "hit": compute_hit(retrieved_titles, supporting),
            "mrr": compute_mrr(retrieved_titles, supporting),
            "retrieval_precision": compute_retrieval_precision(
                retrieved_titles, supporting
            ),
            "retrieval_count": result["total_retrievals"],
            "iter_retgen_iterations": len(result["iterations"]),
            "round_answers": [it["answer"] for it in result["iterations"]],
        }
