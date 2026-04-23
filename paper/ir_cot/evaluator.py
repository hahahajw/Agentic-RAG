"""IR-CoT Evaluator — inherits BaseEvaluator, integrates with checkpoint system."""

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
from paper.ir_cot.ir_cot import IRCoTPipeline

logger = logging.getLogger(__name__)


class IRCoTEvaluator(BaseEvaluator):
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
        k_per_step=4,
        max_steps=10,
        max_paragraphs=15,
    ):
        super().__init__(
            eval_mode="ir_cot",
            llm=llm,
            dataset_type=dataset_type,
            batch_size=batch_size,
            max_workers=max_workers,
            max_retries=max_retries,
        )
        self.topk = topk
        self.max_chunks = max_chunks
        self.use_reranker = use_reranker
        self.k_per_step = k_per_step
        self.max_steps = max_steps
        self.max_paragraphs = max_paragraphs

    def evaluate_single(self, question: NormalizedQuestion) -> dict:
        retriever = MilvusRetriever(
            dataset_type=self.dataset_type,
            topk_propositions=self.topk,
            max_chunks=self.max_chunks,
            use_reranker=self.use_reranker,
        )

        pipeline = IRCoTPipeline(
            llm=self.llm,
            retriever=retriever,
            k_per_step=self.k_per_step,
            max_steps=self.max_steps,
            max_paragraphs=self.max_paragraphs,
        )

        result = pipeline.run(question.question)

        if not result["answer"]:
            return {
                "prediction": None,
                "error": "IR-CoT returned empty answer",
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
                    "chunk_title": para.get("chunk_title", ""),
                    "content": para.get("page_content", ""),
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
            "cot_steps": len(result["cot_sentences"]),
            "stopped_by": result["stopped_by"],
        }
