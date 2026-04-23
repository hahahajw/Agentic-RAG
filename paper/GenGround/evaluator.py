"""GenGround Evaluator — inherits BaseEvaluator, integrates with checkpoint system."""

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
from paper.GenGround.gen_ground import GenGroundPipeline

logger = logging.getLogger(__name__)


class GenGroundEvaluator(BaseEvaluator):
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
        k_per_ref=10,
        max_iterations=5,
    ):
        super().__init__(
            eval_mode="gen_ground",
            llm=llm,
            dataset_type=dataset_type,
            batch_size=batch_size,
            max_workers=max_workers,
            max_retries=max_retries,
        )
        self.topk = topk
        self.max_chunks = max_chunks
        self.use_reranker = use_reranker
        self.k_per_ref = k_per_ref
        self.max_iterations = max_iterations

    def evaluate_single(self, question: NormalizedQuestion) -> dict:
        retriever = MilvusRetriever(
            dataset_type=self.dataset_type,
            topk_propositions=self.topk,
            max_chunks=self.max_chunks,
            use_reranker=self.use_reranker,
        )

        pipeline = GenGroundPipeline(
            llm=self.llm,
            retriever=retriever,
            k_per_ref=self.k_per_ref,
            max_iterations=self.max_iterations,
        )

        result = pipeline.run(question.question)

        if not result["answer"]:
            return {
                "prediction": None,
                "error": "GenGround returned empty answer",
                "chunks": [],
                "context_recall": None,
                "hit": None,
                "mrr": None,
                "retrieval_precision": None,
            }

        # Collect all unique titles from references
        retrieved_titles = []
        seen = set()
        for ref in result.get("references", []):
            for doc, _ in ref.get("results", []):
                title = doc.metadata.get("context_title", "")
                if title and title not in seen:
                    seen.add(title)
                    retrieved_titles.append(title)

        chunks = [{"context_title": t} for t in retrieved_titles]
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
            "gen_ground_iterations": result["iterations"],
            "stopped_by": result["stopped_by"],
        }
