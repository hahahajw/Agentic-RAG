"""
ITER-RETGEN: Iterative Retrieve and Generate for Multi-hop QA

Architecture:
  For t = 1 to T:
    1. Retrieve with (y_{t-1} || q)  -> top-K paragraphs
       (t=1: y_0 = "", only use q)
    2. Generate y_t with CoT using paragraphs + q
  Output: refined concise answer (from all iterations via dedicated prompt)
"""

import re
import logging

from langchain_openai import ChatOpenAI

from Retrieval.milvus_retriever import MilvusRetriever
from paper.iter_retgen.prompts import get_generation_prompt, get_refine_answer_prompt

logger = logging.getLogger(__name__)

MAX_ITERATIONS = 3
K_PER_ITERATION = 5
ANSWER_PATTERN = re.compile(r".* answer is:? (.*)\.?", re.IGNORECASE)


class IterRetGenPipeline:
    """ITER-RETGEN pipeline.

    Each round uses the previous round's generation + original question
    as the retrieval query, then generates a new answer with CoT.
    """

    def __init__(
        self,
        llm: ChatOpenAI,
        retriever: MilvusRetriever,
        k_per_iteration: int = K_PER_ITERATION,
        max_iterations: int = MAX_ITERATIONS,
    ):
        self.llm = llm
        self.retriever = retriever
        self.k_per_iteration = k_per_iteration
        self.max_iterations = max_iterations

    def run(self, question: str) -> dict:
        """Run the ITER-RETGEN pipeline.

        Returns:
            {
                "answer": str,              # Final answer (y_T)
                "iterations": list[dict],   # Per-round details
                "total_retrievals": int,
                "all_paragraphs": list[dict],
            }
        """
        prev_generation = ""  # y_0 = "" (first round uses only original question)
        all_paragraphs = []
        seen_ids = set()
        iterations = []
        total_retrievals = 0

        for t in range(self.max_iterations):
            # === Step 1: Retrieve ===
            if t == 0:
                query = question
            else:
                query = f"{prev_generation}\n{question}"

            results = self.retriever.get_similar_chunk_with_score(query)
            total_retrievals += 1

            round_paragraphs = []
            for doc, score in results:
                cid = doc.metadata.get("chunk_id")
                para = {
                    "chunk_id": cid,
                    "context_title": doc.metadata.get("context_title", ""),
                    "page_content": doc.page_content,
                    "score": score,
                }
                round_paragraphs.append(para)
                if cid and cid not in seen_ids:
                    seen_ids.add(cid)
                    all_paragraphs.append(para)

            # === Step 2: Generate (CoT) ===
            prompt = get_generation_prompt(
                question=question,
                paragraphs=round_paragraphs,
                prev_generation=prev_generation,
                iteration=t + 1,
            )
            response = self.llm.invoke([prompt])
            generated_text = response.content.strip()

            answer = self._extract_answer(generated_text)

            iterations.append(
                {
                    "round": t + 1,
                    "query": query[:200] + "..." if len(query) > 200 else query,
                    "paragraphs": round_paragraphs,
                    "generation": generated_text,
                    "answer": answer,
                }
            )

            prev_generation = generated_text

        # Final answer refinement: use a dedicated prompt to generate a concise answer
        # from all accumulated reasoning, rather than just extracting from y_T
        final_answer = self._refine_answer(question, iterations)

        return {
            "answer": final_answer,
            "iterations": iterations,
            "total_retrievals": total_retrievals,
            "all_paragraphs": all_paragraphs,
        }

    def _refine_answer(self, question: str, iterations: list) -> str:
        """Use a dedicated prompt to generate a concise final answer
        from all accumulated reasoning, rather than just extracting from y_T."""
        if not iterations:
            return ""
        prompt = get_refine_answer_prompt(question=question, iterations=iterations)
        response = self.llm.invoke([prompt])
        content = response.content.strip()
        match = ANSWER_PATTERN.search(content)
        if match:
            return match.group(1).strip()
        return content

    def _extract_answer(self, text: str) -> str:
        """Extract answer from CoT generated text."""
        if not text:
            return ""
        match = ANSWER_PATTERN.search(text)
        if match:
            return match.group(1).strip()
        sentences = re.split(r"[.!?]", text)
        return sentences[-1].strip() if sentences else text
