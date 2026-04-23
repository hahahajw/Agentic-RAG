"""
GenGround: Generate-then-Ground Paradigm

Architecture:
  Generate -> Ground (retrieve per <ref>) -> Revise -> loop until ###Finish[answer]
"""

import re
import logging

from langchain_openai import ChatOpenAI

from Retrieval.milvus_retriever import MilvusRetriever
from paper.GenGround.prompts import get_generate_prompt, get_revise_prompt, get_refine_answer_prompt

logger = logging.getLogger(__name__)

MAX_ITERATIONS = 5
K_PER_REF = 10
MAX_CONTEXT_PARAGRAPHS = 30

REF_PATTERN = re.compile(r"<ref>")
REVISE_PATTERN = re.compile(r"<revise>(.*?)</revise>", re.DOTALL)
FINISH_PATTERN = re.compile(r"###Finish\[(.*?)\]")


class GenGroundPipeline:
    """GenGround pipeline.

    Generate: LLM generates reasoning chain with <ref> tags marking claims needing verification
    Ground:   Retrieve for each <ref> claim
    Revise:   LLM revises incorrect claims using search results
    Loop until ###Finish[answer] or max iterations
    """

    def __init__(
        self,
        llm: ChatOpenAI,
        retriever: MilvusRetriever,
        k_per_ref: int = K_PER_REF,
        max_iterations: int = MAX_ITERATIONS,
        max_context_paragraphs: int = MAX_CONTEXT_PARAGRAPHS,
    ):
        self.llm = llm
        self.retriever = retriever
        self.k_per_ref = k_per_ref
        self.max_iterations = max_iterations
        self.max_context_paragraphs = max_context_paragraphs

    def run(self, question: str) -> dict:
        """Run the GenGround pipeline.

        Returns:
            {
                "answer": str,
                "reasoning_chain": list[str],
                "references": list[dict],
                "revisions": list[str],
                "total_retrievals": int,
                "iterations": int,
                "stopped_by": str,
            }
        """
        all_paragraphs = []
        seen_ids = set()
        reasoning_chain = []
        all_references = []
        all_revisions = []
        total_retrievals = 0
        stopped_by = "max_iterations"

        for iteration in range(self.max_iterations):
            # === Generate Phase ===
            generate_prompt = get_generate_prompt(
                question=question,
                context_paragraphs=all_paragraphs,
                iteration=iteration,
            )
            response = self.llm.invoke([generate_prompt])
            generated_text = response.content.strip()
            reasoning_chain.append(generated_text)

            # Check termination
            finish_match = FINISH_PATTERN.search(generated_text)
            if finish_match:
                stopped_by = "finished"
                break

            # Extract <ref> claims
            refs = self._extract_refs(generated_text)

            if not refs:
                stopped_by = "no_refs"
                break

            # === Ground Phase: retrieve for each <ref> ===
            ref_results = []
            for ref_text in refs:
                query = ref_text.strip()
                results = self.retriever.get_similar_chunk_with_score(query)
                total_retrievals += 1
                ref_results.append({"query": query, "results": results})
                for doc, score in results:
                    cid = doc.metadata.get("chunk_id")
                    if cid and cid not in seen_ids:
                        seen_ids.add(cid)
                        all_paragraphs.append(
                            {
                                "chunk_id": cid,
                                "context_title": doc.metadata.get("context_title", ""),
                                "page_content": doc.page_content[:500],
                            }
                        )

            all_references.extend(ref_results)

            # === Revise Phase ===
            revise_prompt = get_revise_prompt(
                question=question,
                generated_text=generated_text,
                ref_results=ref_results,
            )
            response = self.llm.invoke([revise_prompt])
            revised_text = response.content.strip()

            revisions = REVISE_PATTERN.findall(revised_text)
            all_revisions.extend(revisions)

            # Check paragraph limit
            if len(all_paragraphs) >= self.max_context_paragraphs:
                stopped_by = "max_paragraphs"
                break
        else:
            stopped_by = "max_iterations"

        # Final answer refinement: use a dedicated prompt to generate a concise answer
        # from all accumulated reasoning, rather than extracting from ###Finish or last sentence
        if reasoning_chain:
            final_answer = self._refine_answer(question, reasoning_chain, all_revisions)
        else:
            final_answer = self._extract_answer("")

        return {
            "answer": final_answer,
            "reasoning_chain": reasoning_chain,
            "references": all_references,
            "revisions": all_revisions,
            "total_retrievals": total_retrievals,
            "iterations": iteration + 1,
            "stopped_by": stopped_by,
        }

    def _refine_answer(self, question: str, reasoning_chain: list, revisions: list) -> str:
        """Use a dedicated prompt to generate a concise final answer
        from all accumulated reasoning, rather than extracting from ###Finish or last sentence."""
        prompt = get_refine_answer_prompt(
            question=question,
            reasoning_chain=reasoning_chain,
            revisions=revisions,
        )
        response = self.llm.invoke([prompt])
        content = response.content.strip()
        match = re.search(r"answer is:? (.*)\.?", content, re.IGNORECASE)
        if match:
            return match.group(1).strip()
        return content

    def _extract_refs(self, text: str) -> list[str]:
        """Extract text before each <ref> tag as retrieval queries."""
        refs = []
        parts = text.split("<ref>")
        for part in parts[1:]:
            sentence_match = re.match(r"([^.!?]*[.!?]?)", part)
            if sentence_match:
                refs.append(sentence_match.group(1).strip())
            else:
                refs.append(part[:100])
        return refs

    def _extract_answer(self, text: str) -> str:
        """Extract answer from text."""
        if not text:
            return ""
        finish_match = FINISH_PATTERN.search(text)
        if finish_match:
            return finish_match.group(1).strip()
        answer_match = re.search(r"answer is:? (.*)\.?", text, re.IGNORECASE)
        if answer_match:
            return answer_match.group(1).strip()
        sentences = re.split(r"[.!?]", text)
        return sentences[-1].strip() if sentences else text
