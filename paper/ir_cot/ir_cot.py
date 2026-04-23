"""
IR-CoT: Interleaving Retrieval with Chain-of-Thought Reasoning

Architecture:
  Phase 1: Interleaved Retrieval Loop
    Initial retrieval -> [generate 1 CoT sentence -> retrieve with CoT sentence] x N
  Phase 2: Reader
    Use all collected paragraphs + question -> LLM generates full CoT + answer
"""

import re
import logging

from langchain_openai import ChatOpenAI

from Retrieval.milvus_retriever import MilvusRetriever
from paper.ir_cot.prompts import get_cot_loop_prompt, get_reader_prompt

logger = logging.getLogger(__name__)

MAX_COT_STEPS = 10
MAX_PARAGRAPHS = 15
K_PER_STEP = 4
WH_WORDS = re.compile(
    r"\b(who|what|when|where|why|which|how|does|is|are|were|was|do|did)\b",
    re.IGNORECASE,
)
ANSWER_PATTERN = re.compile(r".* answer is:? (.*)\.?", re.IGNORECASE)


def strip_wh_words(text: str) -> str:
    """Remove WH words to produce better BM25 queries."""
    return WH_WORDS.sub("", text).strip()


def extract_first_sentence(text: str) -> tuple[str, str]:
    """Extract the first sentence, return (sentence, remaining_text)."""
    match = re.match(r"([^\.!?]*[\.!?]?)\s*(.*)", text.strip(), re.DOTALL)
    if match:
        return match.group(1).strip(), match.group(2).strip()
    return text.strip(), ""


class IRCoTPipeline:
    """IR-CoT pipeline.

    Phase 1: Interleaved retrieval loop - each round generates 1 CoT sentence,
              retrieves with that sentence.
    Phase 2: Reader - uses all collected paragraphs to generate full CoT + answer.
    """

    def __init__(
        self,
        llm: ChatOpenAI,
        retriever: MilvusRetriever,
        k_per_step: int = K_PER_STEP,
        max_steps: int = MAX_COT_STEPS,
        max_paragraphs: int = MAX_PARAGRAPHS,
    ):
        self.llm = llm
        self.retriever = retriever
        self.k_per_step = k_per_step
        self.max_steps = max_steps
        self.max_paragraphs = max_paragraphs

    def run(self, question: str) -> dict:
        """Run the IR-CoT pipeline.

        Returns:
            {
                "answer": str,              # Final answer
                "cot_sentences": list[str], # Generated CoT sentences
                "all_paragraphs": list[dict], # All collected paragraphs
                "total_retrievals": int,    # Total retrieval count
                "stopped_by": str,          # "answer_detected", "max_steps", "max_paragraphs"
            }
        """
        all_paragraphs = []
        seen_ids = set()
        cot_sentences = []
        total_retrievals = 0
        stopped_by = "max_steps"

        # Step 0: Initial retrieval with original question
        initial_results = self.retriever.get_similar_chunk_with_score(question)
        total_retrievals += 1
        self._add_paragraphs(initial_results, all_paragraphs, seen_ids)

        # Step 1-N: Iterative loop
        for step in range(self.max_steps):
            prompt = get_cot_loop_prompt(
                question=question,
                paragraphs=all_paragraphs,
                cot_prefix=cot_sentences,
            )

            response = self.llm.invoke([prompt])
            generated = response.content.strip()

            first_sentence, _ = extract_first_sentence(generated)
            cot_sentences.append(first_sentence)

            # Check termination: answer detected
            if ANSWER_PATTERN.match(first_sentence):
                stopped_by = "answer_detected"
                answer_match = ANSWER_PATTERN.search(first_sentence)
                answer = answer_match.group(1).strip() if answer_match else ""
                break

            # Retrieve with CoT sentence as query
            query = strip_wh_words(first_sentence)
            results = self.retriever.get_similar_chunk_with_score(query)
            total_retrievals += 1
            self._add_paragraphs(results, all_paragraphs, seen_ids)

            # Check paragraph limit
            if len(all_paragraphs) >= self.max_paragraphs:
                stopped_by = "max_paragraphs"
                break
        else:
            stopped_by = "max_steps"
            answer = ""

        # Phase 2: Reader answer
        reader_answer = self._run_reader(question, all_paragraphs)
        final_answer = reader_answer or self._extract_answer_from_cot(cot_sentences)

        return {
            "answer": final_answer,
            "cot_sentences": cot_sentences,
            "all_paragraphs": all_paragraphs,
            "total_retrievals": total_retrievals,
            "stopped_by": stopped_by,
        }

    def _add_paragraphs(self, results: list, pool: list, seen: set):
        """Add retrieval results to paragraph pool, deduplicating."""
        for doc, score in results:
            cid = doc.metadata.get("chunk_id")
            if cid and cid not in seen:
                seen.add(cid)
                pool.append(
                    {
                        "chunk_id": cid,
                        "context_title": doc.metadata.get("context_title", ""),
                        "chunk_title": doc.metadata.get("chunk_title", ""),
                        "page_content": doc.page_content,
                    }
                )
                if len(pool) >= self.max_paragraphs:
                    return

    def _run_reader(self, question: str, paragraphs: list) -> str:
        """Phase 2: Reader generates full CoT + answer from scratch."""
        if not paragraphs:
            return ""
        prompt = get_reader_prompt(question=question, paragraphs=paragraphs)
        response = self.llm.invoke([prompt])
        content = response.content.strip()

        match = ANSWER_PATTERN.search(content)
        if match:
            return match.group(1).strip()

        sentences = re.split(r"[.!?]", content)
        return sentences[-1].strip() if sentences else content

    def _extract_answer_from_cot(self, cot_sentences: list) -> str:
        """Extract answer from CoT sentences."""
        for s in reversed(cot_sentences):
            match = ANSWER_PATTERN.search(s)
            if match:
                return match.group(1).strip()
        return ""
