"""ITER-RETGEN Prompt Templates"""

from langchain_core.messages import SystemMessage

ITER_RETGEN_TEMPLATE = """\
You are answering a multi-hop question. Use the provided context and your own knowledge to reason step by step.

### Question
{question}

### Context
{context_text}

{previous_attempt}

### Instructions
Think step by step and provide your reasoning. End your response with "So the answer is: <answer>."

### Response
"""


REFINE_ANSWER_TEMPLATE = """\
You have completed multiple rounds of iterative retrieval and reasoning for this question.

### Question
{question}

### Reasoning History
{reasoning_text}

### Instructions
Based on all the reasoning above, provide a **concise and direct** answer to the question.
- Be as brief as possible — output only the essential answer
- Do NOT include explanations, context, or additional details
- Do NOT repeat the question
- Output ONLY the answer after "So the answer is: "

### Final Answer
"""


def get_refine_answer_prompt(question: str, iterations: list) -> SystemMessage:
    """Generate the final answer refinement prompt."""
    reasoning_parts = []
    for it in iterations:
        round_num = it["round"]
        gen = it["generation"][:500]  # Truncate to avoid excessive context
        reasoning_parts.append(f"--- Round {round_num} ---\n{gen}")
    reasoning_text = "\n\n".join(reasoning_parts)

    return SystemMessage(
        content=REFINE_ANSWER_TEMPLATE.format(
            question=question,
            reasoning_text=reasoning_text,
        )
    )


def get_generation_prompt(
    question: str,
    paragraphs: list,
    prev_generation: str = "",
    iteration: int = 1,
) -> SystemMessage:
    """Generate the prompt for each ITER-RETGEN iteration.

    Args:
        question: The original question
        paragraphs: Retrieved paragraphs for this round
        prev_generation: Previous round's generation (y_{t-1})
        iteration: Current round number (1-based)
    """
    if paragraphs:
        context_text = "\n\n".join(
            f"[{i + 1}] {p.get('context_title', 'Unknown')}: {p['page_content']}"
            for i, p in enumerate(paragraphs)
        )
    else:
        context_text = "(No context available)"

    if prev_generation:
        previous_attempt = (
            f"\n### Previous Attempt (Round {iteration - 1})\n{prev_generation}"
            f"\n\nUse this as additional context, but feel free to correct any errors."
        )
    else:
        previous_attempt = ""

    return SystemMessage(
        content=ITER_RETGEN_TEMPLATE.format(
            question=question,
            context_text=context_text,
            previous_attempt=previous_attempt,
        )
    )
