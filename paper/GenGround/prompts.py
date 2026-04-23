"""GenGround Prompt Templates"""

from langchain_core.messages import SystemMessage

GENERATE_TEMPLATE = """\
You are answering a multi-hop question. Use your knowledge to reason step by step.

When you mention a fact that needs external verification, mark it with <ref>.
When you have enough information, output ###Finish[your_answer].

### Question
{question}

{context_section}

### Instructions
1. Reason step by step about the question
2. Use <ref> to mark facts that need verification
3. Be specific — each <ref> should be a concrete claim
4. If you have the answer, output ###Finish[answer]

### Reasoning
"""

REVISE_TEMPLATE = """\
You generated the following reasoning for the question "{question}":

{generated_text}

Below are the search results for each <ref> claim:

{search_results_text}

### Instructions
1. Compare each <ref> claim with the search results
2. If the claim is correct, keep it as is
3. If the claim is wrong or incomplete, wrap the corrected version in <revise>...</revise>
4. Output the corrected reasoning

### Corrected Reasoning
"""


REFINE_ANSWER_TEMPLATE = """\
You have completed a generate-then-ground reasoning process for this question.

### Question
{question}

### Reasoning Chain
{reasoning_text}

### Verified Revisions
{revisions_text}

### Instructions
Based on the reasoning chain and verified revisions above, provide a **concise and direct** answer to the question.
- Be as brief as possible — output only the essential answer
- Do NOT include explanations, context, or additional details
- Do NOT repeat the question
- Output ONLY the answer after "So the answer is: "

### Final Answer
"""


def get_refine_answer_prompt(question: str, reasoning_chain: list, revisions: list) -> SystemMessage:
    """Generate the final answer refinement prompt."""
    reasoning_text = "\n\n".join(
        f"--- Step {i + 1} ---\n{r[:500]}"
        for i, r in enumerate(reasoning_chain)
    )
    revisions_text = "\n".join(
        f"- {r}" for r in revisions
    ) if revisions else "(No revisions needed)"

    return SystemMessage(
        content=REFINE_ANSWER_TEMPLATE.format(
            question=question,
            reasoning_text=reasoning_text,
            revisions_text=revisions_text,
        )
    )


def get_generate_prompt(
    question: str, context_paragraphs: list, iteration: int
) -> SystemMessage:
    """Generate the Generate Phase prompt."""
    if context_paragraphs:
        context_section = "### Context from Previous Searches\n" + "\n\n".join(
            f"[{i + 1}] {p.get('context_title', 'Unknown')}: {p['page_content']}"
            for i, p in enumerate(context_paragraphs[-20:])
        )
    else:
        context_section = "(No context yet — use your own knowledge to start.)"

    return SystemMessage(
        content=GENERATE_TEMPLATE.format(
            question=question,
            context_section=context_section,
        )
    )


def get_revise_prompt(
    question: str, generated_text: str, ref_results: list
) -> SystemMessage:
    """Generate the Revise Phase prompt."""
    search_results_text = ""
    for i, ref_result in enumerate(ref_results):
        query = ref_result["query"]
        docs = ref_result["results"][:5]
        search_results_text += f'\n\nClaim {i + 1}: "{query}"\n'
        for j, (doc, score) in enumerate(docs):
            title = doc.metadata.get("context_title", "Unknown")
            search_results_text += f"  [{j + 1}] {title}: {doc.page_content[:200]}\n"

    return SystemMessage(
        content=REVISE_TEMPLATE.format(
            question=question,
            generated_text=generated_text,
            search_results_text=search_results_text,
        )
    )
