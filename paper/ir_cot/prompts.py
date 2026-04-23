"""IR-CoT Prompt Templates"""

from langchain_core.messages import SystemMessage

COT_LOOP_TEMPLATE = """\
You are answering a multi-hop question. Below is evidence collected so far.

### Evidence Collected So Far
{paragraphs_text}

If no evidence has been collected yet, use your own knowledge to start reasoning.

### Question
{question}

### Task
Reason step by step based on the evidence above. Generate ONE sentence at a time.
When you have enough information, end with "So the answer is: <answer>."

Continue from where you left off:
"""

NO_CONTEXT_TEMPLATE = """\
You are answering a multi-hop question. Use your knowledge to reason step by step.

### Question
{question}

### Task
Reason step by step. Generate ONE sentence at a time.
When you have enough information, end with "So the answer is: <answer>."
"""

READER_TEMPLATE = """\
You are answering a question based on the evidence below.

### Evidence
{paragraphs_text}

### Question
{question}

### Task
Reason step by step based on the evidence. Then provide a concise final answer.
IMPORTANT: After your reasoning, end with a brief, direct answer in the format:
"So the answer is: <answer>."
Keep the answer as short as possible — just the essential answer, no explanation.

Let's think step by step:
"""


def get_cot_loop_prompt(
    question: str, paragraphs: list, cot_prefix: list
) -> SystemMessage:
    """Generate the prompt for the interleaved retrieval loop."""
    if paragraphs:
        paragraphs_text = "\n\n".join(
            f"Wikipedia Title: {p.get('context_title', p.get('chunk_title', 'Unknown'))}\n{p['page_content']}"
            for p in paragraphs
        )
        template = COT_LOOP_TEMPLATE
    else:
        paragraphs_text = "(No evidence collected yet)"
        template = NO_CONTEXT_TEMPLATE

    if cot_prefix:
        paragraphs_text += f"\n\nReasoning so far: {' '.join(cot_prefix)}"

    return SystemMessage(
        content=template.format(
            paragraphs_text=paragraphs_text,
            question=question,
        )
    )


def get_reader_prompt(question: str, paragraphs: list) -> SystemMessage:
    """Generate the Reader prompt (Phase 2)."""
    paragraphs_text = "\n\n".join(
        f"Wikipedia Title: {p.get('context_title', p.get('chunk_title', 'Unknown'))}\n{p['page_content']}"
        for p in paragraphs
    )
    return SystemMessage(
        content=READER_TEMPLATE.format(
            paragraphs_text=paragraphs_text,
            question=question,
        )
    )
