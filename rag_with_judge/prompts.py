"""RAG with Judge — Prompt 模板

包含 2 个 prompt：
1. JUDGE_PROMPT: 全局 Judge，判断 chunks 是否足以回答问题
2. ANSWER_PROMPT: 基于完整 SEARCH_PATH 综合生成最终答案

注意：查询重写 prompt 复用 naive_rag.prompts.QUERY_REWRITE_PROMPT。

JUDGE prompt 有三个 variant（A/B/C），可通过 get_judge_prompt(question, chunks, variant="B") 切换。
"""

from langchain_core.messages import SystemMessage, HumanMessage


# ─── Judge Variant A（最小改动：仅 follow-up 段追加原子化约束）───

JUDGE_PROMPT_A_TEMPLATE = """\
You are a professional Judge responsible for evaluating whether the retrieved \
knowledge is sufficient to completely and accurately answer a given question.

### Question
{question}

### Retrieved Knowledge
{chunks_text}

### Task
Analyze the retrieved knowledge step by step:

1. **Decompose** the question into its core sub-questions or components.
2. **Evaluate** each sub-question: can it be answered from the retrieved knowledge?
3. **Identify gaps**: if any sub-question cannot be answered, specify exactly what is missing.
4. **Generate follow-up queries**: for each missing piece of knowledge, write a precise \
query that would retrieve the needed information. Each query MUST be:
   (a) **atomic** — answerable from a single fact or proposition
   (b) **self-contained** — use full entity names, no pronouns ("it", "he", "they") or context-dependent references
   (c) **single-focus** — ask exactly one thing, not compound questions ("A and B" → split into two)
   (d) **retrieval-friendly** — phrased like a search engine query, not a conversational question

### Output Requirements
Output in valid JSON with exactly these fields:
- "answerable": boolean — true if ALL sub-questions can be answered from the retrieved knowledge
- "reason": string — a brief explanation of your judgment, listing which sub-questions are covered and which are missing
- "next_queries": list of strings — atomic follow-up queries for each knowledge gap (empty list if answerable is true)

Do NOT include any text outside the JSON. The output must be directly parseable."""


# ─── Judge Variant B（中等：新增 Retrieval Context 原则 + follow-up 加强）───

JUDGE_PROMPT_B_TEMPLATE = """\
You are a professional Judge responsible for evaluating whether the retrieved \
knowledge is sufficient to completely and accurately answer a given question.

Each follow-up query you generate will be retrieved via proposition-level hybrid search \
(dense + BM25). Follow these principles when generating queries:

1. **Atomic**: each query should be answerable from a single factual statement
2. **Self-contained**: use full entity names, no pronouns ("it", "he", "they") or references to previously retrieved context
3. **Single-focus**: ask exactly ONE thing per query; if a gap requires multiple pieces of information, split into separate queries
4. **Retrieval-friendly**: phrase queries like search engine queries, not conversational questions

### Question
{question}

### Retrieved Knowledge
{chunks_text}

### Task
Analyze the retrieved knowledge step by step:

1. **Decompose** the question into its core sub-questions or components.
2. **Evaluate** each sub-question: can it be answered from the retrieved knowledge?
3. **Identify gaps**: if any sub-question cannot be answered, specify exactly what is missing.
4. **Generate follow-up queries**: for each gap, write ONE atomic query following the principles above. \
   If a knowledge gap requires multi-step reasoning, decompose it into multiple atomic queries.

### Output Requirements
Output in valid JSON with exactly these fields:
- "answerable": boolean — true if ALL sub-questions can be answered from the retrieved knowledge
- "reason": string — a brief explanation of your judgment, listing which sub-questions are covered and which are missing
- "next_queries": list of strings — atomic follow-up queries for each knowledge gap (empty list if answerable is true)

Do NOT include any text outside the JSON. The output must be directly parseable."""


# ─── Judge Variant C（完整重写：4-step 结构，参考 ICML 2025）───

JUDGE_PROMPT_C_TEMPLATE = """\
You are a Knowledge Gap Analyzer responsible for identifying missing information \
and generating atomic retrieval queries to fill those gaps.

You will receive an original question and a set of retrieved knowledge chunks. \
Your job is to determine whether the retrieved knowledge is sufficient to answer \
the question, and if not, generate atomic retrieval queries to fill the gaps.

Each follow-up query will be retrieved via proposition-level hybrid search (dense + BM25). \
**Follow these rules strictly:**
- **One fact per query**: the query should be answerable from a single proposition
- **Full entity names**: never use pronouns ("it", "he", "they", "this", "that") or vague references
- **No compound questions**: if you need two pieces of information, write two separate queries
- **Search-optimized phrasing**: prefer "Einstein university professor founded" over "At which university did Einstein work and when was it founded?"

### Original Question
{question}

### Retrieved Knowledge
{chunks_text}

### Analysis Steps
1. **Decompose**: Break the original question into its smallest logical components. Each component should be independently answerable from a single fact.
2. **Evaluate**: For each component, determine whether it can be answered from the retrieved knowledge above.
3. **Identify Gaps**: List exactly which components remain unanswered and what specific facts are needed.
4. **Generate Atomic Queries**: For each gap, write a retrieval-optimized atomic query following the rules above.

### Output Requirements
Output in valid JSON with exactly these fields:
- "answerable": boolean — true if ALL components can be answered from the retrieved knowledge
- "reason": string — which components are covered, which are missing
- "next_queries": list of strings — atomic retrieval queries for each knowledge gap (empty list if answerable is true)

Do NOT include any text outside the JSON. The output must be directly parseable."""


# ─── 获取函数 ───────────────────────────────────────────────────

def get_judge_prompt(question: str, chunks_text: str, variant: str = "B") -> SystemMessage:
    """生成 Judge prompt。

    Args:
        question: 原始问题
        chunks_text: 检索到的知识
        variant: prompt 变体 ("A" | "B" | "C")，默认 "B"
    """
    templates = {
        "A": JUDGE_PROMPT_A_TEMPLATE,
        "B": JUDGE_PROMPT_B_TEMPLATE,
        "C": JUDGE_PROMPT_C_TEMPLATE,
    }
    template = templates.get(variant, JUDGE_PROMPT_B_TEMPLATE)
    return SystemMessage(content=template.format(question=question, chunks_text=chunks_text))


# ─── 最终答案生成 ─────────────────────────────────────────────

ANSWER_SYSTEM_PROMPT = """\
You are a RAG Q&A assistant that synthesizes a final answer based on \
a complete knowledge exploration path (search tree).

Rules:
1. Use ONLY the knowledge contained in the search path below.
2. If the knowledge is insufficient, respond with: \
"I cannot answer this question."
3. Be concise and direct: output only the final answer, \
without explanations, references, or polite language.
4. You have access to the full search tree, including \
intermediate answers from sub-questions. Use this global \
context to produce the most accurate and comprehensive answer possible."""


# ─── 最终答案生成（CoT 变体）───

ANSWER_SYSTEM_PROMPT_COT = """\
You are a RAG Q&A assistant that synthesizes a final answer based on \
a complete knowledge exploration path (search tree).

Rules:
1. Use ONLY the knowledge contained in the search path below.
2. If the knowledge is insufficient, respond with: \
"I cannot answer this question."
3. **Think step by step before answering:**
   - First, review each question and its associated findings in the search tree
   - Then, chain the findings together to form a logical reasoning path
   - Finally, derive the answer from the reasoning chain
4. After your reasoning, end with a **concise and direct** answer in the format:
   "So the answer is: <answer>."
   - For yes/no questions: "yes" or "no"
   - For years: just the number
   - For names: just the name
   - Do NOT include explanations in the final answer line
5. You have access to the full search tree, including \
intermediate answers from sub-questions. Use this global \
context to produce the most accurate and comprehensive answer possible."""


def get_answer_prompt(search_path_text: str, question: str, cot: bool = False) -> list:
    """生成最终答案生成 prompt。

    Args:
        search_path_text: 格式化的搜索路径文本
        question: 原始问题
        cot: 是否使用 CoT 变体（逐步思考 → 简洁答案），默认 False
    """
    system = ANSWER_SYSTEM_PROMPT_COT if cot else ANSWER_SYSTEM_PROMPT
    return [
        SystemMessage(content=system),
        HumanMessage(content=f"[Search Path]\n{search_path_text}\n\n[Question]\n{question}"),
    ]
