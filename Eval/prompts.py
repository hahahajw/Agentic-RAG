"""评估器共享 prompt 模板。"""

# ─── LLM-Only Prompt ─────────────────────────────────────────────

LLM_ONLY_SYSTEM_PROMPT = """\
You are a question-answering assistant. Answer the following question directly and \
concisely. Output only the final answer without explanation, references, or polite language."""

LLM_ONLY_USER_PROMPT = """Question: {question}"""

# ─── Naive RAG Prompt ────────────────────────────────────────────
# 复用 naive_rag/nodes.py 中的 _RAG_SYS_PROMPT 以保持一致性
# 此处不需要重复定义
