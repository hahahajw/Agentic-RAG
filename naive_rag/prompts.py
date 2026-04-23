"""所有 prompt 模板"""

QUERY_REWRITE_PROMPT = """\
You are an expert search query generator. Given a user question, generate 4 alternative versions
that could be used to search for relevant information. Each version should emphasize different
aspects or use different terminology while preserving the original meaning.

Requirements:
- Generate exactly 4 alternative queries
- Each query should be concise and search-optimized
- Use different angles: synonyms, rephrasing, different perspectives
- Output ONLY a JSON array of 4 strings, nothing else

Example:
Input: "What year was the university where Einstein taught founded?"
Output: ["Einstein university professor founding year", "institution where Albert Einstein worked establishment date", "university founded date Einstein faculty", "year of establishment of Einstein's academic institution"]

Input: "{query}"
Output:"""

QUERY_REWRITE_WITH_CONTEXT_PROMPT = """\
You are an expert search query generator. You have some known facts from previous searches that can help you generate more precise queries.

### Known Facts
{context}

### Original Question
{query}

### Task
Generate exactly 4 alternative search queries that could find additional information. Use the known facts to:
1. Replace vague references with specific entity names (e.g., "that university" → "Moscow State University")
2. Build on established facts rather than repeating what's already known
3. Use different angles: synonyms, rephrasing, different perspectives

Output ONLY a JSON array of 4 strings, nothing else."""

FOLLOWUP_PROMPT = """\
Based on the following conversation, suggest 3 potential follow-up questions.
The questions should explore related aspects, implications, or details.

Question: {original_query}
Answer: {answer}

Output ONLY a JSON array of 3 strings, nothing else."""
