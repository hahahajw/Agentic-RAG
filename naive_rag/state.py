"""LangGraph State 定义"""

from typing import List, Annotated
from langchain_core.documents import Document
from langgraph.graph import MessagesState


class NaiveRAGState(MessagesState):
    original_query: str
    rewritten_queries: List[str]
    all_queries: List[str]
    # {query: [(Document, score), ...]} — merged via custom reducer
    retrieval_results: Annotated[dict, lambda old, new: {**old, **new}]
    fused_chunks: List[tuple[Document, float]]
    answer: str
    suggested_followups: List[str]
