"""Agentic RAG — 状态定义

AgenticRAGState: LangGraph StateGraph 使用的 TypedDict
控制流: START → plan → execute → evaluate → (answered → generate_answer → END)
                                      → (progressing → plan 循环)
                                      → (stuck → reflect → plan 循环)
"""

import operator
from typing import Annotated, Any

from langgraph.graph import MessagesState


def _append_history(left: list, right: list) -> list:
    """Redducer for exploration_history: append new rounds."""
    return left + right


class AgenticRAGState(MessagesState):
    """Agentic RAG 的单层 workflow 状态。

    每轮探索追加到 exploration_history，供 Planner 和 Evaluator 参考。
    """

    # 当前问题
    query: str

    # 探索历史（线性链，每轮追加一个 round）
    # 使用 Annotated[list, operator.add] 使节点返回的新列表追加到现有历史
    exploration_history: Annotated[list[dict[str, Any]], operator.add]

    # 当前轮次的中间状态
    plan: dict[str, Any]              # Planner 输出的探索计划
    rewritten_queries: list[str]      # 查询变体
    chunks: list[dict[str, Any]]     # 检索结果
    evaluation: dict[str, Any]       # Evaluator 输出的评估结果

    # 终止条件
    final_answer: str                # 最终答案
    done: bool                       # 是否结束
