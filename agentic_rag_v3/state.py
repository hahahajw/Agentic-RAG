"""Agentic RAG v2 — 状态定义

新增核心字段 sub_questions：记录每个子问题的解决状态。
每个子问题独立检索、独立评估、独立记录答案。
"""

import operator
from typing import Annotated, Any

from langgraph.graph import MessagesState


class AgenticRAGV2State(MessagesState):
    """Agentic RAG v2 的状态。

    与 v1 的核心区别：
    - sub_questions: 逐一追踪每个子问题的解决状态
    - exploration_history: 保留每轮探索记录
    - evaluation: 合成器的评估结果（不再是 Evaluator 的三态评估）
    """

    # 原始问题
    query: str

    # 子问题解决状态（核心新增）
    # 每轮结果追加到此列表，累积所有轮次的子问题解决记录
    # 结构：{"question": str, "status": str, "answer": str, "retrieved_chunks": list}
    # status: "solved" | "unsolved" | "stuck"
    sub_questions: Annotated[list[dict[str, Any]], operator.add]

    # 探索历史（线性链，每轮追加）
    exploration_history: Annotated[list[dict[str, Any]], operator.add]

    # 当前轮次中间状态
    plan: dict[str, Any]              # Planner 输出的计划
    evaluation: dict[str, Any]        # Synthesizer 输出的评估结果

    # 终止条件
    final_answer: str
    done: bool
