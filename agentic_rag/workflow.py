"""Agentic RAG — 构建 LangGraph StateGraph

控制流：
START → plan → execute → evaluate → 条件路由:
  ├─ "answered" → generate_answer → END
  ├─ "progressing" → plan (循环)
  ├─ "stuck" (连续 × 1) → plan (循环)
  └─ "stuck" (连续 >= 2 轮) → reflect → execute (使用 pivot plan)

LangGraph 原生支持条件边实现循环，不需要 Python 递归。
"""

from langgraph.graph import START, END, StateGraph

from agentic_rag.state import AgenticRAGState
from agentic_rag.nodes import (
    plan_node,
    execute_node,
    evaluate_node,
    reflect_node,
    generate_answer_node,
    route_after_evaluate,
    route_after_reflect,
)


def build_agentic_rag_graph() -> StateGraph:
    """构建 Agentic RAG 的闭环 StateGraph。

    节点：
    1. plan: 生成探索计划（假设 + 目标 + 优先级）
    2. execute: 执行检索（重写查询 + 并行检索 + RRF 融合）
    3. evaluate: 评估当前轮次进度
    4. reflect: （条件触发）当连续 stuck 时反思
    5. generate_answer: 基于完整探索历史生成最终答案

    路由：
    - evaluate → {generate_answer | plan | reflect}
    - reflect → {execute | generate_answer}

    返回: 编译好的 CompiledStateGraph
    """
    builder = StateGraph(AgenticRAGState)

    # 添加节点
    builder.add_node("plan", plan_node)
    builder.add_node("execute", execute_node)
    builder.add_node("evaluate", evaluate_node)
    builder.add_node("reflect", reflect_node)
    builder.add_node("generate_answer", generate_answer_node)

    # 添加边
    builder.add_edge(START, "plan")
    builder.add_edge("plan", "execute")
    builder.add_edge("execute", "evaluate")
    builder.add_edge("generate_answer", END)

    # 条件路由
    builder.add_conditional_edges(
        "evaluate",
        route_after_evaluate,
        {
            "generate_answer": "generate_answer",
            "plan": "plan",
            "reflect": "reflect",
        },
    )

    builder.add_conditional_edges(
        "reflect",
        route_after_reflect,
        {
            "execute": "execute",
            "generate_answer": "generate_answer",
        },
    )

    return builder.compile()
