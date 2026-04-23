"""Agentic RAG v3 — 构建 LangGraph StateGraph

控制流：
START → plan → solve_sub_questions → synthesize
  ├─ "complete" → generate_answer → END
  ├─ "incomplete" → plan (补充子问题)
  ├─ "stuck" (连续 × 1) → plan (再试，带小提醒)
  └─ "stuck" (连续 >= 2) → reflect → solve_sub_questions (换角度直接执行)

与 v1 的核心区别：
- v1: plan → execute(RRF 融合) → evaluate(全局三态)
- v2: plan → solve_sub_questions(每个子问题重写→检索→评估) → synthesize(推理链检查)
"""

from langgraph.graph import START, END, StateGraph

from agentic_rag_v3.state import AgenticRAGV2State
from agentic_rag_v3.nodes import (
    plan_node,
    solve_sub_questions,
    synthesize_node,
    reflect_node,
    generate_answer_node,
    route_after_synthesize,
    route_after_reflect,
)


def build_agentic_rag_v3_graph() -> StateGraph:
    """构建 Agentic RAG v3 的闭环 StateGraph。

    节点：
    1. plan: 生成需要解决的子问题（假设 + 子问题 + 优先级）
    2. solve_sub_questions: 逐一检索 + 评估每个子问题
    3. synthesize: 检查推理链完整性
    4. reflect: （条件触发）当连续 stuck 时反思
    5. generate_answer: 基于推理链生成最终答案

    路由：
    - synthesize → {generate_answer | plan | reflect}
    - reflect → {solve_sub_questions | generate_answer}

    返回: 编译好的 CompiledStateGraph
    """
    builder = StateGraph(AgenticRAGV2State)

    # 添加节点
    builder.add_node("plan", plan_node)
    builder.add_node("solve_sub_questions", solve_sub_questions)
    builder.add_node("synthesize", synthesize_node)
    builder.add_node("reflect", reflect_node)
    builder.add_node("generate_answer", generate_answer_node)

    # 添加边
    builder.add_edge(START, "plan")
    builder.add_edge("plan", "solve_sub_questions")
    builder.add_edge("solve_sub_questions", "synthesize")
    builder.add_edge("generate_answer", END)

    # 条件路由
    builder.add_conditional_edges(
        "synthesize",
        route_after_synthesize,
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
            "solve_sub_questions": "solve_sub_questions",
            "generate_answer": "generate_answer",
        },
    )

    return builder.compile()
