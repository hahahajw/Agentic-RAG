"""RAG with Judge — 构建 LangGraph 单层 StateGraph

控制流：START → rewrite_query → batch_retrieve → judge → END

注意：递归控制由外层 Python 函数 (rag_with_judge) 管理，
不在此处使用 LangGraph 的递归或 Send API。
"""

from langgraph.graph import START, END, StateGraph

from rag_with_judge.state import JudgeRAGState
from rag_with_judge.nodes import rewrite_query_node, batch_retrieve_node, judge_node


def build_judge_rag_graph() -> StateGraph:
    """构建单层 RAG with Judge 的 StateGraph。

    每层执行：
    1. rewrite_query: 生成多个查询变体
    2. batch_retrieve: 用所有变体做多路检索
    3. judge: 判断检索到的 chunks 是否足以回答问题

    返回: 编译好的 CompiledStateGraph
    """
    builder = StateGraph(JudgeRAGState)

    # 添加节点
    builder.add_node("rewrite_query", rewrite_query_node)
    builder.add_node("batch_retrieve", batch_retrieve_node)
    builder.add_node("judge", judge_node)

    # 添加边
    builder.add_edge(START, "rewrite_query")
    builder.add_edge("rewrite_query", "batch_retrieve")
    builder.add_edge("batch_retrieve", "judge")
    builder.add_edge("judge", END)

    return builder.compile()
