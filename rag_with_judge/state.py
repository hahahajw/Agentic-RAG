"""RAG with Judge — 状态定义

1. SEARCH_PATH: 嵌套 dict 结构，表示一棵递归搜索树
   - 通过 Python 引用传递，子节点修改自动反映在父节点
   - 可序列化为 JSON，直接用于前端渲染或持久化

2. JudgeRAGState: LangGraph StateGraph 使用的 TypedDict
   - 仅描述「单层」执行所需的状态（rewrite → retrieve → judge）
   - 递归控制由外层 Python 函数管理，不在 LangGraph 中处理
"""

from typing import Any

from langgraph.graph import MessagesState


# ─── SEARCH_PATH 结构约定（文档，非代码）─────────────────────────
#
# SEARCH_PATH = {
#     "question": "原始问题文本",
#     "answer": "最终答案（None 表示尚未生成）",
#     "chunks": [                           # 本轮检索到的 chunks
#         {
#             "chunk_id": "...",
#             "chunk_title": "...",
#             "chunk_summary": "...",
#             "context_title": "...",       # 用于计算检索指标
#             "page_content": "...",
#             "score": 0.5,
#         },
#     ],
#     "answerable": False,                  # Judge 判断结果
#     "next_queries": [                     # 如果 answerable=False
#         { "question": "...", "answer": "...", "chunks": [...], "answerable": True, "next_queries": [] },
#         ...
#     ],
# }
#
# 注意：next_queries 用 list 而非 dict，因为：
#   - 问题文本作为 key 不可靠（特殊字符、长度、重复）
#   - list 在 JSON 序列化后更易读
#   - 并行写入时 append 比 dict key 赋值更安全


class JudgeRAGState(MessagesState):
    """LangGraph 单层 workflow 的状态。

    控制流：START → rewrite_query → batch_retrieve → judge → END
    """

    # 当前问题（由递归函数传入）
    query: str

    # 重写后的查询变体
    rewritten_queries: list[str]

    # 检索结果：[{chunk_id, chunk_title, chunk_summary, context_title,
    #             page_content, score}, ...]
    chunks: list[dict[str, Any]]

    # Judge 判断结果
    answerable: bool
    next_queries: list[str]
    judgement_reason: str
