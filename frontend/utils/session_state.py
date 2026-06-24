"""Session State Key 集中定义。

所有 Streamlit session_state key 在此集中管理，避免各页面各自定义导致冲突。
命名规范: {作用域}_{用途}，如 live_algorithm、result_pipeline。
"""


class SessionKeys:
    """Session State Key 常量——用于 st.session_state 读写。"""

    # ═══════════════════════════════════════════════════════════════
    # 在线问答页
    # ═══════════════════════════════════════════════════════════════

    LIVE_ALGORITHM = "live_algorithm"           # AlgorithmType 值
    LIVE_SEARCH_SOURCE = "live_search_source"   # SearchSource 值
    LIVE_DATASET = "live_dataset"               # str: hotpotqa/2wikimultihopqa/musique
    LIVE_MESSAGES = "live_messages"             # list[dict]: 对话历史
    LIVE_RESULT = "live_result"                 # UnifiedResult | None

    # 模型
    LIVE_BASE_MODEL = "live_base_model"         # str: 基础模型名
    LIVE_REWRITE_MODEL = "live_rewrite_model"   # str
    LIVE_JUDGE_MODEL = "live_judge_model"       # str
    LIVE_ANSWER_MODEL = "live_answer_model"     # str
    LIVE_PLANNER_MODEL = "live_planner_model"   # str
    LIVE_CRITIC_MODEL = "live_critic_model"     # str
    LIVE_SOLVER_MODEL = "live_solver_model"     # str

    # 检索参数
    LIVE_MAX_CHUNKS = "live_max_chunks"         # int: 3-20
    LIVE_MAX_DEPTH = "live_max_depth"           # int: 1-5 (RAG with Judge)
    LIVE_MAX_ROUNDS = "live_max_rounds"         # int: 1-10 (rag_loop)
    LIVE_USE_RERANKER = "live_use_reranker"     # bool

    # Prompt 编辑
    # 动态 key: f"prompt_{algorithm}_{role}"，如 prompt_naive-rag_rewrite
    PROMPT_PREFIX = "prompt_"

    # ═══════════════════════════════════════════════════════════════
    # 实验结果页
    # ═══════════════════════════════════════════════════════════════

    RESULT_PIPELINE = "result_pipeline"         # str: AlgorithmType 值
    RESULT_DATASET = "result_dataset"           # str
    RESULT_SEARCH = "result_search"             # str: 问题搜索文本
    RESULT_PAGE = "result_page"                 # int: 当前页码

    # ═══════════════════════════════════════════════════════════════
    # F1 热力图页
    # ═══════════════════════════════════════════════════════════════

    HEATMAP_DATASET = "heatmap_dataset"         # str