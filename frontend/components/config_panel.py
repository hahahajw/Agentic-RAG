"""算法配置面板组件——Sidebar 中的模型、检索参数、Prompt 编辑。

Usage:
    from frontend.components.config_panel import (
        render_algorithm_selector,
        render_search_source_selector,
        render_model_config,
        render_retrieval_params,
        render_prompt_editor,
    )
"""

from typing import Any

import streamlit as st

from services.result_types import AlgorithmType, SearchSource


# ═══════════════════════════════════════════════════════════════════
# 算法展示信息
# ═══════════════════════════════════════════════════════════════════

ALGORITHM_INFO: dict[AlgorithmType, dict[str, str]] = {
    AlgorithmType.LLM_ONLY: {
        "label": "LLM Only（直接回答）",
        "help": "直接调用大语言模型回答，不进行任何检索",
        "icon": "💬",
    },
    AlgorithmType.NAIVE_RAG: {
        "label": "模块化 RAG",
        "help": "多查询重写 + RRF 融合 + 答案生成",
        "icon": "📊",
    },
    AlgorithmType.RAG_WITH_JUDGE: {
        "label": "递归检索 RAG",
        "help": "Judge 判断知识充足性 + 递归生成 follow-up 问题 + 搜索树探索",
        "icon": "🌳",
    },
    AlgorithmType.RAG_LOOP: {
        "label": "规划-执行-反馈闭环 RAG",
        "help": "Planner-Solver-Critic DAG 闭环，结构收敛 + 答案质量双重保障",
        "icon": "🔄",
    },
}

# 每种算法需要的角色模型
ALGORITHM_ROLES: dict[AlgorithmType, list[str]] = {
    AlgorithmType.LLM_ONLY: ["answer"],
    AlgorithmType.NAIVE_RAG: ["rewrite", "answer"],
    AlgorithmType.RAG_WITH_JUDGE: ["rewrite", "judge", "answer"],
    AlgorithmType.RAG_LOOP: ["planner", "critic", "solver", "answer"],
}

ROLE_LABELS: dict[str, str] = {
    "rewrite": "重写模型",
    "judge": "Judge 模型",
    "answer": "答案模型",
    "planner": "Planner 模型",
    "critic": "Critic 模型",
    "solver": "Solver 模型",
}

MODEL_OPTIONS = [
    "qwen3.6-plus", "qwen3.5-plus", "qwen-plus", "qwen-max",
    "qwen3-max", "qwen3-235b-a22b",
]

DATASET_OPTIONS = ["hotpotqa", "2wikimultihopqa", "musique"]


# ═══════════════════════════════════════════════════════════════════
# 选择器组件
# ═══════════════════════════════════════════════════════════════════


def _resolve_algorithm_index(key: str, options: list[AlgorithmType]) -> int:
    """从 session_state 解析算法选择器的默认索引。

    处理三种情况: 首次加载(int 0)、枚举值(Streamlit 自动存入)、旧字符串值。
    """
    if key not in st.session_state:
        return 0
    val = st.session_state[key]
    if isinstance(val, AlgorithmType) and val in options:
        return options.index(val)
    if isinstance(val, str):
        try:
            alg = AlgorithmType(val)
            if alg in options:
                return options.index(alg)
        except ValueError:
            pass
    if isinstance(val, int) and 0 <= val < len(options):
        return val
    return 0


def render_algorithm_selector(key: str = "live_algorithm") -> AlgorithmType:
    """渲染算法选择器。

    Returns:
        选中的 AlgorithmType
    """
    options = list(ALGORITHM_INFO.keys())
    default_idx = _resolve_algorithm_index(key, options)

    selected = st.radio(
        "选择算法",
        options=options,
        index=default_idx,
        format_func=lambda a: f"{ALGORITHM_INFO[a]['icon']} {ALGORITHM_INFO[a]['label']}",
        help="\n\n".join(f"{a.value}: {ALGORITHM_INFO[a]['help']}" for a in options),
        key=key,
    )
    return selected


def render_search_source_selector(
    algorithm: AlgorithmType,
    key: str = "live_search_source",
) -> SearchSource | None:
    """渲染搜索源选择器。LLM Only 时返回 None。

    Returns:
        选中的 SearchSource，或 None（LLM Only 模式）
    """
    if algorithm == AlgorithmType.LLM_ONLY:
        return None

    options = [
        SearchSource.MILVUS,
        SearchSource.WEB,
        SearchSource.TAVILY,
    ]

    return st.radio(
        "搜索源",
        options=options,
        format_func=lambda s: {
            SearchSource.MILVUS: "Milvus 向量数据库",
            SearchSource.WEB: "网络搜索 (DuckDuckGo)",
            SearchSource.TAVILY: "网络搜索 (Tavily)",
        }.get(s, s.value),
        key=key,
    )


# ═══════════════════════════════════════════════════════════════════
# 模型配置组件
# ═══════════════════════════════════════════════════════════════════


def render_model_config(
    algorithm: AlgorithmType,
    base_key: str = "live_base_model",
    role_prefix: str = "live",
) -> dict[str, str]:
    """渲染算法模型配置区域。

    基础模型 + 算法特定角色模型。

    Returns:
        {role: model_name} dict
    """
    st.subheader("模型配置")

    base_model = st.selectbox(
        "基础模型",
        options=MODEL_OPTIONS,
        key=base_key,
    )

    roles = ALGORITHM_ROLES.get(algorithm, ["answer"])
    role_models = {}

    if len(roles) > 1:
        with st.expander("分角色模型配置", expanded=False):
            for role in roles:
                role_model = st.selectbox(
                    ROLE_LABELS.get(role, role),
                    options=["(使用基础模型)"] + MODEL_OPTIONS,
                    key=f"{role_prefix}_{role}_model",
                )
                if role_model != "(使用基础模型)":
                    role_models[role] = role_model

    # 如果某角色未配置，使用基础模型
    for role in roles:
        if role not in role_models:
            role_models[role] = base_model

    return role_models


# ═══════════════════════════════════════════════════════════════════
# Thinking 模式配置（仅 rag_loop）
# ═══════════════════════════════════════════════════════════════════


def render_thinking_config(algorithm: AlgorithmType) -> dict[str, dict]:
    """渲染 Thinking 模式配置——为每个角色独立设置启用/禁用和 budget。

    对所有算法显示。对于 rag_loop，启用 thinking 的角色使用 `thinking_tool`
    结构化输出方法（替代 function_calling 以兼容 enable_thinking）。
    对于其他算法（Naive RAG、RAG with Judge、LLM Only），thinking 通过
    extra_body 的 enable_thinking=True 启用，无需特殊方法。

    Returns:
        {role: {"enabled": bool, "budget": int}} dict，
        例如 {"planner": {"enabled": True, "budget": 8192}}
    """
    thinking_config: dict[str, dict] = {}

    roles = ALGORITHM_ROLES.get(algorithm, [])
    if not roles:
        return thinking_config

    st.divider()
    with st.expander("Thinking 模式配置", expanded=False):
        st.caption(
            "启用 Thinking 的角色将在请求中设置 `enable_thinking=True`，"
            "并配置思考 token 预算（thinking_budget）。"
            "注意：启用 Thinking 会显著增加延迟（~80s/8192 budget）。"
        )
        for role in roles:
            enabled = st.toggle(
                f"启用 {ROLE_LABELS.get(role, role)} Thinking",
                key=f"think_{role}",
                value=True,
            )
            budget = 8192
            if enabled:
                budget = st.number_input(
                    f"{ROLE_LABELS.get(role, role)} Thinking Budget",
                    min_value=1024, max_value=32768, value=8192, step=1024,
                    key=f"think_budget_{role}",
                )
            thinking_config[role] = {"enabled": enabled, "budget": budget}

    return thinking_config


# ═══════════════════════════════════════════════════════════════════
# 检索参数组件
# ═══════════════════════════════════════════════════════════════════


def render_retrieval_params(
    algorithm: AlgorithmType,
    search_source: SearchSource | None,
) -> dict[str, Any]:
    """渲染检索参配置区域。

    Returns:
        参数字典: max_chunks, dataset_type, max_depth?, max_rounds?, use_reranker?
    """
    st.subheader("检索参数")

    params: dict[str, Any] = {}

    params["max_chunks"] = st.slider(
        "最大 Chunk 数", min_value=3, max_value=20, value=10,
        key="live_max_chunks",
    )

    if search_source == SearchSource.MILVUS:
        params["dataset_type"] = st.selectbox(
            "数据集", options=DATASET_OPTIONS, key="live_dataset",
        )
        params["use_reranker"] = st.toggle("启用 Reranker", value=False, key="live_reranker",
                                        help="使用 qwen3-rerank 模型对 Milvus 检索结果重排序。仅 Milvus 搜索源可用。")

    if algorithm == AlgorithmType.RAG_WITH_JUDGE:
        params["max_depth"] = st.slider(
            "最大递归深度", min_value=1, max_value=5, value=3,
            key="live_max_depth",
        )

    if algorithm == AlgorithmType.RAG_LOOP:
        params["max_rounds"] = st.slider(
            "最大探索轮次", min_value=1, max_value=10, value=5,
            key="live_max_rounds",
        )

    return params


# ═══════════════════════════════════════════════════════════════════
# Prompt 编辑器组件
# ═══════════════════════════════════════════════════════════════════


# Prompt 默认模板映射（从各算法模块导入）
def _get_default_prompts(algorithm: AlgorithmType) -> dict[str, str]:
    """获取算法的默认 prompt 模板。懒加载——仅在用户展开编辑器时导入。"""
    import logging
    logger = logging.getLogger(__name__)
    defaults: dict[str, str] = {}

    if algorithm == AlgorithmType.NAIVE_RAG:
        try:
            from naive_rag.prompts import QUERY_REWRITE_PROMPT
            defaults["rewrite"] = QUERY_REWRITE_PROMPT
        except ImportError as e:
            logger.warning("无法加载 Naive RAG rewrite prompt: %s", e)
        try:
            from naive_rag.nodes import RAG_SYS_PROMPT
            defaults["answer"] = RAG_SYS_PROMPT
        except ImportError as e:
            logger.warning("无法加载 Naive RAG answer prompt: %s", e)

    elif algorithm == AlgorithmType.RAG_WITH_JUDGE:
        try:
            from naive_rag.prompts import QUERY_REWRITE_PROMPT
            defaults["rewrite"] = QUERY_REWRITE_PROMPT
        except ImportError as e:
            logger.warning("无法加载 Judge rewrite prompt: %s", e)
        try:
            from rag_with_judge.prompts import (
                JUDGE_PROMPT_B_TEMPLATE,
                ANSWER_SYSTEM_PROMPT,
            )
            defaults["judge"] = JUDGE_PROMPT_B_TEMPLATE
            defaults["answer"] = ANSWER_SYSTEM_PROMPT
        except ImportError as e:
            logger.warning("无法加载 Judge prompts: %s", e)

    elif algorithm == AlgorithmType.RAG_LOOP:
        try:
            from rag_loop.planner import PLANNER_SYSTEM_TEMPLATE
            defaults["planner"] = PLANNER_SYSTEM_TEMPLATE
        except ImportError as e:
            logger.warning("无法加载 rag_loop planner prompt: %s", e)
        try:
            from rag_loop.critic import CRITIC_SYSTEM_TEMPLATE
            defaults["critic"] = CRITIC_SYSTEM_TEMPLATE
        except ImportError as e:
            logger.warning("无法加载 rag_loop critic prompt: %s", e)
        try:
            from rag_loop.answer_generator import ANSWER_GENERATOR_SYSTEM_TEMPLATE
            defaults["answer"] = ANSWER_GENERATOR_SYSTEM_TEMPLATE
        except ImportError as e:
            logger.warning("无法加载 rag_loop answer prompt: %s", e)

    return defaults


def render_prompt_editor(algorithm: AlgorithmType) -> dict[str, str]:
    """渲染 Prompt 模板编辑器。

    Returns:
        {role: custom_prompt_text} dict，仅包含被修改的 prompt
    """
    st.divider()
    with st.expander("Prompt 模板配置", expanded=False):
        defaults = _get_default_prompts(algorithm)
        if not defaults:
            st.caption("此算法无可编辑的 Prompt 模板")
            return {}

        custom_prompts: dict[str, str] = {}

        for role, default_text in defaults.items():
            state_key = f"prompt_{algorithm.value}_{role}"
            if state_key not in st.session_state:
                st.session_state[state_key] = default_text

            with st.container(border=True):
                st.caption(f"**{ROLE_LABELS.get(role, role)}**")
                new_val = st.text_area(
                    f"编辑 {role} prompt",
                    value=st.session_state[state_key],
                    height=150,
                    key=f"ta_{state_key}",
                    label_visibility="collapsed",
                )
                st.session_state[state_key] = new_val

                cols = st.columns([1, 1])
                if cols[0].button("重置默认", key=f"reset_{state_key}"):
                    st.session_state[state_key] = default_text
                    st.rerun()

                if new_val != default_text:
                    custom_prompts[role] = new_val
                    cols[1].success("✓ 已自定义", icon="✅")

        return custom_prompts