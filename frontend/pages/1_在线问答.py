"""在线问答页——统一 RAG 系统问答界面。

支持 4 种算法 × 2 种搜索源的在线提问，对话式交互。
所有算法通过 services.UnifiedRAGService 统一调用。
"""

import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import streamlit as st
from langchain_openai import ChatOpenAI

from services import UnifiedRAGService, AlgorithmType, SearchSource
from frontend.components.chunk_card import render_chunks_list
from frontend.components.search_tree import render_search_tree, render_search_tree_graph
from frontend.components.dag_viewer import render_dag_result
from frontend.components.graph_viewer import render_node_detail, render_tree_node_detail, _tree_to_graph
from frontend.components.metrics import render_batch_metrics
from frontend.components.config_panel import (
    ALGORITHM_INFO,
    MODEL_OPTIONS,
    ROLE_LABELS,
    render_algorithm_selector,
    render_search_source_selector,
    render_model_config,
    render_retrieval_params,
    render_prompt_editor,
    render_thinking_config,
)
from frontend.utils.ppt_exporter import build_pptx_bytes, build_interactive_html

# ═══════════════════════════════════════════════════════════════════
# 工具函数
# ═══════════════════════════════════════════════════════════════════


def _get_llm(model_name: str, extra_body: dict | None = None) -> ChatOpenAI:
    """创建 LLM 实例。

    Args:
        model_name: 模型名称
        extra_body: 额外的请求体参数。None 时默认禁用 thinking（Qwen API 安全默认值）。
                    传入 {"enable_thinking": True, ...} 时自动设置 timeout/max_retries。
    """
    api_key = st.session_state.get("live_api_key", "") or os.getenv("BL_API_KEY", "")
    base_url = st.session_state.get("live_base_url", "") or os.getenv("BL_BASE_URL", "")
    # 始终显式设置 enable_thinking=False 作为安全基线
    # （Qwen3 系列模型默认启用 thinking，与 function_calling structured output 冲突）
    _body: dict = {"enable_thinking": False}
    thinking_enabled = extra_body is not None and extra_body.get("enable_thinking") is True
    if extra_body is not None:
        _body.update(extra_body)
    kwargs: dict = dict(
        api_key=api_key, base_url=base_url,
        model=model_name, temperature=0.0,
        extra_body=_body,
    )
    if thinking_enabled:
        kwargs["timeout"] = 580.0
        kwargs["max_retries"] = 1
    return ChatOpenAI(**kwargs)


def _render_rewritten_queries(queries: list[str] | None) -> None:
    """渲染重写查询列表（Naive RAG 特有）。"""
    if queries:
        with st.expander(f"重写查询 ({len(queries)} 条)", expanded=False):
            for i, q in enumerate(queries, 1):
                st.markdown(f"{i}. `{q}`")


# ═══════════════════════════════════════════════════════════════════
# 主页面
# ═══════════════════════════════════════════════════════════════════


def main():
    st.title("在线问答")
    st.caption("选择 RAG 算法和搜索源，输入多跳问题进行测试")

    # ── Sidebar 配置区 ──
    with st.sidebar:
        st.header("系统配置")

        # API 配置
        with st.expander("API 配置", expanded=False):
            api_key = st.text_input(
                "API Key", value=os.getenv("BL_API_KEY", ""),
                type="password", key="live_api_key",
            )
            base_url = st.text_input(
                "Base URL", value=os.getenv("BL_BASE_URL", ""),
                key="live_base_url",
            )

        st.divider()

        # 算法选择
        algorithm = render_algorithm_selector(key="live_algorithm")

        # 搜索源（LLM Only 时隐藏）
        search_source = render_search_source_selector(algorithm, key="live_search_source")

        st.divider()

        # 模型配置
        role_models = render_model_config(algorithm)

        # 检索参数
        if search_source is not None:
            retrieval_params = render_retrieval_params(algorithm, search_source)
        else:
            retrieval_params = {}

        # Prompt 编辑器
        custom_prompts = render_prompt_editor(algorithm)

        # Thinking 模式配置（仅 rag_loop 显示）
        thinking_config = render_thinking_config(algorithm)

        st.divider()

        # 清除历史
        if st.button("清除对话历史", type="secondary", use_container_width=True):
            st.session_state.live_messages = []
            st.rerun()

    # ── 初始化 session state ──
    if "live_messages" not in st.session_state:
        st.session_state.live_messages = []

    # ── 显示对话历史 ──
    for idx, msg in enumerate(st.session_state.live_messages):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

            # 渲染结果可视化（仅 assistant 消息）
            if msg["role"] == "assistant" and msg.get("result"):
                result = msg["result"]

                # 错误展示
                if result.get("error"):
                    st.error(f"执行失败: {result['error']}")
                    continue

                # 算法特有可视化
                algo = AlgorithmType(result.get("algorithm", "llm-only"))
                elapsed = result.get("elapsed", 0)
                chunks = result.get("chunks", [])

                # 耗时 + chunk 数
                st.caption(
                    f"⏱ {elapsed:.1f}s | 算法: {ALGORITHM_INFO.get(algo, {}).get('label', algo.value)}"
                    f" | 来源: {len(chunks)} chunks"
                )

                # LLM Only: 仅答案（已在上方显示）
                if algo == AlgorithmType.LLM_ONLY:
                    pass

                # Naive RAG: 重写查询 + chunks
                elif algo == AlgorithmType.NAIVE_RAG:
                    rewritten = result.get("rewritten_queries")
                    _render_rewritten_queries(rewritten)
                    if chunks:
                        render_chunks_list(chunks)

                # RAG with Judge: 搜索树 (expander 视图, 历史回放用)
                elif algo == AlgorithmType.RAG_WITH_JUDGE:
                    rewritten = result.get("rewritten_queries")
                    _render_rewritten_queries(rewritten)
                    search_path = result.get("search_path")
                    if search_path:
                        with st.expander("递归探索树", expanded=False):
                            render_search_tree(search_path)
                    if chunks:
                        render_chunks_list(chunks)

                # rag_loop: DAG 概览 (历史回放用简化视图)
                elif algo == AlgorithmType.RAG_LOOP:
                    pipeline_result = result.get("pipeline_result")
                    if pipeline_result:
                        rounds = pipeline_result.get("total_rounds", "?")
                        searches = pipeline_result.get("total_search_calls", "?")
                        term = pipeline_result.get("termination_reason", "?")
                        st.caption(f"总轮次: {rounds} | 搜索次数: {searches} | 终止: {term}")
                    if chunks:
                        render_chunks_list(chunks)

                # ── 导出按钮 (历史消息也可见) ──
                if result and not result.get("error"):
                    try:
                        pptx_bytes = build_pptx_bytes(result)
                        html_str = build_interactive_html(result)
                        c1, c2 = st.columns(2)
                        with c1:
                            st.download_button(
                                "📥 导出 PPT",
                                pptx_bytes,
                                file_name=f"{(result.get('answer','')[:30] or 'result')}.pptx",
                                mime="application/vnd.openxmlformats-officedocument.presentationml.presentation",
                                key=f"dl_pptx_{idx}",
                            )
                        with c2:
                            st.download_button(
                                "🌐 导出交互式 HTML",
                                html_str.encode("utf-8"),
                                file_name=f"{(result.get('answer','')[:30] or 'result')}.html",
                                mime="text/html",
                                key=f"dl_html_{idx}",
                            )
                    except Exception:
                        pass  # pptx 生成失败不阻塞页面

    # ── 聊天输入 ──
    if prompt := st.chat_input("输入你的多跳问题..."):
        # 添加用户消息
        st.session_state.live_messages.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.markdown(prompt)

        # 执行查询
        with st.chat_message("assistant"):
            status = st.status("正在执行...", expanded=True)
            start_time = time.time()

            try:
                # 创建 LLM 实例 — role_models 始终包含 "answer" (render_model_config 保证)
                base_model = role_models.get("answer", MODEL_OPTIONS[0])

                # 根据 thinking_config 构建每个角色的 extra_body 覆盖和 role_methods
                # 基线 _get_llm 已设置 enable_thinking=False，此处仅需为启用 thinking 的角色覆盖
                role_methods: dict[str, str] = {}
                _role_extra: dict[str, dict | None] = {}

                for role in role_models:
                    tc = thinking_config.get(role, {})
                    if tc.get("enabled"):
                        _role_extra[role] = {
                            "enable_thinking": True,
                            "enable_search": False,
                            "thinking_budget": tc.get("budget", 8192),
                        }
                        # 使用 structured output 的角色需要 thinking_tool 方法
                        # （bind_tools 无 tool_choice，兼容 enable_thinking）
                        if algorithm == AlgorithmType.RAG_LOOP:
                            role_methods[role] = "thinking_tool"
                        elif algorithm == AlgorithmType.RAG_WITH_JUDGE and role == "judge":
                            role_methods[role] = "thinking_tool"
                    else:
                        _role_extra[role] = None  # 使用基线 enable_thinking=False

                llm = _get_llm(base_model, extra_body=_role_extra.get("answer"))

                # 分角色 LLM
                rewrite_llm = _get_llm(role_models["rewrite"], extra_body=_role_extra.get("rewrite")) if "rewrite" in role_models else None
                judge_llm = _get_llm(role_models["judge"], extra_body=_role_extra.get("judge")) if "judge" in role_models else None
                planner_llm = _get_llm(role_models["planner"], extra_body=_role_extra.get("planner")) if "planner" in role_models else None
                critic_llm = _get_llm(role_models["critic"], extra_body=_role_extra.get("critic")) if "critic" in role_models else None
                solver_llm = _get_llm(role_models["solver"], extra_body=_role_extra.get("solver")) if "solver" in role_models else None
                answer_llm = _get_llm(role_models.get("answer", base_model), extra_body=_role_extra.get("answer"))

                status.write("初始化完成，开始执行...")

                # 调用服务层
                service = UnifiedRAGService()

                max_chunks = retrieval_params.get("max_chunks", 8)
                max_depth = retrieval_params.get("max_depth", 3)
                max_rounds = retrieval_params.get("max_rounds", 5)
                dataset_type = retrieval_params.get("dataset_type", "hotpotqa")
                use_reranker = retrieval_params.get("use_reranker", False)

                # 映射 custom_prompts 到 service 参数
                cp_rewrite = custom_prompts.get("rewrite") if custom_prompts else None
                cp_answer = custom_prompts.get("answer") if custom_prompts else None
                cp_judge = custom_prompts.get("judge") if custom_prompts else None
                cp_planner = custom_prompts.get("planner") if custom_prompts else None
                cp_critic = custom_prompts.get("critic") if custom_prompts else None

                # 提示使用了哪些自定义提示词
                if custom_prompts:
                    roles_modified = [ROLE_LABELS.get(r, r) for r in custom_prompts]
                    st.toast(f"使用自定义提示词: {', '.join(roles_modified)}", icon="✏️")

                result = service.query(
                    question=prompt,
                    algorithm=algorithm,
                    search_source=search_source,
                    llm=llm,
                    rewrite_llm=rewrite_llm,
                    judge_llm=judge_llm,
                    answer_llm=answer_llm,
                    planner_llm=planner_llm,
                    critic_llm=critic_llm,
                    solver_llm=solver_llm,
                    dataset_type=dataset_type,
                    max_chunks=max_chunks,
                    max_depth=max_depth,
                    max_rounds=max_rounds,
                    use_reranker=use_reranker,
                    custom_rewrite_prompt=cp_rewrite,
                    custom_answer_prompt=cp_answer,
                    custom_judge_prompt=cp_judge,
                    custom_planner_prompt=cp_planner,
                    custom_critic_prompt=cp_critic,
                    role_methods=role_methods if role_methods else None,
                )

                elapsed = time.time() - start_time
                status.update(label=f"完成 ({elapsed:.1f}s)", state="complete")

                # ── 展示结果 ──
                if result.error:
                    st.error(f"执行失败: {result.error}")
                else:
                    # 答案正文
                    st.markdown("### 答案")
                    st.markdown(
                        result.answer
                        or "*（算法未生成答案文本——请查看下方可视化中各节点的 answer 字段）*"
                    )

                    if result.chunks:
                        render_chunks_list(result.chunks)

                    # 算法特有可视化（非交互部分保留在 chat 内）
                    if result.algorithm == AlgorithmType.NAIVE_RAG:
                        _render_rewritten_queries(result.rewritten_queries)

                    elif result.algorithm == AlgorithmType.RAG_WITH_JUDGE:
                        _render_rewritten_queries(result.rewritten_queries)
                        if result.search_path:
                            # 仅保留 expander 视图（非交互），交互图在下方持久化区块渲染
                            with st.expander("逐节点详情 (展开式)", expanded=False):
                                render_search_tree(result.search_path)

                    elif result.algorithm == AlgorithmType.RAG_LOOP:
                        # DAG 交互式可视化在下方持久化区块渲染，此处跳过
                        pass

                    # 底部指标
                    search_count = None
                    if result.pipeline_result:
                        search_count = result.pipeline_result.get("total_search_calls")
                    st.divider()
                    render_batch_metrics(
                        latency_ms=elapsed * 1000,
                        search_count=search_count,
                        chunk_count=len(result.chunks),
                    )

                    # ── 导出按钮 (当前回答, 无需等历史刷新) ──
                    export_result = {
                        "answer": result.answer,
                        "chunks": result.chunks,
                        "algorithm": result.algorithm.value,
                        "search_source": result.search_source.value if result.search_source else None,
                        "elapsed": elapsed,
                        "error": result.error,
                        "rewritten_queries": result.rewritten_queries,
                        "search_path": result.search_path,
                        "pipeline_result": result.pipeline_result,
                    }
                    try:
                        pptx_bytes = build_pptx_bytes(export_result)
                        html_str = build_interactive_html(export_result)
                        c1, c2 = st.columns(2)
                        with c1:
                            st.download_button(
                                "下载 PPT",
                                pptx_bytes,
                                file_name=f"{(result.answer or 'result')[:30]}.pptx",
                                mime="application/vnd.openxmlformats-officedocument.presentationml.presentation",
                                key="dl_pptx_current",
                            )
                        with c2:
                            st.download_button(
                                "下载交互式 HTML",
                                html_str.encode("utf-8"),
                                file_name=f"{(result.answer or 'result')[:30]}.html",
                                mime="text/html",
                                key="dl_html_current",
                            )
                    except Exception:
                        pass

                # 保存消息
                msg_record = {
                    "role": "assistant",
                    "content": result.answer if not result.error else f"错误: {result.error}",
                    "result": export_result,
                }
                st.session_state.live_messages.append(msg_record)
                # 持久化当前结果，供 widget rerun 时重新渲染交互式可视化
                st.session_state.live_current_result = export_result

            except Exception as e:
                elapsed = time.time() - start_time
                status.update(label=f"错误 ({elapsed:.1f}s)", state="error")

                # 区分错误类型给出中文提示
                err_type = type(e).__name__
                if "AuthenticationError" in err_type or "AuthError" in err_type:
                    error_msg = "API Key 验证失败 —— 请检查 API Key 是否正确配置"
                elif "ConnectionError" in err_type or "Connection" in err_type:
                    error_msg = "无法连接到服务 —— 请检查网络或 Base URL 配置"
                elif "TimeoutError" in err_type or "Timeout" in err_type:
                    error_msg = "请求超时 —— 请稍后重试或切换更快的模型"
                else:
                    error_msg = f"{err_type}: {e}"

                st.error(f"执行失败: {error_msg}")

                st.session_state.live_messages.append({
                    "role": "assistant",
                    "content": f"执行失败: {error_msg}",
                    "result": {"error": error_msg, "elapsed": elapsed},
                })

    # ── 持久化交互式可视化（widget rerun 时不会进入上方 if prompt: 块，故在此渲染）──
    current = st.session_state.get("live_current_result")
    if current and not current.get("error"):
        algo = AlgorithmType(current.get("algorithm", "llm-only"))
        if algo == AlgorithmType.RAG_LOOP and current.get("pipeline_result"):
            st.divider()
            st.markdown("### DAG 闭环执行")
            render_dag_result(current["pipeline_result"], key_suffix="live")
        elif algo == AlgorithmType.RAG_WITH_JUDGE and current.get("search_path"):
            st.divider()
            st.markdown("### 递归探索树")
            sel_key = "tree_selected"
            selected = st.session_state.get(sel_key, "")
            clicked = render_search_tree_graph(current["search_path"], selected=selected, key_suffix="live")
            if clicked:
                st.session_state[sel_key] = clicked
                nodes, _ = _tree_to_graph(current["search_path"])
                node = nodes.get(clicked)
                if node:
                    render_tree_node_detail(node, clicked)


main()