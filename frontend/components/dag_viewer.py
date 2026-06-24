"""DAG 结构可视化组件——用于 rag_loop 的 DAG 演化过程。

将 PipelineResult 中的 DAG 数据渲染为:
  1. 逐轮摘要卡片（节点数、SOLVED 数、搜索次数、终止原因）
  2. 交互式 DAG 结构图 (点击节点查看详情)
  3. 节点详情展开（answer、chunks、critic_health）

Usage:
    from frontend.components.dag_viewer import render_dag_result
    render_dag_result(pipeline_result)
"""

import streamlit as st

from frontend.components.graph_viewer import render_dag_dict, render_node_detail


def render_dag_result(pipeline_result: dict, key_suffix: str = "") -> None:
    """渲染 rag_loop 的完整执行结果。

    Args:
        pipeline_result: _pipeline_result_to_dict() 产出的 dict
        key_suffix: 用于区分同一页面多个 DAG 实例的唯一后缀 (如 question_index)
    """
    if not pipeline_result:
        return

    total_rounds = pipeline_result.get("total_rounds", 0)
    total_searches = pipeline_result.get("total_search_calls", 0)
    termination = pipeline_result.get("termination_reason", "?")
    round_dags = pipeline_result.get("round_dags", [])
    final_dag = pipeline_result.get("final_dag", {})

    # ── 概览指标 ──
    st.markdown("### DAG 闭环执行概览")
    cols = st.columns(4)
    cols[0].metric("总轮次", total_rounds)
    cols[1].metric("总搜索次数", total_searches)
    cols[2].metric("终止原因", _term_label(termination))
    cols[3].metric("最终节点数", len(final_dag.get("nodes", {})))

    # ── 逐轮 DAG 演化 ──
    if round_dags:
        st.markdown("### DAG 逐轮演化")
        tabs = st.tabs([f"Round {i+1}" for i in range(len(round_dags))])
        for i, (tab, dag) in enumerate(zip(tabs, round_dags)):
            with tab:
                _render_dag_round(dag, round_num=i+1)

    # ── DAG 交互式结构 (可选轮次) ──
    if round_dags or final_dag:
        st.markdown("### DAG 交互式结构")

        # 构建可选轮次列表
        round_options = []
        if round_dags:
            round_options.extend([f"Round {i+1}" for i in range(len(round_dags))])
        if final_dag and final_dag not in round_dags:
            round_options.append(f"最终 DAG (Round {total_rounds})")

        if len(round_options) > 1:
            selected_label = st.selectbox(
                "选择轮次查看 DAG 拓扑", round_options, key=f"dag_round_{key_suffix}"
            )
        else:
            selected_label = round_options[0]

        # 解析选中的 DAG
        if "最终" in selected_label:
            display_dag = final_dag
        else:
            idx = int(selected_label.split()[1]) - 1
            display_dag = round_dags[idx] if idx < len(round_dags) else final_dag

        sel_key = f"dag_selected_{key_suffix}" if key_suffix else "dag_selected"
        selected = st.session_state.get(sel_key, "")
        clicked = render_dag_dict(display_dag, selected=selected, key_suffix=key_suffix)
        if clicked:
            st.session_state[sel_key] = clicked
            node = display_dag.get("nodes", {}).get(clicked)
            if node:
                render_node_detail(node, clicked)


def _render_dag_round(dag: dict, round_num: int) -> None:
    """渲染单轮 DAG 快照。"""
    nodes = dag.get("nodes", {})
    edges = dag.get("edges", [])

    solved = sum(1 for n in nodes.values() if n.get("status") == "solved")
    unhealthy = sum(1 for n in nodes.values()
                    if n.get("health") in ("unreliable", "blocked"))

    cols = st.columns(3)
    cols[0].metric("节点数", len(nodes))
    cols[1].metric("SOLVED", solved)
    cols[2].metric("异常节点", unhealthy)

    # 节点列表
    with st.expander(f"节点详情 ({len(nodes)} 个)", expanded=False):
        for nid, node in nodes.items():
            status = node.get("status", "?")
            health = node.get("health", "?")
            question = node.get("question", "")[:80]
            answer = node.get("answer", "")[:100]

            icon = "✅" if status == "solved" else "⏳"
            health_icon = {"healthy": "💚", "needs_verification": "⚠️",
                          "unreliable": "🔴", "blocked": "🚫"}.get(health, "❓")

            st.markdown(
                f"{icon}{health_icon} **{nid}** [{status}] "
                f"*{question}*"
            )
            if answer:
                st.caption(f"  → {answer}")

    # 边列表
    if edges:
        with st.expander(f"边 ({len(edges)} 条)", expanded=False):
            for e in edges:
                etype = e.get("type", "?")
                style = "实线" if etype == "decomposition" else "虚线"
                st.caption(f"{e.get('from', '?')} → {e.get('to', '?')} [{etype}] {style}")


def _term_label(reason: str) -> str:
    """终止原因映射为中文标签。"""
    return {
        "all_conditions_met": "全部条件满足",
        "max_rounds": "达到最大轮次",
        "planner_failure": "Planner 失败",
    }.get(reason, reason)