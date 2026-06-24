"""实验结果浏览页——统一浏览所有 RAG 算法的评估结果。

支持算法: LLM Only / 模块化 RAG / 递归检索 RAG / 规划-执行-反馈闭环 RAG
额外: 系统对比（预生成图表）

使用 frontend/components/ 共享组件渲染可视化。
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd
import streamlit as st

from frontend.data_loader import list_available_datasets, load_results
from frontend.components.metrics import render_aggregate_metrics
from frontend.components.chunk_card import render_chunk_card
from frontend.components.search_tree import render_search_tree, render_search_tree_graph
from frontend.components.dag_viewer import render_dag_result
from frontend.components.graph_viewer import render_node_detail, render_tree_node_detail, _tree_to_graph
from frontend.utils.metrics_aggregator import (
    DIMENSION_DEFS, compute_aggregate, compute_type_breakdown,
    load_benchmark_types, render_metrics_by_dimension, render_type_breakdown_table,
)
from frontend.utils.heatmap_utils import (
    build_heatmap_html, load_question_data, sort_questions,
    DS_KEYS, DS_LABELS, f1_heatmap_png,
)

# 系统对比: 基于 Plotly 的动态图表
from frontend.utils.comparison_data import (
    load_comparison_data, ALGO_KEYS, DS_KEYS as COMP_DS_KEYS,
    ALGO_LABELS as COMP_ALGO_LABELS, DS_LABELS as COMP_DS_LABELS,
    ALGO_COLORS,
)
from frontend.components.comparison_charts import (
    build_answer_quality_chart, build_radar_chart,
    build_retrieval_quality_chart, build_efficiency_chart, build_efficiency_bars,
    build_type_heatmap, type_heatmap_png,
    build_retrieval_cdf_chart,
)

# ═══════════════════════════════════════════════════════════════════
# 管道配置
# ═══════════════════════════════════════════════════════════════════

PIPELINE_CONFIG = {
    "llm-only": {"label": "LLM Only 结果", "icon": "💬"},
    "naive-rag": {"label": "模块化 RAG 结果", "icon": "📊"},
    "rag-with-judge": {"label": "递归检索 RAG 结果", "icon": "🌳"},
    "rag-loop": {"label": "规划-执行-反馈闭环 RAG 结果", "icon": "🔄"},
    "system-comparison": {"label": "系统对比", "icon": "📈"},
}

# ═══════════════════════════════════════════════════════════════════
# 工具函数
# ═══════════════════════════════════════════════════════════════════

def _format_ms(ms: float) -> str:
    if ms < 1000:
        return f"{ms:.0f}ms"
    return f"{ms / 1000:.1f}s"


# ═══════════════════════════════════════════════════════════════════
# 指标汇总
# ═══════════════════════════════════════════════════════════════════

def _render_metrics_summary(results: dict, pipeline: str) -> None:
    """渲染评估汇总指标——按算法类型分维度展示。

    各算法的指标维度由 DIMENSION_DEFS 定义:
      - LLM Only: 生成质量 + 效率指标
      - Naive RAG: 生成质量 + 检索质量(含 MRR) + 效率指标
      - Judge / rag_loop: 生成质量 + 检索质量(无 MRR) + 效率(含搜索深度)
    """
    summary = results.get("summary", {})
    items = results.get("results", [])
    aggregate = compute_aggregate(items)

    st.markdown("### 指标汇总")

    # 维度分组指标
    render_metrics_by_dimension(aggregate, pipeline)


@st.cache_data(ttl=300)
def _load_benchmark_types_cached(ds_key: str) -> list[str]:
    """缓存 benchmark 类型加载——避免每次刷新重新读文件。"""
    return load_benchmark_types(ds_key)


def _render_type_breakdown_section(
    items: list[dict], ds_key: str, pipeline: str
) -> None:
    """渲染问题类型细分表格。

    从 benchmark 数据获取每个问题的问题类型，按类型分组计算聚合指标，
    以表格形式展示各维度指标在不同问题类型上的表现。
    """
    st.markdown("### 问题类型细分")

    types_list = _load_benchmark_types_cached(ds_key)
    if not types_list:
        st.info("无法加载问题类型数据——benchmark 文件缺失")
        return

    breakdown = compute_type_breakdown(items, ds_key)
    if not breakdown:
        st.info("无可用的问题类型数据")
        return

    render_type_breakdown_table(breakdown, pipeline)


# ═══════════════════════════════════════════════════════════════════
# 结果详情（按算法分派）
# ═══════════════════════════════════════════════════════════════════

def _render_result_header(item: dict) -> None:
    """渲染问题 / 标准答案 / 预测结果。"""
    col_q, col_gt, col_pred = st.columns(3)
    col_q.markdown(f"**问题:** {item.get('question', 'N/A')}")
    col_gt.markdown(f"**标准答案:** {item.get('answer', 'N/A')}")
    col_pred.markdown(f"**预测:** {item.get('prediction', 'N/A')}")

    pred = (item.get("prediction") or "").strip().lower()
    ans = (item.get("answer") or "").strip().lower()
    if pred and ans and pred == ans:
        st.success("Exact Match: 正确")
    elif pred:
        st.error("Exact Match: 错误")

    error = item.get("error")
    if error:
        st.error(f"错误: {error}")

    latency = item.get("latency_ms")
    if latency:
        st.caption(f"耗时: {_format_ms(latency)}")


def _render_llm_only_detail(item: dict) -> None:
    """LLM Only: 仅显示预测 vs 答案对比。"""
    em = item.get("em", "?")
    f1 = item.get("f1", "?")
    cols = st.columns(2)
    cols[0].metric("EM", f"{em:.4f}" if isinstance(em, float) else em)
    cols[1].metric("F1", f"{f1:.4f}" if isinstance(f1, float) else f1)


def _render_naive_rag_detail(item: dict) -> None:
    """模块化 RAG: 重写查询 + chunks。"""
    rewritten = item.get("rewritten_queries", [])
    if rewritten:
        with st.expander(f"重写查询 ({len(rewritten)})", expanded=False):
            for i, q in enumerate(rewritten, 1):
                st.markdown(f"{i}. {q}")

    chunks = item.get("chunks", item.get("fused_chunks", []))
    if chunks:
        with st.expander(f"检索来源 ({len(chunks)})", expanded=False):
            for i, c in enumerate(chunks, 1):
                render_chunk_card(c, rank=i)


def _render_judge_detail(item: dict) -> None:
    """递归检索 RAG: 搜索树可视化 (交互式图 + expander 备选)。"""
    search_path = item.get("search_path", {})
    if search_path:
        sel_key = f"judge_selected_{item.get('question_index', 0)}"
        selected = st.session_state.get(sel_key, "")
        qidx = str(item.get("question_index", 0))
        clicked = render_search_tree_graph(search_path, selected=selected, key_suffix=qidx)
        if clicked:
            st.session_state[sel_key] = clicked
            nodes, _ = _tree_to_graph(search_path)
            node = nodes.get(clicked)
            if node:
                render_tree_node_detail(node, clicked)

    chunks = item.get("chunks", [])
    if chunks:
        with st.expander(f"全部来源 ({len(chunks)})", expanded=False):
            for i, c in enumerate(chunks, 1):
                render_chunk_card(c, rank=i)


def _render_rag_loop_detail(item: dict) -> None:
    """规划-执行-反馈闭环 RAG: DAG 可视化。"""
    # 尝试多种可能的数据源
    pipeline_result = item.get("pipeline_result")

    # 从 per-question eval 结果中重建
    if not pipeline_result:
        pipeline_result = _build_pipeline_result_from_item(item)

    if pipeline_result:
        qidx = item.get("question_index", 0)
        render_dag_result(pipeline_result, key_suffix=str(qidx))
    else:
        st.info("暂无 DAG 可视化数据")

    chunks = item.get("chunks", [])
    if chunks:
        with st.expander(f"全部来源 ({len(chunks)})", expanded=False):
            for i, c in enumerate(chunks, 1):
                render_chunk_card(c, rank=i)


def _build_pipeline_result_from_item(item: dict) -> dict | None:
    """从 eval 结果条目重建 rag_loop pipeline_result dict。

    eval 数据中 dag_nodes/dag_edges 是数量(int)，实际 DAG 结构仅存在于 dag_snapshots。
    """
    dag_snapshots = item.get("dag_snapshots", [])
    if not isinstance(dag_snapshots, list) or not dag_snapshots:
        return None

    # 最后一个 snapshot 作为 final_dag
    last = dag_snapshots[-1]
    final_dag = {
        "nodes": last.get("nodes", {}),
        "edges": last.get("edges", []),
    }

    # 所有 snapshots 作为 round_dags
    round_dags = []
    for snap in dag_snapshots:
        if isinstance(snap, dict):
            round_dags.append({
                "nodes": snap.get("nodes", {}),
                "edges": snap.get("edges", []),
            })

    return {
        "total_rounds": item.get("total_rounds", 0),
        "total_search_calls": item.get("total_search_calls", 0),
        "termination_reason": item.get("termination_reason", "?"),
        "final_dag": final_dag,
        "round_dags": round_dags,
    }


def _render_result_detail(item: dict, mode: str) -> None:
    """按算法分派结果详情渲染。"""
    st.markdown("---")
    _render_result_header(item)
    st.markdown("---")

    dispatcher = {
        "llm-only": _render_llm_only_detail,
        "naive-rag": _render_naive_rag_detail,
        "rag-with-judge": _render_judge_detail,
        "rag-loop": _render_rag_loop_detail,
    }

    render_fn = dispatcher.get(mode)
    if render_fn:
        render_fn(item)
    else:
        st.info(f"未知模式: {mode}")


# ═══════════════════════════════════════════════════════════════════
# 系统对比
# ═══════════════════════════════════════════════════════════════════

@st.cache_data(ttl=300)
def _get_comparison_data():
    return load_comparison_data()


def _render_system_comparison(dataset: str) -> None:
    """三 Tab 系统对比: 数据集整体 / 问题类型 / 单题 F1 热力图。

    全部使用 Plotly 交互式图表 (悬停查看精确值，工具栏可缩放/导出 PNG)。
    """
    with st.spinner("加载 4 种算法 × 3 个数据集的评估数据..."):
        data = _get_comparison_data()

    if not data or not data.get("flat"):
        st.warning("未找到评估数据——请先运行至少一个算法的评估。", icon="⚠️")
        return

    tab_overall, tab_types, tab_heatmap = st.tabs([
        "数据集整体", "问题类型", "单题 F1 热力图",
    ])

    # ═════════════════════════════════════════════════════════════
    # Tab 1: 数据集整体
    # ═════════════════════════════════════════════════════════════

    with tab_overall:
        st.markdown("### 数据集整体对比")
        st.caption("4 种算法在 3 个数据集上的综合表现。悬停查看精确值，工具栏可缩放/导出 PNG。")

        _section("EM · F1 · 累计召回率 雷达图")
        st.plotly_chart(build_radar_chart(data), use_container_width=True)

        st.divider()
        _section("回答质量 — EM, F1, Precision, Recall 分组柱状图")
        st.plotly_chart(build_answer_quality_chart(data), use_container_width=True)

        st.divider()
        _section("检索质量 — 累计召回率, 总 Chunk 数, 不重复标题数")
        st.plotly_chart(build_retrieval_quality_chart(data), use_container_width=True)

        st.divider()
        _section("效率 — EM × 平均耗时 (Pareto 前沿)")
        st.plotly_chart(build_efficiency_chart(data), use_container_width=True)

        _section("效率补充 — 平均耗时 / 搜索次数 / 搜索深度")
        st.plotly_chart(build_efficiency_bars(data), use_container_width=True)

        st.divider()
        _section("检索次数累积分布 (CDF)")
        st.caption(
            "横轴 = 检索次数 N, 纵轴 = 检索次数 ≤ N 的问题占比。"
            "阶梯曲线越靠右 → 检索开销越大。"
            "模块化 RAG 检索次数固定, 递归检索 RAG 和 规划-执行-反馈闭环 RAG 的检索次数自适应变化。"
        )
        st.plotly_chart(build_retrieval_cdf_chart(data), use_container_width=True)

    # ═════════════════════════════════════════════════════════════
    # Tab 2: 问题类型
    # ═════════════════════════════════════════════════════════════

    with tab_types:
        st.markdown("### 问题类型对比")
        st.caption("单元格颜色越深 = 指标越高。每行一个数据集，EM 和 F1 并排。展示与下载使用同一 Plotly 渲染，品质一致。")

        for ds in COMP_DS_KEYS:
            st.markdown(f"**{COMP_DS_LABELS.get(ds, ds)}**")
            col1, col2 = st.columns(2)
            with col1:
                st.caption("EM 热力图")
                em_fig = build_type_heatmap(data, ds, "em")
                st.plotly_chart(em_fig, use_container_width=True)
                st.download_button(
                    "下载 PNG", type_heatmap_png(data, ds, "em"),
                    file_name=f"{ds}_EM_heatmap.png", mime="image/png",
                    key=f"dl_em_{ds}",
                )
            with col2:
                st.caption("F1 热力图")
                f1_fig = build_type_heatmap(data, ds, "f1")
                st.plotly_chart(f1_fig, use_container_width=True)
                st.download_button(
                    "下载 PNG", type_heatmap_png(data, ds, "f1"),
                    file_name=f"{ds}_F1_heatmap.png", mime="image/png",
                    key=f"dl_f1_{ds}",
                )

    # ═════════════════════════════════════════════════════════════
    # Tab 3: 单题 F1 热力图 (复用 heatmap_utils)
    # ═════════════════════════════════════════════════════════════

    with tab_heatmap:
        st.markdown("### 逐题 F1 热力图")
        st.caption("4 个系统在每个问题上的 F1 分数，按问题类型分组。绿色 = 正确，灰色 = 错误。")

        hm_dataset = st.selectbox(
            "数据集",
            options=COMP_DS_KEYS,
            format_func=lambda k: COMP_DS_LABELS.get(k, k),
            key="sc_heatmap_ds",
        )
        with st.spinner(f"加载 {COMP_DS_LABELS.get(hm_dataset, hm_dataset)} 热力图数据..."):
            questions = load_question_data(hm_dataset)
            if questions:
                sorted_qs = sort_questions(questions)
                ds_label = COMP_DS_LABELS.get(hm_dataset, hm_dataset)
                html = build_heatmap_html(sorted_qs, hm_dataset, ds_label)
                st.markdown(html, unsafe_allow_html=True)
                png = f1_heatmap_png(sorted_qs, hm_dataset, ds_label)
                st.download_button(
                    "下载 PNG", png,
                    file_name=f"{hm_dataset}_F1_heatmap.png", mime="image/png",
                    key=f"dl_f1_hm_{hm_dataset}",
                )
            else:
                st.warning("未找到热力图数据——请先运行评估。")


def _section(label: str) -> None:
    """渲染图表子标题。"""
    st.caption(f"**{label}**")


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════

def main():
    st.title("实验结果")
    st.caption("浏览所有 RAG 算法的评估结果")

    # Sidebar
    with st.sidebar:
        st.subheader("选择管道")
        pipeline = st.radio(
            "管道",
            options=list(PIPELINE_CONFIG.keys()),
            format_func=lambda k: f"{PIPELINE_CONFIG[k]['icon']} {PIPELINE_CONFIG[k]['label']}",
            index=4,  # 默认: 系统对比
            key="result_pipeline",
        )

        if pipeline == "system-comparison":
            st.subheader("数据集")
            dataset = st.selectbox(
                "数据集",
                COMP_DS_KEYS,
                format_func=lambda d: COMP_DS_LABELS.get(d, d),
                key="sc_dataset",
            )
        else:
            st.subheader("筛选")
            datasets = list_available_datasets(pipeline)
            if not datasets:
                st.warning(f"暂无 {PIPELINE_CONFIG[pipeline]['label']} 数据。")
                return
            dataset = st.selectbox("数据集", options=datasets, index=0, key="result_dataset")

            # Naive RAG 有 Schema A/B 两种融合策略，需选择
            schema = None
            if pipeline == "naive-rag":
                schema = st.radio(
                    "融合策略",
                    options=["a", "b"],
                    format_func=lambda s: f"Schema {s.upper()} ({'客户端 RRF' if s == 'a' else '服务端 RRF'})",
                    horizontal=True,
                    key="result_schema",
                )
            search_text = st.text_input("搜索问题", key="result_search")

    # 系统对比视图
    if pipeline == "system-comparison":
        _render_system_comparison(dataset)
        return

    # 加载数据
    results = load_results(pipeline, dataset, schema=schema)
    if not results or "results" not in results:
        st.warning("未找到评估结果。")
        return

    items = results["results"]

    # 搜索过滤（逐词匹配：每个搜索词独立匹配题目或答案，不需要连续出现）
    if search_text:
        search_terms = search_text.lower().split()
        items = [item for item in items
                 if all(
                     term in item.get("question", "").lower()
                     or term in item.get("answer", "").lower()
                     for term in search_terms
                 )]

    # 指标汇总
    _render_metrics_summary(results, pipeline)
    st.divider()

    # 问题类型细分
    _render_type_breakdown_section(items, dataset, pipeline)
    st.divider()

    # 结果列表（分页）
    st.markdown(f"### 结果 ({len(items)} 个问题)")

    page_size = 20
    total_pages = max(1, (len(items) + page_size - 1) // page_size)
    if total_pages > 1:
        page = st.slider("页码", 1, total_pages, 1, key="result_page")
    else:
        page = 1
        if len(items) == 0:
            st.info("没有匹配的问题，请调整搜索条件。")
            return

    start = (page - 1) * page_size
    end = min(start + page_size, len(items))
    page_items = items[start:end]

    for idx, item in enumerate(page_items, start=start):
        question = item.get("question", "")
        prediction = item.get("prediction", "")
        answer = item.get("answer", "")
        error = item.get("error")

        is_correct = (answer.strip().lower() == prediction.strip().lower()
                      if answer and prediction else False)
        badge = ":green[✓]" if is_correct else ":red[✗]"

        # 算法特有状态
        extra = ""
        if pipeline == "rag-loop":
            extra = f"[{item.get('total_rounds', '?')} 轮]"
        elif pipeline == "rag-with-judge":
            extra = f"[depth={item.get('search_depth', '?')}]"

        if error:
            extra = ":red[Error]"

        label = f"{badge} {extra} — {question[:120]}{'...' if len(question) > 120 else ''}"
        with st.expander(label, expanded=False):
            _render_result_detail(item, pipeline)

    st.caption(f"显示 {start + 1}–{end} / 共 {len(items)} 条结果")


main()