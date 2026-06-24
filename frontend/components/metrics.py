"""指标展示卡片组件。

Usage:
    from frontend.components.metrics import render_metrics_row, render_metric_card
    render_metrics_row({"EM": 0.45, "F1": 0.51, "Hit": 0.89, "MRR": 0.65})
"""

import streamlit as st


def render_metrics_row(metrics: dict[str, float], columns: int = 4) -> None:
    """渲染一行指标卡片。

    Args:
        metrics: {指标名: 值} dict
        columns: 每行列数
    """
    cols = st.columns(columns)
    for i, (name, value) in enumerate(metrics.items()):
        with cols[i % columns]:
            formatted = f"{value:.4f}" if isinstance(value, float) and value < 10 else str(value)
            st.metric(label=name, value=formatted)


def render_aggregate_metrics(aggregate: dict) -> None:
    """渲染评估结果的聚合指标。

    展示 EM、F1、Precision、Recall、Context Recall、Hit、MRR、Retrieval Precision。
    """
    if not aggregate:
        return

    metrics_order = [
        ("EM", "em"),
        ("F1", "f1"),
        ("Precision", "precision"),
        ("Recall", "recall"),
        ("Context Recall", "context_recall"),
        ("Hit@1", "hit"),
        ("MRR", "mrr"),
        ("Retrieval Precision", "retrieval_precision"),
    ]

    cols = st.columns(4)
    for i, (label, key) in enumerate(metrics_order):
        val = aggregate.get(key)
        if val is not None:
            with cols[i % 4]:
                st.metric(label=label, value=f"{val:.4f}")


def render_batch_metrics(latency_ms: float, search_count: int | None = None,
                         chunk_count: int | None = None) -> None:
    """渲染单次查询的执行指标。

    Args:
        latency_ms: 耗时（毫秒）
        search_count: 搜索次数
        chunk_count: 检索 chunk 数
    """
    cols = st.columns(3)
    cols[0].metric("耗时", f"{latency_ms/1000:.1f}s")
    if search_count is not None:
        cols[1].metric("搜索次数", search_count)
    if chunk_count is not None:
        cols[2].metric("检索 Chunk 数", chunk_count)