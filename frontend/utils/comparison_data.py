"""系统对比数据加载模块。

加载 4 种算法在 3 个数据集上的评估结果，计算:
  1. 数据集整体聚合 — 按算法 × 数据集汇总全部指标
  2. 问题类型聚合 — 按算法 × 数据集 × 问题类型分组汇总
  3. 平铺格式 — 方便 Plotly 绘图直接使用

依赖 frontend/data_loader.py 加载 JSON，
依赖 frontend/utils/metrics_aggregator.py 计算聚合。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import streamlit as st

from frontend.data_loader import load_results as _load_results
from frontend.utils.metrics_aggregator import (
    compute_aggregate, compute_type_breakdown,
)

# ═══════════════════════════════════════════════════════════════════
# 常量
# ═══════════════════════════════════════════════════════════════════

ALGO_KEYS = ["llm-only", "naive-rag", "rag-with-judge", "rag-loop"]
DS_KEYS = ["hotpotqa", "2wikimultihopqa", "musique"]

ALGO_LABELS: dict[str, str] = {
    "llm-only": "LLM Only",
    "naive-rag": "模块化 RAG",
    "rag-with-judge": "递归检索 RAG",
    "rag-loop": "规划-执行-反馈闭环 RAG",
}

DS_LABELS: dict[str, str] = {
    "hotpotqa": "HotpotQA",
    "2wikimultihopqa": "2WikiMultihopQA",
    "musique": "MuSiQue",
}

# Okabe-Ito 色盲友好调色板 (Nature-quality, accessible)
ALGO_COLORS: dict[str, str] = {
    "llm-only": "#999999",        # 灰 — 基线
    "naive-rag": "#0072B2",       # 蓝 — 模块化 RAG
    "rag-with-judge": "#E69F00",  # 橙 — 递归检索 RAG
    "rag-loop": "#009E73",        # 绿 — DAG 闭环 RAG
}

ALGO_ORDER = ["llm-only", "naive-rag", "rag-with-judge", "rag-loop"]


# ═══════════════════════════════════════════════════════════════════
# 数据加载
# ═══════════════════════════════════════════════════════════════════


@st.cache_data(ttl=300)
def load_comparison_data() -> dict:
    """加载 4 算法 × 3 数据集的全部评估数据并计算聚合。

    Returns:
        {
            "algorithms": [...],
            "datasets": [...],
            "by_dataset": {
                ds: {
                    algo: {
                        "aggregate": {em, f1, precision, recall, context_recall,
                                      total_chunks, total_distinct_titles,
                                      avg_latency_ms, retrieval_count, search_depth},
                        "types": {type_name: {em, f1, ..., count}},
                        "count": int,
                        "items": list[dict],  # 逐题结果
                    }
                }
            },
            "flat": [  # 数据集 × 算法 级别的平铺记录
                {algo, ds, algo_label, ds_label, color,
                 em, f1, precision, recall,
                 cum_recall, total_chunks, distinct_titles,
                 avg_latency_ms, search_count, search_depth},
                ...
            ],
        }
    """
    by_dataset: dict[str, dict[str, dict]] = {}
    flat: list[dict] = []

    for ds in DS_KEYS:
        by_dataset[ds] = {}
        for algo in ALGO_KEYS:
            entry = _load_algo_dataset(algo, ds)
            by_dataset[ds][algo] = entry

            agg = entry["aggregate"]
            flat.append({
                "algo": algo,
                "ds": ds,
                "algo_label": ALGO_LABELS.get(algo, algo),
                "ds_label": DS_LABELS.get(ds, ds),
                "color": ALGO_COLORS.get(algo, "#888888"),
                "em": agg.get("em", 0),
                "f1": agg.get("f1", 0),
                "precision": agg.get("precision", 0),
                "recall": agg.get("recall", 0),
                "cum_recall": agg.get("context_recall") or 0,
                "total_chunks": agg.get("total_chunks") or 0,
                "distinct_titles": agg.get("total_distinct_titles") or 0,
                "avg_latency_ms": agg.get("avg_latency_ms") or 0,
                "search_count": agg.get("retrieval_count") or 0,
                "search_depth": agg.get("search_depth") or 0,
                "count": entry["count"],
            })

    return {
        "algorithms": ALGO_KEYS,
        "datasets": DS_KEYS,
        "by_dataset": by_dataset,
        "flat": flat,
    }


def _load_algo_dataset(algo: str, ds: str) -> dict:
    """加载单个算法 × 数据集的结果并计算聚合。

    Returns:
        {aggregate, types, items, count}
    """
    # Naive RAG 使用 schema_a
    schema = "a" if algo == "naive-rag" else None
    results = _load_results(algo, ds, schema=schema)

    if not results or "results" not in results:
        return {
            "aggregate": {},
            "types": {},
            "items": [],
            "count": 0,
        }

    items = results["results"]
    aggregate = compute_aggregate(items)
    types = compute_type_breakdown(items, ds)

    return {
        "aggregate": aggregate,
        "types": types,
        "items": items,
        "count": len(items),
    }