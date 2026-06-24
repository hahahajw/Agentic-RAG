"""指标聚合工具 — 按算法类型和维度分组计算、渲染评估指标。

将 Eval 结果 JSON 中的逐题指标聚合为:
  1. 整体汇总 — 按维度分组的平均值
  2. 问题类型细分 — 每种问题类型的维度聚合

维度定义基于用户验收标准:
  - LLM Only: 生成质量 + 效率指标 (2 维度)
  - 模块化 RAG: 生成质量 + 检索质量(含 MRR) + 效率指标 (3 维度)
  - 递归检索 RAG / rag_loop: 生成质量 + 检索质量(无 MRR, 含总Chunk) + 效率(含搜索深度) (3 维度)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import streamlit as st

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent.parent
BENCHMARK_DIR = PROJECT_ROOT / "Data" / "benchmark"

BENCHMARK_FILES = {
    "hotpotqa": "HotpotQA_500_benchmark.json",
    "2wikimultihopqa": "2WikiMultihopQA_500_benchmark.json",
    "musique": "MuSiQue_500_benchmark.json",
    "gb_standards_25": "gb_standards_multihop_25.json",
}

# ═══════════════════════════════════════════════════════════════════
# 维度定义
# ═══════════════════════════════════════════════════════════════════

DIMENSION_DEFS: dict[str, dict] = {
    "llm-only": {
        "generation": {
            "label": "生成质量",
            "metrics": {
                "em": "EM", "f1": "F1",
                "precision": "Precision", "recall": "Recall",
            },
        },
        "efficiency": {
            "label": "效率指标",
            "metrics": {"avg_latency_ms": "平均耗时"},
        },
    },
    "naive-rag": {
        "generation": {
            "label": "生成质量",
            "metrics": {
                "em": "EM", "f1": "F1",
                "precision": "Precision", "recall": "Recall",
            },
        },
        "retrieval": {
            "label": "检索质量",
            "metrics": {
                "context_recall": "Context Recall",
                "hit": "Hit@1",
                "mrr": "MRR",
            },
        },
        "efficiency": {
            "label": "效率指标",
            "metrics": {"avg_latency_ms": "平均耗时"},
        },
    },
    "rag-with-judge": {
        "generation": {
            "label": "生成质量",
            "metrics": {
                "em": "EM", "f1": "F1",
                "precision": "Precision", "recall": "Recall",
            },
        },
        "retrieval": {
            "label": "检索质量 (多轮探索)",
            "metrics": {
                "context_recall": "累计召回率",
                "total_chunks": "总 Chunk 数",
                "total_distinct_titles": "不同标题数",
            },
            "note": "MRR 对多轮探索算法不适用——搜索是递归的，非单次排序",
        },
        "efficiency": {
            "label": "效率指标",
            "metrics": {
                "avg_latency_ms": "平均耗时",
                "retrieval_count": "平均检索次数",
                "search_depth": "平均搜索深度",
            },
        },
    },
    "rag-loop": {
        "generation": {
            "label": "生成质量",
            "metrics": {
                "em": "EM", "f1": "F1",
                "precision": "Precision", "recall": "Recall",
            },
        },
        "retrieval": {
            "label": "检索质量 (多轮探索)",
            "metrics": {
                "context_recall": "累计召回率",
                "total_chunks": "总 Chunk 数",
                "total_distinct_titles": "不同标题数",
            },
            "note": "MRR 对多轮探索算法不适用——基于 DAG 拓扑调度搜索，非单次排序",
        },
        "efficiency": {
            "label": "效率指标",
            "metrics": {
                "avg_latency_ms": "平均耗时",
                "retrieval_count": "平均检索次数",
                "search_depth": "平均搜索深度",
            },
        },
    },
}

# 质量指标: denominator = total (未回答 → 0)
_QUALITY_FIELDS = [
    "em", "f1", "precision", "recall",
    "context_recall", "hit", "mrr", "retrieval_precision",
]

# 效率指标: denominator = answered
_EFFICIENCY_FIELDS = [
    "retrieval_count", "total_chunks",
    "total_distinct_titles", "search_depth", "total_rounds",
]


# ═══════════════════════════════════════════════════════════════════
# 聚合计算
# ═══════════════════════════════════════════════════════════════════

def compute_aggregate(items: list[dict]) -> dict:
    """从结果列表计算各数值字段的均值。

    分母策略:
      - 质量指标 (EM, F1, Context Recall, Hit, MRR 等): 总题数 total
        未回答题贡献 0
      - 时间延迟 (latency_ms): 总题数 total
      - 效率指标 (retrieval_count, search_depth 等): 回答题数 answered

    Args:
        items: eval result 条目列表，每条含 em, f1, latency_ms 等字段

    Returns:
        聚合 dict: {em: 0.45, f1: 0.51, ..., avg_latency_ms: 1234.5}
    """
    if not items:
        return {}

    total = len(items)
    answered = [r for r in items if r.get("prediction") is not None]
    if not answered:
        answered = items

    result: dict = {}

    # 质量指标: denominator = total
    for field in _QUALITY_FIELDS:
        vals = [r.get(field, 0) or 0 for r in items]
        result[field] = sum(vals) / total

    # 时间延迟: denominator = total
    latency_vals = [r.get("latency_ms", 0) or 0 for r in items]
    result["latency_ms"] = sum(latency_vals) / total
    result["avg_latency_ms"] = result["latency_ms"]

    # 效率指标: denominator = answered
    for field in _EFFICIENCY_FIELDS:
        vals = [r[field] for r in answered if r.get(field) is not None]
        if vals:
            result[field] = sum(vals) / len(vals)

    return result


def load_benchmark_types(ds_key: str) -> list[str]:
    """加载 benchmark JSON，返回每个 question_index 对应的问题类型。

    Args:
        ds_key: "hotpotqa" | "2wikimultihopqa" | "musique"

    Returns:
        list[str]，下标 = question_index，值为问题类型名称
    """
    bench_path = BENCHMARK_DIR / BENCHMARK_FILES[ds_key]

    try:
        with open(bench_path, encoding="utf-8") as f:
            benchmark = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logger.warning("无法加载 benchmark 数据: %s — %s", bench_path, e)
        return []

    types = []
    for item in benchmark:
        if ds_key == "musique":
            hop = len(item.get("question_decomposition", []))
            types.append(f"{hop}-hop")
        else:
            types.append(item.get("type", "unknown"))
    return types


def compute_type_breakdown(
    items: list[dict], ds_key: str
) -> dict[str, dict]:
    """按问题类型分组并计算各类型的聚合指标。

    Args:
        items: eval result 条目列表
        ds_key: 数据集 key

    Returns:
        {type_name: {em: 0.52, f1: 0.58, ..., avg_latency_ms: 10200}}
    """
    types = load_benchmark_types(ds_key)
    if not types:
        return {}

    # 按类型分组
    grouped: dict[str, list[dict]] = {}
    for item in items:
        idx = item.get("question_index", 0)
        if idx >= len(types):
            continue
        qtype = types[idx]
        grouped.setdefault(qtype, []).append(item)

    # 对每组计算聚合
    breakdown: dict[str, dict] = {}
    for qtype, group_items in grouped.items():
        breakdown[qtype] = compute_aggregate(group_items)
        breakdown[qtype]["count"] = len(group_items)

    return breakdown


# ═══════════════════════════════════════════════════════════════════
# 渲染函数
# ═══════════════════════════════════════════════════════════════════

def render_metrics_by_dimension(aggregate: dict, pipeline: str) -> None:
    """按维度分组渲染指标卡片。

    Args:
        aggregate: compute_aggregate() 的返回结果
        pipeline: "llm-only" | "naive-rag" | "rag-with-judge" | "rag-loop"
    """
    dims = DIMENSION_DEFS.get(pipeline)
    if not dims:
        st.warning(f"未找到 {pipeline} 的维度定义")
        return

    for dim_key, dim_info in dims.items():
        label = dim_info["label"]
        metrics_map = dim_info["metrics"]
        note = dim_info.get("note")

        with st.container(border=True):
            st.caption(f"**{label}**")
            if note:
                st.caption(f"*注: {note}*")

            # 将指标排列为 4 列
            metric_items = list(metrics_map.items())
            cols = st.columns(min(4, len(metric_items)))
            for i, (field, display_name) in enumerate(metric_items):
                val = aggregate.get(field)
                if val is not None:
                    formatted = _format_metric(val, field)
                    cols[i % 4].metric(label=display_name, value=formatted)
                else:
                    cols[i % 4].metric(label=display_name, value="N/A")


def render_type_breakdown_table(
    breakdown: dict[str, dict], pipeline: str
) -> None:
    """渲染问题类型细分表格。

    Args:
        breakdown: compute_type_breakdown() 的返回结果 {type: {em, f1, ..., count}}
        pipeline: 算法标识
    """
    dims = DIMENSION_DEFS.get(pipeline)
    if not dims or not breakdown:
        return

    # 收集所有待显示的指标（按维度顺序）
    all_metrics: list[tuple[str, str]] = []  # [(field, display_name), ...]
    all_metrics.append(("count", "数量"))
    for dim_info in dims.values():
        all_metrics.extend(dim_info["metrics"].items())

    # 按类型名的自然顺序排序
    sorted_types = sorted(breakdown.keys())

    # 构建 DataFrame
    import pandas as pd
    rows = []
    for qtype in sorted_types:
        agg = breakdown[qtype]
        row = {"问题类型": qtype}
        for field, display_name in all_metrics:
            val = agg.get(field)
            if val is not None:
                row[display_name] = _format_metric(val, field)
            else:
                row[display_name] = "N/A"
        rows.append(row)

    if rows:
        st.dataframe(
            pd.DataFrame(rows),
            use_container_width=True,
            hide_index=True,
        )


# ═══════════════════════════════════════════════════════════════════
# 辅助
# ═══════════════════════════════════════════════════════════════════

def _format_metric(val: float, field: str) -> str:
    """根据字段类型格式化指标值。"""
    if field in ("avg_latency_ms", "latency_ms"):
        if val < 1000:
            return f"{val:.0f}ms"
        return f"{val / 1000:.1f}s"
    if field in ("total_chunks", "total_distinct_titles"):
        return f"{val:.1f}"
    if field in ("retrieval_count", "search_depth", "total_rounds", "count"):
        return f"{val:.2f}"
    # 默认: 比例 (EM, F1, Recall, etc.)
    return f"{val:.4f}"