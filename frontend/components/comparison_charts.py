"""系统对比图表构建器 — 基于 Plotly 的专业可视化。

为系统对比的三个维度生成交互式图表:
  Tab 1 (数据集整体): 雷达图 + 回答质量分组柱状图 + 检索质量柱状图 + 效率散点图
  Tab 2 (问题类型): EM 热力图 + F1 热力图 (每数据集一组)
  Tab 3 (单题): F1 热力图 (由 heatmap_utils 提供)

色板: Okabe-Ito 色盲友好调色板 — 灰(LLM Only) / 蓝(Naive) / 橙(Judge) / 绿(rag_loop)
字体: 系统原生 sans-serif，确保中英文混排一致
网格: 极淡灰色 (#e5e7eb)，最小化非数据元素
"""

from __future__ import annotations

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from frontend.utils.comparison_data import (
    ALGO_KEYS, DS_KEYS, ALGO_LABELS, DS_LABELS, ALGO_COLORS, ALGO_ORDER,
)

# 问题类型中文翻译 (与 heatmap_utils.py 保持一致)
_TYPE_CN = {
    "bridge": "桥接型",
    "comparison": "比较型",
    "bridge_comparison": "桥接比较型",
    "compositional": "组合型",
    "inference": "推理型",
}

# ═══════════════════════════════════════════════════════════════════
# 通用样式 — Nature-quality 设计原则
# ═══════════════════════════════════════════════════════════════════

FONT_FAMILY = "'SF Mono', 'Consolas', 'Cascadia Code', 'Courier New', monospace"
GRID_COLOR = "#e5e7eb"
LEGEND_Y = -0.14

_BASE_LAYOUT = dict(
    font=dict(family=FONT_FAMILY, size=13, color="#333333"),
    plot_bgcolor="white",
    paper_bgcolor="white",
    legend=dict(
        orientation="h", yanchor="top", y=LEGEND_Y, xanchor="center", x=0.5,
        font=dict(size=13, family=FONT_FAMILY, color="#111111"),
        bgcolor="rgba(255,255,255,0.8)",
    ),
)


def _dark_axes(fig: go.Figure) -> None:
    """将图表中所有轴的刻度标签设为深色 + 统一字号 + 等宽字体。"""
    fig.update_xaxes(tickfont=dict(color="#111111", size=12, family=FONT_FAMILY))
    fig.update_yaxes(tickfont=dict(color="#111111", size=12, family=FONT_FAMILY))


def _add_algo_bars(fig, get_value, row, col, show_legend=False):
    """向 subplot 添加 4 条算法柱状图 trace。"""
    for algo in ALGO_ORDER:
        values = [get_value(algo, ds) for ds in DS_KEYS]
        fig.add_trace(go.Bar(
            name=ALGO_LABELS[algo],
            x=[DS_LABELS[d] for d in DS_KEYS],
            y=values,
            marker_color=ALGO_COLORS[algo],
            marker_line=dict(width=0),
            legendgroup=algo,
            showlegend=show_legend,
            hovertemplate=f"{ALGO_LABELS[algo]}<br>%{{x}}: %{{y:.4f}}<extra></extra>",
        ), row=row, col=col)


# ═══════════════════════════════════════════════════════════════════
# Tab 1: 数据集整体
# ═══════════════════════════════════════════════════════════════════


def build_answer_quality_chart(data: dict) -> go.Figure:
    """回答质量: 2×2 分组柱状图 (EM, F1, Precision, Recall)。"""
    flat = _ensure_flat_indexed(data)

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("EM", "F1", "Precision", "Recall"),
        vertical_spacing=0.14, horizontal_spacing=0.08,
    )
    _dark_axes(fig)

    metrics = [
        ("em", 1, 1),
        ("f1", 1, 2),
        ("precision", 2, 1),
        ("recall", 2, 2),
    ]

    for field, r, c in metrics:
        show_legend = (r == 1 and c == 1)
        _add_algo_bars(
            fig,
            get_value=lambda algo, ds, f=field: flat.get((algo, ds), {}).get(f, 0),
            row=r, col=c, show_legend=show_legend,
        )
        fig.update_yaxes(
            range=[0, 1], gridcolor=GRID_COLOR, zeroline=True, zerolinecolor="#cccccc",
            row=r, col=c,
        )

    fig.update_layout(**_BASE_LAYOUT)
    fig.update_layout(height=520)
    return fig


def build_radar_chart(data: dict) -> go.Figure:
    """雷达图: 单一极坐标图，9 轴 = 3 数据集 × 3 指标。

    4 算法在同一图上叠加对比，仅用线条 (无面积填充) 以避免遮挡。
    轴按数据集分组: HotpotQA→2Wiki→MuSiQue，每组内按 CR→F1→EM 排列。

    设计原则:
      - 线条模式 (fill=None)：所有算法同时可见
      - 圆形标记：便于识别各数据点
      - 紧凑轴标签：Hot=HotpotQA, 2Wiki=2WikiMultihopQA, Mus=MuSiQue
    """
    flat = _ensure_flat_indexed(data)

    # 9 轴: CR-Hot, F1-Hot, EM-Hot, CR-2Wiki, F1-2Wiki, EM-2Wiki, CR-Mus, F1-Mus, EM-Mus
    ds_suffixes = {"hotpotqa": "Hot", "2wikimultihopqa": "2Wiki", "musique": "Mus"}
    metric_labels = {"cum_recall": "CR", "f1": "F1", "em": "EM"}
    axis_labels = []
    for ds in DS_KEYS:
        for m in ["cum_recall", "f1", "em"]:
            axis_labels.append(f"{metric_labels[m]}-{ds_suffixes[ds]}")

    fig = go.Figure()

    for algo in ALGO_ORDER:
        values = []
        for ds in DS_KEYS:
            for m in ["cum_recall", "f1", "em"]:
                entry = flat.get((algo, ds), {})
                values.append(entry.get(m, 0))

        # 闭合多边形（仅用于连线，不填充）
        values_closed = values + [values[0]]
        labels_closed = axis_labels + [axis_labels[0]]

        fig.add_trace(go.Scatterpolar(
            r=values_closed,
            theta=labels_closed,
            name=ALGO_LABELS[algo],
            mode="lines+markers",
            marker=dict(size=5, color=ALGO_COLORS[algo]),
            line=dict(width=2, color=ALGO_COLORS[algo]),
            fill=None,  # 关键：无线条间填充，避免遮挡
            legendgroup=algo,
            hovertemplate=(
                f"{ALGO_LABELS[algo]}<br>"
                "%{theta}: %{r:.4f}<extra></extra>"
            ),
        ))

    # 轴分组标注 — 用注释标记三个数据集区域
    annotations = []
    for i, (ds_label, ds_key, start_idx) in enumerate([
        ("HotpotQA", "hotpotqa", 0),
        ("2WikiMultihopQA", "2wikimultihopqa", 3),
        ("MuSiQue", "musique", 6),
    ]):
        center_idx = start_idx + 1
        angle_rad = (2 * 3.14159) * (1 - center_idx / 9)
        annotations.append(dict(
            x=angle_rad, y=1.22,
            text=f"<b>{ds_label}</b>",
            showarrow=False,
            font=dict(size=11, color="#555555", family=FONT_FAMILY),
        ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                range=[0, 1.0],
                gridcolor=GRID_COLOR,
                tickfont=dict(size=12, family=FONT_FAMILY, color="#111111"),
                tickformat=".1f",
                showline=False,
            ),
            angularaxis=dict(
                gridcolor=GRID_COLOR,
                tickfont=dict(size=12, family=FONT_FAMILY, color="#111111"),
                rotation=90,
                direction="clockwise",
            ),
        ),
        font=dict(family=FONT_FAMILY, size=13, color="#333333"),
        plot_bgcolor="white",
        paper_bgcolor="white",
        height=480,
        margin=dict(l=40, r=40, t=60, b=60),
        legend=dict(
            orientation="h", yanchor="top", y=-0.04,
            xanchor="center", x=0.5,
            font=dict(size=13, color="#111111"),
            bgcolor="rgba(255,255,255,0.8)",
        ),
        annotations=annotations,
    )

    return fig


def build_retrieval_quality_chart(data: dict) -> go.Figure:
    """检索质量: 1×3 分组柱状图 (累计召回率, 总 Chunk 数, 不重复标题数)。"""
    flat = _ensure_flat_indexed(data)

    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=("累计召回率", "总 Chunk 数", "不重复标题数"),
        horizontal_spacing=0.1,
    )
    _dark_axes(fig)

    charts = [
        ("cum_recall", 1, 1, [0, 1]),
        ("total_chunks", 1, 2, None),
        ("distinct_titles", 1, 3, None),
    ]

    for field, r, c, yr in charts:
        _add_algo_bars(
            fig,
            get_value=lambda algo, ds, f=field: flat.get((algo, ds), {}).get(f, 0),
            row=r, col=c, show_legend=(c == 1),
        )
        if yr:
            fig.update_yaxes(range=yr, gridcolor=GRID_COLOR, row=r, col=c)
        else:
            fig.update_yaxes(gridcolor=GRID_COLOR, row=r, col=c)

    fig.update_layout(**_BASE_LAYOUT)
    fig.update_layout(height=370)
    return fig


def build_efficiency_chart(data: dict) -> go.Figure:
    """效率: EM×耗时散点图 — 展示「质量-效率」Pareto 前沿。

    每个点 = 一个算法-数据集组合 (4 算法 × 3 数据集 = 12 点)。
    点颜色 = 算法，标签 = 数据集缩写。
    """
    flat_list = _ensure_flat_list(data)

    fig = go.Figure()
    _dark_axes(fig)

    for algo in ALGO_ORDER:
        points = [p for p in flat_list if p["algo"] == algo]
        if not points:
            continue
        fig.add_trace(go.Scatter(
            x=[p["avg_latency_ms"] / 1000 for p in points],
            y=[p["em"] for p in points],
            mode="markers+text",
            name=ALGO_LABELS[algo],
            text=[p["ds_label"][:4] for p in points],
            textposition="top center",
            textfont=dict(size=12, family=FONT_FAMILY, color="#111111"),
            marker=dict(
                size=14, color=ALGO_COLORS[algo],
                line=dict(width=1.5, color="white"),
                opacity=0.9,
            ),
            hovertemplate=(
                f"{ALGO_LABELS[algo]}<br>"
                "数据集: %{text}<br>"
                "EM: %{y:.4f}<br>"
                "耗时: %{x:.1f}s<extra></extra>"
            ),
        ))

    fig.update_layout(
        **_BASE_LAYOUT,
        height=370,
        xaxis=dict(gridcolor=GRID_COLOR, zeroline=False),
        yaxis=dict(gridcolor=GRID_COLOR, range=[0, 0.7]),
    )
    fig.update_layout(legend=dict(orientation="h", yanchor="top", y=-0.18, xanchor="center", x=0.5, font=dict(color="#111111", size=13)))

    return fig


def build_efficiency_bars(data: dict) -> go.Figure:
    """效率补充: 1×3 分组柱状图 (平均耗时, 平均搜索次数, 平均搜索深度)。"""
    flat = _ensure_flat_indexed(data)

    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=("平均耗时 (s)", "平均搜索次数", "平均搜索深度"),
        horizontal_spacing=0.1,
    )
    _dark_axes(fig)

    charts = [
        ("avg_latency_ms", 1, 1, lambda v: v / 1000),
        ("search_count", 1, 2, lambda v: v),
        ("search_depth", 1, 3, lambda v: v),
    ]

    for field, r, c, xform in charts:
        _add_algo_bars(
            fig,
            get_value=lambda algo, ds, f=field: xform(flat.get((algo, ds), {}).get(f, 0)),
            row=r, col=c, show_legend=(c == 1),
        )
        fig.update_yaxes(gridcolor=GRID_COLOR, row=r, col=c)

    fig.update_layout(**_BASE_LAYOUT)
    fig.update_layout(height=370)
    return fig


# ═══════════════════════════════════════════════════════════════════
# Tab 2: 问题类型
# ═══════════════════════════════════════════════════════════════════


# (旧 build_type_heatmap v1 已移除 — v2 定义见下方 line ~432)


def _f1_color(val: float) -> str:
    """F1 热力图同款 5 级绿色渐变。"""
    if val <= 0.001:
        return "#d0d0d0"
    elif val < 0.25:
        return "#b8e6c8"
    elif val < 0.50:
        return "#73c99a"
    elif val < 0.75:
        return "#30a954"
    else:
        return "#1a7a3a"


def build_type_heatmap(data: dict, dataset: str, metric: str) -> go.Figure:
    """问题类型热力图 (Plotly) — 展示与下载统一使用此函数。

    kaleido 渲染品质一致，消除 HTML/CSS vs PNG 的不一致。

    Returns:
        Plotly Figure，用 st.plotly_chart() 展示或 fig.to_image() 下载。
    """
    by_dataset = data.get("by_dataset", {})

    all_types_raw: list[str] = []
    for algo in ALGO_ORDER:
        algo_data = by_dataset.get(dataset, {}).get(algo, {})
        for t in algo_data.get("types", {}):
            if t not in all_types_raw:
                all_types_raw.append(t)

    if not all_types_raw:
        return _empty_figure(f"无 {dataset} 问题类型数据")

    all_types_display = [_TYPE_CN.get(t, t) for t in all_types_raw]
    metric_label = {"em": "EM", "f1": "F1"}.get(metric, metric)

    z = []
    text = []
    for algo in ALGO_ORDER:
        algo_data = by_dataset.get(dataset, {}).get(algo, {})
        types = algo_data.get("types", {})
        row_z = [types.get(t, {}).get(metric, 0) for t in all_types_raw]
        z.append(row_z)
        text.append([f"{v:.4f}" for v in row_z])

    green_scale = [
        [0.0, "#d0d0d0"], [0.25, "#b8e6c8"], [0.5, "#73c99a"],
        [0.75, "#30a954"], [1.0, "#1a7a3a"],
    ]

    fig = go.Figure(data=go.Heatmap(
        z=z, x=[f"<b>{d}</b>" for d in all_types_display],
        y=[f"<b>{ALGO_LABELS[a]}</b>" for a in ALGO_ORDER],
        text=text, texttemplate="%{text}",
        textfont=dict(size=14, family=FONT_FAMILY, color="#111111"),
        colorscale=green_scale, zmin=0, zmax=1,
        showscale=True,
        colorbar=dict(
            title=dict(text=metric_label, font=dict(color="#111111", size=13)),
            tickformat=".2f", tickfont=dict(color="#111111", size=12),
            tickvals=[0, 0.25, 0.5, 0.75, 1.0],
            len=0.85,
        ),
        hovertemplate=(
            "算法: %{y}<br>"
            f"类型: %{{x}}<br>"
            f"{metric_label}: %{{z:.4f}}<extra></extra>"
        ),
    ))

    fig.update_layout(
        font=dict(family=FONT_FAMILY, size=13, color="#333333"),
        plot_bgcolor="white", paper_bgcolor="white",
        height=280, width=1000,
        xaxis=dict(tickfont=dict(family=FONT_FAMILY, size=13, color="#111111")),
        yaxis=dict(tickfont=dict(family=FONT_FAMILY, size=13, color="#111111"),
                    autorange="reversed"),
        margin=dict(l=200, r=40, t=10, b=50),
        legend=dict(
            orientation="h", yanchor="top", y=-0.14,
            xanchor="center", x=0.5,
            font=dict(size=13, family=FONT_FAMILY, color="#111111"),
            bgcolor="rgba(255,255,255,0.8)",
        ),
    )

    return fig


def type_heatmap_png(data: dict, dataset: str, metric: str) -> bytes:
    """从 Plotly 热力图生成高清 PNG（5× 超采样, 480 DPI 等效）。"""
    fig = build_type_heatmap(data, dataset, metric)
    return fig.to_image(format="png", engine="kaleido", scale=5)


def type_heatmap_pdf(data: dict, dataset: str, metric: str) -> bytes:
    """从 Plotly 热力图生成矢量 PDF（无限缩放, 印刷级质量）。"""
    fig = build_type_heatmap(data, dataset, metric)
    return fig.to_image(format="pdf", engine="kaleido", scale=1)


def radar_chart_png(data: dict) -> bytes:
    """雷达图高清 PNG（5× 超采样）。"""
    fig = build_radar_chart(data)
    return fig.to_image(format="png", engine="kaleido", scale=5)


def radar_chart_pdf(data: dict) -> bytes:
    """雷达图矢量 PDF。"""
    fig = build_radar_chart(data)
    return fig.to_image(format="pdf", engine="kaleido", scale=1)


def answer_quality_chart_png(data: dict) -> bytes:
    fig = build_answer_quality_chart(data)
    return fig.to_image(format="png", engine="kaleido", scale=5)


def answer_quality_chart_pdf(data: dict) -> bytes:
    fig = build_answer_quality_chart(data)
    return fig.to_image(format="pdf", engine="kaleido", scale=1)


def retrieval_quality_chart_png(data: dict) -> bytes:
    fig = build_retrieval_quality_chart(data)
    return fig.to_image(format="png", engine="kaleido", scale=5)


def retrieval_quality_chart_pdf(data: dict) -> bytes:
    fig = build_retrieval_quality_chart(data)
    return fig.to_image(format="pdf", engine="kaleido", scale=1)


def efficiency_chart_png(data: dict) -> bytes:
    fig = build_efficiency_chart(data)
    return fig.to_image(format="png", engine="kaleido", scale=5)


def efficiency_chart_pdf(data: dict) -> bytes:
    fig = build_efficiency_chart(data)
    return fig.to_image(format="pdf", engine="kaleido", scale=1)


def efficiency_bars_png(data: dict) -> bytes:
    fig = build_efficiency_bars(data)
    return fig.to_image(format="png", engine="kaleido", scale=5)


def efficiency_bars_pdf(data: dict) -> bytes:
    fig = build_efficiency_bars(data)
    return fig.to_image(format="pdf", engine="kaleido", scale=1)


# ═══════════════════════════════════════════════════════════════════
# 检索次数 CDF (累积分布函数)
# ═══════════════════════════════════════════════════════════════════

_CDF_ALGOS = ["naive-rag", "rag-with-judge", "rag-loop"]


def build_retrieval_cdf_chart(data: dict) -> go.Figure:
    """检索次数累积分布函数 (CDF) — 3×1 子图, 每数据集 3 条阶梯曲线。

    横轴 = 检索次数 N (一次搜索流水线 = 一次检索)
    纵轴 = 检索次数 ≤ N 的问题占比
    阶梯曲线越靠右 → 检索开销越大。

    数据来源: data["by_dataset"][ds][algo]["items"][].retrieval_count
      模块化 RAG: 始终 1 (一条流水线)
      递归检索 RAG: 搜索树节点数
      rag_loop: Solver 搜索调用总次数
    """
    by_dataset = data.get("by_dataset", {})
    ds_labels = {"hotpotqa": "HotpotQA", "2wikimultihopqa": "2WikiMultihopQA", "musique": "MuSiQue"}

    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=[ds_labels[ds] for ds in DS_KEYS],
        vertical_spacing=0.12,
    )
    _dark_axes(fig)

    # ── CDF 曲线 ──
    endpoint_data: dict[tuple[int, str], int] = {}  # (row, algo) → max_count

    for row, ds in enumerate(DS_KEYS, 1):
        show_leg = (row == 1)
        for algo in _CDF_ALGOS:
            items = by_dataset.get(ds, {}).get(algo, {}).get("items", [])
            counts = _extract_retrieval_counts(items, algo)
            if not counts:
                continue

            sorted_c = sorted(counts)
            n = len(sorted_c)
            cdf_y = [(i + 1) / n for i in range(n)]

            x_vals = [0] + sorted_c
            y_vals = [0.0] + cdf_y

            fig.add_trace(go.Scatter(
                x=x_vals, y=y_vals,
                mode="lines",
                line=dict(shape="hv", width=2, color=ALGO_COLORS[algo]),
                name=ALGO_LABELS[algo],
                legendgroup=algo,
                showlegend=show_leg,
                hovertemplate=(
                    f"{ALGO_LABELS[algo]}<br>"
                    "检索次数: %{x}<br>"
                    "累积占比: %{y:.3f}<extra></extra>"
                ),
            ), row=row, col=1)

            if algo in ("rag-with-judge", "rag-loop"):
                endpoint_data[(row, algo)] = max(sorted_c)

        fig.update_yaxes(
            range=[0, 1.02], gridcolor=GRID_COLOR,
            tickformat=".0%",
            title="累计占比" if row == 2 else "",
            title_font=dict(size=13, family=FONT_FAMILY, color="#222222"),
            row=row, col=1,
        )
        fig.update_xaxes(
            gridcolor=GRID_COLOR,
            title="检索次数" if row == 3 else "",
            title_font=dict(size=13, family=FONT_FAMILY, color="#222222"),
            row=row, col=1,
        )

    # ── 终点标注 ──

    for row, ds in enumerate(DS_KEYS, 1):
        x_ref = "x" if row == 1 else f"x{row}"
        y_ref = "y" if row == 1 else f"y{row}"

        # ── 终点 (递归检索 RAG & 闭环 RAG 的 max 检索次数) ──
        for algo in ["rag-with-judge", "rag-loop"]:
            key = (row, algo)
            if key not in endpoint_data:
                continue
            max_c = endpoint_data[key]
            c = ALGO_COLORS[algo]

            fig.add_trace(go.Scatter(
                x=[max_c], y=[1.0], mode="markers",
                marker=dict(size=10, color=c, symbol="diamond",
                            line=dict(width=1.5, color="white")),
                showlegend=False,
                hovertemplate=(
                    f"{ALGO_LABELS[algo]}<br>"
                    f"最大检索次数: %{{x}}<br>累计占比: 100%<extra></extra>"
                ),
            ), row=row, col=1)

            fig.add_annotation(
                x=max_c, y=1.0, xref=x_ref, yref=y_ref,
                text=f"max={max_c}",
                showarrow=True,
                arrowhead=1, arrowsize=1, arrowwidth=1,
                arrowcolor=c,
                ax=0, ay=-22,
                font=dict(size=10, family=FONT_FAMILY, color=c),
                bgcolor="rgba(255,255,255,0.85)", borderpad=2,
            )

    fig.update_layout(**_BASE_LAYOUT)
    fig.update_layout(height=750, margin=dict(b=50, l=40, r=20, t=30))
    fig.update_layout(legend=dict(y=-0.08))

    return fig


def _extract_retrieval_counts(items: list[dict], algo: str) -> list[int]:
    """从逐题 items 中提取检索次数。

    一次搜索流水线 = 一次检索:
      - 模块化 RAG: 始终为 1 (多查询重写 + 并行搜索 + RRF 融合 = 一条流水线)
      - 递归检索 RAG: 搜索树节点数 (每次 Judge 判断后搜索子问题)
      - 规划-执行-反馈闭环 RAG: Solver 搜索调用总次数
    如果 eval JSON 中没有 retrieval_count 字段 (旧数据), 回退到 1 (Naive RAG)。
    """
    counts = []
    for it in items:
        rc = it.get("retrieval_count")
        if rc is not None and isinstance(rc, (int, float)) and rc > 0:
            counts.append(int(rc))
        elif algo == "naive-rag":
            counts.append(1)  # 一条搜索流水线 = 一次检索
    return counts


def retrieval_cdf_chart_png(data: dict) -> bytes:
    """检索次数 CDF 高清 PNG（5× 超采样）。"""
    fig = build_retrieval_cdf_chart(data)
    return fig.to_image(format="png", engine="kaleido", scale=5)


def retrieval_cdf_chart_pdf(data: dict) -> bytes:
    """检索次数 CDF 矢量 PDF。"""
    fig = build_retrieval_cdf_chart(data)
    return fig.to_image(format="pdf", engine="kaleido", scale=1)


def build_type_heatmap_html(data: dict, dataset: str, metric: str) -> str:
    """问题类型热力图 (HTML/CSS) — 与 F1 热力图字体/风格一致。

    布局:
      - 标题 (数据集 — EM/F1)
      - 数据表格 (算法名 × 类型，类型标签在下方)
      - 连续渐变颜色条

    Args:
        data: comparison_data (by_dataset 结构)
        dataset: 数据集 key
        metric: 指标字段名

    Returns:
        HTML 字符串。
    """
    by_dataset = data.get("by_dataset", {})

    all_types_raw: list[str] = []
    for algo in ALGO_ORDER:
        algo_data = by_dataset.get(dataset, {}).get(algo, {})
        for t in algo_data.get("types", {}):
            if t not in all_types_raw:
                all_types_raw.append(t)

    if not all_types_raw:
        return ""

    all_types_display = [_TYPE_CN.get(t, t) for t in all_types_raw]
    metric_label = {"em": "EM", "f1": "F1"}.get(metric, metric)

    rows: list[tuple[str, list[float]]] = []
    for algo in ALGO_ORDER:
        algo_data = by_dataset.get(dataset, {}).get(algo, {})
        types = algo_data.get("types", {})
        values = [types.get(t, {}).get(metric, 0) for t in all_types_raw]
        rows.append((ALGO_LABELS[algo], values))

    n_types = len(all_types_display)
    cell_w = max(80, min(140, 600 // n_types))
    cell_pad = cell_w // 3

    # 计算表格大致高度，用于颜色条匹配
    bar_h = 40 + len(rows) * 35  # 行高 + 类型标签行

    parts = [
        '<div style="'
        "font-family: 'SF Mono', 'Consolas', monospace; "
        'background: white; padding: 12px 20px 16px; '
        'overflow-x: auto;'
        '">',
        f'<div style="font-size:14px;font-weight:700;color:#222;margin-bottom:10px;">'
        f'{DS_LABELS.get(dataset, dataset)} &mdash; {metric_label}</div>',
        # flex 容器: 表格 + 颜色条
        '<div style="display:flex;align-items:stretch;gap:12px;">',
        # 左侧: 表格
        '<div>',
        '<table style="border-collapse:collapse;">',
    ]

    # 数据行 (无表头)
    for algo_label, values in rows:
        parts.append('<tr>')
        parts.append(
            f'<td style="text-align:right;padding:8px 12px 8px 0;'
            f'font-size:13px;font-weight:700;color:#222;white-space:nowrap;">'
            f'{algo_label}</td>'
        )
        for v in values:
            bg = _f1_color(v)
            parts.append(
                f'<td style="background:{bg};text-align:center;'
                f'padding:8px {cell_pad}px;font-size:13px;color:#111;">{v:.4f}</td>'
            )
        parts.append('</tr>')

    # 类型标签行
    parts.append(
        '<tr><td style="text-align:right;padding:6px 12px 0 0;'
        'font-size:12px;font-weight:700;color:#222;"></td>'
    )
    for td in all_types_display:
        parts.append(
            f'<td style="text-align:center;padding:6px {cell_pad}px 0;'
            f'font-size:12px;font-weight:700;color:#222;">{td}</td>'
        )
    parts.append('</tr>')

    parts.append('</table>')
    parts.append('</div>')  # 关闭左侧表格容器

    # 右侧: 垂直渐变颜色条
    parts.append(
        '<div style="display:flex;flex-direction:column;align-items:center;'
        'justify-content:flex-start;min-width:36px;">'
        # 标签: 1
        '<span style="font-size:11px;color:#555;font-weight:700;">1</span>'
        # 渐变条
        f'<div style="width:16px;height:{bar_h}px;'
        'background:linear-gradient(to bottom, #1a7a3a, #30a954, #73c99a, #b8e6c8, #d0d0d0);'
        'border:1px solid #ccc;border-radius:2px;position:relative;">'
        # 0.5 标记线
        f'<div style="position:absolute;top:50%;left:-4px;right:-4px;'
        'border-top:1px dashed #888;"></div>'
        '</div>'
        # 标签: 0.5
        '<span style="font-size:10px;color:#888;">.5</span>'
        # 填满空间
        f'<div style="flex:1;"></div>'
        # 标签: 0 (底部对齐，需要撑开空间)
        '<span style="font-size:11px;color:#555;font-weight:700;margin-top:auto;">0</span>'
        '</div>'
    )

    parts.append('</div>')  # 关闭 flex 容器
    parts.append('</div>')
    return "\n".join(parts)
# ═══════════════════════════════════════════════════════════════════


def _ensure_flat_indexed(data: dict) -> dict[tuple[str, str], dict]:
    """将 comparison_data 的 flat 列表转为 {(algo, ds): entry} 索引。"""
    flat_list = data.get("flat", [])
    if isinstance(flat_list, list):
        return {(e["algo"], e["ds"]): e for e in flat_list}
    return flat_list  # 已经是索引格式


def _ensure_flat_list(data: dict) -> list[dict]:
    """确保 flat 是列表格式。"""
    flat = data.get("flat", [])
    if isinstance(flat, dict):
        return list(flat.values())
    return flat


def _empty_figure(message: str) -> go.Figure:
    """返回一个显示提示信息的空图。"""
    fig = go.Figure()
    fig.add_annotation(
        text=message, x=0.5, y=0.5, showarrow=False,
        font=dict(size=14, color="#94a3b8"),
    )
    fig.update_layout(**_BASE_LAYOUT, height=150)
    return fig