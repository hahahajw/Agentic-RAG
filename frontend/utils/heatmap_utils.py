"""F1 热力图工具 — 从 3_F1热力图.py 提取的纯计算/渲染函数。

供 2_实验结果.py（系统对比-逐题热力图 tab）和 3_F1热力图.py（独立页面）共用。

核心函数:
  - load_question_data(ds_key) → 加载 4 系统的逐题 F1 向量
  - build_heatmap_html(sorted_qs, ds_key, ds_label) → 生成热力图 HTML
"""

from __future__ import annotations

import json
import math
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
BENCHMARK_DIR = PROJECT_ROOT / "Data" / "benchmark"

# ═══════════════════════════════════════════════════════════════════
# 常量
# ═══════════════════════════════════════════════════════════════════

SYSTEM_KEYS = ["llm_only", "naive_rag_a", "rag_with_judge", "rag_loop"]
SYSTEM_LABELS = ["LLM Only", "模块化 RAG", "递归检索 RAG", "规划-执行-反馈 RAG"]
SYSTEM_COLORS = ["#4A90D9", "#F5A623", "#50B948", "#E25555"]

SYSTEM_MODE_MAP = {
    "llm_only": "llm-only",
    "naive_rag_a": "naive-rag",
    "rag_with_judge": "rag-with-judge",
    "rag_loop": "rag-loop",
}

DS_KEYS = ["hotpotqa", "2wikimultihopqa", "musique"]
DS_LABELS = ["HotpotQA", "2WikiMultihopQA", "MuSiQue"]

BENCHMARK_FILES = {
    "hotpotqa": "HotpotQA_500_benchmark.json",
    "2wikimultihopqa": "2WikiMultihopQA_500_benchmark.json",
    "musique": "MuSiQue_500_benchmark.json",
}

TYPE_CN = {
    "bridge": "桥接型",
    "comparison": "比较型",
    "bridge_comparison": "桥接比较型",
    "compositional": "组合型",
    "inference": "推理型",
}

COLS = 50
CELL = 16
GAP = 1
STEP = CELL + GAP

F1_COLORS = ["#d0d0d0", "#b8e6c8", "#73c99a", "#30a954", "#1a7a3a"]
F1_LABELS = ["0", "0.25", "0.50", "0.75", "1.0"]

# 连续色阶断点: (位置, R, G, B)
_F1_STOPS = [
    (0.0, 0xd0, 0xd0, 0xd0),
    (0.25, 0xb8, 0xe6, 0xc8),
    (0.50, 0x73, 0xc9, 0x9a),
    (0.75, 0x30, 0xa9, 0x54),
    (1.0, 0x1a, 0x7a, 0x3a),
]


def f1_color_hex(f1: float) -> str:
    """F1 值 → 连续绿色 hex 颜色（在 5 个断点间线性插值）。"""
    f1 = max(0.0, min(1.0, f1))
    for i in range(len(_F1_STOPS) - 1):
        p0, r0, g0, b0 = _F1_STOPS[i]
        p1, r1, g1, b1 = _F1_STOPS[i + 1]
        if f1 <= p1:
            t = (f1 - p0) / (p1 - p0)
            r = int(r0 + t * (r1 - r0))
            g = int(g0 + t * (g1 - g0))
            b = int(b0 + t * (b1 - b0))
            return f"#{r:02x}{g:02x}{b:02x}"
    return "#1a7a3a"

TYPE_COLORS = [
    "#5b8c5a", "#e6b800", "#4a90d9", "#d9534f", "#9b59b6",
    "#1abc9c", "#f39c12", "#3498db", "#e74c3c", "#8e44ad",
]


# ═══════════════════════════════════════════════════════════════════
# 数据加载
# ═══════════════════════════════════════════════════════════════════

def load_question_data(ds_key: str) -> list[dict]:
    """加载 4 个系统的逐题 F1 向量，与 benchmark 问题类型合并。

    Args:
        ds_key: "hotpotqa" | "2wikimultihopqa" | "musique"

    Returns:
        [{idx, type, f1_vector: {llm_only: 0.5, naive_rag_a: 0.6, ...}}, ...]
        加载失败(任一系统缺数据)返回空列表
    """
    from frontend.data_loader import load_results

    # 加载 4 系统的 eval 结果
    system_data: dict[str, dict[int, dict]] = {}
    for sk in SYSTEM_KEYS:
        mode_name = SYSTEM_MODE_MAP[sk]
        schema = "a" if sk == "naive_rag_a" else None
        raw = load_results(mode_name, ds_key, schema)
        if raw is None or "results" not in raw:
            return []
        system_data[sk] = {r["question_index"]: r for r in raw["results"]}

    # 加载 benchmark
    bench_path = BENCHMARK_DIR / BENCHMARK_FILES[ds_key]
    with open(bench_path, encoding="utf-8") as f:
        benchmark = json.load(f)

    def get_type(item):
        if ds_key == "musique":
            hop = len(item.get("question_decomposition", []))
            return f"{hop}-hop"
        return item.get("type", "unknown")

    questions = []
    n = len(system_data[SYSTEM_KEYS[0]])
    for idx in range(n):
        f1_vec = {}
        for sk in SYSTEM_KEYS:
            r = system_data[sk].get(idx, {})
            f1_vec[sk] = float(r.get("f1", 0) or 0)
        b_item = benchmark[idx] if idx < len(benchmark) else {}
        q_type = get_type(b_item) if b_item else "unknown"
        questions.append({"idx": idx, "type": q_type, "f1_vector": f1_vec})

    return questions


# ═══════════════════════════════════════════════════════════════════
# 排序与分组
# ═══════════════════════════════════════════════════════════════════

def sort_questions(questions: list[dict]) -> list[dict]:
    """按 (type, idx) 排序。"""
    return sorted(questions, key=lambda q: (q["type"], q["idx"]))


def group_by_type(sorted_qs: list[dict]) -> list[tuple[str, list[dict]]]:
    """将已排序的问题按类型分组。

    Returns:
        [(type_name, [question_dict, ...]), ...]
    """
    groups = []
    cur_type = None
    cur_list: list[dict] = []
    for q in sorted_qs:
        if q["type"] != cur_type:
            if cur_type is not None:
                groups.append((cur_type, cur_list))
            cur_type = q["type"]
            cur_list = [q]
        else:
            cur_list.append(q)
    if cur_list:
        groups.append((cur_type, cur_list))
    return groups


# ═══════════════════════════════════════════════════════════════════
# 辅助
# ═══════════════════════════════════════════════════════════════════

def f1_class(f1: float) -> str:
    """将 F1 分数映射为 CSS class (c0-c4)。"""
    if f1 <= 0.0:
        return "c0"
    elif f1 <= 0.25:
        return "c1"
    elif f1 <= 0.50:
        return "c2"
    elif f1 <= 0.75:
        return "c3"
    else:
        return "c4"


def display_name(raw_type: str, ds_key: str) -> str:
    """问题类型的中文展示名。"""
    if ds_key == "musique":
        return raw_type
    return TYPE_CN.get(raw_type, raw_type)


# ═══════════════════════════════════════════════════════════════════
# HTML 生成
# ═══════════════════════════════════════════════════════════════════

def build_heatmap_html(
    sorted_qs: list[dict], ds_key: str, ds_label: str
) -> str:
    """生成完整的 F1 热力图 HTML。

    Args:
        sorted_qs: sort_questions() 的输出
        ds_key: 数据集 key
        ds_label: 数据集展示名

    Returns:
        完整的 HTML 字符串，可直接用 st.markdown(..., unsafe_allow_html=True) 渲染
    """
    n = len(sorted_qs)
    type_groups = group_by_type(sorted_qs)
    type_ranges = [(tname, len(qs)) for tname, qs in type_groups]

    max_grid_w = COLS * CELL + (COLS - 1) * GAP
    SYS_GAP = 5

    html_parts = [
        f"""<div id="hm-root" style="font-family: 'SF Mono', 'Consolas', monospace; background: white; padding: 24px 32px; width: 100%; overflow-x: auto;">
<style>
.hm-inner {{ width: fit-content; min-width: 1020px; }}
.hm-cell {{ position: absolute; width: {CELL}px; height: {CELL}px; border-radius: 3px; }}
.hm-sys-row {{ display: flex; align-items: center; margin-bottom: {SYS_GAP}px; }}
.hm-label {{ width: 140px; font-weight: 600; font-size: 13px; text-align: right; padding-right: 12px; flex-shrink: 0; white-space: nowrap; color: #222; }}
.hm-grid {{ position: relative; }}
.hm-border {{ position: absolute; inset: -0.5px; border: 1px solid #ccc; pointer-events: none; border-radius: 3px; }}
.hm-type-header {{ font-size: 14px; font-weight: 700; color: #222; margin: 18px 0 10px; font-family: 'SF Mono', 'Consolas', monospace; display: flex; align-items: center; gap: 8px; }}
.hm-type-line {{ flex: 1; height: 1px; background: #ddd; }}
.hm-type-divider {{ border-top: 1px dashed #ccc; margin: 4px 0; }}
.hm-title {{ font-size: 16px; font-weight: 700; color: #222; margin-bottom: 16px; font-family: 'SF Mono', 'Consolas', monospace; }}
.hm-legend {{ margin-top: 24px; }}
.hm-dist-title {{ font-size: 12px; font-weight: 700; margin-bottom: 6px; font-family: 'SF Mono', 'Consolas', monospace; }}
.hm-dist {{ width: {max_grid_w}px; height: 14px; border-radius: 2px; overflow: hidden; display: flex; }}
.hm-dist-seg {{ height: 100%; }}
.hm-f1-title {{ font-size: 12px; font-weight: 700; margin-top: 12px; margin-bottom: 6px; font-family: 'SF Mono', 'Consolas', monospace; }}
.hm-f1-row {{ display: flex; align-items: center; gap: 10px; }}
.hm-f1-block {{ width: 14px; height: 14px; border-radius: 3px; border: 1px solid #888; flex-shrink: 0; }}
.hm-f1-label {{ font-size: 11px; color: #555; font-family: 'SF Mono', 'Consolas', monospace; }}
.hm-f1-arrow {{ font-size: 11px; color: #888; margin-left: 8px; font-family: 'SF Mono', 'Consolas', monospace; }}
.hm-type-row {{ display: flex; flex-wrap: wrap; gap: 14px; margin-top: 10px; }}
.hm-type-item {{ display: flex; align-items: center; gap: 4px; }}
.hm-type-swatch {{ width: 10px; height: 10px; border-radius: 2px; border: 1px solid #888; flex-shrink: 0; }}
.hm-type-name {{ font-size: 10px; color: #555; font-family: 'SF Mono', 'Consolas', monospace; }}
</style>
<div class="hm-inner">"""
    ]

    html_parts.append(
        f'<div class="hm-title">F1 Contribution Heatmap &mdash; {ds_label} (N={n})</div>'
    )

    for gi, (tname, tqs) in enumerate(type_groups):
        tc = TYPE_COLORS[gi % len(TYPE_COLORS)]
        tn = len(tqs)
        tdisplay = display_name(tname, ds_key)

        html_parts.append(
            f'<div class="hm-type-header"><span>{tdisplay}</span> '
            f'<span style="color:#888;font-weight:400;font-size:12px">(N={tn})</span>'
            f'<span class="hm-type-line"></span></div>'
        )

        trows = math.ceil(tn / COLS)
        tgrid_h = trows * CELL + (trows - 1) * GAP
        tcols = min(tn, COLS)
        tgrid_w = tcols * CELL + (tcols - 1) * GAP

        for s in range(4):
            label = SYSTEM_LABELS[s]
            sk = SYSTEM_KEYS[s]

            html_parts.append('<div class="hm-sys-row">')
            html_parts.append(f'<div class="hm-label">{label}</div>')
            html_parts.append(
                f'<div class="hm-grid" style="width:{tgrid_w}px;height:{tgrid_h}px;">'
            )
            html_parts.append('<div class="hm-border"></div>')

            for j, q in enumerate(tqs):
                col = j % COLS
                row = j // COLS
                f1 = q["f1_vector"][sk]
                bg = f1_color_hex(f1)
                x = col * STEP
                y = row * STEP
                html_parts.append(
                    f'<div class="hm-cell" style="left:{x}px;top:{y}px;background:{bg};"></div>'
                )

            html_parts.append('</div></div>')

        if gi < len(type_groups) - 1:
            html_parts.append('<div class="hm-type-divider"></div>')

    # 底部图例
    html_parts.append('<div class="hm-legend">')
    html_parts.append('<div class="hm-dist-title">问题类型分布</div>')
    html_parts.append('<div class="hm-dist">')
    for idx, (_tname, tcount) in enumerate(type_ranges):
        frac = tcount / n
        tc = TYPE_COLORS[idx % len(TYPE_COLORS)]
        html_parts.append(
            f'<div class="hm-dist-seg" style="width:{frac*100:.4f}%;background:{tc};"></div>'
        )
    html_parts.append('</div>')

    html_parts.append('<div class="hm-f1-title">F1 Score</div>')
    html_parts.append(
        '<div style="display:flex;align-items:center;gap:6px;font-size:11px;color:#555;">'
        '<span>0</span>'
        '<span style="display:inline-block;width:200px;height:14px;'
        'background:linear-gradient(to right, #d0d0d0, #b8e6c8, #73c99a, #30a954, #1a7a3a);'
        'border:1px solid #ccc;border-radius:2px;"></span>'
        '<span>1</span>'
        '<span style="font-size:11px;color:#888;margin-left:6px;">&rarr; F1 递增</span>'
        '</div>'
    )

    html_parts.append('<div class="hm-type-row">')
    for idx, (tname, tcount) in enumerate(type_ranges):
        tc = TYPE_COLORS[idx % len(TYPE_COLORS)]
        tdisplay = display_name(tname, ds_key)
        html_parts.append(
            f'<div class="hm-type-item">'
            f'<div class="hm-type-swatch" style="background:{tc};"></div>'
            f'<span class="hm-type-name">{tdisplay} ({tcount})</span>'
            f'</div>'
        )
    html_parts.append('</div>')

    html_parts.append("</div></div>")
    return "".join(html_parts)


# ═══════════════════════════════════════════════════════════════════
# PNG/PDF 导出 (Plotly + kaleido, 5x 超采样 / 矢量 PDF)
# ═══════════════════════════════════════════════════════════════════

def f1_heatmap_png(sorted_qs: list[dict], ds_key: str, ds_label: str) -> bytes:
    """从 F1 热力图数据直接生成高清 PNG（Plotly + kaleido，不经过浏览器）。

    简洁网格布局: 行=题目(按类型分组), 列=4 系统, 连续绿色渐变。
    磁盘缓存: 首次生成后存入 Eval/figures/，后续直接读取。
    3x 超采样，高清完整。
    """
    import json
    import hashlib
    import plotly.graph_objects as go

    n = len(sorted_qs)
    if n == 0:
        return b""

    # ── 磁盘缓存 ──
    figures_dir = PROJECT_ROOT / "Eval" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    cache_key = hashlib.md5(
        json.dumps([(q["idx"], q["type"], q["f1_vector"]) for q in sorted_qs]).encode()
    ).hexdigest()[:12]
    cache_path = figures_dir / f"{ds_key}_F1_heatmap_{cache_key}.png"
    if cache_path.exists():
        return cache_path.read_bytes()

    # ── 构建数据 ──
    # 按类型分组，组间插入空行作为分隔
    type_groups = group_by_type(sorted_qs)
    z = []
    for gi, (tname, tqs) in enumerate(type_groups):
        for q in tqs:
            z.append([
                q["f1_vector"].get("llm_only", 0),
                q["f1_vector"].get("naive_rag_a", 0),
                q["f1_vector"].get("rag_with_judge", 0),
                q["f1_vector"].get("rag_loop", 0),
            ])
        # 组间分隔空行 (最后一组不加)
        if gi < len(type_groups) - 1:
            z.append([None, None, None, None])

    green_scale = [
        [0.0, "#d0d0d0"], [0.25, "#b8e6c8"], [0.5, "#73c99a"],
        [0.75, "#30a954"], [1.0, "#1a7a3a"],
    ]

    # ── 构建图表 ──
    row_h = 10  # 每行像素高度
    total_h = len(z) * row_h + 60

    fig = go.Figure(data=go.Heatmap(
        z=z,
        x=[f"<b>{l}</b>" for l in SYSTEM_LABELS],
        colorscale=green_scale, zmin=0, zmax=1,
        showscale=True,
        colorbar=dict(
            title=dict(text="F1", font=dict(color="#111111", size=13)),
            tickformat=".2f", tickfont=dict(color="#111111", size=12),
            tickvals=[0, 0.25, 0.5, 0.75, 1.0], len=0.9,
        ),
    ))

    fig.update_layout(
        font=dict(family="'SF Mono', 'Consolas', monospace", size=11, color="#333333"),
        plot_bgcolor="white", paper_bgcolor="white",
        height=total_h, width=800,
        xaxis=dict(tickfont=dict(size=12, color="#111111"), side="top"),
        yaxis=dict(showticklabels=False, autorange="reversed"),
        margin=dict(l=10, r=40, t=30, b=10),
    )

    png = fig.to_image(format="png", engine="kaleido", scale=5)
    cache_path.write_bytes(png)
    return png


# ═══════════════════════════════════════════════════════════════════
# Playwright 高清导出 (委托 generate_heatmap_images.py)
# ═══════════════════════════════════════════════════════════════════

def generate_heatmap_files(ds_key: str, *,
                           formats: tuple[str, ...] = ("png", "pdf"),
                           output_dir: str | None = None) -> dict[str, str]:
    """使用 Playwright 生成高清热力图 PNG/PDF（5× 超采样, 480 DPI）。

    委托 frontend/pages/generate_heatmap_images.py 的 Playwright 渲染引擎。

    Args:
        ds_key: "hotpotqa" | "2wikimultihopqa" | "musique"
        formats: 输出格式
        output_dir: 输出目录 (默认 analyse_v2/figures/)

    Returns:
        {"png": "/path/to/file.png", "pdf": "/path/to/file.pdf"}
    """
    import importlib
    module = importlib.import_module("frontend.pages.generate_heatmap_images")
    result = module.generate_one(ds_key, formats=formats, output_dir=output_dir)
    return {fmt: str(p) for fmt, p in result.items()}