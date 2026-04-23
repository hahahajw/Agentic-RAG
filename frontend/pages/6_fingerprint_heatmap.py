"""
6_fingerprint_heatmap.py — F1 贡献热力图：按题型分网格的 HTML/CSS 渲染。

核心设计：
- 纯 HTML/CSS 渲染，前端展示与论文图片同源
- 每个题型 = 独立网格块，4 系统共享该题型的题目排序
- 图片导出：Playwright MCP 对当前页面截图（PNG/PDF）
- GitHub 风格 4 级绿色梯度 + 灰色背景(F1=0) + 圆角方块

数据源：
  - Eval/{mode}_data/result/{dataset}.json
  - Data/benchmark/{Dataset}_500_benchmark.json

用法: streamlit run frontend/app.py → 侧边栏选 "F1 Heatmap"
      图片导出：Playwright MCP → browser_take_screenshot / page.pdf()
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import json
import math
import streamlit as st

# ── 路径 ──
PROJECT_ROOT = Path(__file__).parent.parent.parent
BENCHMARK_DIR = PROJECT_ROOT / "Data" / "benchmark"

# ── 系统配置 ──
SYSTEM_KEYS = ["llm_only", "naive_rag_a", "rag_with_judge", "agentic_rag_v3"]
SYSTEM_LABELS = ["Qwen3.6-plus", "Modular RAG", "RAG with Judge", "Agentic RAG"]
SYSTEM_COLORS = ["#4A90D9", "#F5A623", "#50B948", "#E25555"]

DS_KEYS = ["hotpotqa", "2wikimultihopqa", "musique"]
DS_LABELS = ["HotpotQA", "2WikiMultihopQA", "MuSiQue"]

EVAL_PATHS = {
    "llm_only": "Eval/llm_only_data/result/{ds}.json",
    "naive_rag_a": "Eval/naive_rag_data/result/{ds}_schema_a.json",
    "rag_with_judge": "Eval/rag_with_judge_data/result/{ds}.json",
    "agentic_rag_v3": "Eval/agentic_rag_v3_data/result/{ds}.json",
}

BENCHMARK_FILES = {
    "hotpotqa": "HotpotQA_500_benchmark.json",
    "2wikimultihopqa": "2WikiMultihopQA_500_benchmark.json",
    "musique": "MuSiQue_500_benchmark.json",
}

# ── 题型中文化映射 ──
TYPE_CN = {
    # HotpotQA
    "bridge": "桥接型",
    "comparison": "比较型",
    # 2WikiMultihopQA
    "bridge_comparison": "桥接比较型",
    "compositional": "组合型",
    "inference": "推理型",
    # MuSiQue: 跳数不翻译，保持 "2-hop" / "3-hop" / "4-hop"
}

# ── 网格参数 ──
COLS = 50
CELL = 16       # px
GAP = 1         # px
STEP = CELL + GAP  # 17px

# ── F1 颜色 ──
F1_COLORS = ["#d0d0d0", "#b8e6c8", "#73c99a", "#30a954", "#1a7a3a"]
F1_LABELS = ["0", "0.25", "0.50", "0.75", "1.0"]

# ── 题型颜色 ──
TYPE_COLORS = [
    "#5b8c5a", "#e6b800", "#4a90d9", "#d9534f", "#9b59b6",
    "#1abc9c", "#f39c12", "#3498db", "#e74c3c", "#8e44ad",
]


@st.cache_data(ttl=120)
def load_question_data(ds_key: str) -> list[dict]:
    system_data = {}
    for sk in SYSTEM_KEYS:
        rel = EVAL_PATHS[sk].format(ds=ds_key)
        path = PROJECT_ROOT / rel
        with open(path, encoding="utf-8") as f:
            raw = json.load(f)
        system_data[sk] = {r["question_index"]: r for r in raw["results"]}

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


def sort_questions(questions: list[dict]) -> list[dict]:
    """按题型分组，组内保持原始 benchmark 索引顺序。"""
    return sorted(questions, key=lambda q: (q["type"], q["idx"]))


def f1_class(f1: float) -> str:
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
    """返回显示用的题型名称（HotpotQA/2Wiki 中文化，MuSiQue 保持英文）"""
    if ds_key == "musique":
        return raw_type  # "2-hop", "3-hop", "4-hop"
    return TYPE_CN.get(raw_type, raw_type)


def group_by_type(sorted_qs: list[dict]) -> list[tuple[str, list[dict]]]:
    groups = []
    cur_type = None
    cur_list = []
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


def _type_label(tname: str, ds_key: str) -> str:
    """题型标签，带颜色名（用于 HTML 的 color: 属性，非显示文本）"""
    return tname


def _type_display(tname: str, ds_key: str) -> str:
    """题型显示文本"""
    return display_name(tname, ds_key)


def build_heatmap_html(sorted_qs: list[dict], ds_key: str, ds_label: str) -> str:
    n = len(sorted_qs)
    type_groups = group_by_type(sorted_qs)
    type_ranges = [(tname, len(qs)) for tname, qs in type_groups]

    max_grid_w = COLS * CELL + (COLS - 1) * GAP  # 845

    SYS_GAP = 5  # 同题型内系统间距 (px)，约 1/3 小方格

    html_parts = []

    html_parts.append(f"""<div id="hm-root" style="font-family: 'SF Mono', 'Consolas', monospace; background: white; padding: 24px 32px; width: 100%; overflow-x: auto;">
<style>
.hm-inner {{ width: fit-content; min-width: 1020px; }}
.hm-cell {{ position: absolute; width: {CELL}px; height: {CELL}px; border-radius: 3px; }}
.c0 {{ background: #d0d0d0; }}
.c1 {{ background: #b8e6c8; }}
.c2 {{ background: #73c99a; }}
.c3 {{ background: #30a954; }}
.c4 {{ background: #1a7a3a; }}
.hm-sys-row {{ display: flex; align-items: center; margin-bottom: {SYS_GAP}px; }}
.hm-label {{ width: 140px; font-weight: 700; font-size: 13px; text-align: right; padding-right: 12px; flex-shrink: 0; white-space: nowrap; color: #222; }}
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
<div class="hm-inner">""")

    html_parts.append(f'<div class="hm-title">F1 Contribution Heatmap &mdash; {ds_label} (N={n})</div>')

    # ── 按题型分块渲染 ──
    for gi, (tname, tqs) in enumerate(type_groups):
        tc = TYPE_COLORS[gi % len(TYPE_COLORS)]
        tn = len(tqs)
        tdisplay = _type_display(tname, ds_key)

        html_parts.append(f'<div class="hm-type-header"><span>{tdisplay}</span> <span style="color:#888;font-weight:400;font-size:12px">(N={tn})</span><span class="hm-type-line"></span></div>')

        trows = math.ceil(tn / COLS)
        tgrid_h = trows * CELL + (trows - 1) * GAP
        tcols = min(tn, COLS)
        tgrid_w = tcols * CELL + (tcols - 1) * GAP

        for s in range(4):
            label = SYSTEM_LABELS[s]
            sk = SYSTEM_KEYS[s]

            html_parts.append(f'<div class="hm-sys-row">')
            html_parts.append(f'<div class="hm-label">{label}</div>')
            html_parts.append(f'<div class="hm-grid" style="width:{tgrid_w}px;height:{tgrid_h}px;">')
            html_parts.append(f'<div class="hm-border"></div>')

            for j, q in enumerate(tqs):
                col = j % COLS
                row = j // COLS
                f1 = q["f1_vector"][sk]
                cls = f1_class(f1)
                x = col * STEP
                y = row * STEP
                html_parts.append(f'<div class="hm-cell {cls}" style="left:{x}px;top:{y}px;"></div>')

            html_parts.append(f'</div></div>')  # close grid, sys-row

        if gi < len(type_groups) - 1:
            html_parts.append('<div class="hm-type-divider"></div>')

    # ── 底部图例 ──
    html_parts.append('<div class="hm-legend">')

    html_parts.append('<div class="hm-dist-title">Question Type Distribution</div>')
    html_parts.append('<div class="hm-dist">')
    for idx, (tname, tcount) in enumerate(type_ranges):
        frac = tcount / n
        tc = TYPE_COLORS[idx % len(TYPE_COLORS)]
        html_parts.append(f'<div class="hm-dist-seg" style="width:{frac*100:.4f}%;background:{tc};"></div>')
    html_parts.append('</div>')

    html_parts.append('<div class="hm-f1-title">F1 Score</div>')
    html_parts.append('<div class="hm-f1-row">')
    for k in range(5):
        html_parts.append(f'<div class="hm-f1-block c{k}"></div>')
        html_parts.append(f'<span class="hm-f1-label">{F1_LABELS[k]}</span>')
    html_parts.append('<span class="hm-f1-arrow">&rarr; increasing F1</span>')
    html_parts.append('</div>')

    html_parts.append('<div class="hm-type-row">')
    for idx, (tname, tcount) in enumerate(type_ranges):
        tc = TYPE_COLORS[idx % len(TYPE_COLORS)]
        tdisplay = _type_display(tname, ds_key)
        html_parts.append(f'<div class="hm-type-item"><div class="hm-type-swatch" style="background:{tc};"></div><span class="hm-type-name">{tdisplay} ({tcount})</span></div>')
    html_parts.append('</div>')

    html_parts.append("</div></div>")  # close hm-inner, hm-root
    return "".join(html_parts)


# ── Streamlit 页面 ──
st.set_page_config(page_title="F1 Heatmap", page_icon="🟩", layout="wide")

st.sidebar.title("F1 Heatmap")
st.sidebar.info("""
**图片导出方法：**
1. 侧边栏选择数据集
2. 在 Playwright MCP 中截图保存：
   - `browser_take_screenshot` → PNG
   - `browser_run_code` + `page.pdf()` → PDF
""")

ds_choice = st.sidebar.selectbox(
    "Select Dataset",
    options=DS_KEYS,
    format_func=lambda k: dict(zip(DS_KEYS, DS_LABELS))[k],
)

ds_key = ds_choice
ds_label = dict(zip(DS_KEYS, DS_LABELS))[ds_key]

with st.spinner(f"Loading {ds_label} data..."):
    questions = load_question_data(ds_key)
    type_counts = {}
    for q in questions:
        type_counts[q["type"]] = type_counts.get(q["type"], 0) + 1
    st.sidebar.info(f"Questions: 500\nTypes: {', '.join(f'{k}({v})' for k, v in sorted(type_counts.items()))}")
    sorted_qs = sort_questions(questions)

html = build_heatmap_html(sorted_qs, ds_key, ds_label)
st.markdown(html, unsafe_allow_html=True)
