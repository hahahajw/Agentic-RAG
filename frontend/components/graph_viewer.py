"""交互式图可视化组件——自包含 iframe + 悬浮详情卡片。

支持:
  - 搜索树 (RAG with Judge SEARCH_PATH)
  - DAG (rag_loop PipelineResult)
  - 点击节点 → 悬浮卡片出现 (问题/状态/答案/chunks)
  - 再次点击同节点或关闭按钮 → 卡片消失

技术方案:
  - Python 端 Sugiyama 层级布局 (BFS 分层 + 重心法排序)
  - st.components.v1.html 自包含 iframe — 纯客户端交互, 零服务端往返
  - SVG onclick → JS toggleCard → CSS 悬浮卡片
"""

from __future__ import annotations

import json
import streamlit as st
import streamlit.components.v1 as components

# ═══════════════════════════════════════════════════════════════════
# 布局常量
# ═══════════════════════════════════════════════════════════════════

NODE_W = 220
NODE_H = 56
LAYER_GAP = 90
NODE_GAP = 30
PAD = 40
IFRAME_HEIGHT = 520

STATUS_COLORS = {
    "solved": "#22c55e", "unsolved": "#f59e0b",
    "blocked": "#ef4444", "failed_search": "#94a3b8", "empty_search": "#94a3b8",
}

HEALTH_COLORS = {
    "healthy": "#22c55e", "needs_verification": "#f59e0b",
    "unreliable": "#ef4444", "blocked": "#6b7280",
}


# ═══════════════════════════════════════════════════════════════════
# 图例/副标题
# ═══════════════════════════════════════════════════════════════════

def _subtitle(graph_type: str) -> str:
    if graph_type == "tree":
        return "点击节点查看详情 · 绿=可回答 橙=不可回答"
    return "点击节点查看详情 · 实线=分解 虚线=依赖"


# ═══════════════════════════════════════════════════════════════════
# 公共 API
# ═══════════════════════════════════════════════════════════════════

def render_search_tree(search_path: dict, selected: str = "",
                       key_suffix: str = "") -> str | None:
    """渲染搜索树交互式图 (RAG with Judge)。"""
    nodes, edges = _tree_to_graph(search_path)
    title = search_path.get("question", "搜索树")[:60]
    return _render_graph(nodes, edges, "tree", title, key_suffix)


def render_dag(pipeline_result: dict, key_suffix: str = "") -> str | None:
    """渲染 DAG (rag_loop)。"""
    dag = pipeline_result.get("final_dag", {})
    return render_dag_dict(dag, key_suffix=key_suffix)


def render_dag_dict(dag: dict, selected: str = "", height: int = 500,
                    key_suffix: str = "") -> str | None:
    """渲染 DAG dict。"""
    nodes = dag.get("nodes", {})
    edges = dag.get("edges", [])
    return _render_graph(nodes, edges, "dag", "DAG 结构", key_suffix)


def render_tree_dict(nodes: dict, edges: list, title: str = "图结构",
                     selected: str = "", height: int = 500,
                     key_suffix: str = "") -> str | None:
    """渲染通用 (nodes, edges) 图。"""
    return _render_graph(nodes, edges, "dag", title, key_suffix)


# ═══════════════════════════════════════════════════════════════════
# 数据转换: SEARCH_PATH → (nodes, edges)
# ═══════════════════════════════════════════════════════════════════

def _tree_to_graph(
    path: dict, parent_id: str | None = None, counter: list[int] | None = None
) -> tuple[dict[str, dict], list[dict[str, str]]]:
    if counter is None:
        counter = [0]
    q = path.get("question", "?")[:60]
    node_id = f"t{counter[0]}"
    counter[0] += 1
    nodes = {
        node_id: {
            "id": node_id, "question": q,
            "status": "solved" if path.get("answerable") else "unsolved",
            "answer": path.get("answer", "") or "",
            "detail": {
                "answerable": path.get("answerable", False),
                "judgement_reason": path.get("judgement_reason", ""),
                "chunks": path.get("chunks", []),
                "answer": path.get("answer", "") or "",
            },
        }
    }
    edges = []
    if parent_id:
        edges.append({"from": parent_id, "to": node_id, "type": "decomposition"})
    for child in path.get("next_queries", []):
        if isinstance(child, dict):
            cn, ce = _tree_to_graph(child, node_id, counter)
            nodes.update(cn); edges.extend(ce)
    return nodes, edges


# ═══════════════════════════════════════════════════════════════════
# Python 端布局算法
# ═══════════════════════════════════════════════════════════════════

def _compute_layout(nodes, edges):
    nids = list(nodes.keys())
    if not nids:
        return {"layers": [], "positions": {}, "svgW": 0, "svgH": 0}
    indeg = {nid: 0 for nid in nids}
    for e in edges:
        to_id = e.get("to", "")
        if to_id in indeg: indeg[to_id] += 1
    roots = [n for n in nids if indeg[n] == 0]
    start = roots if roots else [nids[0]]
    layer = {}
    queue = list(start)
    for r in start: layer[r] = 0
    while queue:
        u = queue.pop(0); nl = layer[u] + 1
        for e in edges:
            if e.get("from", "") == u:
                t = e.get("to", "")
                if t not in layer or layer[t] < nl:
                    layer[t] = nl; queue.append(t)
    for n in nids:
        if n not in layer: layer[n] = 0
    max_l = max(layer.values()) if layer else 0
    groups = []
    for i in range(max_l + 1):
        groups.append([n for n in nids if layer[n] == i])
    for i in range(1, len(groups)):
        prev = groups[i - 1]
        def _bary(nid):
            nb = [e.get("from","") for e in edges if e.get("to","") == nid and e.get("from","") in prev]
            return sum(prev.index(n) for n in nb) / len(nb) if nb else len(prev) / 2
        groups[i].sort(key=_bary)
    max_n = max(len(g) for g in groups) if groups else 1
    sw = max_n * (NODE_W + NODE_GAP) + PAD * 2
    sh = len(groups) * (NODE_H + LAYER_GAP) + PAD * 2
    pos = {}
    for li, grp in enumerate(groups):
        tw = len(grp) * NODE_W + (len(grp) - 1) * NODE_GAP
        sx = (sw - tw) / 2
        for ni, nid in enumerate(grp):
            pos[nid] = {"x": sx + ni * (NODE_W + NODE_GAP), "y": PAD + li * (NODE_H + LAYER_GAP)}
    return {"layers": groups, "positions": pos, "svgW": sw, "svgH": sh}


# ═══════════════════════════════════════════════════════════════════
# HTML 转义
# ═══════════════════════════════════════════════════════════════════

def _esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;").replace("'", "&#39;")


# ═══════════════════════════════════════════════════════════════════
# 自包含 iframe HTML 生成 (核心)
# ═══════════════════════════════════════════════════════════════════

def _build_iframe_html(nodes, edges, graph_type, title):
    """生成自包含 HTML: SVG 图 + JS 悬浮卡片交互。"""
    layout = _compute_layout(nodes, edges)
    positions = layout["positions"]
    sw, sh = layout["svgW"], layout["svgH"]
    iframe_h = max(IFRAME_HEIGHT, sh + 80)

    # 状态标签
    if graph_type == "tree":
        st_lbl = {"solved": "可回答", "unsolved": "不可回答"}
        leg = [("可回答", STATUS_COLORS["solved"]), ("不可回答", STATUS_COLORS["unsolved"])]
    else:
        st_lbl = {"solved": "SOLVED", "unsolved": "UNSOLVED", "blocked": "BLOCKED"}
        leg = [("SOLVED", STATUS_COLORS["solved"]), ("UNSOLVED", STATUS_COLORS["unsolved"]),
               ("BLOCKED", STATUS_COLORS["blocked"])]

    # ── 构建节点数据 JSON (供 JS 填充悬浮卡片) ──
    node_data = {}
    for nid, node in nodes.items():
        entry = {
            "id": nid,
            "question": (node.get("question") or "")[:200],
            "status": node.get("status", "unsolved"),
            "answer": (node.get("answer") or "")[:500],
        }
        if graph_type == "tree":
            d = node.get("detail", {})
            entry["answerable"] = d.get("answerable", False)
            entry["judgement_reason"] = (d.get("judgement_reason") or "")[:300]
            entry["chunks"] = [
                {"title": c.get("chunk_title", "?"), "content": (c.get("page_content") or "")[:300]}
                for c in d.get("chunks", [])
            ]
        else:
            entry["health"] = node.get("health", "healthy")
            entry["planner_rationale"] = (node.get("planner_rationale") or "")[:200]
            entry["search_query"] = (node.get("search_query") or "")[:200]
            entry["solver_judgment"] = (node.get("solver_judgment") or "")[:200]
            entry["critic_factual_notes"] = (node.get("critic_factual_notes") or "")[:300]
            entry["critic_normative_advice"] = (node.get("critic_normative_advice") or "")[:300]
            entry["round_created"] = node.get("round_created", 0)
            entry["round_last_updated"] = node.get("round_last_updated", 0)
            # retrieved_chunks: live query 有完整 ChunkInfo, eval 数据用 retrieved_chunks_summary
            rc = node.get("retrieved_chunks", [])
            if rc and isinstance(rc, list) and len(rc) > 0 and isinstance(rc[0], dict) and "page_content" in rc[0]:
                entry["retrieved_chunks"] = [
                    {"title": c.get("chunk_title", "?"), "content": c.get("page_content", ""),
                     "summary": c.get("chunk_summary", ""), "context": c.get("context_title", "")}
                    for c in rc
                ]
            else:
                summary = node.get("retrieved_chunks_summary", [])
                entry["retrieved_chunks"] = [
                    {"title": c.get("chunk_title", "?"), "content": c.get("page_content", ""),
                     "summary": c.get("chunk_summary", ""), "context": c.get("context_title", "")}
                    for c in (summary if isinstance(summary, list) else [])
                ]
            entry["retrieved_chunks_count"] = node.get("retrieved_chunks_count", len(entry["retrieved_chunks"]))
            entry["supporting_chunks"] = node.get("supporting_chunks", [])
            entry["supporting_chunks_count"] = node.get("supporting_chunks_count", 0)
        node_data[nid] = entry

    node_data_json = json.dumps(node_data, ensure_ascii=False)
    positions_json = json.dumps(positions, ensure_ascii=False)
    status_colors_json = json.dumps(STATUS_COLORS, ensure_ascii=False)
    health_colors_json = json.dumps(HEALTH_COLORS, ensure_ascii=False)

    # ── 构建 SVG 元素字符串 ──
    svg_parts = []
    svg_parts.append(f'<div class="graph-container" id="graph-container">')
    svg_parts.append(f'<div class="zoom-ctrl">')
    svg_parts.append(f'<button onclick="zoomIn()" title="Zoom in">+</button>')
    svg_parts.append(f'<button onclick="zoomOut()" title="Zoom out">–</button>')
    svg_parts.append(f'<button class="reset-btn" onclick="zoomReset()" title="Reset zoom">↺</button>')
    svg_parts.append(f'</div>')
    svg_parts.append(f'<div class="svg-scroll-area">')
    svg_parts.append(f'<svg id="graph" viewBox="0 0 {sw} {sh}" preserveAspectRatio="xMidYMid meet" style="max-width:{sw}px;">')
    svg_parts.append(f'<g id="transform-layer">')

    # 边
    centers = {}
    for nid, p in positions.items():
        centers[nid] = (p["x"] + NODE_W / 2, p["y"] + NODE_H / 2)
    for e in edges:
        fid, tid = e.get("from", ""), e.get("to", "")
        if fid not in centers or tid not in centers: continue
        fx, fy = centers[fid]; tx, ty = centers[tid]
        dy = ty - fy; cp = abs(dy) * 0.4
        d = f"M {fx} {fy + NODE_H / 2} C {fx} {fy + NODE_H / 2 + cp}, {tx} {ty - NODE_H / 2 - cp}, {tx} {ty - NODE_H / 2}"
        ec = "es-d" if e.get("type") == "dependency" else "es-s"
        svg_parts.append(f'<path d="{d}" class="{ec}"/>')
        if dy > 10:
            svg_parts.append(f'<polygon points="{tx},{ty - NODE_H / 2} {tx-5},{ty - NODE_H / 2 - 8} {tx+5},{ty - NODE_H / 2 - 8}" class="ar"/>')

    # 节点 (带 onclick)
    for nid, node in nodes.items():
        p = positions.get(nid)
        if not p: continue
        x, y = p["x"], p["y"]
        st = node.get("status", "unsolved")
        bg = STATUS_COLORS.get(st, "#94a3b8")
        h = node.get("health", "healthy")
        hc = HEALTH_COLORS.get(h, HEALTH_COLORS["healthy"])
        q = _esc((node.get("question") or "")[:30])
        lbl = st_lbl.get(st, st)
        onclick = f"toggleCard(&#39;{_esc(nid)}&#39;,{x},{y})"
        svg_parts.append(f'<g class="nd" data-nid="{_esc(nid)}" onclick="{onclick}" style="cursor:pointer">')
        svg_parts.append(f'<g transform="translate({x},{y})">')
        svg_parts.append(f'<rect class="nb" width="{NODE_W}" height="{NODE_H}" rx="8" fill="{bg}"/>')
        svg_parts.append(f'<circle cx="10" cy="10" r="5" fill="{hc}" stroke="rgba(255,255,255,.6)" stroke-width="1.5"/>')
        svg_parts.append(f'<text class="nt" x="20" y="16">{_esc(nid)}</text>')
        svg_parts.append(f'<text class="nq" x="12" y="34">{q}</text>')
        svg_parts.append(f'<text class="nq" x="12" y="48">{_esc(lbl)}</text>')
        svg_parts.append('</g></g>')

    # 图例
    ly = sh - 16
    for i, (lbl, clr) in enumerate(leg):
        lx = PAD + i * 100
        svg_parts.append(f'<circle cx="{lx}" cy="{ly}" r="5" fill="{clr}"/>')
        svg_parts.append(f'<text class="lg" x="{lx+10}" y="{ly+4}">{lbl}</text>')

    svg_parts.append('</g>')
    svg_parts.append('</svg>')
    svg_parts.append('</div>')
    svg_parts.append('</div>')
    svg_str = "\n".join(svg_parts)

    # ── 组装完整 HTML ──
    html = f'''<!DOCTYPE html>
<html lang="zh"><head><meta charset="utf-8"><style>
*{{margin:0;padding:0;box-sizing:border-box;}}
body{{font:13px -apple-system,BlinkMacSystemFont,'Segoe UI','Noto Sans SC',sans-serif;background:#f8f9fa;overflow:hidden;}}
h2{{font-size:15px;font-weight:700;color:#1f2937;padding:16px 20px 0;margin:0;}}
p.st{{font-size:11px;color:#9ca3af;padding:4px 20px 12px;margin:0;}}
.graph-container{{position:relative;overflow:hidden;background:#f8f9fa;border-radius:8px;}}
.svg-scroll-area{{overflow:hidden;width:100%;}}
svg{{display:block;width:100%;height:auto;}}
.zoom-ctrl{{position:absolute;top:8px;right:8px;display:flex;gap:4px;z-index:10;}}
.zoom-ctrl button{{width:28px;height:28px;border:1px solid #d1d5db;border-radius:6px;background:rgba(255,255,255,.85);cursor:pointer;font-size:15px;line-height:28px;text-align:center;padding:0;color:#374151;user-select:none;}}
.zoom-ctrl button:hover{{background:#f3f4f6;border-color:#9ca3af;}}
.zoom-ctrl button.reset-btn{{font-size:11px;width:auto;padding:0 6px;}}
.nd{{}}
.nd.sel .nb{{stroke:#2563eb;stroke-width:3px;}}
.nb{{stroke:rgba(0,0,0,.08);stroke-width:1.5px;transition:stroke .15s;}}
.nt{{font:bold 11px sans-serif;fill:#fff;}}
.nq{{font:10px sans-serif;fill:rgba(255,255,255,.9);}}
.es-s{{stroke:#9ca3af;stroke-width:2;fill:none;}}
.es-d{{stroke:#c4c9d0;stroke-width:2;stroke-dasharray:6 3;fill:none;}}
.ar{{fill:#9ca3af;}}
.lg{{font:10px sans-serif;fill:#6b7280;}}
.card{{
  display:none;position:absolute;z-index:1000;
  background:#fff;border:1px solid #e5e7eb;border-radius:12px;
  padding:16px;box-shadow:0 4px 16px rgba(0,0,0,.15);
  max-width:400px;max-height:380px;overflow-y:auto;
  font-size:12px;line-height:1.5;
}}
.card.show{{display:block;}}
.card h3{{font-size:14px;margin:0 0 6px;color:#1f2937;}}
.card .close{{
  position:absolute;top:8px;right:12px;
  background:none;border:none;font-size:18px;cursor:pointer;color:#9ca3af;
  padding:2px 6px;border-radius:4px;
}}
.card .close:hover{{background:#f3f4f6;color:#1f2937;}}
.card .field{{margin:4px 0;}}
.card .label{{font-weight:600;color:#6b7280;}}
.card .val{{color:#1f2937;word-break:break-word;}}
.card .chunk{{margin:4px 0;padding:6px 8px;background:#f9fafb;border-radius:6px;border-left:3px solid #e5e7eb;}}
.card .chunk .ct{{font-weight:600;color:#374151;}}
.card .chunk .cc{{font-size:11px;color:#6b7280;margin-top:2px;}}
.chips{{display:flex;flex-wrap:wrap;gap:4px;margin:4px 0;}}
.chip{{font-size:10px;padding:2px 6px;border-radius:10px;font-weight:600;}}
.chip-ok{{background:#dcfce7;color:#166534;}}
.chip-warn{{background:#fef3c7;color:#92400e;}}
.chip-bad{{background:#fee2e2;color:#991b1b;}}
.sect{{margin:8px 0;padding:8px 10px;background:#f9fafb;border-radius:8px;border:1px solid #f3f4f6;}}
.sect-h{{font-size:11px;font-weight:700;color:#6b7280;margin-bottom:4px;text-transform:uppercase;letter-spacing:.5px;}}
</style></head>
<body>
<h2>{_esc(title)}</h2>
<p class="st">{_esc(_subtitle(graph_type))}</p>
{svg_str}
<div id="card" class="card"></div>
<script>
var NODES = {node_data_json};
var POSITIONS = {positions_json};
var STATUS_COLORS = {status_colors_json};
var HEALTH_COLORS = {health_colors_json};
var GRAPH_TYPE = "{graph_type}";
var NODE_W = {NODE_W};
var NODE_H = {NODE_H};
var OPEN_ID = null;

// ── Zoom/Pan state ──
var scale = 1;
var panX = 0, panY = 0;
var MIN_SCALE = 0.2;
var MAX_SCALE = 3.0;
var ZOOM_STEP = 1.2;
var isPanning = false;
var panStartX = 0, panStartY = 0;

// SVG viewBox dimensions for coordinate mapping
var svgViewBoxW = {sw};
var svgViewBoxH = {sh};

function updateTransform() {{
  var layer = document.getElementById('transform-layer');
  if (layer) {{
    layer.setAttribute('transform', 'translate(' + panX + ',' + panY + ') scale(' + scale + ')');
  }}
}}

var graphContainer = document.getElementById('graph-container');
graphContainer.addEventListener('wheel', function(e) {{
  e.preventDefault();
  var rect = graphContainer.getBoundingClientRect();
  var mx = e.clientX - rect.left;
  var my = e.clientY - rect.top;
  var oldScale = scale;
  if (e.deltaY < 0) {{
    scale = Math.min(MAX_SCALE, scale * ZOOM_STEP);
  }} else {{
    scale = Math.max(MIN_SCALE, scale / ZOOM_STEP);
  }}
  var ds = scale / oldScale;
  panX = mx - ds * (mx - panX);
  panY = my - ds * (my - panY);
  updateTransform();
}}, {{passive: false}});

graphContainer.addEventListener('mousedown', function(e) {{
  if (e.target.closest('.nd') || e.target.closest('.card') || e.target.closest('.zoom-ctrl')) return;
  isPanning = true;
  panStartX = e.clientX - panX;
  panStartY = e.clientY - panY;
  graphContainer.style.cursor = 'grabbing';
  e.preventDefault();
}});

document.addEventListener('mousemove', function(e) {{
  if (!isPanning) return;
  panX = e.clientX - panStartX;
  panY = e.clientY - panStartY;
  updateTransform();
}});

document.addEventListener('mouseup', function() {{
  if (isPanning) {{
    isPanning = false;
    graphContainer.style.cursor = '';
  }}
}});

function zoomIn() {{
  scale = Math.min(MAX_SCALE, scale * ZOOM_STEP);
  updateTransform();
}}

function zoomOut() {{
  scale = Math.max(MIN_SCALE, scale / ZOOM_STEP);
  updateTransform();
}}

function zoomReset() {{
  scale = 1;
  panX = 0;
  panY = 0;
  updateTransform();
}}

if (window.ResizeObserver) {{
  var ro = new ResizeObserver(function() {{
    // No-op: SVG max-width + viewBox handles responsive sizing natively.
    // ResizeObserver is kept as a hook for future dynamic layout adjustments.
  }});
  ro.observe(document.getElementById('graph-container'));
}}

function toggleCard(nid, x, y) {{
  if (OPEN_ID === nid) {{ closeCard(); return; }}
  closeCard();
  OPEN_ID = nid;
  // 高亮
  var prev = document.querySelector('.nd.sel');
  if (prev) prev.classList.remove('sel');
  var el = document.querySelector('[data-nid="' + nid + '"]');
  if (el) el.classList.add('sel');
  // 填充卡片
  var node = NODES[nid];
  if (!node) return;
  var card = document.getElementById('card');
  var h = '';
  h += '<button class="close" onclick="closeCard()">&times;</button>';
  h += '<h3>' + nid + ': ' + (node.question||'').substring(0,80) + '</h3>';
  // 状态 chips
  var st = node.status||'unsolved';
  var stIcon = st==='solved'?'✅':st==='blocked'?'🚫':'⏳';
  var stLabel = GRAPH_TYPE==='tree'?(st==='solved'?'可回答':'不可回答'):st.toUpperCase();
  h += '<div class="chips"><span class="chip chip-'+(st==='solved'?'ok':st==='blocked'?'bad':'warn')+'">'+stIcon+' '+stLabel+'</span>';
  if (node.health) {{
    var hi = node.health==='healthy'?'💚':node.health==='needs_verification'?'⚠️':node.health==='unreliable'?'🔴':'🚫';
    h += '<span class="chip chip-'+(node.health==='healthy'?'ok':node.health==='unreliable'||node.health==='blocked'?'bad':'warn')+'">'+hi+' '+node.health+'</span>';
  }}
  h += '</div>';

  if (GRAPH_TYPE === 'dag') {{
    // ══ Planner ══
    var pParts = [];
    if (node.question) pParts.push('<div class="field"><span class="label">Question:</span> <span class="val">'+escH(node.question)+'</span></div>');
    if (node.planner_rationale) pParts.push('<div class="field"><span class="label">Rationale:</span> <span class="val">'+escH(node.planner_rationale)+'</span></div>');
    if (pParts.length) h += '<div class="sect"><div class="sect-h">📐 Planner</div>'+pParts.join('')+'</div>';

    // ══ Solver ══
    var sParts = [];
    if (node.answer) sParts.push('<div class="field"><span class="label">Answer:</span> <span class="val">'+escH(node.answer)+'</span></div>');
    if (node.search_query) sParts.push('<div class="field"><span class="label">Search Query:</span> <span class="val">'+escH(node.search_query)+'</span></div>');
    if (node.solver_judgment) sParts.push('<div class="field"><span class="label">Judgment:</span> <span class="val">'+escH(node.solver_judgment)+'</span></div>');
    // Retrieved chunks
    var rcCount = node.retrieved_chunks_count || (node.retrieved_chunks ? node.retrieved_chunks.length : 0);
    if (node.retrieved_chunks && node.retrieved_chunks.length > 0) {{
      sParts.push('<div class="field"><span class="label">Retrieved Chunks ('+node.retrieved_chunks.length+'):</span></div>');
      node.retrieved_chunks.forEach(function(c,i) {{
        sParts.push('<div class="chunk"><div class="ct">'+(i+1)+'. '+escH(c.title||'?')+'</div>');
        if (c.context) sParts.push('<div class="cc" style="color:#9ca3af;font-size:10px;">'+escH(c.context)+'</div>');
        if (c.summary) sParts.push('<div class="cc" style="color:#6b7280;">'+escH(c.summary)+'</div>');
        if (c.content) sParts.push('<div class="cc">'+escH(c.content)+'</div>');
        sParts.push('</div>');
      }});
    }} else if (rcCount > 0) {{
      sParts.push('<div class="field"><span class="label">Retrieved Chunks:</span> <span class="val">'+rcCount+' chunks (summary only in eval data)</span></div>');
    }}
    // Supporting chunks
    var supChunks = node.supporting_chunks || [];
    var supCount = node.supporting_chunks_count || supChunks.length;
    if (supCount > 0) {{
      var supText = supCount + ' chunk' + (supCount > 1 ? 's' : '');
      if (supChunks.length > 0 && supChunks.length <= 10) supText += ': ' + supChunks.join(', ');
      sParts.push('<div class="field"><span class="label">Supporting Chunks:</span> <span class="val">'+escH(supText)+'</span></div>');
    }}
    if (sParts.length) h += '<div class="sect"><div class="sect-h">🔍 Solver</div>'+sParts.join('')+'</div>';

    // ══ Critic ══
    var cParts = [];
    if (node.critic_factual_notes) cParts.push('<div class="field"><span class="label">Factual Notes:</span> <span class="val">'+escH(node.critic_factual_notes)+'</span></div>');
    if (node.critic_normative_advice) cParts.push('<div class="field"><span class="label">Normative Advice:</span> <span class="val">'+escH(node.critic_normative_advice)+'</span></div>');
    if (cParts.length) h += '<div class="sect"><div class="sect-h">🪞 Critic</div>'+cParts.join('')+'</div>';

    // ══ System ══
    if (node.round_created != null) {{
      h += '<div class="sect"><div class="sect-h">⚙️ System</div>';
      h += '<div class="field"><span class="label">Created:</span> <span class="val">Round '+node.round_created+'</span></div>';
      h += '<div class="field"><span class="label">Last Updated:</span> <span class="val">Round '+node.round_last_updated+'</span></div>';
      h += '</div>';
    }}
  }} else {{
    // ══ Tree (Judge) ══
    if (node.answer) h += '<div class="field"><span class="label">答案:</span> <span class="val">'+escH(node.answer)+'</span></div>';
    if (node.judgement_reason) h += '<div class="field"><span class="label">Judge 理由:</span> <span class="val">'+escH(node.judgement_reason)+'</span></div>';
    if (node.chunks && node.chunks.length > 0) {{
      h += '<div class="field"><span class="label">Chunks ('+node.chunks.length+'):</span></div>';
      node.chunks.forEach(function(c,i) {{
        h += '<div class="chunk"><div class="ct">'+(i+1)+'. '+escH(c.title||'?')+'</div>';
        if (c.content) h += '<div class="cc">'+escH(c.content)+'</div>';
        h += '</div>';
      }});
    }}
  }}
  card.innerHTML = h;
  // 定位卡片: 先测量高度, 再放到正确位置 (避免底部节点被截断)
  var svg = document.getElementById('graph');
  var svgRect = svg.getBoundingClientRect();
  // viewBox-to-display scale factors
  var sx = svgRect.width / svgViewBoxW;
  var sy = svgRect.height / svgViewBoxH;
  // 将 SVG 坐标变换为视觉坐标 (考虑 viewBox 映射 + zoom/pan)
  var visualX = (x * scale + panX) * sx;
  var visualY = (y * scale + panY) * sy;
  var visualNodeW = NODE_W * scale * sx;
  var cx = svgRect.left + visualX + visualNodeW + 12;
  // 先放右侧; 若超出视口则放左侧
  if (cx + 400 > window.innerWidth - 20) cx = svgRect.left + visualX - 412;
  // 临时定位到视口外以测量高度
  card.style.left = '-9999px';
  card.style.top = '0px';
  card.classList.add('show');
  var cardH = card.getBoundingClientRect().height;
  card.classList.remove('show');
  // 用实际高度计算垂直位置
  var cy = svgRect.top + visualY;
  if (cy + cardH > window.innerHeight - 10) cy = window.innerHeight - cardH - 16;
  if (cy < 10) cy = 10;
  card.style.left = cx + 'px';
  card.style.top = cy + 'px';
  card.classList.add('show');
}}

function closeCard() {{
  var prev = document.querySelector('.nd.sel');
  if (prev) prev.classList.remove('sel');
  document.getElementById('card').classList.remove('show');
  OPEN_ID = null;
}}

function escH(s) {{
  return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}}

// 点击空白处关闭卡片
document.addEventListener('click', function(e) {{
  if (!e.target.closest('.nd') && !e.target.closest('.card')) closeCard();
}});
</script>
</body></html>'''

    return html, iframe_h


# ═══════════════════════════════════════════════════════════════════
# 核心渲染
# ═══════════════════════════════════════════════════════════════════

_next_key_id = 0


def _render_graph(nodes, edges, graph_type, title, key_suffix=""):
    """渲染交互式图。

    双层交互:
      1. 自包含 iframe: 点击节点 → 悬浮卡片 (纯客户端, 即时响应)
      2. st.selectbox: 下拉选节点 → Streamlit 详情卡片 (服务端渲染, 更全面)

    返回选中节点 ID 供调用方展示 Streamlit 详情卡片。
    """
    global _next_key_id
    if not nodes:
        return None

    if not key_suffix:
        key_suffix = str(_next_key_id)
        _next_key_id += 1

    # ── 1. iframe 悬浮卡片交互 ──
    iframe_html, iframe_h = _build_iframe_html(nodes, edges, graph_type, title)
    components.html(iframe_html, height=iframe_h, scrolling=False)

    # ── 2. selectbox 节点选择器 (独立于 iframe, 用于 Streamlit 详情展示) ──
    node_options = ["(无)"] + list(nodes.keys())
    node_labels = ["(选择节点查看 Streamlit 详情卡片)"]
    for nid in nodes:
        q = (nodes[nid].get("question") or "")[:50]
        st_label = nodes[nid].get("status", "unsolved")
        icon = "✅" if st_label == "solved" else "⏳"
        node_labels.append(f"{icon} {nid}: {q}")

    sel_key = f"_graph_sel_{graph_type}_{key_suffix}"
    picked = st.selectbox(
        "选择节点查看 Streamlit 详情",
        options=node_options,
        format_func=lambda n: node_labels[node_options.index(n)] if n in node_options else n,
        key=sel_key,
    )

    if picked and picked != "(无)":
        return picked
    return None


# ═══════════════════════════════════════════════════════════════════
# 节点详情渲染 (Streamlit 端——保留作为备选/补充视图)
# ═══════════════════════════════════════════════════════════════════

def render_node_detail(node: dict, node_id: str) -> None:
    """渲染 DAG 节点详情 — DAGNode 全部字段。"""
    with st.container(border=True):
        st.markdown(f"**节点 `{node_id}`**")
        c1, c2, c3 = st.columns(3)
        status = node.get("status", "?")
        icon = {"solved": "✅", "unsolved": "⏳", "blocked": "🚫"}.get(status, "❓")
        c1.metric("状态", f"{icon} {status}")
        health = node.get("health") or node.get("critic_health")
        c2.metric("健康度", str(health) if health else "N/A")
        rc = node.get("round_created", "?"); ru = node.get("round_last_updated", "?")
        c3.metric("轮次", f"创建 R{rc} / 更新 R{ru}")
        question = node.get("question", "")
        if question: st.markdown(f"**问题:** {question[:200]}")
        rationale = node.get("planner_rationale", "")
        if rationale: st.caption(f"Planner 理由: {rationale[:200]}")
        answer = node.get("answer", "")
        if answer: st.markdown(f"**答案:** {answer[:500]}")
        sq = node.get("search_query", "")
        if sq: st.caption(f"搜索查询: {sq[:200]}")
        sj = node.get("solver_judgment", "")
        if sj: st.caption(f"Solver 判断: {sj[:200]}")
        retrieved = node.get("retrieved_chunks", [])
        if retrieved:
            with st.expander(f"检索 Chunks ({len(retrieved)})", expanded=False):
                for i, c in enumerate(retrieved[:10], 1):
                    if isinstance(c, dict):
                        st.caption(f"{i}. {c.get('chunk_title','?')[:80]}")
                        ct = c.get("page_content", "")[:200]
                        if ct: st.caption(f"   {ct}")
        cn = node.get("critic_factual_notes", "")
        if cn: st.caption(f"Critic 事实核查: {cn[:300]}")
        ca = node.get("critic_normative_advice", "")
        if ca: st.caption(f"Critic 建议: {ca[:300]}")


def render_tree_node_detail(node: dict, node_id: str) -> None:
    """渲染搜索树节点详情。"""
    with st.container(border=True):
        st.markdown(f"**节点 `{node_id}`**")
        detail = node.get("detail", {})
        c1, c2 = st.columns(2)
        answerable = detail.get("answerable", False)
        icon = "✅ 可回答" if answerable else "🔄 不可回答"
        c1.metric("Judge 判断", icon)
        c2.metric("问题", node.get("question", "")[:80])
        reason = detail.get("judgement_reason", "")
        if reason: st.caption(f"Judge 理由: {reason[:300]}")
        answer = detail.get("answer") or node.get("answer", "")
        if answer: st.markdown(f"**答案:** {answer[:500]}")
        tree_chunks = detail.get("chunks", [])
        if tree_chunks:
            with st.expander(f"召回的 Chunks ({len(tree_chunks)})", expanded=True):
                for i, c in enumerate(tree_chunks, 1):
                    st.markdown(f"**{i}. {c.get('chunk_title','Unknown')}**")
                    src = c.get("source_url", "")
                    if src: st.caption(f"来源: {src[:120]}")
                    ct = c.get("page_content", "")
                    if ct: st.markdown(ct[:500])
        else:
            st.caption("该节点未召回任何 chunks")