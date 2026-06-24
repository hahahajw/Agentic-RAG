#!/usr/bin/env python3
"""Generate publication-quality DAG evolution figures for thesis.

Produces two complementary visualizations from rag_loop eval JSON:
  1. Small Multiples DAG — side-by-side compact DAG snapshots showing
     structural evolution (node additions, edge changes) across rounds.
  2. Swimlane State Matrix — node × round grid encoding status (fill color)
     and health (hatch pattern), with bold borders on state-change cells.

Output: PDF (vector, for LaTeX \\includegraphics) + PNG (300 DPI preview).

Usage:
    uv run python frontend/pages/generate_dag_figures.py \
        --input Eval/rag_loop_data/result/gb_standards_25_1/0000_Q1.json \
        --output-dir figures/ \
        --format pdf png

Requires: matplotlib (standard scientific Python dependency).
"""

from __future__ import annotations

import argparse
import json
import textwrap
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.lines import Line2D
from matplotlib import font_manager as fm

# ═══════════════════════════════════════════════════════════════════════════════
# Nature-journal PALETTE  (from nature-figure skill references/api.md)
# ═══════════════════════════════════════════════════════════════════════════════

PALETTE = {
    "blue_main":      "#0F4D92",
    "blue_secondary": "#3775BA",
    "green_1": "#DDF3DE",
    "green_2": "#AADCA9",
    "green_3": "#8BCF8B",
    "red_1":   "#F6CFCB",
    "red_2":   "#E9A6A1",
    "red_strong": "#B64342",
    "neutral_light": "#CFCECE",
    "neutral_mid":   "#767676",
    "neutral_dark":  "#4D4D4D",
    "neutral_black": "#272727",
    "gold":   "#FFD700",
    "teal":   "#42949E",
    "violet": "#9A4D8E",
}

# ── Semantic status mapping onto Nature palette ──
#   solved   → green_3   (verified positive outcome)
#   unsolved → red_2     (pending, needs attention — warm but not alarming)
#   blocked  → red_strong (hard failure, stuck)
#   N0 root  → blue_main  (structural anchor node)
STATUS_COLORS: dict[str, str] = {
    "solved":         PALETTE["green_3"],
    "unsolved":       PALETTE["red_2"],
    "blocked":        PALETTE["red_strong"],
    "failed_search":  PALETTE["neutral_mid"],
    "empty_search":   PALETTE["neutral_mid"],
}

# Lighter tints for swimlane cell backgrounds
STATUS_LIGHT: dict[str, str] = {
    "solved":         "#E8F5E9",
    "unsolved":       "#FDF0EF",
    "blocked":        "#FDE8E5",
    "failed_search":  "#F2F2F2",
    "empty_search":   "#F2F2F2",
}

HEALTH_HATCH: dict[str, str] = {
    "healthy":              "",
    "needs_verification":   "///",
    "unreliable":           "xxx",
    "blocked":              "xxx",
}

EDGE_STYLE = {
    "decomposition": {"ls": "-",  "lw": 1.0},
    "dependency":    {"ls": "--", "lw": 0.8},
}

# Neutral tokens — all from PALETTE for consistency
GRID_COLOR      = PALETTE["neutral_light"]
TEXT_PRIMARY    = PALETTE["neutral_black"]
TEXT_SECONDARY  = PALETTE["neutral_dark"]
TEXT_MUTED      = PALETTE["neutral_mid"]
BORDER_DEFAULT  = PALETTE["neutral_light"]
BORDER_CHANGE   = PALETTE["blue_main"]
MISSING_FILL    = "#F2F2F2"
EDGE_COLOR       = PALETTE["neutral_mid"]

# Layout constants
NODE_W = 2.4        # inches
NODE_H = 0.72       # inches
LAYER_GAP = 1.3     # vertical gap between layers
NODE_GAP = 0.35     # horizontal gap between nodes in same layer
PANEL_PAD = 0.25    # padding inside each panel


# ═══════════════════════════════════════════════════════════════════════════════
# Font Setup
# ═══════════════════════════════════════════════════════════════════════════════

def _setup_fonts() -> dict[str, fm.FontProperties]:
    """Configure fonts for Nature-journal Chinese figures.

    Nature standard: sans-serif, Arial first. CJK: Microsoft YaHei (Windows) or SimHei.
    Sets rcParams for editable SVG/PDF text and frameless legends.
    """
    # ── MANDATORY Nature-figure rcParams (api.md) ──
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = [
        "Arial", "Microsoft YaHei", "SimHei", "DejaVu Sans",
    ]
    plt.rcParams["svg.fonttype"] = "none"    # editable <text> nodes in SVG
    plt.rcParams["pdf.fonttype"] = 42        # editable TrueType in PDF
    plt.rcParams["ps.fonttype"] = 42
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["legend.frameon"] = False   # frameless legends
    plt.rcParams["axes.spines.right"] = False
    plt.rcParams["axes.spines.top"] = False

    # Determine best CJK font available
    system_fonts = {f.name for f in fm.fontManager.ttflist}
    cn_family = None
    for name in ["Microsoft YaHei", "SimHei"]:
        if name in system_fonts:
            cn_family = name
            break

    fp_cn = fm.FontProperties(family=cn_family or "sans-serif", size=9)
    fp_en = fm.FontProperties(family="Arial", size=8)
    fp_mono = fm.FontProperties(family="Consolas", size=8)
    return {"cn": fp_cn, "en": fp_en, "mono": fp_mono}


# ═══════════════════════════════════════════════════════════════════════════════
# Data Loading
# ═══════════════════════════════════════════════════════════════════════════════

def load_data(path: Path) -> list[dict]:
    """Load dag_snapshots from eval result JSON."""
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    if "dag_snapshots" in raw:
        snapshots = raw["dag_snapshots"]
    elif "round_dags" in raw:
        snapshots = raw["round_dags"]
    else:
        raise ValueError(f"No dag_snapshots or round_dags found in {path}")

    return sorted(snapshots, key=lambda s: s.get("round_number", 0))


# ═══════════════════════════════════════════════════════════════════════════════
# Node Label Extraction
# ═══════════════════════════════════════════════════════════════════════════════

def extract_short_labels(snapshots: list[dict]) -> dict[str, str]:
    """Generate short display labels for each node ID.

    Strategy — ordered by priority:
      1. N0 → 'Q (根节点)'
      2. Extract GB/T code + keyword from question
      3. Extract key concept pair from question (e.g. '熔炼分析 vs 成品分析')
      4. Short truncation of question text
    """
    all_nodes: dict[str, dict] = {}
    for snap in snapshots:
        for nid, node in snap.get("nodes", {}).items():
            if nid not in all_nodes:
                all_nodes[nid] = node

    labels: dict[str, str] = {}
    for nid, node in all_nodes.items():
        if nid == "N0":
            labels[nid] = "Q (根节点)"
            continue

        q = node.get("question", "")
        label = _extract_key_entity(q)
        if not label:
            label = q[:16] + "…" if len(q) > 16 else q
        labels[nid] = label

    return labels


def _extract_key_entity(question: str) -> str:
    """Extract a short label (≤18 chars) from a Chinese standards question."""
    import re

    # ── Strategy 1: Match GB/T code + following keyword ──
    m = re.search(r"GB/?T\s*\d+[—\-—―]*\d*", question)
    if m:
        std = m.group(0).replace("—", "-").replace("—", "-")
        rest = question[m.end():].strip().rstrip("。，的")
        # Look for a short qualifying term right after the standard code
        keywords = ["偏差", "允许偏差", "特殊规定", "表", "条款", "脚注"]
        for kw in keywords:
            idx = question.find(kw)
            if 0 < idx < len(question):
                # Get a window around the keyword
                label = f"{std}\n{kw}"
                if len(label) <= 20:
                    return label[:20]
        label = std[:18]
        return label

    # ── Strategy 2: Match 'X 与 Y' / 'X vs Y' patterns ──
    vs_patterns = [
        r"(熔炼分析).*(成品分析).*(?:定义|区别|差异)",
        r"(成品分析).*(熔炼分析).*(?:定义|区别|差异)",
    ]
    for pat in vs_patterns:
        m = re.search(pat, question)
        if m:
            g1, g2 = m.group(1), m.group(2)
            label = f"{g1} vs {g2}"
            return label[:18]

    # ── Strategy 3: Match '是否' questions and extract the core subject ──
    m = re.search(r"([^，。是否]+)(?:是否|有没有|可不可以)", question)
    if m:
        subj = m.group(1).strip().rstrip("的")
        label = subj[-14:] if len(subj) > 14 else subj
        return label[:18]

    # ── Strategy 4: Just take the first 16 chars ──
    clean = question.replace("　", " ").strip()
    return clean[:16]


# ═══════════════════════════════════════════════════════════════════════════════
# Change Detection
# ═══════════════════════════════════════════════════════════════════════════════

def detect_changes(snapshots: list[dict]) -> dict:
    """Detect all state changes between consecutive rounds.

    Returns:
        {
            "first_appear": {node_id: round_idx},
            "status_changes": {(node_id, round_idx): old_status},
            "health_changes": {(node_id, round_idx): old_health},
            "question_changes": {(node_id, round_idx): True},  # relabel
        }
    """
    first_appear: dict[str, int] = {}
    status_changes: dict[tuple[str, int], str] = {}
    health_changes: dict[tuple[str, int], str] = {}
    question_changes: dict[tuple[str, int], bool] = {}

    prev_nodes: dict[str, dict] = {}

    for ridx, snap in enumerate(snapshots):
        curr_nodes = snap.get("nodes", {})

        for nid, node in curr_nodes.items():
            if nid not in first_appear:
                first_appear[nid] = ridx

            if nid in prev_nodes:
                prev = prev_nodes[nid]
                if node.get("status") != prev.get("status"):
                    status_changes[(nid, ridx)] = prev.get("status", "")
                if node.get("health") != prev.get("health"):
                    health_changes[(nid, ridx)] = prev.get("health", "")
                if node.get("question") != prev.get("question"):
                    question_changes[(nid, ridx)] = True

        prev_nodes = dict(curr_nodes)

    return {
        "first_appear": first_appear,
        "status_changes": status_changes,
        "health_changes": health_changes,
        "question_changes": question_changes,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# DAG Layout (Sugiyama-style: BFS layering + barycentric ordering)
# ═══════════════════════════════════════════════════════════════════════════════

def compute_dag_layout(nodes: dict, edges: list[dict]) -> dict[str, tuple[float, float]]:
    """Compute (x, y) positions for each node. Returns {node_id: (x, y)}."""
    if not nodes:
        return {}

    # Build adjacency
    children: dict[str, list[str]] = defaultdict(list)
    parents: dict[str, list[str]] = defaultdict(list)
    for e in edges:
        f, t = e.get("from", ""), e.get("to", "")
        if f in nodes and t in nodes:
            children[f].append(t)
            parents[t].append(f)

    # Find roots (no parents)
    roots = [nid for nid in nodes if not parents[nid]]
    if not roots:
        roots = [next(iter(nodes))]

    # BFS layering
    layers: dict[str, int] = {}
    queue = list(roots)
    for r in roots:
        layers[r] = 0

    visited = set()
    while queue:
        nid = queue.pop(0)
        if nid in visited:
            continue
        visited.add(nid)
        for child in children.get(nid, []):
            if child not in layers:
                layers[child] = layers[nid] + 1
            else:
                layers[child] = max(layers[child], layers[nid] + 1)
            if child not in visited:
                queue.append(child)

    # Assign layer 0 to any unvisited nodes
    for nid in nodes:
        if nid not in layers:
            layers[nid] = 0

    # Group by layer
    layer_groups: dict[int, list[str]] = defaultdict(list)
    for nid, layer in layers.items():
        layer_groups[layer].append(nid)

    # Barycentric ordering within layers
    max_layer = max(layer_groups.keys()) if layer_groups else 0
    for l in range(1, max_layer + 1):
        group = layer_groups[l]
        prev_group = layer_groups.get(l - 1, [])
        prev_pos = {nid: i for i, nid in enumerate(prev_group)}

        def barycenter(nid):
            pars = parents.get(nid, [])
            if not pars:
                return 0
            return sum(prev_pos.get(p, 0) for p in pars) / len(pars)

        group.sort(key=barycenter)
        layer_groups[l] = group

    # Assign coordinates
    positions: dict[str, tuple[float, float]] = {}
    for layer, group in layer_groups.items():
        n = len(group)
        total_w = n * NODE_W + (n - 1) * NODE_GAP
        start_x = -total_w / 2
        for i, nid in enumerate(group):
            x = start_x + i * (NODE_W + NODE_GAP) + NODE_W / 2
            y = -layer * LAYER_GAP  # negative y = lower
            positions[nid] = (x, y)

    return positions


# ═══════════════════════════════════════════════════════════════════════════════
# Drawing Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _draw_node_rect(
    ax, x: float, y: float, w: float, h: float,
    status: str, health: str,
    label_top: str, label_bot: str,
    font_top: float = 8, font_bot: float = 6.5,
    border_color: str = BORDER_DEFAULT, border_width: float = 1.0,
    fonts: dict | None = None,
):
    """Draw a single DAG node as a rounded rectangle with status color."""
    fp_cn = fonts["cn"] if fonts else fm.FontProperties()
    fp_mono = fonts["mono"] if fonts else fm.FontProperties()

    fill = STATUS_COLORS.get(status, PALETTE["neutral_mid"])
    hatch = HEALTH_HATCH.get(health, "")

    # Background rect (lighter tint)
    bg_rect = FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle="round,pad=0.04",
        facecolor=STATUS_LIGHT.get(status, PALETTE["neutral_light"]),
        edgecolor=border_color, linewidth=border_width,
        zorder=2,
    )
    ax.add_patch(bg_rect)

    # Status color bar on left edge
    bar_w = 0.08
    bar_rect = FancyBboxPatch(
        (x - w / 2 + 0.02, y - h / 2 + 0.04), bar_w, h - 0.08,
        boxstyle="round,pad=0.02",
        facecolor=fill, edgecolor="none",
        zorder=3,
    )
    ax.add_patch(bar_rect)

    # Health hatch overlay (subtle)
    if hatch:
        hatch_rect = FancyBboxPatch(
            (x - w / 2, y - h / 2), w, h,
            boxstyle="round,pad=0.04",
            facecolor="none", edgecolor=fill,
            linewidth=0, hatch=hatch,
            alpha=0.15, zorder=3,
        )
        ax.add_patch(hatch_rect)

    # Top label (node ID, monospace)
    ax.text(
        x - w / 2 + 0.18, y + h * 0.16,
        label_top,
        fontsize=font_top, fontproperties=fp_mono,
        color=TEXT_PRIMARY, va="center", ha="left",
        fontweight="bold", zorder=4,
    )

    # Bottom label (short question, Chinese font)
    bot_text = textwrap.shorten(label_bot, width=22, placeholder="…")
    ax.text(
        x - w / 2 + 0.18, y - h * 0.22,
        bot_text,
        fontsize=font_bot, fontproperties=fp_cn,
        color=TEXT_SECONDARY, va="center", ha="left",
        zorder=4,
    )


def _draw_edge(
    ax,
    x1: float, y1: float, x2: float, y2: float,
    edge_type: str = "decomposition",
    color: str = None,
):
    """Draw a curved edge between two nodes."""
    style = EDGE_STYLE.get(edge_type, EDGE_STYLE["decomposition"])

    # Control points for Bezier curve
    dy = abs(y2 - y1)
    ctrl_offset = max(dy * 0.35, 0.15)

    arrow = FancyArrowPatch(
        (x1, y1 - NODE_H / 2),
        (x2, y2 + NODE_H / 2),
        connectionstyle=f"arc3,rad=0.0",
        arrowstyle="->,head_width=4,head_length=3",
        color=color,
        linewidth=style["lw"],
        linestyle=style["ls"],
        zorder=1,
        shrinkA=2, shrinkB=2,
    )
    ax.add_patch(arrow)


def _draw_round_label(ax, round_num: int, y_pos: float, fontsize: float = 10):
    """Draw round label above a panel."""
    fp_cn = fm.FontProperties(family="sans-serif", weight="bold")
    ax.text(
        0, y_pos, f"Round {round_num}",
        fontsize=fontsize, fontproperties=fp_cn,
        color=TEXT_PRIMARY, va="bottom", ha="center",
        fontweight="bold",
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 1: Small Multiples DAG
# ═══════════════════════════════════════════════════════════════════════════════

def generate_small_multiples(
    snapshots: list[dict],
    labels: dict[str, str],
    changes: dict,
    output_dir: Path,
    formats: list[str],
    fonts: dict,
) -> list[Path]:
    """Generate side-by-side compact DAG snapshots with 2-row thesis layout."""
    fp_cn = fonts["cn"]
    fp_mono = fonts["mono"]

    n_rounds = len(snapshots)
    n_cols = min(n_rounds, 4)
    n_rows = (n_rounds + n_cols - 1) // n_cols

    panel_w = 2.8
    panel_h = 3.0
    fig_w = panel_w * n_cols + 0.6
    fig_h = panel_h * n_rows + 1.0

    fig = plt.figure(figsize=(fig_w, fig_h), facecolor="white")

    # Compute global layout using the final snapshot
    final_snap = snapshots[-1]
    global_positions = compute_dag_layout(
        final_snap.get("nodes", {}),
        final_snap.get("edges", []),
    )

    if global_positions:
        xs = [p[0] for p in global_positions.values()]
        ys = [p[1] for p in global_positions.values()]
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        x_range = max(x_max - x_min, 0.01)
        y_range = max(y_max - y_min, 0.01)
    else:
        x_range = y_range = 1.0
        x_min = y_min = 0

    gs = fig.add_gridspec(
        n_rows, n_cols,
        left=0.03, right=0.97,
        top=0.90, bottom=0.12,
        wspace=0.12, hspace=0.35,
    )

    # Track identical rounds for structural-freeze annotation
    prev_node_ids: set[str] = set()
    prev_statuses: dict[str, str] = {}
    freeze_start: int | None = None

    for ridx, snap in enumerate(snapshots):
        row = ridx // n_cols
        col = ridx % n_cols
        ax = fig.add_subplot(gs[row, col])

        round_num = snap.get("round_number", ridx + 1)
        nodes = snap.get("nodes", {})
        edges = snap.get("edges", [])
        curr_node_ids = set(nodes.keys())

        # Detect structural freeze: same nodes + same statuses as previous round
        curr_statuses = {nid: n.get("status", "") for nid, n in nodes.items()}
        is_frozen = (
            ridx > 0
            and curr_node_ids == prev_node_ids
            and curr_statuses == prev_statuses
        )

        # Round label
        label_text = f"R{round_num}"
        if is_frozen and freeze_start is None:
            freeze_start = ridx
        ax.set_title(
            label_text,
            fontsize=12, fontweight="bold",
            color=TEXT_PRIMARY if not is_frozen else TEXT_MUTED, pad=5,
            fontproperties=fp_cn,
        )

        # Subtitle
        n_nodes = len(nodes)
        n_solved = sum(1 for n in nodes.values() if n.get("status") == "solved")
        ax.text(
            0.5, 1.03, f"{n_nodes}节点 {n_solved}solved",
            transform=ax.transAxes, fontsize=7, color=TEXT_MUTED,
            ha="center", va="bottom", fontproperties=fp_cn,
        )

        # Positions
        positions = {}
        for nid in nodes:
            if nid in global_positions:
                positions[nid] = global_positions[nid]
            else:
                local_pos = compute_dag_layout(nodes, edges)
                global_positions.update(local_pos)
                positions.update(local_pos)

        def to_panel(pos):
            px = (pos[0] - x_min) / x_range * 0.72 + 0.14
            py = (pos[1] - y_min) / y_range * 0.58 + 0.22
            return px, py

        panel_pos = {nid: to_panel(p) for nid, p in positions.items() if nid in nodes}

        # ── Draw edges ──
        for edge in edges:
            f, t = edge.get("from", ""), edge.get("to", "")
            if f in panel_pos and t in panel_pos:
                fx, fy = panel_pos[f]
                tx, ty = panel_pos[t]
                estyle = EDGE_STYLE.get(edge.get("type", "decomposition"),
                                        EDGE_STYLE["decomposition"])
                ax.annotate(
                    "", xy=(tx, ty + 0.028), xytext=(fx, fy - 0.028),
                    arrowprops=dict(
                        arrowstyle="-|>,head_width=3,head_length=2.5",
                        color=EDGE_COLOR, lw=estyle["lw"], ls=estyle["ls"],
                        shrinkA=3, shrinkB=3,
                    ), zorder=1,
                )

        # ── Draw nodes ──
        for nid in nodes:
            if nid not in panel_pos:
                continue
            node = nodes[nid]
            px, py = panel_pos[nid]
            status = node.get("status", "unsolved")
            health = node.get("health", "healthy")

            is_changed = (
                (nid, ridx) in changes["status_changes"]
                or (nid, ridx) in changes["health_changes"]
                or (nid, ridx) in changes["question_changes"]
            )
            is_new = ridx == changes["first_appear"].get(nid, 0) and ridx > 0

            fill = STATUS_COLORS.get(status, PALETTE["neutral_mid"])
            border_c = BORDER_CHANGE if is_changed else (PALETTE["neutral_light"] if is_frozen else BORDER_DEFAULT)
            border_w = 2.5 if is_changed else 0.8

            nw, nh = 0.19, 0.058
            rect = FancyBboxPatch(
                (px - nw / 2, py - nh / 2), nw, nh,
                boxstyle="round,pad=0.004",
                facecolor=fill, edgecolor=border_c,
                linewidth=border_w, zorder=5,
            )
            ax.add_patch(rect)

            # Hatch for health
            hatch = HEALTH_HATCH.get(health, "")
            if hatch:
                ax.add_patch(FancyBboxPatch(
                    (px - nw / 2, py - nh / 2), nw, nh,
                    boxstyle="round,pad=0.004",
                    facecolor="none", edgecolor="none",
                    hatch=hatch, linewidth=0, alpha=0.2, zorder=6,
                ))

            # Node ID
            ax.text(
                px, py, nid,
                fontsize=7, fontweight="bold", fontproperties=fp_mono,
                color="white", va="center", ha="center", zorder=7,
            )
            # Short label below
            short_label = labels.get(nid, "")
            if short_label:
                lbl = short_label.replace("\n", " ")[:14]
                ax.text(
                    px, py - nh / 2 - 0.012, lbl,
                    fontsize=5.5, fontproperties=fp_cn,
                    color=TEXT_SECONDARY, va="top", ha="center", zorder=7,
                )

            # New indicator
            if is_new:
                ax.text(
                    px + nw / 2 + 0.008, py + nh / 2, "N",
                    fontsize=5.5, color=PALETTE["gold"], fontweight="bold",
                    fontfamily="DejaVu Sans",
                    va="top", ha="left", zorder=7,
                )

        # Freeze marker
        if is_frozen:
            ax.text(
                0.98, 0.02, "(冻结)",
                transform=ax.transAxes, fontsize=6.5,
                color=TEXT_MUTED, ha="right", va="bottom",
                fontproperties=fp_cn, style="italic",
            )

        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_aspect("equal"); ax.axis("off")
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color(GRID_COLOR if not is_frozen else PALETTE["neutral_light"])
            spine.set_linewidth(0.5)

        prev_node_ids = curr_node_ids
        prev_statuses = curr_statuses

    # Hide unused subplots
    for idx in range(n_rounds, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        ax = fig.add_subplot(gs[row, col])
        ax.axis("off")

    # ── Shared legend ──
    _add_sm_legend(fig, fonts)

    # ── Title ──
    fig.suptitle(
        "DAG 结构逐轮演化",
        fontsize=14, fontweight="bold", fontproperties=fp_cn, y=0.96,
    )

    output_files = []
    for fmt in formats:
        fname = f"dag_small_multiples.{fmt}"
        fpath = output_dir / fname
        fig.savefig(fpath, dpi=600, bbox_inches="tight",
                    facecolor="white", edgecolor="none")
        output_files.append(fpath)

    plt.close(fig)
    return output_files


def _add_sm_legend(fig, fonts):
    """Add compact legend below small multiples figure."""
    fp_cn = fonts["cn"]
    items = [
        mpatches.Patch(facecolor=STATUS_COLORS["solved"], edgecolor=BORDER_DEFAULT,
                       label="solved (已解决)"),
        mpatches.Patch(facecolor=STATUS_COLORS["unsolved"], edgecolor=BORDER_DEFAULT,
                       label="unsolved (待解)"),
        mpatches.Patch(facecolor=STATUS_COLORS["blocked"], edgecolor=BORDER_DEFAULT,
                       label="blocked (阻塞)"),
        Line2D([0], [0], color=EDGE_COLOR, lw=1.2, ls="-",
               label="decomposition 边"),
        Line2D([0], [0], color=EDGE_COLOR, lw=1.0, ls="--",
               label="dependency 边"),
        mpatches.Patch(facecolor="none", edgecolor=BORDER_CHANGE,
                       linewidth=2.5, label="状态变化 (粗边框)"),
    ]
    fig.legend(
        handles=items, loc="lower center", ncol=3, fontsize=7.5,
        frameon=True, fancybox=True, edgecolor=GRID_COLOR,
        bbox_to_anchor=(0.5, 0.03), prop=fp_cn,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 2: Swimlane State Matrix
# ═══════════════════════════════════════════════════════════════════════════════

def generate_swimlane_matrix(
    snapshots: list[dict],
    labels: dict[str, str],
    changes: dict,
    output_dir: Path,
    formats: list[str],
    fonts: dict,
) -> list[Path]:
    """Generate node × round state evolution matrix."""

    n_rounds = len(snapshots)

    # Determine node order: by first appearance round, then by ID
    node_order: list[str] = sorted(
        changes["first_appear"].keys(),
        key=lambda nid: (changes["first_appear"][nid], nid),
    )
    n_nodes = len(node_order)

    # Figure sizing
    cell_w = 1.65      # inches per cell
    cell_h = 0.85      # inches per cell
    label_col_w = 2.8  # inches for left label column
    margin = 1.0

    fig_w = label_col_w + cell_w * n_rounds + margin
    fig_h = cell_h * n_nodes + 2.0  # extra for header + legend

    fig, ax = plt.subplots(figsize=(fig_w, fig_h), facecolor="white")

    # ── Draw column headers (round labels) ──
    for c in range(n_rounds):
        round_num = snapshots[c].get("round_number", c + 1)
        x = label_col_w + c * cell_w + cell_w / 2
        y = cell_h * n_nodes + 0.35
        ax.text(
            x, y, f"R{round_num}",
            fontsize=11, fontweight="bold",
            fontproperties=fonts["mono"],
            color=TEXT_PRIMARY, va="bottom", ha="center",
        )

    # ── Draw rows ──
    for ridx_row, nid in enumerate(node_order):
        row_y = (n_nodes - 1 - ridx_row) * cell_h  # bottom-up

        # Row label (left column)
        short_label = labels.get(nid, nid)
        label_text = textwrap.shorten(short_label, width=20, placeholder="…")

        # Node ID
        ax.text(
            label_col_w - 0.15, row_y + cell_h * 0.62,
            nid,
            fontsize=10, fontweight="bold",
            fontproperties=fonts["mono"],
            color=TEXT_PRIMARY, va="center", ha="right",
        )
        # Short description
        ax.text(
            label_col_w - 0.15, row_y + cell_h * 0.30,
            label_text,
            fontsize=8, fontproperties=fonts["cn"],
            color=TEXT_SECONDARY, va="center", ha="right",
        )

        # Draw horizontal separator line
        if ridx_row < n_nodes - 1:
            ax.axhline(
                y=row_y - 0.02,
                xmin=0.02, xmax=0.98,
                color=GRID_COLOR, linewidth=0.5,
            )

        # ── Draw cells for each round ──
        for c in range(n_rounds):
            snap = snapshots[c]
            nodes = snap.get("nodes", {})
            cell_x = label_col_w + c * cell_w

            if nid not in nodes:
                # Node doesn't exist in this round — draw empty cell
                rect = FancyBboxPatch(
                    (cell_x + 0.04, row_y + 0.04),
                    cell_w - 0.08, cell_h - 0.08,
                    boxstyle="round,pad=0.03",
                    facecolor=MISSING_FILL,
                    edgecolor=PALETTE["neutral_light"],
                    linewidth=0.5,
                    linestyle=":",
                    zorder=2,
                )
                ax.add_patch(rect)

                # Dash indicator
                ax.text(
                    cell_x + cell_w / 2, row_y + cell_h / 2,
                    "—",
                    fontsize=12, color=TEXT_MUTED,
                    va="center", ha="center", zorder=3,
                )
                continue

            node = nodes[nid]
            status = node.get("status", "unsolved")
            health = node.get("health", "healthy")

            # Detect changes
            has_status_change = (nid, c) in changes["status_changes"]
            has_health_change = (nid, c) in changes["health_changes"]
            has_question_change = (nid, c) in changes["question_changes"]
            is_first = c == changes["first_appear"].get(nid, 0)
            is_changed = has_status_change or has_health_change or has_question_change

            # Cell rectangle
            fill = STATUS_COLORS.get(status, PALETTE["neutral_mid"])
            border_c = BORDER_CHANGE if is_changed else BORDER_DEFAULT
            border_w = 2.5 if is_changed else 0.8

            rect = FancyBboxPatch(
                (cell_x + 0.04, row_y + 0.04),
                cell_w - 0.08, cell_h - 0.08,
                boxstyle="round,pad=0.03",
                facecolor=fill,
                edgecolor=border_c,
                linewidth=border_w,
                zorder=2,
            )
            ax.add_patch(rect)

            # Health hatch overlay
            hatch = HEALTH_HATCH.get(health, "")
            if hatch:
                hatch_rect = FancyBboxPatch(
                    (cell_x + 0.04, row_y + 0.04),
                    cell_w - 0.08, cell_h - 0.08,
                    boxstyle="round,pad=0.03",
                    facecolor="none",
                    edgecolor=(1, 1, 1, 0.25),
                    hatch=hatch, linewidth=0,
                    zorder=3,
                )
                ax.add_patch(hatch_rect)

            # Status text
            status_cn = {
                "solved": "已解决",
                "unsolved": "待解",
                "blocked": "阻塞",
            }.get(status, status)

            ax.text(
                cell_x + cell_w / 2, row_y + cell_h * 0.55,
                status_cn,
                fontsize=8.5, fontweight="bold",
                fontproperties=fonts["cn"],
                color="white", va="center", ha="center",
                zorder=4,
            )

            # Health text
            health_cn = {
                "healthy": "健康",
                "needs_verification": "待验证",
                "unreliable": "不可靠",
                "blocked": "阻塞",
            }.get(health, health)

            ax.text(
                cell_x + cell_w / 2, row_y + cell_h * 0.26,
                health_cn,
                fontsize=6.5, fontproperties=fonts["cn"],
                color=(1, 1, 1, 0.85), va="center", ha="center",
                zorder=4,
            )

            # Event annotation (top-right corner of cell)
            annotation = ""
            if is_first and c > 0:
                annotation = "★ 新增"
            elif has_question_change:
                annotation = "> 重标"
            elif has_status_change and not is_first:
                old_status = changes["status_changes"].get((nid, c), "")
                old_cn = {"solved": "已解", "unsolved": "待解", "blocked": "阻塞"}.get(old_status, old_status[:3])
                annotation = f"← {old_cn}"

            if annotation:
                ax.text(
                    cell_x + cell_w - 0.08, row_y + cell_h - 0.06,
                    annotation,
                    fontsize=5.5, fontproperties=fonts["cn"],
                    color=PALETTE["gold"], va="top", ha="right",
                    fontweight="bold", zorder=5,
                )

    # ── Column separator lines ──
    for c in range(n_rounds + 1):
        x = label_col_w + c * cell_w
        ax.axvline(
            x=x, ymin=0.02, ymax=0.95,
            color=GRID_COLOR, linewidth=0.3, alpha=0.5,
        )

    # ── Axis setup ──
    ax.set_xlim(0, fig_w - 0.3)
    ax.set_ylim(-0.3, cell_h * n_nodes + 0.8)
    ax.set_aspect("equal")
    ax.axis("off")

    # ── Legend ──
    _add_swimlane_legend(fig, ax, fonts, n_nodes, cell_h)

    # Save
    output_files = []
    for fmt in formats:
        fname = f"dag_swimlane_matrix.{fmt}"
        fpath = output_dir / fname
        fig.savefig(fpath, dpi=600, bbox_inches="tight",
                    facecolor="white", edgecolor="none")
        output_files.append(fpath)

    plt.close(fig)
    return output_files


def _add_swimlane_legend(fig, ax, fonts, n_nodes, cell_h):
    """Add compact legend below swimlane matrix."""
    fp_cn = fonts["cn"]
    items = [
        mpatches.Patch(facecolor=STATUS_COLORS["solved"], edgecolor=BORDER_DEFAULT,
                       label="solved"),
        mpatches.Patch(facecolor=STATUS_COLORS["unsolved"], edgecolor=BORDER_DEFAULT,
                       label="unsolved"),
        mpatches.Patch(facecolor=STATUS_COLORS["blocked"], edgecolor=BORDER_DEFAULT,
                       label="blocked"),
        mpatches.Patch(facecolor=PALETTE["neutral_light"], edgecolor=BORDER_DEFAULT, hatch="///",
                       label="needs_ver."),
        mpatches.Patch(facecolor="none", edgecolor=BORDER_CHANGE, linewidth=2.5,
                       label="状态变化"),
    ]
    fig.legend(
        handles=items, loc="lower center", ncol=3, fontsize=7.5,
        frameon=True, fancybox=True, edgecolor=GRID_COLOR,
        bbox_to_anchor=(0.52, 0.02), prop=fp_cn,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 3: Search × Chunk Co-occurrence Heatmap
# ═══════════════════════════════════════════════════════════════════════════════

def _load_search_trace(input_path: Path) -> tuple[list[dict], dict[str, str], dict[int, set[str]]]:
    """Extract search_trace, chunk_id→title map, and supporting chunk sets from JSON.

    Returns:
        searches: [{round, query_short, chunk_ids, node_label}]
        chunk_titles: {chunk_id: short_title}
        supporting_map: {search_idx: set(supporting_chunk_ids)}
    """
    with open(input_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    # ── Build chunk_id → title map from top-level chunks ──
    chunk_titles: dict[str, str] = {}
    for c in raw.get("chunks", []):
        cid = c.get("chunk_id", "")
        title = c.get("context_title", c.get("chunk_title", ""))
        # Strip common prefixes and truncate
        title = title.replace("[PDF] ", "").replace("中华人民共和国国家标准", "国标")
        if len(title) > 24:
            title = title[:22] + "…"
        if cid:
            chunk_titles[cid] = title or cid[:10]

    # ── Collect all unique chunk IDs from search_trace + build title fallbacks ──
    search_trace = raw.get("search_trace", [])
    all_cids: set[str] = set()
    for entry in search_trace:
        for cid in entry.get("chunk_ids", []):
            all_cids.add(cid)
            if cid not in chunk_titles:
                chunk_titles[cid] = cid[:10]

    # ── Build supporting chunk map: which chunks were marked as supporting ──
    supporting_map: dict[int, set[str]] = {}
    dag_snapshots = raw.get("dag_snapshots", raw.get("round_dags", []))
    for snap in dag_snapshots:
        for nid, node in snap.get("nodes", {}).items():
            sup = set(node.get("supporting_chunks", []))
            if not sup:
                continue
            rnd = snap.get("round_number", 0) - 1  # 0-indexed
            # Map supporting chunks to the search that retrieved them
            for s_idx, entry in enumerate(search_trace):
                if entry.get("round") == snap.get("round_number"):
                    entry_cids = set(entry.get("chunk_ids", []))
                    overlap = sup & entry_cids
                    if overlap:
                        if s_idx not in supporting_map:
                            supporting_map[s_idx] = set()
                        supporting_map[s_idx] |= overlap

    # ── Build search entries with node labels ──
    # Map round → searches to infer which DAG node each search targets
    searches: list[dict] = []
    for s_idx, entry in enumerate(search_trace):
        rnd = entry.get("round", 0)
        query = entry.get("query", "")
        # Shorten query to key topic
        query_short = _shorten_query(query)
        # Infer node label from round_summaries or query content
        node_label = _infer_node_label(query, rnd)
        searches.append({
            "round": rnd,
            "query_short": query_short,
            "chunk_ids": entry.get("chunk_ids", []),
            "node_label": node_label,
            "chunks_returned": entry.get("chunks_returned", 0),
        })

    return searches, chunk_titles, supporting_map


def _shorten_query(query: str) -> str:
    """Shorten a Chinese query to ≤28 chars for row label."""
    # Remove trailing punctuation
    q = query.rstrip("？?。.")
    if "？" in q:
        q = q.split("？")[0]
    if len(q) <= 28:
        return q
    return q[:26] + "…"


def _infer_node_label(query: str, round_num: int) -> str:
    """Infer which DAG node a search query targets."""
    if "熔炼分析" in query and ("成品分析" in query or "区别" in query or "定义" in query):
        return "N1_2"
    if "GB/T 700" in query and ("特殊" in query or "0.22" in query or "需方" in query):
        return "N2_1"
    if "GB/T 222" in query or "偏差" in query or "表1" in query:
        return "N1_1"
    if "成品分析" in query and "允许偏差" in query:
        return "N1_1"
    return f"R{round_num}"


def generate_search_heatmap(
    input_path: Path,
    output_dir: Path,
    formats: list[str],
    fonts: dict,
) -> list[Path]:
    """Generate search × chunk co-occurrence heatmap."""
    fp_cn = fonts["cn"]
    fp_mono = fonts["mono"]

    searches, chunk_titles, supporting_map = _load_search_trace(input_path)
    n_searches = len(searches)
    if n_searches == 0:
        print("  (no search_trace data found)")
        return []

    # ── Sort chunk IDs by first appearance round ──
    first_appear: dict[str, int] = {}
    for s_idx, s in enumerate(searches):
        for cid in s["chunk_ids"]:
            if cid not in first_appear:
                first_appear[cid] = s_idx
    sorted_cids = sorted(first_appear.keys(), key=lambda c: first_appear[c])
    n_chunks = len(sorted_cids)

    # ── Build presence matrix ──
    matrix = [[0] * n_chunks for _ in range(n_searches)]
    support_matrix = [[0] * n_chunks for _ in range(n_searches)]
    for s_idx, s in enumerate(searches):
        sup = supporting_map.get(s_idx, set())
        for c_idx, cid in enumerate(sorted_cids):
            if cid in s["chunk_ids"]:
                matrix[s_idx][c_idx] = 1
            if cid in sup:
                support_matrix[s_idx][c_idx] = 1

    # ── Figure layout ──
    cell_size = 0.28      # inches per cell
    row_label_w = 4.8     # left label column
    col_label_h = 1.6     # top label row (angled text)
    legend_h = 0.4

    fig_w = row_label_w + cell_size * n_chunks + 1.0
    fig_h = col_label_h + cell_size * n_searches + legend_h + 0.8

    fig, ax = plt.subplots(figsize=(fig_w, fig_h), facecolor="white")

    # ── Draw cells ──
    for s_idx in range(n_searches):
        for c_idx in range(n_chunks):
            x = row_label_w + c_idx * cell_size
            y = legend_h + (n_searches - 1 - s_idx) * cell_size  # bottom-up

            present = matrix[s_idx][c_idx]
            is_support = support_matrix[s_idx][c_idx]

            if present:
                if is_support:
                    # Supporting chunk: green fill + star marker
                    face = PALETTE["green_3"]
                    edge = PALETTE["green_2"]
                    lw = 1.2
                else:
                    # Regular hit: blue fill
                    face = PALETTE["blue_main"]
                    edge = PALETTE["blue_secondary"]
                    lw = 0.5
            else:
                # Missing: light neutral
                face = "#F5F5F5"
                edge = PALETTE["neutral_light"]
                lw = 0.3

            rect = FancyBboxPatch(
                (x + 0.02, y + 0.02), cell_size - 0.04, cell_size - 0.04,
                boxstyle="round,pad=0.015",
                facecolor=face, edgecolor=edge, linewidth=lw, zorder=3,
            )
            ax.add_patch(rect)

            # Star marker for supporting chunks
            if is_support:
                ax.text(
                    x + cell_size / 2, y + cell_size / 2, "S",
                    fontsize=6, color="white", fontweight="bold",
                    fontfamily="DejaVu Sans",
                    va="center", ha="center", zorder=4,
                )

    # ── Row labels (search events) ──
    prev_round = 0
    for s_idx, s in enumerate(searches):
        y_center = legend_h + (n_searches - 1 - s_idx) * cell_size + cell_size / 2

        # Round separator
        if s["round"] != prev_round and s_idx > 0:
            sep_y = legend_h + (n_searches - 1 - s_idx) * cell_size + cell_size + 0.04
            ax.axhline(y=sep_y, xmin=0.01, xmax=0.99,
                       color=PALETTE["neutral_light"], linewidth=1.5, zorder=1)
        prev_round = s["round"]

        # Round + Node label
        row_text = f"R{s['round']} {s['node_label']}"
        ax.text(
            row_label_w - 0.06, y_center + 0.02,
            row_text,
            fontsize=6.5, fontweight="bold", fontproperties=fp_mono,
            color=PALETTE["neutral_black"], va="bottom", ha="right",
        )
        # Query topic
        ax.text(
            row_label_w - 0.06, y_center - 0.07,
            s["query_short"],
            fontsize=5.8, fontproperties=fp_cn,
            color=PALETTE["neutral_dark"], va="top", ha="right",
        )

    # ── Column labels (chunk IDs + document titles) ──
    for c_idx, cid in enumerate(sorted_cids):
        x_center = row_label_w + c_idx * cell_size + cell_size / 2
        title = chunk_titles.get(cid, cid[:8])

        # Chunk ID (monospace)
        ax.text(
            x_center, legend_h + n_searches * cell_size + 0.12,
            cid[:10],
            fontsize=5, fontproperties=fp_mono,
            color=PALETTE["neutral_mid"], va="bottom", ha="center",
            rotation=45,
        )
        # Document title
        ax.text(
            x_center, legend_h + n_searches * cell_size + 0.30,
            title,
            fontsize=5.2, fontproperties=fp_cn,
            color=PALETTE["neutral_dark"], va="bottom", ha="center",
            rotation=45,
        )

    # ── Stagnation bracket for N1_1 repeated searches ──
    # Find consecutive rows with same node_label
    stagnation_groups = []
    i = 0
    while i < n_searches:
        node = searches[i]["node_label"]
        j = i + 1
        while j < n_searches and searches[j]["node_label"] == node:
            j += 1
        if j - i >= 3 and node == "N1_1":  # 3+ consecutive same-node searches
            stagnation_groups.append((i, j - 1))
        i = j

    for start_idx, end_idx in stagnation_groups:
        y_top = legend_h + (n_searches - 1 - start_idx) * cell_size + cell_size + 0.02
        y_bot = legend_h + (n_searches - 1 - end_idx) * cell_size - 0.02
        # Right-side bracket
        bracket_x = row_label_w + n_chunks * cell_size + 0.15
        ax.plot([bracket_x, bracket_x], [y_bot, y_top],
                color=PALETTE["red_strong"], linewidth=1.5, zorder=2)
        ax.plot([bracket_x, bracket_x + 0.12], [y_top, y_top],
                color=PALETTE["red_strong"], linewidth=1.5, zorder=2)
        ax.plot([bracket_x, bracket_x + 0.12], [y_bot, y_bot],
                color=PALETTE["red_strong"], linewidth=1.5, zorder=2)
        ax.text(
            bracket_x + 0.20, (y_top + y_bot) / 2,
            "搜索停滞带\n(KB 无覆盖)",
            fontsize=6.5, fontproperties=fp_cn,
            color=PALETTE["red_strong"], va="center", ha="left",
            fontweight="bold",
        )

    # ── Legend ──
    legend_items = [
        mpatches.Patch(facecolor=PALETTE["blue_main"], edgecolor=PALETTE["blue_secondary"],
                       label="chunk 命中", linewidth=0.5),
        mpatches.Patch(facecolor=PALETTE["green_3"], edgecolor=PALETTE["green_2"],
                       label="supporting chunk [S]", linewidth=1.2),
        mpatches.Patch(facecolor="#F5F5F5", edgecolor=PALETTE["neutral_light"],
                       label="未命中", linewidth=0.3),
    ]
    fig.legend(
        handles=legend_items, loc="lower center", ncol=3, fontsize=7,
        frameon=False, bbox_to_anchor=(0.5, 0.005), prop=fp_cn,
    )

    # ── Axis setup ──
    ax.set_xlim(0, row_label_w + n_chunks * cell_size + 1.2)
    ax.set_ylim(0, legend_h + n_searches * cell_size + col_label_h)
    ax.set_aspect("equal")
    ax.axis("off")

    # ── Save ──
    output_files = []
    for fmt in formats:
        fname = f"dag_search_heatmap.{fmt}"
        fpath = output_dir / fname
        fig.savefig(fpath, dpi=600, bbox_inches="tight",
                    facecolor="white", edgecolor="none")
        output_files.append(fpath)

    plt.close(fig)
    return output_files

def main():
    parser = argparse.ArgumentParser(
        description="Generate publication-quality DAG evolution figures.",
    )
    parser.add_argument(
        "--input", "-i", required=True, type=Path,
        help="Path to eval result JSON (containing dag_snapshots).",
    )
    parser.add_argument(
        "--output-dir", "-o", default="figures/", type=Path,
        help="Output directory for generated figures.",
    )
    parser.add_argument(
        "--format", "-f", nargs="+", default=["svg", "pdf", "png"],
        choices=["pdf", "png", "svg"],
        help="Output format(s). Nature standard: SVG (primary) + PDF + PNG. Default: svg pdf png.",
    )
    parser.add_argument(
        "--figure-type", "-t", default="both",
        choices=["both", "small-multiples", "swimlane", "search-heatmap"],
        help="Which figure(s) to generate. Default: both (small-multiples + swimlane).",
    )

    args = parser.parse_args()

    # Setup
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fonts = _setup_fonts()

    # Load data
    print(f"Loading data from {args.input} ...")
    snapshots = load_data(args.input)
    n_rounds = len(snapshots)

    if n_rounds == 0:
        print("Error: No DAG snapshots found in the input file.")
        return

    print(f"Found {n_rounds} rounds, "
          f"{len(snapshots[-1].get('nodes', {}))} nodes in final DAG.")

    # Extract labels and detect changes
    labels = extract_short_labels(snapshots)
    changes = detect_changes(snapshots)

    print(f"Nodes: {', '.join(labels.keys())}")
    for nid, lbl in labels.items():
        print(f"  {nid}: {lbl}")
    n_status_changes = len(changes["status_changes"])
    n_health_changes = len(changes["health_changes"])
    print(f"Changes detected: {n_status_changes} status, {n_health_changes} health")

    # Generate figures
    all_outputs: list[Path] = []

    if args.figure_type in ("both", "small-multiples"):
        print("\nGenerating Small Multiples DAG ...")
        files = generate_small_multiples(
            snapshots, labels, changes,
            args.output_dir, args.format, fonts,
        )
        all_outputs.extend(files)
        for f in files:
            print(f"  → {f}")

    if args.figure_type in ("both", "swimlane"):
        print("\nGenerating Swimlane State Matrix ...")
        files = generate_swimlane_matrix(
            snapshots, labels, changes,
            args.output_dir, args.format, fonts,
        )
        all_outputs.extend(files)
        for f in files:
            print(f"  → {f}")

    if args.figure_type in ("search-heatmap",):
        print("\nGenerating Search × Chunk Co-occurrence Heatmap ...")
        files = generate_search_heatmap(
            args.input, args.output_dir, args.format, fonts,
        )
        all_outputs.extend(files)
        for f in files:
            print(f"  → {f}")

    print(f"\nDone! {len(all_outputs)} file(s) saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
