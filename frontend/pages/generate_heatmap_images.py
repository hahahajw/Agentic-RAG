"""
Generate heatmap images directly from eval data using Playwright rendering.
Uses heatmap_utils.py for data loading and HTML generation (shared with frontend),
Playwright for high-res PNG capture at 5× scale (480 DPI), and Pillow for PDF.

Usage:
    # Generate PNG+PDF for all 3 datasets
    uv run python frontend/pages/generate_heatmap_images.py

    # Single dataset, specific format
    uv run python frontend/pages/generate_heatmap_images.py --dataset hotpotqa
    uv run python frontend/pages/generate_heatmap_images.py --format png
    uv run python frontend/pages/generate_heatmap_images.py --format pdf
    uv run python frontend/pages/generate_heatmap_images.py --output ./my_figures/

    # As a Python module
    from frontend.pages.generate_heatmap_images import generate_all, generate_one
    generate_one("hotpotqa", formats=("png", "pdf"))
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

# Ensure project root on path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from frontend.utils.heatmap_utils import (
    load_question_data,
    sort_questions,
    build_heatmap_html,
    group_by_type,
    DS_KEYS,
    DS_LABELS,
    COLS,
    CELL,
    GAP,
)

# ── Defaults ──
DEFAULT_OUTPUT_DIR = Path(__file__).parent.parent.parent / "analyse_v2" / "figures"
SCALE = 8  # 8× → 768 DPI PDF (8 × 96 DPI browser resolution)


# ═══════════════════════════════════════════════════════════════════
# Layout calculation
# ═══════════════════════════════════════════════════════════════════

def _calc_content_width() -> int:
    """Calculate exact content width in pixels."""
    max_grid_w = COLS * CELL + (COLS - 1) * GAP  # 845px
    label_w = 152  # 140 label + 12 padding
    padding_lr = 32 * 2
    return label_w + max_grid_w + padding_lr


def _calc_content_height(sorted_qs: list[dict]) -> int:
    """Calculate exact content height in pixels (matches HTML layout)."""
    type_groups = group_by_type(sorted_qs)
    h = 0
    h += 16 + 16  # Title: font-size + margin-bottom
    for i, (_, tqs) in enumerate(type_groups):
        h += 24  # Type header
        tn = len(tqs)
        trows = math.ceil(tn / COLS)
        for _ in range(4):  # 4 system rows
            h += trows * CELL + (trows - 1) * GAP + 5  # +SYS_GAP
        if i < len(type_groups) - 1:
            h += 5  # Divider: margin 4 + border 1
    h += 24  # Legend margin-top
    h += 12 + 6  # dist title + margin
    h += 14  # dist bar
    h += 12 + 12 + 6  # f1 title margin + font + margin
    h += 14  # f1 row (blocks)
    h += 10  # type row margin-bottom
    h += 24  # Bottom padding
    return h


# ═══════════════════════════════════════════════════════════════════
# Playwright-based rendering
# ═══════════════════════════════════════════════════════════════════

def _render_html_to_png(html: str, content_w: int, content_h: int,
                        output_path: Path) -> None:
    """Render HTML heatmap to high-res PNG using headless Chromium.

    Uses Playwright's device_scale_factor for native high-DPI rendering
    (no CSS transform tricks that break full_page scroll dimensions).
    """
    from playwright.sync_api import sync_playwright

    # Render at native CSS size with device_scale_factor for pixel density.
    # No CSS transform — avoids scroll-height clipping with tall content.
    full_html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ background: white; }}
    </style>
</head>
<body>
{html}
</body>
</html>"""

    vp_w = content_w + 20
    vp_h = max(900, min(content_h, 4000))  # Viewport covers most content; full_page captures rest

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(
            viewport={"width": vp_w, "height": vp_h},
            device_scale_factor=SCALE,
        )
        page = context.new_page()
        page.set_content(full_html, wait_until="networkidle")
        page.wait_for_timeout(300)
        page.screenshot(path=str(output_path), full_page=True)
        page.close()
        context.close()
        browser.close()

    # Report output dimensions
    from PIL import Image
    img = Image.open(output_path)
    print(f"  -> PNG: {output_path.name}  ({img.width}x{img.height}px, {SCALE}x device scale)")
    img.close()


def _png_to_pdf(png_path: Path, pdf_path: Path) -> None:
    """Convert high-res PNG to single-page PDF at correct physical dimensions.

    Reads actual PNG pixel dimensions and embeds at effective DPI = 96 * SCALE.
    """
    from PIL import Image

    effective_dpi = int(96 * SCALE)

    img = Image.open(png_path)
    img.info["dpi"] = (effective_dpi, effective_dpi)
    img.save(pdf_path, "PDF", resolution=effective_dpi)

    pdf_w_in = img.width / effective_dpi
    pdf_h_in = img.height / effective_dpi
    print(f"  -> PDF: {pdf_path.name}  "
          f"({img.width}x{img.height}px @ {effective_dpi} DPI "
          f"-> {pdf_w_in:.1f}x{pdf_h_in:.1f}in, single page)")
    img.close()


# ═══════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════

def generate_one(ds_key: str, *,
                 formats: tuple[str, ...] = ("png", "pdf"),
                 output_dir: Path | str | None = None) -> dict[str, Path]:
    """Generate heatmap images for a single dataset.

    Args:
        ds_key: "hotpotqa" | "2wikimultihopqa" | "musique"
        formats: which formats to generate ("png", "pdf")
        output_dir: where to save files (default: analyse_v2/figures/)

    Returns:
        {"png": Path, "pdf": Path} — paths to generated files
    """
    if ds_key not in DS_KEYS:
        raise ValueError(f"Unknown dataset: {ds_key}. Choose from {DS_KEYS}")

    out_dir = Path(output_dir) if output_dir else DEFAULT_OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    ds_label = dict(zip(DS_KEYS, DS_LABELS))[ds_key]

    # Load data via shared heatmap_utils functions
    questions = load_question_data(ds_key)
    if not questions:
        raise RuntimeError(f"No eval data found for {ds_label}. Run evaluation first.")

    sorted_qs = sort_questions(questions)
    heatmap_html = build_heatmap_html(sorted_qs, ds_key, ds_label)
    content_w = _calc_content_width()

    result: dict[str, Path] = {}

    if "png" in formats:
        png_path = out_dir / f"heatmap_f1_{ds_label}.png"
        _render_html_to_png(heatmap_html, content_w, _calc_content_height(sorted_qs), png_path)
        result["png"] = png_path

    if "pdf" in formats:
        pdf_path = out_dir / f"heatmap_f1_{ds_label}.pdf"
        png_for_pdf = result.get("png") or out_dir / f"heatmap_f1_{ds_label}.png"
        if not png_for_pdf.exists():
            _render_html_to_png(heatmap_html, content_w, _calc_content_height(sorted_qs), png_for_pdf)
            result["png"] = png_for_pdf
        _png_to_pdf(png_for_pdf, pdf_path)
        result["pdf"] = pdf_path

    return result


def generate_all(*, formats: tuple[str, ...] = ("png", "pdf"),
                 output_dir: Path | str | None = None) -> dict[str, dict[str, Path]]:
    """Generate heatmap images for all 3 datasets.

    Returns:
        {ds_key: {"png": Path, "pdf": Path}}
    """
    results = {}
    for ds_key in DS_KEYS:
        ds_label = dict(zip(DS_KEYS, DS_LABELS))[ds_key]
        print(f"Processing {ds_label}...")
        try:
            results[ds_key] = generate_one(ds_key, formats=formats, output_dir=output_dir)
        except RuntimeError as e:
            print(f"  SKIP {ds_label}: {e}")
    return results


# ═══════════════════════════════════════════════════════════════════
# CLI entry point
# ═══════════════════════════════════════════════════════════════════

_DEFAULT_SCALE = 8  # Default render scale (8x -> 768 DPI)


def main():
    parser = argparse.ArgumentParser(
        description="Generate high-res F1 heatmap PNG/PDF using Playwright"
    )
    parser.add_argument(
        "--dataset", choices=DS_KEYS, default=None,
        help="Single dataset to process (default: all 3)"
    )
    parser.add_argument(
        "--format", choices=["png", "pdf", "both"], default="both",
        help="Output format (default: both)"
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})"
    )
    parser.add_argument(
        "--scale", type=int, default=_DEFAULT_SCALE,
        help=f"Render scale factor — higher = sharper PDF (default: {_DEFAULT_SCALE}× → {96*_DEFAULT_SCALE} DPI)"
    )
    args = parser.parse_args()

    # Allow overriding scale for this run
    global SCALE
    SCALE = args.scale

    formats = ("png", "pdf") if args.format == "both" else (args.format,)

    if args.dataset:
        generate_one(args.dataset, formats=formats, output_dir=args.output)
    else:
        generate_all(formats=formats, output_dir=args.output)

    print("Done.")


if __name__ == "__main__":
    main()