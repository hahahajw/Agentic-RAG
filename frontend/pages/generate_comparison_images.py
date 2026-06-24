"""
Generate system comparison chart images from eval data using Plotly + kaleido.
All charts are vector PDF (infinitely scalable) + high-res PNG (5× supersampling).

Exports:
  数据集整体 (Tab 1): radar chart, answer quality, retrieval quality, efficiency scatter, efficiency bars
  问题类型   (Tab 2): EM heatmap, F1 heatmap (per dataset)

Data-driven cache: when eval data changes, re-running this script regenerates all images.

Usage:
    # Generate all charts (PNG + PDF)
    uv run python frontend/pages/generate_comparison_images.py

    # Specific formats or datasets
    uv run python frontend/pages/generate_comparison_images.py --format png
    uv run python frontend/pages/generate_comparison_images.py --dataset hotpotqa
    uv run python frontend/pages/generate_comparison_images.py --output ./my_figures/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import streamlit as st  # noqa: E402 — needed for data_loader cache context

from frontend.utils.comparison_data import load_comparison_data, DS_KEYS, DS_LABELS  # noqa: E402
from frontend.components.comparison_charts import (  # noqa: E402
    radar_chart_png, radar_chart_pdf,
    type_heatmap_png, type_heatmap_pdf,
    answer_quality_chart_png, answer_quality_chart_pdf,
    retrieval_quality_chart_png, retrieval_quality_chart_pdf,
    efficiency_chart_png, efficiency_chart_pdf,
    efficiency_bars_png, efficiency_bars_pdf,
    retrieval_cdf_chart_png, retrieval_cdf_chart_pdf,
)

DEFAULT_OUTPUT_DIR = Path(__file__).parent.parent.parent / "analyse_v2" / "figures" / "comparison"


# ═══════════════════════════════════════════════════════════════════
# Export helpers
# ═══════════════════════════════════════════════════════════════════

def _save(name: str, png_bytes: bytes, pdf_bytes: bytes,
          out_dir: Path, formats: tuple[str, ...]) -> None:
    """Save PNG and/or PDF bytes to disk."""
    if "png" in formats and png_bytes:
        path = out_dir / f"{name}.png"
        path.write_bytes(png_bytes)
        print(f"  → PNG: {path.name}  ({len(png_bytes):,} bytes)")

    if "pdf" in formats and pdf_bytes:
        path = out_dir / f"{name}.pdf"
        path.write_bytes(pdf_bytes)
        print(f"  → PDF: {path.name}  ({len(pdf_bytes):,} bytes)")


# ═══════════════════════════════════════════════════════════════════
# Main generation
# ═══════════════════════════════════════════════════════════════════

def _save_chart(name: str, png_fn, pdf_fn, *fn_args,
               out_dir: Path, formats: tuple[str, ...]) -> None:
    """Save a chart as PNG/PDF with error recovery for kaleido crashes."""
    import time
    if "png" in formats:
        try:
            png_bytes = png_fn(*fn_args)
            path = out_dir / f"{name}.png"
            path.write_bytes(png_bytes)
            print(f"  -> PNG: {path.name}  ({len(png_bytes):,} bytes)")
        except Exception as e:
            print(f"  !! PNG failed ({name}): {e}")
            time.sleep(1)  # Let kaleido recover

    if "pdf" in formats:
        try:
            pdf_bytes = pdf_fn(*fn_args)
            path = out_dir / f"{name}.pdf"
            path.write_bytes(pdf_bytes)
            print(f"  -> PDF: {path.name}  ({len(pdf_bytes):,} bytes)")
        except Exception as e:
            print(f"  !! PDF failed ({name}): {e}")
            time.sleep(1)


def generate_all(*, formats: tuple[str, ...] = ("png", "pdf"),
                 output_dir: Path | str | None = None) -> None:
    """Generate all comparison chart images."""
    import time

    out_dir = Path(output_dir) if output_dir else DEFAULT_OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading comparison data...")
    data = load_comparison_data()

    if not data or not data.get("flat"):
        print("ERROR: No comparison data available. Run evaluations first.")
        return

    # ── Tab 1: 数据集整体 ──
    print("\n═══ 数据集整体 (Tab 1) ═══")

    print("  Radar chart...")
    _save_chart("radar_chart", radar_chart_png, radar_chart_pdf, data, out_dir=out_dir, formats=formats)
    time.sleep(0.5)

    print("  Answer quality...")
    _save_chart("answer_quality", answer_quality_chart_png, answer_quality_chart_pdf, data, out_dir=out_dir, formats=formats)
    time.sleep(0.5)

    print("  Retrieval quality...")
    _save_chart("retrieval_quality", retrieval_quality_chart_png, retrieval_quality_chart_pdf, data, out_dir=out_dir, formats=formats)
    time.sleep(0.5)

    print("  Efficiency scatter...")
    _save_chart("efficiency_scatter", efficiency_chart_png, efficiency_chart_pdf, data, out_dir=out_dir, formats=formats)
    time.sleep(0.5)

    print("  Efficiency bars...")
    _save_chart("efficiency_bars", efficiency_bars_png, efficiency_bars_pdf, data, out_dir=out_dir, formats=formats)
    time.sleep(0.5)

    print("  Retrieval count CDF...")
    _save_chart("retrieval_cdf", retrieval_cdf_chart_png, retrieval_cdf_chart_pdf, data, out_dir=out_dir, formats=formats)
    time.sleep(0.5)

    # ── Tab 2: 问题类型热力图 (每数据集 × EM/F1) ──
    print("\n═══ 问题类型热力图 (Tab 2) ═══")

    for ds in DS_KEYS:
        ds_label = DS_LABELS.get(ds, ds)
        has_data = any(
            data.get("by_dataset", {}).get(ds, {}).get(a, {}).get("types")
            for a in ["llm-only", "naive-rag", "rag-with-judge", "rag-loop"]
        )
        if not has_data:
            print(f"  SKIP {ds_label}: no type data")
            continue

        for metric, m_label in [("em", "EM"), ("f1", "F1")]:
            name = f"type_heatmap_{m_label}_{ds_label}"
            print(f"  {name}...")
            _save_chart(name, type_heatmap_png, type_heatmap_pdf, data, ds, metric,
                       out_dir=out_dir, formats=formats)
            time.sleep(0.3)

    print(f"\nDone. Files saved to: {out_dir}")


# ═══════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Generate high-res comparison chart PNG/PDF using Plotly+kaleido"
    )
    parser.add_argument(
        "--format", choices=["png", "pdf", "both"], default="both",
        help="Output format (default: both)"
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})"
    )
    args = parser.parse_args()

    formats = ("png", "pdf") if args.format == "both" else (args.format,)
    generate_all(formats=formats, output_dir=args.output)


if __name__ == "__main__":
    main()