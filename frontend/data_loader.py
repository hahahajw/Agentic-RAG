"""Data loading module for Agentic RAG frontend.

支持两种评估结果格式:
  1. 单文件格式: Eval/{mode}_data/result/{dataset}.json (含 summary + results 数组)
  2. 逐题格式: Eval/{mode}_data/result/{dataset}/ (含 _summary.json + 逐题 JSON)
"""

import json
from pathlib import Path
from typing import Optional

import streamlit as st

EVAL_DIR = Path(__file__).parent.parent / "Eval"

# Mode aliases: frontend mode name → actual directory name (without _data suffix)
MODE_ALIASES: dict[str, str] = {
    "agentic-rag": "agentic_rag_v3",  # 已弃用，仅向后兼容旧 eval 数据
    "llm-only": "llm_only",
    "rag-loop": "rag_loop",
}


def _resolve_mode_dir(mode: str) -> str:
    """Resolve a frontend mode name to the actual directory name."""
    return MODE_ALIASES.get(mode, mode.replace("-", "_"))


_REVERSE_ALIASES: dict[str, str] = {v: k for k, v in MODE_ALIASES.items()}


def _discover_result_files() -> list[tuple[str, str, Optional[str]]]:
    """Scan Eval/*_data/result/ and return all available (mode, dataset, schema) combos.

    支持两种格式:
      - 单文件: result/{dataset}.json
      - 逐题目录: result/{dataset}/_summary.json
    """
    results = []
    if not EVAL_DIR.exists():
        return results

    for data_dir in sorted(EVAL_DIR.glob("*_data")):
        raw_mode_underscore = data_dir.stem.replace("_data", "")
        raw_mode_dash = raw_mode_underscore.replace("_", "-")
        mode = _REVERSE_ALIASES.get(raw_mode_underscore, raw_mode_dash)
        result_dir = data_dir / "result"
        if not result_dir.exists():
            continue

        # ── 格式 1: 单 JSON 文件 ──
        for fp in sorted(result_dir.glob("*.json")):
            name = fp.stem
            if name.endswith("_metrics") or name.endswith("_comparison"):
                continue
            if name.startswith("_"):  # skip _summary.json etc
                continue
            try:
                with open(fp, encoding="utf-8") as test_f:
                    peek = json.load(test_f)
                if not isinstance(peek, dict) or "results" not in peek:
                    continue
            except (json.JSONDecodeError, IOError):
                continue
            if "_schema_" in name:
                parts = name.split("_schema_")
                if len(parts) == 2:
                    results.append((mode, parts[0], parts[1]))
                else:
                    results.append((mode, name, None))
            else:
                results.append((mode, name, None))

        # ── 格式 2: 逐题目录 ──
        for sub_dir in sorted(result_dir.glob("*/")):
            if not sub_dir.is_dir():
                continue
            dataset_name = sub_dir.name
            summary_file = sub_dir / "_summary.json"
            if not summary_file.exists():
                continue
            # 确保有至少一个逐题 JSON
            q_files = sorted(sub_dir.glob("[0-9]*.json"))
            if not q_files:
                continue
            results.append((mode, dataset_name, None))

    return sorted(set(results))


@st.cache_data(ttl=60)
def list_available_results() -> list[tuple[str, str, Optional[str]]]:
    """Return all available (mode, dataset, schema) combinations."""
    return _discover_result_files()


@st.cache_data(ttl=60)
def list_available_datasets(mode: str) -> list[str]:
    """Return available datasets for a given mode."""
    combos = list_available_results()
    datasets = set()
    for m, ds, _ in combos:
        if m == mode:
            datasets.add(ds)
    return sorted(datasets)


@st.cache_data(ttl=60)
def load_results(mode: str, dataset: str, schema: Optional[str] = None) -> Optional[dict]:
    """Load result data for a given mode/dataset/schema. Returns None if not found.

    自动检测单文件格式或逐题目录格式。
    """
    dir_name = _resolve_mode_dir(mode)
    data_dir = EVAL_DIR / f"{dir_name}_data" / "result"

    # ── 先尝试逐题目录格式 ──
    per_q_dir = data_dir / dataset
    if per_q_dir.is_dir():
        summary_file = per_q_dir / "_summary.json"
        if summary_file.exists():
            return _load_per_question_results(per_q_dir)

    # ── 再尝试单文件格式 ──
    if schema:
        filename = f"{dataset}_schema_{schema}.json"
    else:
        filename = f"{dataset}.json"

    path = data_dir / filename
    if not path.exists():
        return None

    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _load_per_question_results(directory: Path) -> Optional[dict]:
    """从逐题目录加载结果并组装为统一格式。

    Returns:
        {"summary": {...}, "results": [...]}  — 与单文件格式兼容
    """
    summary_file = directory / "_summary.json"
    with open(summary_file, encoding="utf-8") as f:
        summary_data = json.load(f)

    # 加载所有逐题 JSON（跳过 _summary.json 和 _metrics.json）
    results = []
    for qf in sorted(directory.glob("[0-9]*.json")):
        try:
            with open(qf, encoding="utf-8") as f:
                results.append(json.load(f))
        except (json.JSONDecodeError, IOError):
            continue

    return {
        "summary": summary_data,
        "results": results,
    }


@st.cache_data(ttl=60)
def load_checkpoint(mode: str, dataset: str, schema: Optional[str] = None) -> Optional[dict]:
    """Load checkpoint file for a given mode/dataset/schema."""
    dir_name = _resolve_mode_dir(mode)
    data_dir = f"{dir_name}_data"

    if schema:
        filename = f"{dataset}_schema_{schema}.json"
    else:
        filename = f"{dataset}.json"

    path = EVAL_DIR / data_dir / "checkpoint" / filename
    if not path.exists():
        return None

    with open(path, encoding="utf-8") as f:
        return json.load(f)