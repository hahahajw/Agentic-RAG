"""Data loading module for Agentic RAG frontend."""

import json
from pathlib import Path
from typing import Optional

import streamlit as st

EVAL_DIR = Path(__file__).parent.parent / "Eval"

# Mode aliases: frontend mode name → actual directory name (without _data suffix)
# e.g. "agentic-rag" → "agentic_rag_v3" means the directory is agentic_rag_v3_data
MODE_ALIASES: dict[str, str] = {
    "agentic-rag": "agentic_rag_v3",
}


def _resolve_mode_dir(mode: str) -> str:
    """Resolve a frontend mode name to the actual directory name."""
    return MODE_ALIASES.get(mode, mode.replace("-", "_"))


# Reverse alias: actual directory stem → frontend mode name
_REVERSE_ALIASES: dict[str, str] = {v: k for k, v in MODE_ALIASES.items()}


def _discover_result_files() -> list[tuple[str, str, Optional[str]]]:
    """Scan Eval/*_data/result/ and return all available (mode, dataset, schema) combos."""
    results = []
    if not EVAL_DIR.exists():
        return results

    for data_dir in sorted(EVAL_DIR.glob("*_data")):
        # Map actual directory stem to frontend mode name
        raw_mode_underscore = data_dir.stem.replace("_data", "")
        raw_mode_dash = raw_mode_underscore.replace("_", "-")
        mode = _REVERSE_ALIASES.get(raw_mode_underscore, raw_mode_dash)
        result_dir = data_dir / "result"
        if not result_dir.exists():
            continue
        for fp in sorted(result_dir.glob("*.json")):
            name = fp.stem
            # Skip metrics files, causal analysis, and other non-standard result files
            if name.endswith("_metrics") or name.endswith("_comparison"):
                continue
            # Quick validation: file must have "summary" + "results" keys
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
    """Load result file for a given mode/dataset/schema. Returns None if not found."""
    dir_name = _resolve_mode_dir(mode)
    data_dir = f"{dir_name}_data"

    if schema:
        filename = f"{dataset}_schema_{schema}.json"
    else:
        filename = f"{dataset}.json"

    path = EVAL_DIR / data_dir / "result" / filename
    if not path.exists():
        return None

    with open(path, encoding="utf-8") as f:
        return json.load(f)


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
