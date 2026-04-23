"""对比 Schema A 与 Schema B 融合策略的检索质量与因果分析。

读取两个 schema 的结果文件（含召回 chunks 和答案），结合 benchmark 中的
supporting_facts / paragraph_support_idx，计算：
  - Context Recall@K
  - Hit@K（Binary）
  - Precision@K
  - MRR（Mean Reciprocal Rank）
  - Retrieval→Answer 因果分析

输出到 Eval/naive_rag_data/result/{dataset}_comparison.json

用法：
    # 单个数据集
    uv run python Eval/compare_schemes.py --dataset hotpotqa

    # 所有数据集
    uv run python Eval/compare_schemes.py --all

    # 限制前 N 题（快速验证）
    uv run python Eval/compare_schemes.py --dataset hotpotqa --limit 10
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

# 确保项目根目录在 sys.path 上
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from Eval.metrics import (
    compute_context_recall, compute_hit, compute_mrr, compute_retrieval_precision,
    extract_supporting_titles,
    exact_match_score,
    f1_score,
)

# Windows 控制台编码修复
if sys.platform == "win32":
    os.system("")
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

logger = logging.getLogger(__name__)

# ─── 常量 ────────────────────────────────────────────────────────

VALID_DATASETS = ["hotpotqa", "2wikimultihopqa", "musique"]

DATASET_BENCHMARK_FILES = {
    "hotpotqa": "Data/benchmark/HotpotQA_500_benchmark.json",
    "2wikimultihopqa": "Data/benchmark/2WikiMultihopQA_500_benchmark.json",
    "musique": "Data/benchmark/MuSiQue_500_benchmark.json",
}

RESULT_DIR = Path("Eval/naive_rag_data/result")


# ─── 因果分析分类 ─────────────────────────────────────────────────

def classify_question(
    chunks_a: List[dict],
    chunks_b: List[dict],
    supporting_titles: Set[str],
) -> str:
    """将问题分类为 A-only / B-only / Both / Neither。

    分类依据：是否召回了任意一个支撑段落的 title。
    """
    if not supporting_titles:
        return "no_ground_truth"

    titles_a = {c.get("context_title", "") for c in chunks_a}
    titles_b = {c.get("context_title", "") for c in chunks_b}

    a_hit = bool(titles_a & supporting_titles)
    b_hit = bool(titles_b & supporting_titles)

    if a_hit and not b_hit:
        return "A_only"
    elif b_hit and not a_hit:
        return "B_only"
    elif a_hit and b_hit:
        return "Both"
    else:
        return "Neither"


def analyze_neither_subcategory(
    pred_a: Optional[str],
    pred_b: Optional[str],
    answer: str,
    supporting_titles: Set[str],
) -> str:
    """Neither 进一步拆解：区分检索失败 vs 生成失败。

    子类别:
        - both_correct: LLM 自身知识足够，两者都答对
        - a_correct_b_wrong: A 答对 B 答错
        - b_correct_a_wrong: B 答对 A 答错
        - both_wrong: 两者都答错，纯检索或知识盲区
    """
    em_a = exact_match_score(pred_a, answer) if pred_a else 0.0
    em_b = exact_match_score(pred_b, answer) if pred_b else 0.0

    if em_a > 0 and em_b > 0:
        return "both_correct"
    elif em_a > 0:
        return "a_correct_b_wrong"
    elif em_b > 0:
        return "b_correct_a_wrong"
    else:
        return "both_wrong"


# ─── 完整数据集分析 ──────────────────────────────────────────────

def load_result_file(path: Path) -> List[dict]:
    """加载结果文件，兼容新旧格式。

    新格式: {"summary": {...}, "results": [...]}
    旧格式: [...]
    """
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        return data.get("results", [])
    elif isinstance(data, list):
        return data
    else:
        raise ValueError(f"未知结果文件格式: {path}")


def analyze_dataset(
    dataset_type: str,
    benchmark_path: str,
    schema_a_path: Path,
    schema_b_path: Path,
    limit: Optional[int] = None,
) -> dict:
    """分析单个数据集的 Schema A vs Schema B 对比。

    Returns:
        包含 schema_a_metrics, schema_b_metrics, causal_analysis, cases 的字典。
    """
    # 加载数据
    with open(benchmark_path, encoding="utf-8") as f:
        benchmark = json.load(f)

    results_a = load_result_file(schema_a_path)
    results_b = load_result_file(schema_b_path)

    # 按 question_index 对齐
    map_a = {r["question_index"]: r for r in results_a}
    map_b = {r["question_index"]: r for r in results_b}

    # 确定要分析的问题索引
    all_indices = sorted(set(map_a.keys()) & set(map_b.keys()))
    if limit is not None:
        all_indices = all_indices[:limit]

    logger.info("加载 %d 个对齐的问题（限制: %s）", len(all_indices), limit)

    # 逐问题分析
    metrics_a: List[dict] = []
    metrics_b: List[dict] = []
    categories: Dict[str, List[dict]] = {
        "A_only": [],
        "B_only": [],
        "Both": [],
        "Neither": [],
        "no_ground_truth": [],
    }

    for idx in all_indices:
        qa = map_a[idx]
        qb = map_b[idx]
        raw = benchmark[idx]

        supporting_titles = extract_supporting_titles(raw, dataset_type)
        chunks_a = qa.get("chunks", [])
        chunks_b = qb.get("chunks", [])
        answer = raw.get("answer", qa.get("answer", ""))
        pred_a = qa.get("prediction")
        pred_b = qb.get("prediction")

        # 检索指标（优先从 result 文件读取，不存在时重新计算）
        if "context_recall" in qa:
            m_a = {
                "context_recall": qa["context_recall"],
                "hit": qa["hit"],
                "precision": qa.get("retrieval_precision"),
                "mrr": qa["mrr"],
            }
        else:
            titles_a = [c.get("context_title", "") for c in chunks_a]
            m_a = {
                "context_recall": compute_context_recall(titles_a, supporting_titles),
                "hit": compute_hit(titles_a, supporting_titles),
                "precision": compute_retrieval_precision(titles_a, supporting_titles),
                "mrr": compute_mrr(titles_a, supporting_titles),
            }

        if "context_recall" in qb:
            m_b = {
                "context_recall": qb["context_recall"],
                "hit": qb["hit"],
                "precision": qb.get("retrieval_precision"),
                "mrr": qb["mrr"],
            }
        else:
            titles_b = [c.get("context_title", "") for c in chunks_b]
            m_b = {
                "context_recall": compute_context_recall(titles_b, supporting_titles),
                "hit": compute_hit(titles_b, supporting_titles),
                "precision": compute_retrieval_precision(titles_b, supporting_titles),
                "mrr": compute_mrr(titles_b, supporting_titles),
            }

        # 生成指标（EM / F1）
        em_a = exact_match_score(pred_a, answer) if pred_a else 0.0
        em_b = exact_match_score(pred_b, answer) if pred_b else 0.0
        f1_a = f1_score(pred_a, answer)[0] if pred_a else 0.0
        f1_b = f1_score(pred_b, answer)[0] if pred_b else 0.0

        m_a["em"] = em_a
        m_a["f1"] = f1_a
        m_b["em"] = em_b
        m_b["f1"] = f1_b

        metrics_a.append(m_a)
        metrics_b.append(m_b)

        # 因果分析分类
        cat = classify_question(chunks_a, chunks_b, supporting_titles)
        case_entry = {
            "question_index": idx,
            "question": raw.get("question", qa.get("question", "")),
            "answer": answer,
            "pred_a": pred_a,
            "pred_b": pred_b,
            "em_a": em_a,
            "em_b": em_b,
            "supporting_titles": sorted(supporting_titles),
        }
        if cat == "Neither":
            case_entry["subcategory"] = analyze_neither_subcategory(
                pred_a, pred_b, answer, supporting_titles
            )
        else:
            case_entry["subcategory"] = None

        # 记录 A-only / B-only 的典型案例（用于输出报告）
        if cat in ("A_only", "B_only", "Neither"):
            case_entry["retrieved_titles_a"] = sorted(
                {c.get("context_title", "") for c in chunks_a}
            )
            case_entry["retrieved_titles_b"] = sorted(
                {c.get("context_title", "") for c in chunks_b}
            )

        categories[cat].append(case_entry)

    # 汇总指标
    def avg_metric(items: List[dict], key: str) -> float:
        vals = [x[key] for x in items if x.get(key) is not None]
        return round(sum(vals) / len(vals), 6) if vals else 0.0

    retrieval_keys = ["context_recall", "hit", "precision", "mrr"]
    generation_keys = ["em", "f1"]

    summary_a = {k: avg_metric(metrics_a, k) for k in retrieval_keys + generation_keys}
    summary_b = {k: avg_metric(metrics_b, k) for k in retrieval_keys + generation_keys}

    # 因果分析汇总 — 固定顺序，空类别也显示
    causal_summary = {}
    for cat in ("A_only", "B_only", "Both", "Neither"):
        cases = categories.get(cat, [])
        if cases:
            ems_a = [c["em_a"] for c in cases]
            ems_b = [c["em_b"] for c in cases]
            info = {
                "count": len(cases),
                "em_mean_a": round(sum(ems_a) / len(ems_a), 6),
                "em_mean_b": round(sum(ems_b) / len(ems_b), 6),
            }
            if cat == "Neither":
                subcats = {}
                for c in cases:
                    sub = c.get("subcategory", "unknown")
                    subcats[sub] = subcats.get(sub, 0) + 1
                info["subcategories"] = subcats
            causal_summary[cat] = info
        else:
            causal_summary[cat] = {
                "count": 0,
                "em_mean_a": None,
                "em_mean_b": None,
            }

    # 典型案例：各取前 20 个
    max_cases = 20
    return {
        "dataset_type": dataset_type,
        "n_questions": len(all_indices),
        "schema_a_metrics": summary_a,
        "schema_b_metrics": summary_b,
        "causal_summary": causal_summary,
        "cases": {
            cat: categories.get(cat, [])[:max_cases]
            for cat in ("A_only", "B_only", "Both", "Neither")
        },
    }


# ─── 报告打印 ─────────────────────────────────────────────────────

def print_report(result: dict):
    """打印人类可读的对比报告。"""
    ds = result["dataset_type"]
    n = result["n_questions"]
    print(f"\n{'='*70}")
    print(f"  {ds}  —  Schema A vs Schema B 对比（{n} 题）")
    print(f"{'='*70}")

    a = result["schema_a_metrics"]
    b = result["schema_b_metrics"]

    print(f"\n{'指标':<20} {'Schema A':>12} {'Schema B':>12} {'差值':>10}")
    print(f"{'-'*56}")
    labels = {
        "context_recall": "Context Recall@K",
        "hit": "Hit@K",
        "precision": "Precision@K",
        "mrr": "MRR",
        "em": "EM",
        "f1": "F1",
    }
    for key, label in labels.items():
        va = a.get(key, 0)
        vb = b.get(key, 0)
        diff = vb - va
        sign = "+" if diff >= 0 else ""
        print(f"{label:<20} {va:>12.4f} {vb:>12.4f} {sign}{diff:>9.4f}")

    # 因果分析 — 固定顺序显示全部四类
    causal = result.get("causal_summary", {})
    print(f"\n{'因果分类':<20} {'数量':>6} {'EM(A)':>8} {'EM(B)':>8}")
    print(f"{'-'*44}")
    for cat in ("A_only", "B_only", "Both", "Neither"):
        info = causal.get(cat)
        if info is None:
            continue
        cnt = info.get("count", 0)
        em_a = info.get("em_mean_a")
        em_b = info.get("em_mean_b")
        em_a_str = f"{em_a:.4f}" if em_a is not None else " — "
        em_b_str = f"{em_b:.4f}" if em_b is not None else " — "
        print(f"{cat:<20} {cnt:>6} {em_a_str:>8} {em_b_str:>8}")
        if cat == "Neither" and "subcategories" in info and info["subcategories"]:
            for sub, cnt_sub in info["subcategories"].items():
                print(f"    {sub:<18} {cnt_sub:>6}")

    # A-only / B-only 典型案例
    cases = result.get("cases", {})
    for cat in ("A_only", "B_only"):
        cat_cases = cases.get(cat, [])
        print(f"\n--- {cat} 典型案例")
        if not cat_cases:
            print("  （无）")
            continue
        for c in cat_cases[:5]:
            print(f"  Q[{c['question_index']}]: {c['question'][:80]}")
            print(f"    Answer: {c['answer']}")
            print(f"    Pred(A): {c['pred_a']}  |  EM={c['em_a']:.0f}")
            print(f"    Pred(B): {c['pred_b']}  |  EM={c['em_b']:.0f}")
            if c['retrieved_titles_a']:
                print(f"    A retrieved: {c['retrieved_titles_a'][:5]}")
            if c['retrieved_titles_b']:
                print(f"    B retrieved: {c['retrieved_titles_b'][:5]}")
            print()

    print(f"\n结果已保存至: {RESULT_DIR / f'{ds}_comparison.json'}")


# ─── CLI ─────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="对比 Schema A 与 Schema B 的检索融合策略"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="分析所有数据集",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=VALID_DATASETS,
        help="分析单个数据集",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="限制分析前 N 个问题（用于快速验证）",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    if args.all:
        datasets = VALID_DATASETS
    elif args.dataset:
        datasets = [args.dataset]
    else:
        parser.print_help()
        print("\n请指定 --all 或 --dataset 参数")
        return

    for ds in datasets:
        benchmark_path = DATASET_BENCHMARK_FILES[ds]
        schema_a_path = RESULT_DIR / f"{ds}_schema_a.json"
        schema_b_path = RESULT_DIR / f"{ds}_schema_b.json"

        if not schema_a_path.exists():
            logger.error("Schema A 结果文件不存在: %s", schema_a_path)
            continue
        if not schema_b_path.exists():
            logger.error("Schema B 结果文件不存在: %s", schema_b_path)
            continue

        logger.info("分析数据集: %s", ds)
        result = analyze_dataset(
            dataset_type=ds,
            benchmark_path=benchmark_path,
            schema_a_path=schema_a_path,
            schema_b_path=schema_b_path,
            limit=args.limit,
        )

        # 保存
        output_path = RESULT_DIR / f"{ds}_comparison.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        print_report(result)


if __name__ == "__main__":
    main()
