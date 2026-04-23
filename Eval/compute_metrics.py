"""从评估结果计算 EM/F1 + 检索质量指标（@K 曲线）。

读取 run_eval.py 产出的 result JSON，结合原始 benchmark 数据中的 supporting titles，
计算每个问题和整体的：
- EM / F1（生成质量）
- Context Recall@K, Hit@K, MRR@K, Retrieval Precision@K（检索质量，K ∈ [1,3,5,8,10]）

用法：
    uv run python Eval/compute_metrics.py --mode naive-rag --dataset hotpotqa
    uv run python Eval/compute_metrics.py --mode naive-rag --dataset hotpotqa --schema a
    uv run python Eval/compute_metrics.py --mode naive-rag --dataset hotpotqa --ks 1 3 5 8 10 20
"""

import argparse
import json
import logging
import sys
from pathlib import Path


# 确保项目根目录在 sys.path 上
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

from Eval.metrics import (
    exact_match_score,
    f1_score,
    compute_context_recall,
    compute_hit,
    compute_mrr,
    compute_retrieval_precision,
    extract_supporting_titles,
)

# 默认 @K 值
DEFAULT_KS = [1, 3, 5, 8, 10]


# 数据集名 → 实际文件名的大小写映射
_DATASET_FILE_MAP = {
    "hotpotqa": "HotpotQA_500_benchmark.json",
    "2wikimultihopqa": "2WikiMultihopQA_500_benchmark.json",
    "musique": "MuSiQue_500_benchmark.json",
}


def load_benchmark_raw(dataset_type: str) -> list[dict]:
    """加载原始 benchmark JSON（不经过 NormalizedQuestion 包装）。"""
    source_dir = Path("Data/benchmark")

    # 优先 chunked 版本
    base = _DATASET_FILE_MAP.get(dataset_type, f"{dataset_type}_500_benchmark.json")
    chunked = base.replace(".json", "_chunked.json")

    for name in [chunked, base]:
        p = source_dir / name
        if p.exists():
            with open(p, encoding="utf-8") as f:
                return json.load(f)

    logger.error("找不到 %s 的 benchmark 文件", dataset_type)
    sys.exit(1)


def compute_metrics(result_path: Path, benchmark: list[dict], ks: list[int]) -> dict:
    """读取结果并计算 EM/F1 + 检索指标 @K。

    Returns:
        包含整体、逐 K 聚合、逐问题得分的 metrics 字典
    """
    with open(result_path, encoding="utf-8") as f:
        data = json.load(f)

    results = data.get("results", data) if isinstance(data, dict) else data

    if not results:
        logger.warning("没有可评估的结果。")
        return {"total": 0, "answered": 0, "unanswered": 0,
                "aggregate": {}, "per_k": {}, "per_question": []}

    total = len(results)
    answered = 0

    # 逐问题收集
    em_list, f1_list, prec_list, rec_list = [], [], [], []
    per_question = []

    for item in results:
        prediction = item.get("prediction")
        ground_truth = item.get("answer", "")
        idx = item["question_index"]

        if prediction is None or prediction == "":
            per_question.append({
                "question_index": idx,
                "em": None, "f1": None,
                "precision": None, "recall": None,
                "retrieval": {},
                "status": "unanswered",
            })
            continue

        answered += 1
        cur_em = exact_match_score(prediction, ground_truth)
        cur_f1, cur_prec, cur_rec = f1_score(prediction, ground_truth)

        em_list.append(cur_em)
        f1_list.append(cur_f1)
        prec_list.append(cur_prec)
        rec_list.append(cur_rec)

        # 检索指标 @K
        chunks = item.get("chunks", [])
        retrieved_titles = [c["context_title"] for c in chunks]

        if not retrieved_titles:
            # LLM-only 或未检索到内容：所有 @K = 0
            retrieval_at_k = {
                str(k): {"context_recall": 0.0, "hit": 0, "mrr": 0.0, "retrieval_precision": 0.0}
                for k in ks
            }
        else:
            supporting = extract_supporting_titles(benchmark[idx], item.get("dataset_type", "hotpotqa"))
            retrieval_at_k = {}
            for k in ks:
                retrieval_at_k[str(k)] = {
                    "context_recall": compute_context_recall(retrieved_titles, supporting, top_k=k),
                    "hit": compute_hit(retrieved_titles, supporting, top_k=k),
                    "mrr": compute_mrr(retrieved_titles, supporting, top_k=k),
                    "retrieval_precision": compute_retrieval_precision(retrieved_titles, supporting, top_k=k),
                }

        per_question.append({
            "question_index": idx,
            "em": cur_em,
            "f1": round(cur_f1, 6),
            "precision": round(cur_prec, 6),
            "recall": round(cur_rec, 6),
            "retrieval": retrieval_at_k,
            "status": "answered",
        })

    unanswered = total - answered

    # 整体 EM/F1
    aggregate = {
        "em": round(sum(em_list) / len(em_list), 6) if em_list else 0.0,
        "f1": round(sum(f1_list) / len(f1_list), 6) if f1_list else 0.0,
        "precision": round(sum(prec_list) / len(prec_list), 6) if prec_list else 0.0,
        "recall": round(sum(rec_list) / len(rec_list), 6) if rec_list else 0.0,
    }

    # 按 K 聚合检索指标
    per_k = {}
    for k in ks:
        k_str = str(k)
        vals = [pq["retrieval"][k_str] for pq in per_question if pq["status"] == "answered"]
        if not vals:
            continue
        per_k[k_str] = {
            "context_recall": round(sum(v["context_recall"] for v in vals) / len(vals), 6),
            "hit": round(sum(v["hit"] for v in vals) / len(vals), 6),
            "mrr": round(sum(v["mrr"] for v in vals) / len(vals), 6),
            "retrieval_precision": round(sum(v["retrieval_precision"] for v in vals) / len(vals), 6),
        }

    return {
        "total": total,
        "answered": answered,
        "unanswered": unanswered,
        "aggregate": aggregate,
        "per_k": per_k,
        "per_question": per_question,
    }


def print_summary(metrics: dict, mode: str, dataset: str):
    """打印可读摘要。"""
    print(f"\n{'='*70}")
    print(f"模式:     {mode}  |  数据集: {dataset}")
    print(f"总数:     {metrics['total']}  |  已回答: {metrics['answered']}  |  未回答: {metrics['unanswered']}")
    print()

    agg = metrics["aggregate"]
    print(f"{'生成质量':=^50}")
    print(f"  EM:         {agg['em']:.4f}")
    print(f"  F1:         {agg['f1']:.4f}")
    print(f"  Precision:  {agg['precision']:.4f}")
    print(f"  Recall:     {agg['recall']:.4f}")
    print()

    if metrics["per_k"]:
        print(f"{'检索质量 @K':=^50}")
        header = f"  {'K':>4}  {'CtxRecall':>10}  {'Hit@K':>8}  {'MRR@K':>8}  {'RetPrec':>10}"
        print(header)
        print(f"  {'-'*4}  {'-'*10}  {'-'*8}  {'-'*8}  {'-'*10}")
        for k_str in sorted(metrics["per_k"].keys(), key=int):
            v = metrics["per_k"][k_str]
            print(f"  {k_str:>4}  {v['context_recall']:>10.4f}  {v['hit']:>8.4f}  {v['mrr']:>8.4f}  {v['retrieval_precision']:>10.4f}")
    print(f"\n{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(description="从评估结果计算 EM/F1 + 检索质量指标（@K 曲线）")
    parser.add_argument("--mode", required=True, choices=["llm-only", "naive-rag", "rag-with-judge"],
                        help="评估模式")
    parser.add_argument("--dataset", required=True,
                        choices=["hotpotqa", "2wikimultihopqa", "musique"],
                        help="数据集名称")
    parser.add_argument("--schema", choices=["a", "b"],
                        help="Naive RAG 的 schema（仅 naive-rag 模式有效）")
    parser.add_argument("--ks", type=int, nargs="+", default=None,
                        help="检索指标 @K 值（默认: 1 3 5 8 10）")
    parser.add_argument("--base-dir", default="Eval",
                        help="评估文件根目录（默认: Eval）")
    args = parser.parse_args()

    ks = args.ks or DEFAULT_KS

    # 确定结果文件名
    mode_dir = args.mode.replace("-", "_")
    data_dir = f"{mode_dir}_data"

    if args.mode == "naive-rag" and args.schema:
        result_name = f"{args.dataset}_schema_{args.schema}.json"
    else:
        result_name = f"{args.dataset}.json"

    result_path = Path(args.base_dir) / data_dir / "result" / result_name

    if not result_path.exists():
        logger.error("结果文件不存在: %s", result_path)
        sys.exit(1)

    # 加载 benchmark 原始数据（用于 extract_supporting_titles）
    benchmark = load_benchmark_raw(args.dataset)

    logger.info("正在计算指标: %s/%s（来源: %s, K=%s）", args.mode, args.dataset, result_path, ks)
    metrics = compute_metrics(result_path, benchmark, ks)

    # 打印摘要
    print_summary(metrics, args.mode, args.dataset)

    # 写入指标文件
    metrics_path = Path(args.base_dir) / data_dir / "result" / f"{Path(result_name).stem}_metrics.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    logger.info("指标已写入: %s", metrics_path)


if __name__ == "__main__":
    main()
