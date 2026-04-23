"""对已有 rag_with_judge 结果做答案后处理提取。

将原始长答案迁移到 _prediction 键，
从 "So the answer is: <answer>." 中提取简洁答案写入 prediction，
重新计算 EM/F1 指标并更新 aggregate。

用法：
    uv run python Eval/extract_cot_answers.py
    uv run python Eval/extract_cot_answers.py --mode rag-with-judge --dataset hotpotqa
    uv run python Eval/extract_cot_answers.py --mode agentic-rag-v3 --dataset 2wikimultihopqa
"""

import json
import logging
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from Eval.metrics import exact_match_score, f1_score

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

VALID_MODES = ["rag-with-judge", "agentic-rag-v3"]
VALID_DATASETS = ["hotpotqa", "2wikimultihopqa", "musique"]


def extract_answer(text: str) -> str:
    """从 CoT 输出中提取 'So the answer is: <answer>.' 中的答案。

    prompt 明确指定格式: "So the answer is: <answer>."
    用 re.MULTILINE 使 $ 匹配每行末尾，处理多行 CoT 输出。
    """
    match = re.search(r'So the answer is:\s*(.+?)\.?\s*$', text, re.IGNORECASE | re.MULTILINE)
    if match:
        answer = match.group(1).strip()
        answer = re.sub(r'[*_`"\\]+', '', answer)  # 去除 markdown 和引号
        return answer.strip()
    # fallback: 取最后一句（LLM 未遵循格式时）
    sentences = re.split(r'[.!?]', text)
    return sentences[-1].strip() if sentences else text


def process_result_file(result_path: Path) -> dict:
    """处理单个结果文件，原地修改并返回统计。"""
    with open(result_path, encoding="utf-8") as f:
        data = json.load(f)

    results = data.get("results", data)
    if not results:
        logger.warning("没有可处理的结果: %s", result_path)
        return {"total": 0, "extracted": 0, "skipped": 0}

    total = 0
    extracted = 0
    skipped = 0
    em_before, em_after = [], []
    f1_before, f1_after = [], []

    for item in results:
        pred = item.get("prediction")
        if not pred:
            continue

        total += 1
        ground_truth = item.get("answer", "")

        # 计算提取前的指标
        em_b = exact_match_score(pred, ground_truth)
        f1_b, _, _ = f1_score(pred, ground_truth)
        em_before.append(em_b)
        f1_before.append(f1_b)

        # # 已处理过的跳过
        # if "_prediction" in item:
        #     skipped += 1
        #     em_after.append(em_b)
        #     f1_after.append(f1_b)
        #     continue

        # 提取简洁答案
        short = extract_answer(pred)
        item["_prediction"] = pred
        item["prediction"] = short

        # 计算提取后的指标
        em_a = exact_match_score(short, ground_truth)
        f1_a, _, _ = f1_score(short, ground_truth)
        em_after.append(em_a)
        f1_after.append(f1_a)

        if em_a > em_b:
            extracted += 1
            logger.info(
                "  [%d] EM 提升: '%s' → '%s' (GT='%s')",
                item["question_index"], pred[:60], short, ground_truth,
            )

    # 更新 summary
    if "summary" in data:
        n = len(em_after)
        if n > 0:
            data["summary"]["aggregate"]["em"] = round(
                sum(em_after) / n, 6
            )
            data["summary"]["aggregate"]["f1"] = round(
                sum(f1_after) / n, 6
            )
            data["summary"]["aggregate"]["precision"] = round(
                sum(f1_after) / n, 6
            )  # 简化：precision 同步更新

    # 写回
    tmp = result_path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    tmp.replace(result_path)

    return {
        "total": total,
        "extracted": extracted,
        "skipped": skipped,
        "em_before": round(sum(em_before) / len(em_before), 6) if em_before else 0,
        "em_after": round(sum(em_after) / len(em_after), 6) if em_after else 0,
        "f1_before": round(sum(f1_before) / len(f1_before), 6) if f1_before else 0,
        "f1_after": round(sum(f1_after) / len(f1_after), 6) if f1_after else 0,
    }


def main():
    import argparse

    parser = argparse.ArgumentParser(description="从 rag_with_judge / agentic_rag_v3 结果中提取简洁答案")
    parser.add_argument("--mode", choices=VALID_MODES, default="rag-with-judge",
                        help="评估模式")
    parser.add_argument("--dataset", choices=VALID_DATASETS,
                        help="数据集名称（不提供则处理全部）")
    parser.add_argument("--base-dir", default="Eval",
                        help="评估文件根目录")
    args = parser.parse_args()

    datasets = [args.dataset] if args.dataset else VALID_DATASETS
    mode_dir = args.mode.replace("-", "_")

    for ds in datasets:
        result_path = Path(args.base_dir) / f"{mode_dir}_data" / "result" / f"{ds}.json"
        if not result_path.exists():
            logger.warning("跳过（文件不存在）: %s", result_path)
            continue

        logger.info("处理: %s", result_path)
        stats = process_result_file(result_path)

        logger.info(
            "完成: %s/%s — 总计=%d, EM 提升=%d, EM: %.4f → %.4f, F1: %.4f → %.4f",
            args.mode, ds,
            stats["total"], stats["extracted"],
            stats["em_before"], stats["em_after"],
            stats["f1_before"], stats["f1_after"],
        )

    logger.info("全部完成。")


if __name__ == "__main__":
    main()
