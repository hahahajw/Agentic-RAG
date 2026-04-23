"""支持 checkpoint 的 RAG 评估 CLI 入口。

使用示例：
    # LLM-only 基线
    uv run python Eval/run_eval.py --mode llm-only --dataset hotpotqa

    # Naive RAG（Scheme B）
    uv run python Eval/run_eval.py --mode naive-rag --dataset hotpotqa

    # 仅重试失败的问题
    uv run python Eval/run_eval.py --mode llm-only --dataset hotpotqa --retry-failed

    # 强制清空重跑
    uv run python Eval/run_eval.py --mode llm-only --dataset hotpotqa --force

    # 查看进度摘要（不处理）
    uv run python Eval/run_eval.py --mode llm-only --dataset hotpotqa --summary

    # 自定义 batch 大小和 worker 数量
    uv run python Eval/run_eval.py --mode naive-rag --dataset hotpotqa --batch-size 10 --max-workers 3
"""

import argparse
import json
import logging
import sys
from pathlib import Path

# 确保项目根目录在 sys.path 上，使 Eval 包导入正常工作
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


VALID_MODES = ["llm-only", "naive-rag", "rag-with-judge", "agentic-rag", "agentic-rag-v2", "agentic-rag-v3", "ir-cot", "iter-retgen", "gen-ground"]
VALID_DATASETS = ["hotpotqa", "2wikimultihopqa", "musique"]

# ChatOpenAI 构造函数的有效 kwargs 白名单
VALID_CHAT_OPENAI_PARAMS = {
    "temperature", "max_tokens", "max_completion_tokens", "top_p",
    "frequency_penalty", "presence_penalty", "stop", "seed",
    "extra_body", "extra_headers", "extra_query",
    "timeout", "max_retries", "streaming",
    "api_version", "organization", "model",
}


def _parse_model_params(json_str: str | None) -> dict:
    """解析并校验 --model-params JSON 字符串。"""
    if json_str is None:
        return {}
    import re

    json_str = json_str.strip()
    # PowerShell/Windows 可能将外层单引号作为内容传递，自动剥离
    if len(json_str) >= 2 and json_str[0] == json_str[-1] and json_str[0] in ("'", '"'):
        json_str = json_str[1:-1]
    json_str = json_str.strip()
    # PowerShell 可能吃掉 JSON 内部的双引号（如 "{...}" 嵌套时），
    # 把 "key": 变成 key: — 用正则将裸露的键名包裹双引号
    if '"' not in json_str:
        json_str = re.sub(r'(?<=[{,])\s*(\w+)\s*:', r' "\1":', json_str)
    try:
        params = json.loads(json_str)
    except json.JSONDecodeError as e:
        logger.error("--model-params JSON 格式错误: %s", e)
        sys.exit(1)
    if not isinstance(params, dict):
        logger.error("--model-params 必须是 JSON 对象，而非 %s", type(params).__name__)
        sys.exit(1)
    invalid = set(params.keys()) - VALID_CHAT_OPENAI_PARAMS
    if invalid:
        logger.error("--model-params 包含未知参数: %s", sorted(invalid))
        sys.exit(1)
    return params


def make_llm(model: str = "qwen3.6-plus", model_params: dict | None = None) -> ChatOpenAI:
    """从环境变量创建 LLM 实例。"""
    import os
    defaults = {
        "temperature": 0.0,
        "extra_body": {"enable_thinking": False, "enable_search": False},
    }
    merged = {**defaults, **(model_params or {})}
    return ChatOpenAI(
        api_key=os.getenv("BL_API_KEY"),
        base_url=os.getenv("BL_BASE_URL"),
        model=model,
        **merged,
    )


def show_summary(mode: str, dataset: str, schema: str = "b", base_dir: str = "Eval"):
    """显示进度摘要，不进行处理。"""
    from Eval.checkpoint import EvalCheckpointManager
    from Eval.base import load_benchmark

    benchmark = load_benchmark(dataset)
    mode_dir = mode.replace("-", "_")
    # naive-rag 模式下文件名包含 schema 信息
    if mode == "naive-rag":
        dataset_name = f"{dataset}_schema_{schema}"
    elif mode in ("agentic-rag", "agentic-rag-v2", "agentic-rag-v3"):
        dataset_name = dataset
    else:
        dataset_name = dataset
    mgr = EvalCheckpointManager(mode_dir, dataset_name, base_dir)
    mgr.load()
    print(mgr.get_summary(len(benchmark)))


def run_eval(mode: str, dataset: str, llm: ChatOpenAI,
             batch_size: int, max_workers: int, max_retries: int,
             retry_failed: bool, force: bool,
             model_params: dict, role_params: dict,
             topk: int = 50, max_chunks: int = 8,
             use_reranker: bool = False,
             schema: str = "b",
             max_depth: int = 3,
             max_rounds: int = 5,
             rewrite_model: str = None, suggest_model: str = None,
             judge_model: str = None, answer_model: str = None,
             judge_variant: str = "B",
             planner_model: str = None, evaluator_model: str = None,
             reflector_model: str = None, synthesizer_model: str = None,
             # 自定义 Milvus collection
             custom_collection: str = None,
             custom_dense_field: str = None,
             custom_text_field: str = None,
             custom_sparse_field: str = None,
             cot: bool = False):
    """运行单个 (mode, dataset) 对的评估。"""
    mode_dir = mode.replace("-", "_")

    if mode == "llm-only":
        from Eval.llm_only import LLMOnlyEvaluator
        evaluator = LLMOnlyEvaluator(
            llm=llm,
            dataset_type=dataset,
            batch_size=batch_size,
            max_workers=max_workers,
            max_retries=max_retries,
        )
    elif mode == "naive-rag":
        from Eval.naive_rag import NaiveRAGEvaluator
        evaluator = NaiveRAGEvaluator(
            llm=llm,
            dataset_type=dataset,
            batch_size=batch_size,
            max_workers=max_workers,
            max_retries=max_retries,
            topk=topk,
            max_chunks=max_chunks,
            use_reranker=use_reranker,
            scheme=schema,
            rewrite_model=rewrite_model,
            suggest_model=suggest_model,
            model_params=model_params,
            role_params=role_params,
            custom_collection=custom_collection,
            custom_dense_field=custom_dense_field,
            custom_text_field=custom_text_field,
            custom_sparse_field=custom_sparse_field,
        )
    elif mode == "rag-with-judge":
        from rag_with_judge.evaluator import JudgeRAGEvaluator
        evaluator = JudgeRAGEvaluator(
            llm=llm,
            dataset_type=dataset,
            batch_size=batch_size,
            max_workers=max_workers,
            max_retries=max_retries,
            topk=topk,
            max_chunks=max_chunks,
            max_depth=max_depth,
            use_reranker=use_reranker,
            rewrite_model=rewrite_model,
            judge_model=judge_model,
            answer_model=answer_model,
            judge_variant=judge_variant,
            cot=cot,
            model_params=model_params,
            role_params=role_params,
            custom_collection=custom_collection,
            custom_dense_field=custom_dense_field,
            custom_text_field=custom_text_field,
            custom_sparse_field=custom_sparse_field,
        )
    elif mode == "agentic-rag":
        from agentic_rag.evaluator import AgenticRAGEvaluator
        evaluator = AgenticRAGEvaluator(
            llm=llm,
            dataset_type=dataset,
            batch_size=batch_size,
            max_workers=max_workers,
            max_retries=max_retries,
            topk=topk,
            max_chunks=max_chunks,
            max_rounds=max_rounds,
            use_reranker=use_reranker,
            planner_model=planner_model,
            evaluator_model=evaluator_model,
            answer_model=answer_model,
            reflector_model=reflector_model,
            rewrite_model=rewrite_model,
            model_params=model_params,
            role_params=role_params,
            custom_collection=custom_collection,
            custom_dense_field=custom_dense_field,
            custom_text_field=custom_text_field,
            custom_sparse_field=custom_sparse_field,
        )
    elif mode == "agentic-rag-v2":
        from agentic_rag_v2.evaluator import AgenticRAGV2Evaluator
        evaluator = AgenticRAGV2Evaluator(
            llm=llm,
            dataset_type=dataset,
            batch_size=batch_size,
            max_workers=max_workers,
            max_retries=max_retries,
            topk=topk,
            max_chunks=max_chunks,
            max_rounds=max_rounds,
            use_reranker=use_reranker,
            planner_model=planner_model,
            evaluator_model=evaluator_model,
            answer_model=answer_model,
            reflector_model=reflector_model,
            rewrite_model=rewrite_model,
            synthesizer_model=synthesizer_model,
            model_params=model_params,
            role_params=role_params,
            custom_collection=custom_collection,
            custom_dense_field=custom_dense_field,
            custom_text_field=custom_text_field,
            custom_sparse_field=custom_sparse_field,
        )
    elif mode == "agentic-rag-v3":
        from agentic_rag_v3.evaluator import AgenticRAGV3Evaluator
        evaluator = AgenticRAGV3Evaluator(
            llm=llm,
            dataset_type=dataset,
            batch_size=batch_size,
            max_workers=max_workers,
            max_retries=max_retries,
            topk=topk,
            max_chunks=max_chunks,
            max_rounds=max_rounds,
            use_reranker=use_reranker,
            planner_model=planner_model,
            evaluator_model=evaluator_model,
            answer_model=answer_model,
            reflector_model=reflector_model,
            rewrite_model=rewrite_model,
            synthesizer_model=synthesizer_model,
            cot=cot,
            model_params=model_params,
            role_params=role_params,
            custom_collection=custom_collection,
            custom_dense_field=custom_dense_field,
            custom_text_field=custom_text_field,
            custom_sparse_field=custom_sparse_field,
        )
    elif mode == "ir-cot":
        from paper.ir_cot.evaluator import IRCoTEvaluator
        evaluator = IRCoTEvaluator(
            llm=llm,
            dataset_type=dataset,
            batch_size=batch_size,
            max_workers=max_workers,
            max_retries=max_retries,
            topk=topk,
            max_chunks=max_chunks,
            use_reranker=use_reranker,
        )
    elif mode == "iter-retgen":
        from paper.iter_retgen.evaluator import IterRetGenEvaluator
        evaluator = IterRetGenEvaluator(
            llm=llm,
            dataset_type=dataset,
            batch_size=batch_size,
            max_workers=max_workers,
            max_retries=max_retries,
            topk=topk,
            max_chunks=max_chunks,
            use_reranker=use_reranker,
        )
    elif mode == "gen-ground":
        from paper.GenGround.evaluator import GenGroundEvaluator
        evaluator = GenGroundEvaluator(
            llm=llm,
            dataset_type=dataset,
            batch_size=batch_size,
            max_workers=max_workers,
            max_retries=max_retries,
            topk=topk,
            max_chunks=max_chunks,
            use_reranker=use_reranker,
        )
    else:
        logger.error("未知模式: %s", mode)
        sys.exit(1)

    evaluator.run(retry_failed=retry_failed, force=force)


def main():
    parser = argparse.ArgumentParser(
        description="支持 Checkpoint 的 RAG 评估系统",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--mode", required=True, choices=VALID_MODES,
                        help="评估模式")
    parser.add_argument("--dataset", required=True, choices=VALID_DATASETS,
                        help="要评估的数据集")
    parser.add_argument("--batch-size", type=int, default=20,
                        help="每批处理的问题数（默认: 20）")
    parser.add_argument("--max-workers", type=int, default=5,
                        help="每批并行 worker 数（默认: 5）")
    parser.add_argument("--max-retries", type=int, default=2,
                        help="瞬态错误自动重试次数（默认: 2）")
    parser.add_argument("--model", default="qwen3.6-plus",
                        help="LLM 模型名称（默认: qwen-max）")
    parser.add_argument("--retry-failed", action="store_true",
                        help="仅处理之前失败的问题")
    parser.add_argument("--force", action="store_true",
                        help="忽略 checkpoint，全部重跑")
    parser.add_argument("--summary", action="store_true",
                        help="显示进度摘要，不处理")
    parser.add_argument("--topk", type=int, default=50,
                        help="每次查询检索的 proposition 数（仅 Naive RAG / Agentic RAG，默认: 50）")
    parser.add_argument("--max-chunks", type=int, default=8,
                        help="最终返回的 chunk 数（仅 Naive RAG / Agentic RAG，默认: 8）")
    parser.add_argument("--rerank", action="store_true",
                        help="启用 Reranker（仅 Naive RAG / Agentic RAG）")
    parser.add_argument("--schema", type=str, default="b", choices=["a", "b"],
                        help="Naive RAG 融合策略：a=Client-Side RRF, b=AnnSearchRequest-Level（默认: b）")
    parser.add_argument("--rewrite-model", type=str, default=None,
                        help="问题重写模型（仅 Naive RAG / RAG with Judge / Agentic RAG v2，默认: 同 --model）")
    parser.add_argument("--suggest-model", type=str, default=None,
                        help="后续问题生成模型（仅 Naive RAG，默认: 同 --model）")
    parser.add_argument("--judge-model", type=str, default=None,
                        help="Judge 模型（仅 RAG with Judge，默认: 同 --model）")
    parser.add_argument("--answer-model", type=str, default=None,
                        help="回答模型（仅 Naive RAG / RAG with Judge / Agentic RAG，默认: 同 --model）")
    parser.add_argument("--judge-variant", type=str, default="B", choices=["A", "B", "C"],
                        help="Judge prompt 变体（仅 RAG with Judge：A=最小改动, B=中等, C=完整重写，默认: B）")
    parser.add_argument("--max-depth", type=int, default=3,
                        help="RAG with Judge 最大递归深度（默认: 3）")
    parser.add_argument("--max-rounds", type=int, default=5,
                        help="Agentic RAG / Agentic RAG v2 最大探索轮次（默认: 5）")
    parser.add_argument("--planner-model", type=str, default=None,
                        help="Planner 模型（仅 Agentic RAG / Agentic RAG v2，默认: 同 --model）")
    parser.add_argument("--evaluator-model", type=str, default=None,
                        help="Evaluator 模型（仅 Agentic RAG / Agentic RAG v2，默认: 同 --model）")
    parser.add_argument("--reflector-model", type=str, default=None,
                        help="Reflector 模型（仅 Agentic RAG / Agentic RAG v2，默认: 同 --model）")
    parser.add_argument("--synthesizer-model", type=str, default=None,
                        help="Synthesizer 模型（仅 Agentic RAG v2，默认: 同 --model）")
    # 自定义 Milvus collection
    parser.add_argument("--custom-collection", type=str, default=None,
                        help="自定义 Milvus collection 名称（提供后使用 CustomMilvusRetriever）")
    parser.add_argument("--dense-field", type=str, default="embedding",
                        help="dense 向量字段名（仅 --custom-collection，默认: embedding）")
    parser.add_argument("--text-field", type=str, default="proposition_text",
                        help="文本内容字段名（仅 --custom-collection，默认: proposition_text）")
    parser.add_argument("--sparse-field", type=str, default=None,
                        help="sparse/BM25 向量字段名（仅 --custom-collection，不提供则仅 dense 检索）")
    parser.add_argument("--model-params", type=str, default=None,
                        help="JSON 字符串，应用于所有自定义模型的 ChatOpenAI 参数（全局默认），"
                             "例如: '{\"temperature\": 0.7, \"max_tokens\": 4096}'")
    # 角色专用参数
    parser.add_argument("--planner-params", type=str, default=None,
                        help="Planner 角色专属参数（覆盖 --model-params）")
    parser.add_argument("--evaluator-params", type=str, default=None,
                        help="Evaluator 角色专属参数（覆盖 --model-params）")
    parser.add_argument("--reflector-params", type=str, default=None,
                        help="Reflector 角色专属参数（覆盖 --model-params）")
    parser.add_argument("--synthesizer-params", type=str, default=None,
                        help="Synthesizer 角色专属参数（覆盖 --model-params）")
    parser.add_argument("--rewrite-params", type=str, default=None,
                        help="Rewrite 角色专属参数（覆盖 --model-params）")
    parser.add_argument("--answer-params", type=str, default=None,
                        help="Answer 角色专属参数（覆盖 --model-params）")
    parser.add_argument("--judge-params", type=str, default=None,
                        help="Judge 角色专属参数（覆盖 --model-params）")
    parser.add_argument("--suggest-params", type=str, default=None,
                        help="Suggest 角色专属参数（覆盖 --model-params）")
    parser.add_argument("--cot", action="store_true",
                        help="启用 CoT 逐步推理 → 简洁答案（仅 RAG with Judge / Agentic RAG V3）")

    args = parser.parse_args()

    model_params = _parse_model_params(args.model_params)
    role_params = {
        "planner": _parse_model_params(args.planner_params),
        "evaluator": _parse_model_params(args.evaluator_params),
        "reflector": _parse_model_params(args.reflector_params),
        "synthesizer": _parse_model_params(args.synthesizer_params),
        "rewrite": _parse_model_params(args.rewrite_params),
        "answer": _parse_model_params(args.answer_params),
        "judge": _parse_model_params(args.judge_params),
        "suggest": _parse_model_params(args.suggest_params),
    }
    llm = make_llm(args.model, model_params=model_params)

    if args.summary:
        show_summary(args.mode, args.dataset, schema=args.schema)
        return

    logger.info("开始评估: 模式=%s, 数据集=%s, 模型=%s, 全局参数=%s",
                 args.mode, args.dataset, args.model, model_params or "默认")
    if args.retry_failed:
        logger.info("模式: 仅重试失败问题")
    if args.force:
        logger.info("模式: 强制重跑")

    run_eval(
        mode=args.mode,
        dataset=args.dataset,
        llm=llm,
        batch_size=args.batch_size,
        max_workers=args.max_workers,
        max_retries=args.max_retries,
        retry_failed=args.retry_failed,
        force=args.force,
        model_params=model_params,
        role_params=role_params,
        topk=args.topk,
        max_chunks=args.max_chunks,
        use_reranker=args.rerank,
        schema=args.schema,
        max_depth=args.max_depth,
        max_rounds=args.max_rounds,
        rewrite_model=args.rewrite_model,
        suggest_model=args.suggest_model,
        judge_model=args.judge_model,
        answer_model=args.answer_model,
        judge_variant=args.judge_variant,
        planner_model=args.planner_model,
        evaluator_model=args.evaluator_model,
        reflector_model=args.reflector_model,
        synthesizer_model=args.synthesizer_model,
        # 自定义 collection
        custom_collection=args.custom_collection,
        custom_dense_field=args.dense_field,
        custom_text_field=args.text_field,
        custom_sparse_field=args.sparse_field,
        cot=args.cot,
    )

    logger.info("评估完成: %s/%s。", args.mode, args.dataset)


if __name__ == "__main__":
    main()
