"""CLI 入口 — 快速测试 Naive RAG 工作流"""

import argparse
import json
import logging
import os
import time

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from .workflow import get_workflow

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Naive RAG with Multi-Query Retrieval")
    parser.add_argument("--query", type=str, required=True, help="User question")
    parser.add_argument("--scheme", type=str, default="a", choices=["a", "b"],
                        help="Fusion scheme: a=Client-Side RRF, b=AnnSearchRequest-Level")
    parser.add_argument("--dataset", type=str, default="hotpotqa",
                        choices=["hotpotqa", "2wikimultihopqa", "musique"],
                        help="Dataset collection to query")
    parser.add_argument("--topk", type=int, default=50, help="Top-K propositions per query")
    parser.add_argument("--max-chunks", type=int, default=8, help="Max chunks in final output")
    parser.add_argument("--model", type=str, default="qwen3.5-plus", help="LLM model for answer generation")
    parser.add_argument("--rewrite-model", type=str, default=None, help="问题重写模型（默认: 同 --model）")
    parser.add_argument("--suggest-model", type=str, default=None, help="后续问题生成模型（默认: 同 --model）")
    parser.add_argument("--temp", type=float, default=0.0, help="LLM temperature")
    parser.add_argument("--rerank", action="store_true", help="Enable Reranker (qwen3-rerank)")

    args = parser.parse_args()

    # Setup LLMs
    def make_llm(model_name: str) -> ChatOpenAI:
        return ChatOpenAI(
            api_key=os.getenv("BL_API_KEY"),
            base_url=os.getenv("BL_BASE_URL"),
            model=model_name,
            temperature=args.temp,
        )

    answer_llm = make_llm(args.model)
    rewrite_llm = make_llm(args.rewrite_model) if args.rewrite_model else None
    suggest_llm = make_llm(args.suggest_model) if args.suggest_model else None

    # Build workflow
    app = get_workflow(scheme=args.scheme)

    # Run
    start_time = time.time()
    config = {
        "configurable": {
            "llm": answer_llm,
            "dataset_type": args.dataset,
            "topk_propositions": args.topk,
            "max_chunks": args.max_chunks,
            "use_reranker": args.rerank,
        },
    }
    if rewrite_llm:
        config["configurable"]["rewrite_llm"] = rewrite_llm
    if suggest_llm:
        config["configurable"]["suggest_llm"] = suggest_llm

    result = app.invoke(
        {"original_query": args.query, "messages": []},
        config=config,
    )
    elapsed = time.time() - start_time

    # Print results
    print(f"\n{'='*60}")
    print(f"Scheme: {args.scheme.upper()}")
    print(f"Query: {args.query}")
    print(f"Dataset: {args.dataset}")
    print(f"Reranker: {'Yes' if args.rerank else 'No'}")
    print(f"Time: {elapsed:.2f}s")
    print(f"{'='*60}")

    # Rewritten queries
    if result.get("rewritten_queries"):
        print(f"\nRewritten queries ({len(result['rewritten_queries'])}):")
        for i, q in enumerate(result["rewritten_queries"], 1):
            print(f"  {i}. {q}")

    # All queries
    if result.get("all_queries"):
        print(f"\nAll queries used for retrieval ({len(result['all_queries'])}):")
        for i, q in enumerate(result["all_queries"], 1):
            print(f"  {i}. {q}")

    # Fused chunks
    fused = result.get("fused_chunks", [])
    print(f"\nRetrieved {len(fused)} chunks:")
    for i, (doc, score) in enumerate(fused):
        title = doc.metadata.get("chunk_title", "N/A")
        print(f"  [{i+1}] Score: {score:.6f} | {title}")
        print(f"      {doc.page_content[:200]}...")

    # Answer
    print(f"\n{'='*60}")
    print("ANSWER:")
    print(result.get("answer", "No answer generated"))
    print(f"{'='*60}")

    # Followups
    followups = result.get("suggested_followups", [])
    if followups:
        print(f"\nSuggested follow-up questions ({len(followups)}):")
        for i, q in enumerate(followups, 1):
            print(f"  {i}. {q}")


if __name__ == "__main__":
    main()
