"""CLI 入口：query — 在线检索查询"""

import argparse
import logging
import sys
from pathlib import Path

# 确保项目根目录在 sys.path 中
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from Index.milvus_config import get_all_dataset_types


def cmd_query(args):
    """查询检索"""
    from Retrieval.milvus_retriever import MilvusRetriever

    retriever = MilvusRetriever(
        dataset_type=args.dataset,
        topk_propositions=args.topk,
        max_chunks=args.max_chunks,
        use_reranker=args.rerank,
    )

    results = retriever.get_similar_chunk_with_score(args.query)
    if not results:
        print("No results found.")
        return

    for i, (doc, score) in enumerate(results):
        print(f"\n{'=' * 60}")
        print(f"Chunk {i + 1}  (score: {score:.4f})")
        print(f"  Title:       {doc.metadata.get('chunk_title', 'N/A')}")
        print(f"  Source:      {doc.metadata.get('context_title', 'N/A')}")
        print(f"  Chunk ID:    {doc.metadata.get('chunk_id', 'N/A')}")
        print(f"  Propositions: {doc.metadata.get('aggregated_propositions', 0)}")
        print(f"  Question:    {doc.metadata.get('question_id', 'N/A')}")
        print(f"--- Content ---")
        print(doc.page_content)
        if not args.show_props:
            continue
        if doc.metadata.get("aggregated_propositions", 0) > 0:
            print(f"\n--- Detailed Propositions ---")
            props = [p.strip() for p in doc.page_content.split(". ") if p.strip()]
            for prop in props:
                if not prop.endswith("."):
                    prop += "."
                print(f"  - {prop}")


def main():
    parser = argparse.ArgumentParser(description="Milvus 检索模块 CLI")
    parser.add_argument("query", help="查询问题")
    parser.add_argument("--dataset", required=True, choices=get_all_dataset_types(), help="目标数据集")
    parser.add_argument("--topk", type=int, default=50, help="Stage 1 检索 proposition 数")
    parser.add_argument("--max-chunks", type=int, default=8, help="Stage 2 返回 chunk 数")
    parser.add_argument("--show-props", action="store_true", help="显示详细 proposition 列表")
    parser.add_argument("--rerank", action="store_true", help="使用 Reranker 重排")

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cmd_query(args)


if __name__ == "__main__":
    main()
