"""CLI 入口：自定义 Milvus collection 检索查询"""

import argparse
import logging
import sys
from pathlib import Path

# 确保项目根目录在 sys.path 中
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


def cmd_query(args):
    """查询检索"""
    from custom.retriever import CustomMilvusRetriever

    retriever = CustomMilvusRetriever(
        collection_name=args.collection,
        dense_field=args.dense_field,
        text_field=args.text_field,
        sparse_field=args.sparse_field if args.sparse_field else None,
        topk=args.topk,
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
        cid = doc.metadata.get("chunk_id", "N/A")
        print(f"  Chunk ID:    {cid}")
        print(f"  Aggregated:  {doc.metadata.get('aggregated_propositions', 0)} propositions")
        # 打印所有可用的 metadata 键
        extra_meta = {k: v for k, v in doc.metadata.items() if k not in ("chunk_id", "aggregated_propositions", "id")}
        if extra_meta:
            print(f"  Metadata:    {extra_meta}")
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
    parser = argparse.ArgumentParser(description="自定义 Milvus Collection 检索 CLI")
    parser.add_argument("query", help="查询问题")
    parser.add_argument("--collection", required=True, help="Milvus collection 名称")
    parser.add_argument("--dense-field", default="embedding", help="dense 向量字段名")
    parser.add_argument("--text-field", default="proposition_text", help="文本内容字段名")
    parser.add_argument("--sparse-field", default=None, help="sparse/BM25 向量字段名")
    parser.add_argument("--topk", type=int, default=50, help="Stage 1 检索 proposition 数")
    parser.add_argument("--max-chunks", type=int, default=8, help="最终返回 chunk 数")
    parser.add_argument("--show-props", action="store_true", help="显示详细 proposition 列表")
    parser.add_argument("--rerank", action="store_true", help="使用 Reranker 重排")

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cmd_query(args)


if __name__ == "__main__":
    main()
