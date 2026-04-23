"""CLI 入口：build / stats / drop / create — 索引管理

每个数据集独立 collection，命令默认操作全部数据集，
可通过 --dataset 指定。
"""

import argparse
import json
import logging
import sys
from pathlib import Path

# 确保项目根目录在 sys.path 中
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from Index.milvus_config import get_all_dataset_types, get_collection_name
from Index.milvus_schema import drop_collection, get_collection_stats, create_collection

logger = logging.getLogger(__name__)


def cmd_build(args):
    """构建索引：展平 → 嵌入 → 插入（per-dataset）"""
    from Index.milvus_ingest import build_index

    datasets = args.dataset if args.dataset else None
    result = build_index(
        dataset_types=datasets,
        limit=args.limit,
        rebuild=args.rebuild,
        save_interval=getattr(args, "save_every", 1) or 1,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


def cmd_stats(args):
    """显示所有 collection 统计信息"""
    stats = get_collection_stats()
    print(json.dumps(stats, indent=2, ensure_ascii=False))


def cmd_drop(args):
    """删除指定数据集的 collection（默认全部）"""
    datasets = args.dataset if args.dataset else get_all_dataset_types()

    if not args.force:
        names = [get_collection_name(ds) for ds in datasets]
        confirm = input(f"Drop collections: {names}? (yes/no): ")
        if confirm.strip().lower() != "yes":
            print("Aborted.")
            return

    for ds in datasets:
        drop_collection(ds)
        print(f"Dropped: {get_collection_name(ds)}")


def cmd_create(args):
    """仅创建 collection（不含数据），用于预建"""
    datasets = args.dataset if args.dataset else get_all_dataset_types()
    for ds in datasets:
        create_collection(dataset_type=ds, drop_old=args.rebuild)
        print(f"Created: {get_collection_name(ds)}")


def main():
    parser = argparse.ArgumentParser(description="Milvus 索引管理 CLI")
    subparsers = parser.add_subparsers(dest="command", help="可用命令")

    # build
    build_p = subparsers.add_parser("build", help="构建索引（展平 → 嵌入 → 插入）")
    build_p.add_argument("--dataset", nargs="+", choices=get_all_dataset_types(), help="指定数据集")
    build_p.add_argument("--limit", type=int, help="限制问题数（测试用）")
    build_p.add_argument("--rebuild", action="store_true", help="重建指定数据集的 collection")
    build_p.add_argument("--save-every", type=int, default=1, help="每 N 个 embedding batch 写入一次状态文件 (默认 1)")

    # stats
    subparsers.add_parser("stats", help="所有 collection 统计信息")

    # drop
    drop_p = subparsers.add_parser("drop", help="删除 collection")
    drop_p.add_argument("--dataset", nargs="+", choices=get_all_dataset_types(), help="指定数据集（默认全部）")
    drop_p.add_argument("--force", action="store_true", help="跳过确认")

    # create
    create_p = subparsers.add_parser("create", help="仅创建 collection（不含数据）")
    create_p.add_argument("--dataset", nargs="+", choices=get_all_dataset_types(), help="指定数据集（默认全部）")
    create_p.add_argument("--rebuild", action="store_true", help="已存在则重建")

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(1)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    commands = {"build": cmd_build, "stats": cmd_stats, "drop": cmd_drop, "create": cmd_create}
    commands[args.command](args)


if __name__ == "__main__":
    main()
