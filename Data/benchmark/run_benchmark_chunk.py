"""
Benchmark Chunker CLI - 命令行入口。

支持使用 V3（默认）、V2 或 V1 处理器处理 benchmark 数据集。

用法：
    # 测试模式（3 个问题）
    uv run python Data/benchmark/run_benchmark_chunk.py --test

    # 处理所有数据集
    uv run python Data/benchmark/run_benchmark_chunk.py --all

    # 单个数据集
    uv run python Data/benchmark/run_benchmark_chunk.py --dataset hotpotqa

    # 重试失败的
    uv run python Data/benchmark/run_benchmark_chunk.py --retry-failed

    # 使用 V2 处理器
    uv run python Data/benchmark/run_benchmark_chunk.py --all --use-v2

    # 使用 V1 处理器
    uv run python Data/benchmark/run_benchmark_chunk.py --all --use-v1
"""

import argparse
import os
import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark Chunker CLI - 处理 MultiHop QA 基准数据集"
    )

    # 模式选择
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--test",
        action="store_true",
        help="测试模式：每个数据集处理 3 个问题"
    )
    mode_group.add_argument(
        "--all",
        action="store_true",
        help="处理所有数据集（HotpotQA, 2WikiMultihopQA, MuSiQue）"
    )
    mode_group.add_argument(
        "--dataset",
        type=str,
        choices=["hotpotqa", "2wikimultihop", "musique"],
        help="处理单个数据集"
    )

    # 处理器版本
    parser.add_argument(
        "--use-v2",
        action="store_true",
        help="使用 V2 处理器"
    )
    parser.add_argument(
        "--use-v1",
        action="store_true",
        help="使用 V1 处理器"
    )
    parser.add_argument(
        "--use-v3",
        action="store_true",
        default=True,
        help="使用 V3 处理器（逐问题 + 问题内并行，默认启用）"
    )

    # 其他选项
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="只重试之前失败的单元"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="强制重新处理所有单元（忽略 checkpoint）"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="限制处理的问题数量（用于测试）"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="输出目录（默认为输入文件所在目录）"
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=3,
        help="并行工作线程数（默认 3）"
    )

    args = parser.parse_args()

    # 确定处理器版本
    if args.use_v1:
        use_version = "v1"
    elif args.use_v2:
        use_version = "v2"
    else:
        use_version = "v3"  # 默认 V3

    # 确定限制数量
    limit = args.limit
    if args.test and not limit:
        limit = 3

    # 导入依赖
    from dotenv import load_dotenv
    from langchain_openai import ChatOpenAI

    load_dotenv()

    # 初始化 LLM
    print(f"初始化 LLM (qwen3.5-plus)...")
    llm = ChatOpenAI(
        api_key=os.getenv("BL_API_KEY"),
        base_url=os.getenv("BL_BASE_URL"),
        model='qwen3.6-plus',
        temperature=0.0,
        extra_body={"enable_thinking": False}
    )

    # 根据参数选择处理方式
    if use_version == "v3":
        print("使用 V3 处理器（逐问题 + 问题内并行）\n")
        from Index.benchmark_chunker_v3 import BenchmarkChunkProcessorV3
    elif use_version == "v2":
        print("使用 V2 处理器（全局批处理）\n")
        from Index.benchmark_chunker_v2 import BenchmarkChunkProcessorV2
    else:
        print("使用 V1 处理器（原始版本）\n")
        from Index.benchmark_chunker import BenchmarkChunkProcessor

    def process_dataset(dataset_type: str, input_path: str):
        """处理单个数据集"""
        output_path = None
        if args.output_dir:
            output_path = Path(args.output_dir) / Path(input_path).name.replace('.json', '_chunked.json')

        if use_version == "v3":
            processor = BenchmarkChunkProcessorV3(
                llm=llm,
                dataset_type=dataset_type,
                input_path=input_path,
                output_path=output_path,
                chunk_max_workers=args.max_workers
            )
        elif use_version == "v2":
            processor = BenchmarkChunkProcessorV2(
                llm=llm,
                dataset_type=dataset_type,
                input_path=input_path,
                output_path=output_path,
                max_workers=args.max_workers
            )
        else:
            processor = BenchmarkChunkProcessor(
                llm=llm,
                dataset_type=dataset_type,
                input_path=input_path,
                output_path=output_path
            )

        return processor.process(limit=limit, retry_failed=args.retry_failed, force=args.force)

    # 执行处理
    if args.all:
        datasets = [
            ('hotpotqa', 'Data/benchmark/HotpotQA_500_benchmark.json'),
            ('2wikimultihop', 'Data/benchmark/2WikiMultihopQA_500_benchmark.json'),
            ('musique', 'Data/benchmark/MuSiQue_500_benchmark.json')
        ]

        all_stats = {}
        for dataset_type, input_path in datasets:
            print(f"\n{'='*70}")
            print(f"处理数据集：{dataset_type}")
            print(f"{'='*70}")
            stats = process_dataset(dataset_type, input_path)
            all_stats[dataset_type] = stats

        print(f"\n{'='*70}")
        print("所有数据集处理完成!")
        print(f"{'='*70}")
        for dataset_type, stats in all_stats.items():
            print(f"\n{dataset_type}:")
            print(f"  处理问题数：{stats.get('processed_items', stats.get('processed', 0))}")
            print(f"  完成：{stats.get('completed', 0)} | 部分完成：{stats.get('partial', 0)} | 失败：{stats.get('failed', 0)}")
            print(f"  命题提取：成功 {stats.get('propositions_success', 0)} 个，失败 {stats.get('propositions_failed', 0)} 个")
            print(f"  分块处理：成功 {stats.get('chunks_success', 0)} 个，失败 {stats.get('chunks_failed', 0)} 个")

    elif args.dataset:
        dataset_paths = {
            'hotpotqa': 'Data/benchmark/HotpotQA_500_benchmark.json',
            '2wikimultihop': 'Data/benchmark/2WikiMultihopQA_500_benchmark.json',
            'musique': 'Data/benchmark/MuSiQue_500_benchmark.json'
        }

        input_path = dataset_paths[args.dataset]
        stats = process_dataset(args.dataset, input_path)

        print(f"\n{args.dataset} 处理完成!")
        print(f"  处理问题数：{stats.get('processed_items', stats.get('processed', 0))}")

    elif args.test:
        # 测试模式：只处理 HotpotQA 的前 3 个问题
        print("测试模式：处理 HotpotQA 的前 3 个问题")
        input_path = 'Data/benchmark/HotpotQA_500_benchmark.json'
        stats = process_dataset('hotpotqa', input_path)

        print(f"\n测试处理完成!")
        print(f"  处理问题数：{stats.get('processed_items', stats.get('processed', 0))}")
        print(f"  命题提取：成功 {stats.get('propositions_success', 0)} 个")
        print(f"  分块处理：成功 {stats.get('chunks_success', 0)} 个")

    else:
        parser.print_help()
        print("\n请指定 --test、--all 或 --dataset 参数")


if __name__ == '__main__':
    main()
