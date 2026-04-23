import json
import random
import os


def sample_from_json(file_path, output_path, sample_size=500, seed=42):
    """Sample from JSON array format"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    random.seed(seed)
    sampled = random.sample(data, min(sample_size, len(data)))

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(sampled, f, indent=2, ensure_ascii=False)


def sample_from_jsonl(file_path, output_path, sample_size=500, seed=42):
    """Sample from JSONL format"""
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # Parse all JSON objects
    data = [json.loads(line.strip()) for line in lines if line.strip()]

    random.seed(seed)
    sampled = random.sample(data, min(sample_size, len(data)))

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(sampled, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    # Set random seed for reproducibility
    RANDOM_SEED = 42

    print("Starting benchmark dataset creation...")

    # 2WikiMultihopQA
    print("Sampling from 2WikiMultihopQA...")
    sample_from_json(
        "2WikiMultihopQA_dev.json",
        "benchmark/2WikiMultihopQA_500_benchmark.json",
        sample_size=500,
        seed=RANDOM_SEED
    )

    # HotpotQA
    print("Sampling from HotpotQA...")
    sample_from_json(
        "hotpot_dev_distractor_v1.json",
        "benchmark/HotpotQA_500_benchmark.json",
        sample_size=500,
        seed=RANDOM_SEED
    )

    # MuSiQue
    print("Sampling from MuSiQue...")
    sample_from_jsonl(
        "musique_ans_v1.0_dev.jsonl",
        "benchmark/MuSiQue_500_benchmark.json",
        sample_size=500,
        seed=RANDOM_SEED
    )

    print("\nBenchmark datasets created successfully!")
