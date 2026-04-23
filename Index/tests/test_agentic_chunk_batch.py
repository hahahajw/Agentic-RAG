# -*- coding: utf-8 -*-
"""
AgenticChunk 批处理方法测试套件。

本测试套件验证：
1. get_propositions_batch - 顺序保持、与单条处理的等价性
2. chunk_batch - 顺序保持、无交叉污染、与单条处理的等价性
3. 批处理方法中的自定义 prompt 替换

运行方式：pytest Index/tests/test_agentic_chunk_batch.py -v
或：python Index/tests/test_agentic_chunk_batch.py
"""

import sys
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from Index.agentic_chunk import AgenticChunk, DEFAULT_PROPOSITION_PROMPT_TEMPLATE
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI


def create_chunker():
    """Create a fresh AgenticChunk instance for testing."""
    return AgenticChunk(create_llm())


def create_llm():
    """Create the default LLM for testing."""
    return ChatOpenAI(
        api_key=os.getenv("BL_API_KEY"),
        base_url=os.getenv("BL_BASE_URL"),
        model='qwen-max-latest',
        temperature=0.0
    )


# ==================== get_propositions_batch Tests ====================

def test_get_propositions_batch_order_preservation():
    """
    测试 1: 验证 get_propositions_batch 保持输入输出顺序一致

    Given: 3 个不同的文本段落
    When: 使用 get_propositions_batch 批量处理
    Then: 输出列表的顺序与输入列表顺序完全一致
    """
    chunker = create_chunker()

    # 准备 3 个有明显区别的文本
    texts = [
        "The year is 2023. John lives in New York.",  # 文本 1: 关于时间和地点
        "Maria likes to eat pizza and pasta.",         # 文本 2: 关于食物
        "The conference was held in Paris on Monday."  # 文本 3: 关于事件
    ]

    # 批量处理
    all_props = chunker.get_propositions_batch(texts)

    # 验证 1: 输出长度等于输入长度
    assert len(all_props) == len(texts), \
        f"输出长度 {len(all_props)} 不等于输入长度 {len(texts)}"

    # 验证 2: 每个元素都是列表
    assert all(isinstance(props, list) for props in all_props), \
        "输出中所有元素都应该是列表"

    print("[PASS] Test 1: Order Preservation")


def test_get_propositions_batch_vs_single():
    """
    测试 2：验证批处理与单条处理等价
    """
    chunker = create_chunker()

    texts = [
        "The year is 2023. John lives in New York.",
        "Maria likes to eat pizza and pasta.",
        "The conference was held in Paris on Monday."
    ]

    # 单条处理
    single_results = []
    for text in texts:
        props = chunker.get_propositions(text)
        single_results.append(props)

    # 批处理
    batch_results = chunker.get_propositions_batch(texts)

    # 验证 1：相同长度
    assert len(batch_results) == len(single_results), \
        f"Batch length {len(batch_results)} != Single length {len(single_results)}"

    # 验证 2：每个文本的命题数量相同
    for i, (batch_props, single_props) in enumerate(zip(batch_results, single_results)):
        assert len(batch_props) == len(single_props), \
            f"Text {i}: Batch count {len(batch_props)} != Single count {len(single_props)}"

    # 验证 3：命题内容相同
    for i, (batch_props, single_props) in enumerate(zip(batch_results, single_results)):
        assert batch_props == single_props, \
            f"Text {i}: Batch and Single content mismatch"

    print("[PASS] Test 2: Batch vs Single")


def test_get_propositions_batch_empty_input():
    """
    测试 3：空输入处理
    """
    chunker = create_chunker()

    empty_result = chunker.get_propositions_batch([])
    assert empty_result == [], "Empty input should return empty list"

    print("[PASS] Test 3: Empty Input")


def test_get_propositions_batch_custom_prompt():
    """
    测试 4：批处理中的自定义 prompt 替换
    """
    chunker = create_chunker()

    custom_prompt = ChatPromptTemplate.from_messages([
        ("system", "Extract ONLY propositions that contain numbers or dates."),
        ("user", "{input}"),
    ])

    texts = [
        "The year is 2023. John is 30 years old.",
        "Maria likes pizza.",
        "The event was on January 1st."
    ]

    results = chunker.get_propositions_batch(texts, prompt_template=custom_prompt)
    assert len(results) == len(texts), "Output length should match input"

    print("[PASS] Test 4: Custom Prompt")


# ==================== chunk_batch Tests ====================

def test_chunk_batch_order_preservation():
    """
    测试 5: 验证 chunk_batch 保持输入输出顺序一致

    Given: 3 组来自不同文本的命题
    When: 使用 chunk_batch 批量处理
    Then: 输出列表的顺序与输入列表顺序完全一致
    """
    chunker = create_chunker()

    # 准备 3 组命题，每组有明显区别
    prop_lists = [
        ["The year is 2023.", "John lives in New York."],      # 文本 1
        ["Maria likes pizza.", "Maria likes pasta."],          # 文本 2
        ["The conference was in Paris.", "It was on Monday."]  # 文本 3
    ]

    # 批量处理
    results = chunker.chunk_batch(prop_lists)

    # 验证 1: 输出长度等于输入长度
    assert len(results) == len(prop_lists), \
        f"输出长度 {len(results)} 不等于输入长度 {len(prop_lists)}"

    # 验证 2: 每个元素都是字典
    assert all(isinstance(r, dict) for r in results), \
        "输出中所有元素都应该是字典"

    # 验证 3: 每个字典都有正确的键值结构
    for i, chunks in enumerate(results):
        for chunk_id, chunk in chunks.items():
            assert 'chunk_id' in chunk, f"文本 {i}: chunk 缺少 chunk_id"
            assert 'title' in chunk, f"文本 {i}: chunk 缺少 title"
            assert 'summary' in chunk, f"文本 {i}: chunk 缺少 summary"
            assert 'propositions' in chunk, f"文本 {i}: chunk 缺少 propositions"

    print("[PASS] 测试 5 通过：顺序保持 - 输出长度和结构正确")


def test_chunk_batch_no_cross_contamination():
    """
    测试 6: 验证不同文本之间的 chunk 不会交叉污染

    Given: 3 组主题完全不同的命题
    When: 使用 chunk_batch 处理
    Then: 每组命题的 chunk 应该只包含该组的命题，不会混入其他组的命题
    """
    chunker = create_chunker()

    # 准备 3 组主题完全不同的命题
    prop_lists = [
        # 文本 1: 关于日期和时间
        ["The year is 2023.", "The month is October.", "Today is Monday."],

        # 文本 2: 关于食物
        ["Pizza is Italian.", "Pasta has sauce.", "Sushi is Japanese."],

        # 文本 3: 关于地点
        ["Paris is in France.", "Tokyo is in Japan.", "NYC is in USA."]
    ]

    # 批量处理
    results = chunker.chunk_batch(prop_lists)

    # 收集所有命题用于验证
    all_props_by_text = []
    for chunks in results:
        props_in_this_text = []
        for chunk in chunks.values():
            props_in_this_text.extend(chunk['propositions'])
        all_props_by_text.append(set(props_in_this_text))

    # 验证：任意两组命题之间不应该有重叠
    for i in range(len(all_props_by_text)):
        for j in range(i + 1, len(all_props_by_text)):
            overlap = all_props_by_text[i].intersection(all_props_by_text[j])
            assert len(overlap) == 0, \
                f"文本 {i} 和文本 {j} 之间有命题重叠：{overlap}"

    print("[PASS] 测试 6 通过：没有交叉污染 - 各组命题完全独立")


def test_chunk_batch_vs_single():
    """
    测试 7: 验证批处理与单条处理结果等价

    Given: 3 组命题
    When: 分别使用 chunk 和 chunk_batch 处理
    Then: 两种方法得到的 chunk 数量应该相同
    """
    chunker = create_chunker()

    prop_lists = [
        ["The year is 2023.", "John lives in New York."],
        ["Maria likes pizza.", "Maria likes pasta."],
        ["The conference was in Paris.", "It was on Monday."]
    ]

    # 单条处理
    single_results = []
    for props in prop_lists:
        fresh_chunker = AgenticChunk(llm=chunker.llm)
        chunks = fresh_chunker.chunk(props)
        single_results.append(chunks)

    # 批处理
    batch_results = chunker.chunk_batch(prop_lists)

    # 验证 1: 长度相同
    assert len(batch_results) == len(single_results), \
        f"批处理结果长度 {len(batch_results)} 不等于单条处理长度 {len(single_results)}"

    # 验证 2: 每个文本的 chunk 数量相同
    for i, (batch_chunks, single_chunks) in enumerate(zip(batch_results, single_results)):
        assert len(batch_chunks) == len(single_chunks), \
            f"文本 {i}: 批处理 chunk 数 {len(batch_chunks)} 不等于单条处理 {len(single_chunks)}"

    print("[PASS] 测试 7 通过：批处理与单条处理结果等价")


def test_chunk_batch_with_custom_prompts():
    """
    测试 8: 验证自定义 prompt 可以在 chunk_batch 中正确传递

    Given: 多个自定义 prompt
    When: 在 chunk_batch 中传递自定义 prompt
    Then: 所有 prompt 应该正确传递给内部方法
    """
    chunker = create_chunker()

    # 自定义 prompt - 简化版
    custom_find_prompt = ChatPromptTemplate.from_messages([
        ("system", "Find a matching chunk or return None."),
        ("user", "Chunks: {chunk_outline}\nProposition: {proposition}"),
    ])

    prop_lists = [
        ["Proposition A.", "Proposition B."],
        ["Proposition C."]
    ]

    # 使用自定义 prompt 进行批处理
    results = chunker.chunk_batch(
        prop_lists,
        find_chunk_prompt_template=custom_find_prompt
    )

    # 验证：输出结构正确
    assert len(results) == len(prop_lists), "输出长度应该与输入一致"
    assert all(isinstance(r, dict) for r in results), "输出应该都是字典"

    print("[PASS] 测试 8 通过：自定义 prompt 可以正确传递")


def test_chunk_batch_access_patterns():
    """
    测试 9: 验证 chunk_batch 返回结果的各种访问方式

    Given: chunk_batch 的返回结果
    When: 使用不同的访问方式
    Then: 所有访问方式都应该正确工作
    """
    chunker = create_chunker()

    prop_lists = [
        ["The year is 2023.", "The month is October."],
        ["Maria likes pizza."]
    ]

    results = chunker.chunk_batch(prop_lists)

    # 访问方式 1: 通过索引访问特定文本的 chunks
    text1_chunks = results[0]
    assert isinstance(text1_chunks, dict), "应该可以通过索引访问"

    # 访问方式 2: 通过 chunk_id 访问特定 chunk
    chunk_ids = list(text1_chunks.keys())
    assert len(chunk_ids) > 0, "应该有至少一个 chunk"
    first_chunk = text1_chunks[chunk_ids[0]]
    assert 'chunk_id' in first_chunk, "chunk 应该有 chunk_id"

    # 访问方式 3: 转换为列表
    chunk_list = list(text1_chunks.values())
    assert isinstance(chunk_list, list), "应该可以转换为列表"
    assert len(chunk_list) == len(chunk_ids), "列表长度应该等于 key 数量"

    print("[PASS] 测试 9 通过：所有访问方式都正确")


# ==================== 综合测试 ====================

def test_full_pipeline_batch():
    """
    测试 10: 完整流程测试 - 从原始文本到最终分块

    Given: 多个原始文本段落
    When: 使用 get_propositions_batch 提取命题，再使用 chunk_batch 分块
    Then: 整个流程应该正确完成，结果结构正确
    """
    chunker = create_chunker()

    # 原始文本
    texts = [
        "The year is 2023. John lives in New York. He works as a developer.",
        "Maria is a chef. She loves cooking Italian food. Her specialty is pasta.",
        "The conference was in Paris. It lasted three days. Many people attended."
    ]

    # Step 1: 批量提取命题
    all_props = chunker.get_propositions_batch(texts)

    # 验证 Step 1
    assert len(all_props) == len(texts), "命题数量应该与文本数量一致"
    assert all(len(props) > 0 for props in all_props), "每个文本都应该有命题"

    # Step 2: 批量分块
    all_chunks = chunker.chunk_batch(all_props)

    # 验证 Step 2
    assert len(all_chunks) == len(all_props), "分块数量应该与命题组数一致"

    # 验证最终结果
    for i, chunks in enumerate(all_chunks):
        print(f"\n文本 {i+1} 的分块结果:")
        for chunk_id, chunk in chunks.items():
            print(f"  Chunk {chunk['title']}: {chunk['propositions']}")

    print("\n[PASS] 测试 10 通过：完整流程测试成功")


# ==================== 运行所有测试 ====================

def run_all_tests():
    """运行所有批处理测试。"""
    print("=" * 60)
    print("AgenticChunk 批处理测试套件")
    print("=" * 60)

    tests = [
        ("测试 1: 顺序保持（命题）", test_get_propositions_batch_order_preservation),
        ("测试 2: 批处理 vs 单条处理（命题）", test_get_propositions_batch_vs_single),
        ("测试 3: 空输入处理", test_get_propositions_batch_empty_input),
        ("测试 4: 自定义 prompt（命题）", test_get_propositions_batch_custom_prompt),
        ("测试 5: 顺序保持（分块）", test_chunk_batch_order_preservation),
        ("测试 6: 无交叉污染", test_chunk_batch_no_cross_contamination),
        ("测试 7: 批处理 vs 单条处理（分块）", test_chunk_batch_vs_single),
        ("测试 8: 自定义 prompt（分块）", test_chunk_batch_with_custom_prompts),
        ("测试 9: 访问方式", test_chunk_batch_access_patterns),
        ("测试 10: 完整流程", test_full_pipeline_batch),
    ]

    passed = 0
    failed = 0

    for test_name, test_func in tests:
        print(f"\n{'='*60}")
        print(f"运行：{test_name}")
        print('='*60)

        try:
            test_func()
            passed += 1
        except AssertionError as e:
            failed += 1
            print(f"[FAIL] {test_name}: {e}")
        except Exception as e:
            failed += 1
            print(f"[ERROR] {test_name}: {type(e).__name__}: {e}")

    print("\n" + "="*60)
    print(f"结果：{passed} 通过，{failed} 失败，共 {passed + failed} 项")
    print("="*60)

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
