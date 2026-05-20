"""
测试SimpleStaticSlicer的正确性
"""

import sys
sys.path.insert(0, '/home/pengfei/code/cocoslicer')

from generate_python_dataset import SimpleStaticSlicer
import json


def test_slicer_on_dataset_sample():
    """在数据集样本上测试切片器"""
    filepath = "/home/pengfei/code/cocoslicer/data/train-example-codenet-python.jsonl"

    with open(filepath, 'r') as f:
        # 测试第一个样本
        data = json.loads(f.readline())

    print("=" * 80)
    print("测试SimpleStaticSlicer在数据集样本上的表现")
    print("=" * 80)

    print(f"\nEID: {data['eid']}")
    print(f"变量: {data['variable']} at line {data['line_number']}")

    # 使用SimpleStaticSlicer重新计算
    slicer = SimpleStaticSlicer(data['code'])

    # 注意：数据集中line_number是0-indexed，但slicer需要1-indexed
    computed_backward = slicer.backward_slice(data['line_number'] + 1, data['variable'])
    computed_forward = slicer.forward_slice(data['line_number'] + 1, data['variable'])

    print(f"\n【Backward Slice】")
    print(f"数据集: {sorted(data['backward_slice'])}")
    print(f"计算的: {sorted(computed_backward)}")
    print(f"匹配: {sorted(data['backward_slice']) == sorted(computed_backward)}")

    print(f"\n【Forward Slice】")
    print(f"数据集: {sorted(data['forward_slice'])}")
    print(f"计算的: {sorted(computed_forward)}")
    print(f"匹配: {sorted(data['forward_slice']) == sorted(computed_forward)}")

    # 显示代码
    print(f"\n【代码】")
    code_lines = data['code'].split('\n')
    for i, line in enumerate(code_lines[:30]):  # 只显示前30行
        marker = ""
        if i == data['line_number']:
            marker = " ← TARGET"
        elif i in data['backward_slice']:
            marker = " (B)"
        elif i in data['forward_slice']:
            marker = " (F)"
        print(f"{i:2d}: {line}{marker}")

    # 详细分析依赖
    print(f"\n【依赖分析】")
    print(f"数据依赖:")
    for line, deps in sorted(slicer.data_deps.items()):
        if deps:
            print(f"  Line {line-1}: 依赖于 {sorted([d-1 for d in deps])}")

    print(f"\n控制依赖:")
    for line, deps in sorted(slicer.control_deps.items()):
        if deps:
            print(f"  Line {line-1}: 依赖于 {sorted([d-1 for d in deps])}")


def test_multiple_samples():
    """测试多个样本"""
    filepath = "/home/pengfei/code/cocoslicer/data/train-example-codenet-python.jsonl"

    print("\n" + "=" * 80)
    print("批量测试数据集一致性")
    print("=" * 80)

    total = 0
    backward_match = 0
    forward_match = 0
    both_match = 0

    with open(filepath, 'r') as f:
        for i, line in enumerate(f):
            if i >= 100:  # 只测试前100个
                break

            data = json.loads(line)

            try:
                slicer = SimpleStaticSlicer(data['code'])
                computed_backward = slicer.backward_slice(data['line_number'] + 1, data['variable'])
                computed_forward = slicer.forward_slice(data['line_number'] + 1, data['variable'])

                total += 1

                if sorted(data['backward_slice']) == sorted(computed_backward):
                    backward_match += 1

                if sorted(data['forward_slice']) == sorted(computed_forward):
                    forward_match += 1

                if sorted(data['backward_slice']) == sorted(computed_backward) and \
                   sorted(data['forward_slice']) == sorted(computed_forward):
                    both_match += 1

            except Exception as e:
                print(f"Error on sample {i}: {e}")
                continue

    print(f"\n【结果】")
    print(f"总样本数: {total}")
    print(f"Backward匹配: {backward_match}/{total} ({backward_match/total*100:.1f}%)")
    print(f"Forward匹配: {forward_match}/{total} ({forward_match/total*100:.1f}%)")
    print(f"两者都匹配: {both_match}/{total} ({both_match/total*100:.1f}%)")

    if both_match == total:
        print("\n✓ 数据集完全一致！切片器工作正确。")
    else:
        print(f"\n✗ 有 {total - both_match} 个样本不匹配，可能存在问题。")


if __name__ == "__main__":
    test_slicer_on_dataset_sample()
    test_multiple_samples()
