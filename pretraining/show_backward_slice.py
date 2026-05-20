"""
展示数据集样本及其后向切片
"""

import json
import sys
sys.path.insert(0, '/home/pengfei/code/cocoslicer')
from generate_python_dataset import SimpleStaticSlicer


def show_backward_slice(data):
    """清晰展示一个样本的后向切片"""
    print("=" * 80)
    print("【样本信息】")
    print("=" * 80)
    print(f"EID: {data['eid']}")
    print(f"切片变量: {data['variable']}")
    print(f"变量位置: Line {data['line_number']}, Column {data['variable_loc'][1]}")
    print(f"\nBackward Slice包含 {len(data['backward_slice'])} 行: {data['backward_slice']}")

    # 解析切片器以获取依赖信息
    slicer = SimpleStaticSlicer(data['code'])

    print("\n" + "=" * 80)
    print("【完整代码 - 标注Backward Slice】")
    print("=" * 80)
    print("说明:")
    print("  [*] = 在Backward Slice中")
    print("  [ ] = 不在Backward Slice中")
    print("  >>> = 切片目标行")
    print()

    code_lines = data['code'].split('\n')
    for i, line in enumerate(code_lines):
        in_slice = "[*]" if i in data['backward_slice'] else "[ ]"
        is_target = ">>>" if i == data['line_number'] else "   "

        # 高亮目标行
        print(f"{in_slice} {i:3d} {is_target} {line}")

    print("\n" + "=" * 80)
    print("【Backward Slice详细内容】")
    print("=" * 80)

    for idx, line_num in enumerate(data['backward_slice'], 1):
        if line_num < len(code_lines):
            is_target = " ← 切片目标" if line_num == data['line_number'] else ""
            print(f"{idx}. Line {line_num}: {code_lines[line_num]}{is_target}")

    print("\n" + "=" * 80)
    print("【依赖关系分析】")
    print("=" * 80)

    target_line_1indexed = data['line_number'] + 1

    print(f"\n目标: Line {data['line_number']} (1-indexed: {target_line_1indexed})")
    print(f"变量: {data['variable']}")

    # 显示数据依赖
    print(f"\n【数据依赖】")
    if target_line_1indexed in slicer.data_deps:
        deps = slicer.data_deps[target_line_1indexed]
        if deps:
            print(f"Line {data['line_number']} 数据依赖于:")
            for dep in sorted(deps):
                dep_0indexed = dep - 1
                if dep_0indexed < len(code_lines):
                    print(f"  → Line {dep_0indexed}: {code_lines[dep_0indexed]}")
        else:
            print("  无直接数据依赖")
    else:
        print("  无数据依赖")

    # 显示控制依赖
    print(f"\n【控制依赖】")
    if target_line_1indexed in slicer.control_deps:
        deps = slicer.control_deps[target_line_1indexed]
        if deps:
            print(f"Line {data['line_number']} 控制依赖于:")
            for dep in sorted(deps):
                dep_0indexed = dep - 1
                if dep_0indexed < len(code_lines):
                    print(f"  → Line {dep_0indexed}: {code_lines[dep_0indexed]}")
        else:
            print("  无控制依赖")
    else:
        print("  无控制依赖")

    # 递归展示完整依赖链
    print(f"\n【完整依赖链】")
    print("展示如何从目标行递归追溯到所有Backward Slice行:\n")

    visited = set()

    def trace_dependencies(line_1indexed, indent=0):
        """递归追踪依赖"""
        line_0indexed = line_1indexed - 1

        if line_1indexed in visited or line_0indexed < 0 or line_0indexed >= len(code_lines):
            return

        visited.add(line_1indexed)

        prefix = "  " * indent
        code_preview = code_lines[line_0indexed][:60]
        print(f"{prefix}Line {line_0indexed}: {code_preview}")

        # 追踪数据依赖
        if line_1indexed in slicer.data_deps and slicer.data_deps[line_1indexed]:
            for dep in sorted(slicer.data_deps[line_1indexed]):
                if dep not in visited:
                    print(f"{prefix}  ↓ 数据依赖")
                    trace_dependencies(dep, indent + 1)

        # 追踪控制依赖
        if line_1indexed in slicer.control_deps and slicer.control_deps[line_1indexed]:
            for dep in sorted(slicer.control_deps[line_1indexed]):
                if dep not in visited:
                    print(f"{prefix}  ↓ 控制依赖")
                    trace_dependencies(dep, indent + 1)

    trace_dependencies(target_line_1indexed)

    print("\n" + "=" * 80)
    print("【为什么这些行在Backward Slice中？】")
    print("=" * 80)
    print("""
Backward Slice包含所有影响目标变量值的语句。

包含规则:
1. 目标行本身（变量定义或使用的位置）
2. 数据依赖: 如果目标行使用了变量x，则定义x的行也在切片中
3. 控制依赖: 如果目标行在if/for/while块中，则控制语句也在切片中
4. 传递闭包: 递归应用上述规则，直到没有新的依赖

用途:
- 理解一个变量的值是如何被计算出来的
- 调试: 找到影响某个值的所有代码
- 程序理解: 追踪数据流
    """)


if __name__ == "__main__":
    filepath = "/home/pengfei/code/cocoslicer/data/train-example-codenet-python.jsonl"

    # 读取几个有代表性的样本
    with open(filepath, 'r') as f:
        lines = f.readlines()

    # 样本1: 第一个样本
    print("\n" + "█" * 80)
    print("█ 样本1: 复杂的数学计算")
    print("█" * 80)
    data1 = json.loads(lines[0])
    show_backward_slice(data1)

    # 样本2: 找一个控制依赖明显的样本
    print("\n\n" + "█" * 80)
    print("█ 样本2: 控制流依赖")
    print("█" * 80)

    # 找一个在循环中的变量
    for line in lines[10:50]:
        data = json.loads(line)
        code_lines = data['code'].split('\n')
        target_line = code_lines[data['line_number']] if data['line_number'] < len(code_lines) else ""

        # 找一个在循环中的赋值
        if 'for ' in data['code'] and '=' in target_line and len(data['backward_slice']) >= 3:
            show_backward_slice(data)
            break

    # 样本3: 简单样本
    print("\n\n" + "█" * 80)
    print("█ 样本3: 简单的输入处理")
    print("█" * 80)

    for line in lines[1:10]:
        data = json.loads(line)
        if len(data['backward_slice']) <= 5:
            show_backward_slice(data)
            break
