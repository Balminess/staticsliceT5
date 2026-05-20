"""
验证Python数据集中的切片是否正确
"""

import json
import ast
import random
from typing import Set, Dict, List


class VariableAnalyzer(ast.NodeVisitor):
    """分析变量的定义和使用"""
    def __init__(self):
        self.defs = {}  # {line_number: set of variables}
        self.uses = {}  # {line_number: set of variables}
        self.current_line = None

    def visit(self, node):
        if hasattr(node, 'lineno'):
            old_line = self.current_line
            self.current_line = node.lineno - 1  # 转换为0-based
            if self.current_line not in self.defs:
                self.defs[self.current_line] = set()
            if self.current_line not in self.uses:
                self.uses[self.current_line] = set()
            result = super().visit(node)
            self.current_line = old_line
            return result
        return super().visit(node)

    def visit_Name(self, node):
        if self.current_line is not None:
            if isinstance(node.ctx, ast.Store):
                self.defs[self.current_line].add(node.id)
            elif isinstance(node.ctx, ast.Load):
                self.uses[self.current_line].add(node.id)
        self.generic_visit(node)

    def visit_FunctionDef(self, node):
        if self.current_line is not None:
            self.defs[self.current_line].add(node.name)
        self.generic_visit(node)

    def visit_ClassDef(self, node):
        if self.current_line is not None:
            self.defs[self.current_line].add(node.name)
        self.generic_visit(node)


def compute_backward_slice(code: str, variable: str, line_number: int) -> Set[int]:
    """计算backward slice"""
    try:
        tree = ast.parse(code)
    except:
        return set()

    analyzer = VariableAnalyzer()
    analyzer.visit(tree)

    # 构建依赖图
    deps = {}  # {line: set of lines it depends on}
    for line in range(len(code.split('\n'))):
        deps[line] = set()

    var_last_def = {}  # {variable: line where it was last defined}

    for line in sorted(analyzer.defs.keys()):
        # RAW依赖: 如果使用了某个变量，依赖于最后定义它的行
        for var in analyzer.uses.get(line, set()):
            if var in var_last_def:
                deps[line].add(var_last_def[var])

        # 更新变量最后定义的位置
        for var in analyzer.defs.get(line, set()):
            var_last_def[var] = line

    # 从target line开始，递归收集所有依赖
    slice_lines = set()
    to_visit = [line_number]
    visited = set()

    while to_visit:
        current = to_visit.pop()
        if current in visited:
            continue
        visited.add(current)
        slice_lines.add(current)

        # 添加所有依赖的行
        for dep_line in deps.get(current, set()):
            if dep_line not in visited:
                to_visit.append(dep_line)

    return slice_lines


def compute_forward_slice(code: str, variable: str, line_number: int) -> Set[int]:
    """计算forward slice"""
    try:
        tree = ast.parse(code)
    except:
        return set()

    analyzer = VariableAnalyzer()
    analyzer.visit(tree)

    # 找到在line_number定义的所有变量
    defined_vars = analyzer.defs.get(line_number, set())
    if not defined_vars:
        return {line_number}

    # 构建前向依赖图: 哪些行使用了这些变量
    forward_deps = {}  # {line: set of lines that use variables defined here}

    for line in range(len(code.split('\n'))):
        forward_deps[line] = set()

    var_last_def = {}
    for line in sorted(analyzer.defs.keys()):
        # 对于每个使用变量的行，记录它依赖于定义该变量的行
        for var in analyzer.uses.get(line, set()):
            if var in var_last_def:
                def_line = var_last_def[var]
                forward_deps[def_line].add(line)

        # 更新变量定义位置
        for var in analyzer.defs.get(line, set()):
            var_last_def[var] = line

    # 从line_number开始前向传播
    slice_lines = set()
    to_visit = [line_number]
    visited = set()

    while to_visit:
        current = to_visit.pop()
        if current in visited:
            continue
        visited.add(current)
        slice_lines.add(current)

        # 添加所有使用该行定义变量的行
        for user_line in forward_deps.get(current, set()):
            if user_line not in visited:
                to_visit.append(user_line)

    return slice_lines


def verify_sample(data: dict) -> dict:
    """验证单个样本"""
    code = data['code']
    variable = data['variable']
    line_number = data['line_number']
    expected_backward = set(data['backward_slice'])
    expected_forward = set(data['forward_slice'])

    # 计算实际的slice
    computed_backward = compute_backward_slice(code, variable, line_number)
    computed_forward = compute_forward_slice(code, variable, line_number)

    result = {
        'eid': data['eid'],
        'variable': variable,
        'line_number': line_number,
        'backward_correct': computed_backward == expected_backward,
        'forward_correct': computed_forward == expected_forward,
        'backward_expected': sorted(expected_backward),
        'backward_computed': sorted(computed_backward),
        'backward_missing': sorted(expected_backward - computed_backward),
        'backward_extra': sorted(computed_backward - expected_backward),
        'forward_expected': sorted(expected_forward),
        'forward_computed': sorted(computed_forward),
        'forward_missing': sorted(expected_forward - computed_forward),
        'forward_extra': sorted(computed_forward - expected_forward),
    }

    return result


def verify_dataset(filepath: str, num_samples: int = 20):
    """验证数据集中的切片"""
    print("=" * 80)
    print(f"验证切片正确性: {filepath}")
    print("=" * 80)

    with open(filepath, 'r') as f:
        lines = f.readlines()

    # 随机抽样
    sample_indices = random.sample(range(len(lines)), min(num_samples, len(lines)))

    results = []
    for idx in sample_indices:
        data = json.loads(lines[idx])
        result = verify_sample(data)
        results.append(result)

    # 统计
    backward_correct = sum(1 for r in results if r['backward_correct'])
    forward_correct = sum(1 for r in results if r['forward_correct'])
    both_correct = sum(1 for r in results if r['backward_correct'] and r['forward_correct'])

    print(f"\n【验证结果】")
    print(f"样本数: {len(results)}")
    print(f"Backward切片正确: {backward_correct}/{len(results)} ({backward_correct/len(results)*100:.1f}%)")
    print(f"Forward切片正确: {forward_correct}/{len(results)} ({forward_correct/len(results)*100:.1f}%)")
    print(f"两者都正确: {both_correct}/{len(results)} ({both_correct/len(results)*100:.1f}%)")

    # 显示错误样本
    print(f"\n【详细检查】")
    for i, result in enumerate(results[:10], 1):
        print(f"\n样本 {i}: {result['eid']}")
        print(f"  变量: {result['variable']} at line {result['line_number']}")

        # Backward slice
        print(f"  Backward切片: {'✓' if result['backward_correct'] else '✗'}")
        if not result['backward_correct']:
            print(f"    期望: {result['backward_expected']}")
            print(f"    计算: {result['backward_computed']}")
            if result['backward_missing']:
                print(f"    缺失: {result['backward_missing']}")
            if result['backward_extra']:
                print(f"    多余: {result['backward_extra']}")

        # Forward slice
        print(f"  Forward切片: {'✓' if result['forward_correct'] else '✗'}")
        if not result['forward_correct']:
            print(f"    期望: {result['forward_expected']}")
            print(f"    计算: {result['forward_computed']}")
            if result['forward_missing']:
                print(f"    缺失: {result['forward_missing']}")
            if result['forward_extra']:
                print(f"    多余: {result['forward_extra']}")

    # 显示一个完整样本进行人工验证
    print(f"\n{'=' * 80}")
    print("【人工验证样本】")
    print("=" * 80)

    # 选择一个切片不正确的样本（如果有），否则选第一个
    verify_idx = 0
    for i, r in enumerate(results):
        if not r['backward_correct'] or not r['forward_correct']:
            verify_idx = i
            break

    sample = results[verify_idx]
    data = json.loads(lines[sample_indices[verify_idx]])

    print(f"\nEID: {data['eid']}")
    print(f"变量: {data['variable']} at line {data['line_number']}")

    print(f"\n【代码】")
    code_lines = data['code'].split('\n')
    for i, line in enumerate(code_lines):
        marker = ""
        if i == data['line_number']:
            marker = " ← TARGET"
        elif i in data['backward_slice']:
            marker = " (B)"
        elif i in data['forward_slice']:
            marker = " (F)"

        print(f"{i:2d}: {line}{marker}")

    print(f"\n【Backward Slice】")
    print(f"期望: {sample['backward_expected']}")
    print(f"计算: {sample['backward_computed']}")
    for line_num in sample['backward_expected']:
        print(f"  {line_num}: {data['back_lined_code'][sample['backward_expected'].index(line_num)]}")

    print(f"\n【Forward Slice】")
    print(f"期望: {sample['forward_expected']}")
    print(f"计算: {sample['forward_computed']}")
    for line_num in sample['forward_expected']:
        print(f"  {line_num}: {data['forward_lined_code'][sample['forward_expected'].index(line_num)]}")


if __name__ == "__main__":
    filepath = "/home/pengfei/code/cocoslicer/data/train-example-codenet-python.jsonl"
    verify_dataset(filepath, num_samples=20)
