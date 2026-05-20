"""
简化的AST-based Order Permutation
原理：在同一control flow里，不存在依赖关系的语句可以交换位置
"""

import javalang
from typing import List, Set, Tuple
from collections import defaultdict


class Statement:
    def __init__(self, index: int, code: str, node):
        self.index = index
        self.code = code
        self.node = node
        self.defs = set()  # 定义的变量
        self.uses = set()  # 使用的变量

    def __repr__(self):
        return f"S{self.index}: {self.code[:40]}"


def extract_all_variables(node, result_set: Set[str]):
    """递归提取AST节点中的所有变量引用"""
    if isinstance(node, javalang.tree.MemberReference):
        # 处理 sc.nextInt() 或 简单变量 a
        if node.qualifier:
            result_set.add(node.qualifier)
        if node.member:
            result_set.add(node.member)

    elif isinstance(node, javalang.tree.MethodInvocation):
        # 处理 sc.nextInt() 中的sc
        if node.qualifier:
            result_set.add(node.qualifier)

    # 递归所有子节点
    if hasattr(node, 'children'):
        for child in node.children:
            if isinstance(child, list):
                for item in child:
                    if isinstance(item, javalang.tree.Node):
                        extract_all_variables(item, result_set)
            elif isinstance(child, javalang.tree.Node):
                extract_all_variables(child, result_set)


def analyze_statement(stmt: Statement):
    """分析语句的def和use"""
    node = stmt.node

    # 1. 提取定义（左值）
    if isinstance(node, javalang.tree.LocalVariableDeclaration):
        # int a, b, c;
        for declarator in node.declarators:
            stmt.defs.add(declarator.name)
            # 初始化表达式中的uses
            if declarator.initializer:
                extract_all_variables(declarator.initializer, stmt.uses)

    elif isinstance(node, javalang.tree.StatementExpression):
        if isinstance(node.expression, javalang.tree.Assignment):
            # a = expr
            target = node.expression.expressionl
            if isinstance(target, javalang.tree.MemberReference):
                stmt.defs.add(target.member)
            # 右值的uses
            extract_all_variables(node.expression.value, stmt.uses)
        else:
            # 其他表达式语句（如方法调用）
            extract_all_variables(node.expression, stmt.uses)

    # 2. 提取使用（if, for等的条件和内部所有变量）
    if isinstance(node, javalang.tree.IfStatement):
        # if条件中的变量
        extract_all_variables(node.condition, stmt.uses)
        # if块内部的所有变量使用
        if node.then_statement:
            extract_all_variables(node.then_statement, stmt.uses)
        if node.else_statement:
            extract_all_variables(node.else_statement, stmt.uses)

    elif isinstance(node, javalang.tree.ForStatement):
        if node.control:
            extract_all_variables(node.control, stmt.uses)
        # for块内部的所有变量使用
        if node.body:
            extract_all_variables(node.body, stmt.uses)

    # 移除自身定义的变量（在use中）
    stmt.uses -= stmt.defs


def build_dependency_graph(statements: List[Statement]) -> dict:
    """
    构建依赖图
    返回: {stmt_index: set of依赖的stmt_index}
    """
    deps = defaultdict(set)
    var_last_def = {}  # var -> stmt_index

    for stmt in statements:
        # 对于使用的每个变量，找到最后定义它的语句
        for var in stmt.uses:
            if var in var_last_def:
                deps[stmt.index].add(var_last_def[var])

        # 更新变量的最后定义
        for var in stmt.defs:
            # WAW: 如果变量已被定义过，当前语句依赖于前一个定义
            if var in var_last_def:
                deps[stmt.index].add(var_last_def[var])
            var_last_def[var] = stmt.index

    return deps


def find_swappable_pairs(statements: List[Statement], deps: dict) -> List[Tuple[int, int]]:
    """找出可以交换的语句对（同一control flow中无依赖的相邻或邻近语句）"""
    swappable = []

    for i in range(len(statements)):
        for j in range(i + 1, len(statements)):
            # 检查i和j是否可以交换
            # 条件1: i和j之间没有直接依赖
            i_depends_on_j = j in deps[i]
            j_depends_on_i = i in deps[j]

            if i_depends_on_j or j_depends_on_i:
                continue

            # 条件2: 它们之间的任何语句k都不能依赖于i或j
            # 如果k依赖于i，那么交换后i会在k之后，破坏依赖
            # 如果k依赖于j，那么交换后j会在k之前，破坏依赖
            has_intermediate_dep = False
            for k in range(i + 1, j):
                if i in deps[k] or j in deps[k]:
                    has_intermediate_dep = True
                    break

            # 条件3: i和j也不能依赖于它们之间的任何语句
            # 如果i依赖于k，那么交换后i会在k之前，破坏依赖
            # 如果j依赖于k，那么交换后j会在k之后，破坏依赖
            if not has_intermediate_dep:
                for k in range(i + 1, j):
                    if k in deps[i] or k in deps[j]:
                        has_intermediate_dep = True
                        break

            if not has_intermediate_dep:
                swappable.append((i, j))

    return swappable


def generate_permutation(java_code: str) -> Tuple[str, List[int]]:
    """
    生成dataflow等价的代码变体

    Returns:
        (变体代码, 交换的行号)
    """
    try:
        tree = javalang.parse.parse(java_code)
    except:
        return java_code, []

    lines = java_code.split('\n')
    statements = []

    # 提取main方法中的语句
    for path, node in tree:
        if isinstance(node, javalang.tree.MethodDeclaration):
            if node.name == 'main' and node.body:
                for idx, stmt_node in enumerate(node.body):
                    if hasattr(stmt_node, 'position') and stmt_node.position:
                        line_num = stmt_node.position.line - 1
                        code = lines[line_num].strip()
                        stmt = Statement(idx, code, stmt_node)
                        analyze_statement(stmt)
                        statements.append(stmt)
                break

    if len(statements) < 2:
        return java_code, []

    # 构建依赖图
    deps = build_dependency_graph(statements)

    # 找可交换对
    swappable = find_swappable_pairs(statements, deps)

    if not swappable:
        return java_code, []

    # 选择第一对交换
    i, j = swappable[0]
    stmt_i = statements[i]
    stmt_j = statements[j]

    # 交换代码行
    line_i = stmt_i.node.position.line - 1
    line_j = stmt_j.node.position.line - 1

    new_lines = lines[:]
    new_lines[line_i], new_lines[line_j] = new_lines[line_j], new_lines[line_i]

    return '\n'.join(new_lines), [line_i, line_j]


def test_on_jsonl(jsonl_file: str, num_samples: int = 10):
    """在JSONL数据上测试"""
    import json

    print("="*80)
    print(f"简化AST Permutation测试 - {num_samples}个样本")
    print("="*80)

    success = 0

    with open(jsonl_file, 'r') as f:
        for i, line in enumerate(f):
            if i >= num_samples:
                break

            sample = json.loads(line)
            eid = sample['eid']
            code = sample['code']

            # 添加类包装
            full_code = f"public class Solution {{\n{code}\n}}"

            variant, swapped = generate_permutation(full_code)

            if swapped:
                success += 1
                print(f"[{i+1}] ✓ {eid}: 交换行 {swapped[0]+1} ↔ {swapped[1]+1}")
            else:
                print(f"[{i+1}] ✗ {eid}: 无可交换语句")

    print("="*80)
    print(f"成功率: {success}/{num_samples} ({success/num_samples*100:.1f}%)")
    print("="*80)


if __name__ == "__main__":
    # 测试单个文件
    test_file = '/home/pengfei/code/cocoslicer/slice/example_1.java'

    with open(test_file, 'r') as f:
        code = f.read()

    print("测试单个文件:")
    print("="*80)

    # 分析
    try:
        tree = javalang.parse.parse(code)
        statements = []

        for path, node in tree:
            if isinstance(node, javalang.tree.MethodDeclaration) and node.name == 'main':
                if node.body:
                    for idx, stmt_node in enumerate(node.body):
                        if hasattr(stmt_node, 'position'):
                            lines = code.split('\n')
                            line_num = stmt_node.position.line - 1
                            stmt = Statement(idx, lines[line_num].strip(), stmt_node)
                            analyze_statement(stmt)
                            statements.append(stmt)
                    break

        print(f"\n提取了 {len(statements)} 个语句:")
        for stmt in statements:
            print(f"  {stmt}")
            print(f"    defs: {stmt.defs}")
            print(f"    uses: {stmt.uses}")

        deps = build_dependency_graph(statements)
        print(f"\n依赖关系:")
        for i, d in deps.items():
            if d:
                print(f"  S{i} -> {d}")

        swappable = find_swappable_pairs(statements, deps)
        print(f"\n可交换对: {len(swappable)}")
        for i, j in swappable[:5]:
            print(f"  S{i} ↔ S{j}")

    except Exception as e:
        print(f"错误: {e}")

    print("\n" + "="*80)
    print("测试JSONL数据:")
    test_on_jsonl('/home/pengfei/code/cocoslicer/data/test-example-codenet-java.jsonl', 10)
