"""
DFG-based Span Denoising for Code Pre-training

原理：Mask DFG中的有意义单元，从单个表达式到整个依赖链
类似AST-T5，但基于数据流图而不是AST
"""

import javalang
from typing import List, Set, Tuple, Dict
from collections import defaultdict
import random


class Statement:
    def __init__(self, index: int, code: str, node, line_num: int):
        self.index = index
        self.code = code
        self.node = node
        self.line_num = line_num
        self.defs = set()
        self.uses = set()
        self.size = len(code.split())  # token数量估计

    def __repr__(self):
        return f"S{self.index}[{self.size}tok]: {self.code[:40]}"


class DFGUnit:
    """DFG中的可masking单元"""
    def __init__(self, statements: List[Statement], unit_type: str):
        self.statements = statements
        self.unit_type = unit_type  # "single", "chain", "subtree"
        self.size = sum(s.size for s in statements)
        self.indices = [s.index for s in statements]

    def __repr__(self):
        return f"{self.unit_type}[{self.size}tok]: S{self.indices}"


def extract_all_variables(node, result_set: Set[str]):
    """递归提取AST节点中的所有变量引用"""
    if isinstance(node, javalang.tree.MemberReference):
        if node.qualifier:
            result_set.add(node.qualifier)
        if node.member:
            result_set.add(node.member)
    elif isinstance(node, javalang.tree.MethodInvocation):
        if node.qualifier:
            result_set.add(node.qualifier)

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

    if isinstance(node, javalang.tree.LocalVariableDeclaration):
        for declarator in node.declarators:
            stmt.defs.add(declarator.name)
            if declarator.initializer:
                extract_all_variables(declarator.initializer, stmt.uses)

    elif isinstance(node, javalang.tree.StatementExpression):
        if isinstance(node.expression, javalang.tree.Assignment):
            target = node.expression.expressionl
            if isinstance(target, javalang.tree.MemberReference):
                stmt.defs.add(target.member)
            extract_all_variables(node.expression.value, stmt.uses)
        else:
            extract_all_variables(node.expression, stmt.uses)

    if isinstance(node, javalang.tree.IfStatement):
        extract_all_variables(node.condition, stmt.uses)
        if node.then_statement:
            extract_all_variables(node.then_statement, stmt.uses)
        if node.else_statement:
            extract_all_variables(node.else_statement, stmt.uses)

    elif isinstance(node, javalang.tree.ForStatement):
        if node.control:
            extract_all_variables(node.control, stmt.uses)
        if node.body:
            extract_all_variables(node.body, stmt.uses)

    stmt.uses -= stmt.defs


def build_dependency_graph(statements: List[Statement]) -> Dict[int, Set[int]]:
    """构建依赖图: {stmt_index: set of它依赖的stmt_index}"""
    deps = defaultdict(set)
    var_last_def = {}

    for stmt in statements:
        for var in stmt.uses:
            if var in var_last_def:
                deps[stmt.index].add(var_last_def[var])

        for var in stmt.defs:
            if var in var_last_def:
                deps[stmt.index].add(var_last_def[var])
            var_last_def[var] = stmt.index

    return deps


def find_dependency_chains(statements: List[Statement], deps: Dict[int, Set[int]]) -> List[List[int]]:
    """识别依赖链: A -> B -> C"""
    chains = []
    visited = set()

    for stmt in statements:
        if stmt.index in visited:
            continue

        # 找到当前语句的完整依赖链
        chain = [stmt.index]
        current = stmt.index

        # 向前追溯依赖
        while deps[current]:
            # 选择直接依赖（只有一个依赖时）
            if len(deps[current]) == 1:
                prev = list(deps[current])[0]
                if prev not in chain:
                    chain.insert(0, prev)
                    current = prev
                else:
                    break
            else:
                break

        if len(chain) >= 2:
            chains.append(chain)
            visited.update(chain)

    return chains


def find_dependency_subtrees(statements: List[Statement], deps: Dict[int, Set[int]]) -> List[Set[int]]:
    """识别依赖子树: 一个语句及其所有传递依赖"""
    subtrees = []

    for stmt in statements:
        if not deps[stmt.index]:
            continue

        # BFS找到所有传递依赖
        subtree = {stmt.index}
        queue = list(deps[stmt.index])
        visited = set(queue)

        while queue:
            current = queue.pop(0)
            subtree.add(current)
            for dep in deps[current]:
                if dep not in visited:
                    visited.add(dep)
                    queue.append(dep)

        if len(subtree) >= 2:
            subtrees.append(subtree)

    return subtrees


def build_dfg_units(statements: List[Statement], deps: Dict[int, Set[int]]) -> List[DFGUnit]:
    """构建DFG单元列表"""
    units = []
    stmt_dict = {s.index: s for s in statements}

    # 1. 依赖链
    chains = find_dependency_chains(statements, deps)
    for chain in chains:
        chain_stmts = [stmt_dict[i] for i in chain]
        units.append(DFGUnit(chain_stmts, "chain"))

    # 2. 依赖子树
    subtrees = find_dependency_subtrees(statements, deps)
    for subtree in subtrees:
        subtree_stmts = [stmt_dict[i] for i in sorted(subtree)]
        units.append(DFGUnit(subtree_stmts, "subtree"))

    # 3. 单个语句
    for stmt in statements:
        units.append(DFGUnit([stmt], "single"))

    # 按大小排序
    units.sort(key=lambda u: u.size, reverse=True)

    return units


def mask_dfg_span(statements: List[Statement], deps: Dict[int, Set[int]],
                  mask_ratio: float = 0.15, theta: int = 5) -> Tuple[List[int], str]:
    """
    Mask DFG单元

    Args:
        statements: 语句列表
        deps: 依赖图
        mask_ratio: mask比例
        theta: 粒度控制阈值（小于theta的单元优先mask）

    Returns:
        (masked_indices, mask_type)
    """
    total_tokens = sum(s.size for s in statements)
    m = int(total_tokens * mask_ratio)

    # 构建DFG单元
    units = build_dfg_units(statements, deps)

    # 分离大单元和小单元
    large_units = [u for u in units if u.size > theta]
    small_units = [u for u in units if u.size <= theta]

    masked_indices = set()
    m_remaining = m

    # 优先mask小单元（更细粒度）
    random.shuffle(small_units)
    for unit in small_units:
        if m_remaining <= 0:
            break
        # 避免重复mask
        if not any(idx in masked_indices for idx in unit.indices):
            if unit.size <= m_remaining:
                masked_indices.update(unit.indices)
                m_remaining -= unit.size

    # 如果还有quota，mask大单元
    if m_remaining > 0:
        random.shuffle(large_units)
        for unit in large_units:
            if m_remaining <= 0:
                break
            if not any(idx in masked_indices for idx in unit.indices):
                # 部分mask大单元
                available = [idx for idx in unit.indices if idx not in masked_indices]
                num_to_mask = min(len(available), max(1, m_remaining // 5))
                selected = random.sample(available, num_to_mask)
                masked_indices.update(selected)
                m_remaining -= sum(statements[idx].size for idx in selected)

    return list(masked_indices), "dfg_span"


def generate_masked_code(java_code: str, mask_ratio: float = 0.15,
                        theta: int = 5) -> Tuple[str, str, List[int]]:
    """
    生成masked代码

    Returns:
        (masked_code, original_code, masked_indices)
    """
    try:
        tree = javalang.parse.parse(java_code)
    except:
        return java_code, java_code, []

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
                        stmt = Statement(idx, code, stmt_node, line_num)
                        analyze_statement(stmt)
                        statements.append(stmt)
                break

    if len(statements) < 2:
        return java_code, java_code, []

    # 构建依赖图
    deps = build_dependency_graph(statements)

    # Mask DFG单元
    masked_indices, mask_type = mask_dfg_span(statements, deps, mask_ratio, theta)

    if not masked_indices:
        return java_code, java_code, []

    # 生成masked代码
    masked_lines = lines[:]
    for idx in masked_indices:
        stmt = statements[idx]
        original_line = lines[stmt.line_num]
        indent = len(original_line) - len(original_line.lstrip())
        masked_lines[stmt.line_num] = " " * indent + "<mask>"

    return '\n'.join(masked_lines), java_code, masked_indices


def test_on_sample(java_code: str):
    """测试单个样本"""
    print("=" * 80)
    print("DFG-based Span Denoising 测试")
    print("=" * 80)

    masked_code, original_code, masked_indices = generate_masked_code(
        java_code, mask_ratio=0.15, theta=5
    )

    print(f"\n【原始代码】")
    print(original_code)

    print(f"\n【Masked代码】")
    print(masked_code)

    print(f"\n【Mask信息】")
    print(f"Masked语句索引: {masked_indices}")
    num_lines = original_code.count('\n')
    print(f"Mask比例: {len(masked_indices)}/{num_lines} 行")


if __name__ == "__main__":
    # 测试样本
    test_code = """public class Solution {
public static void main(String[] args){
Scanner sc = new Scanner(System.in);
int a = sc.nextInt();
int b = sc.nextInt();
int c = a * b;
if(c % 2 == 0){
System.out.println("Even");
}else{
System.out.println("Odd");
}
}
}"""

    print("\n示例1: 小theta (更多细粒度mask)")
    print("-" * 80)
    masked_code, _, masked_idx = generate_masked_code(test_code, mask_ratio=0.3, theta=3)
    print(f"Masked indices: {masked_idx}")
    print(masked_code)

    print("\n\n示例2: 大theta (更粗粒度mask)")
    print("-" * 80)
    masked_code, _, masked_idx = generate_masked_code(test_code, mask_ratio=0.3, theta=10)
    print(f"Masked indices: {masked_idx}")
    print(masked_code)
