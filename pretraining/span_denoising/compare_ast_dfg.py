"""
对比AST和DFG的区别
通过实际代码运行展示两者的不同
"""

import javalang
from typing import List, Set, Dict
from collections import defaultdict


class Statement:
    def __init__(self, index: int, code: str, node, line_num: int):
        self.index = index
        self.code = code
        self.node = node
        self.line_num = line_num
        self.defs = set()
        self.uses = set()

    def __repr__(self):
        return f"S{self.index}: {self.code[:50]}"


def extract_all_variables(node, result_set: Set[str]):
    """递归提取变量"""
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


def print_ast_structure(java_code: str):
    """打印AST结构 - 语法树"""
    print("=" * 80)
    print("AST (Abstract Syntax Tree) - 抽象语法树")
    print("=" * 80)
    print("关注点: 代码的语法结构")
    print()

    tree = javalang.parse.parse(java_code)

    def print_node(node, indent=0, parent_type=""):
        prefix = "  " * indent
        node_type = type(node).__name__

        # 打印节点类型
        print(f"{prefix}├─ {node_type}", end="")

        # 打印关键信息
        if isinstance(node, javalang.tree.MethodDeclaration):
            print(f" (name: {node.name})", end="")
        elif isinstance(node, javalang.tree.LocalVariableDeclaration):
            var_names = [d.name for d in node.declarators]
            print(f" (vars: {var_names})", end="")
        elif isinstance(node, javalang.tree.MemberReference):
            print(f" (member: {node.member})", end="")
        elif isinstance(node, javalang.tree.IfStatement):
            print(f" (condition)", end="")

        print()

        # 递归子节点
        if hasattr(node, 'children'):
            for child in node.children:
                if isinstance(child, list):
                    for item in child:
                        if isinstance(item, javalang.tree.Node):
                            print_node(item, indent + 1, node_type)
                elif isinstance(child, javalang.tree.Node):
                    print_node(child, indent + 1, node_type)

    for path, node in tree:
        if isinstance(node, javalang.tree.ClassDeclaration):
            print(f"ClassDeclaration: {node.name}")
            for member in node.body:
                if isinstance(member, javalang.tree.MethodDeclaration):
                    print(f"  └─ MethodDeclaration: {member.name}")
                    if member.body:
                        for idx, stmt in enumerate(member.body):
                            print(f"      └─ Statement {idx}")
                            print_node(stmt, indent=4)
            break


def print_dfg_structure(java_code: str):
    """打印DFG结构 - 数据流图"""
    print("\n" + "=" * 80)
    print("DFG (Data Flow Graph) - 数据流图")
    print("=" * 80)
    print("关注点: 变量的定义-使用关系")
    print()

    tree = javalang.parse.parse(java_code)
    lines = java_code.split('\n')
    statements = []

    # 提取语句
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

    # 打印每个语句的def-use
    print("【语句的Def-Use分析】")
    for stmt in statements:
        print(f"\nS{stmt.index}: {stmt.code}")
        print(f"  ├─ DEFS (定义的变量): {stmt.defs if stmt.defs else '无'}")
        print(f"  └─ USES (使用的变量): {stmt.uses if stmt.uses else '无'}")

    # 构建依赖图
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

    # 打印依赖图
    print("\n" + "-" * 80)
    print("【数据依赖图】")
    for stmt in statements:
        if deps[stmt.index]:
            dep_str = ", ".join([f"S{d}" for d in sorted(deps[stmt.index])])
            print(f"S{stmt.index} 依赖于 → [{dep_str}]")
        else:
            print(f"S{stmt.index} → 无依赖")

    # 识别依赖链
    print("\n" + "-" * 80)
    print("【依赖链识别】")
    chains = []
    visited = set()

    for stmt in statements:
        if stmt.index in visited:
            continue

        chain = [stmt.index]
        current = stmt.index

        while deps[current] and len(deps[current]) == 1:
            prev = list(deps[current])[0]
            if prev not in chain:
                chain.insert(0, prev)
                current = prev
            else:
                break

        if len(chain) >= 2:
            chains.append(chain)
            visited.update(chain)

    if chains:
        for i, chain in enumerate(chains, 1):
            chain_str = " → ".join([f"S{s}" for s in chain])
            print(f"Chain {i}: {chain_str}")
    else:
        print("未发现明显的依赖链")


def compare_ast_dfg():
    """对比AST和DFG"""
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

    print("\n测试代码:")
    print("-" * 80)
    for i, line in enumerate(test_code.split('\n'), 1):
        print(f"{i:2d}: {line}")
    print()

    # 打印AST
    print_ast_structure(test_code)

    # 打印DFG
    print_dfg_structure(test_code)

    # 总结对比
    print("\n" + "=" * 80)
    print("总结: AST vs DFG")
    print("=" * 80)
    print("""
AST (抽象语法树):
  - 关注: 代码的语法结构（怎么写的）
  - 结构: 树形层次结构
  - 节点: ClassDeclaration, MethodDeclaration, IfStatement, etc.
  - 用途: 语法分析、代码格式化、语法高亮
  - 例子: if语句包含condition和then_statement两个子节点

DFG (数据流图):
  - 关注: 变量的定义-使用关系（数据怎么流动的）
  - 结构: 有向图（依赖关系）
  - 节点: 语句的def-use集合
  - 用途: 程序优化、死代码消除、变量追踪
  - 例子: "int c = a * b" 依赖于定义a和b的语句

Masking策略差异:
  - AST-T5: Mask语法子树（比如整个if块、整个方法调用）
  - DFG Span: Mask依赖链（比如a的定义和所有使用a的语句）
    """)


if __name__ == "__main__":
    compare_ast_dfg()
