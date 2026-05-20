"""
直观对比AST-based和DFG-based的masking策略
"""


def demo_ast_masking():
    """演示AST-based masking"""
    print("=" * 80)
    print("AST-based Masking (类似AST-T5)")
    print("=" * 80)
    print("策略: Mask整个语法子树\n")

    code = """Scanner sc = new Scanner(System.in);
int a = sc.nextInt();
int b = sc.nextInt();
int c = a * b;
if(c % 2 == 0){
    System.out.println("Even");
}else{
    System.out.println("Odd");
}"""

    print("原始代码:")
    print(code)

    print("\n" + "-" * 80)
    print("AST视角: 识别语法子树")
    print("-" * 80)
    print("""
MethodInvocation (调用nextInt)
  ├─ qualifier: sc
  └─ method: nextInt

BinaryOperation (二元运算)
  ├─ left: a
  ├─ operator: *
  └─ right: b

IfStatement (if语句块)
  ├─ condition: c % 2 == 0
  ├─ then: println("Even")
  └─ else: println("Odd")
    """)

    print("-" * 80)
    print("AST Masking示例1: Mask整个if块")
    print("-" * 80)
    masked = """Scanner sc = new Scanner(System.in);
int a = sc.nextInt();
int b = sc.nextInt();
int c = a * b;
<mask>"""
    print(masked)

    print("\n" + "-" * 80)
    print("AST Masking示例2: Mask方法调用")
    print("-" * 80)
    masked = """Scanner sc = new Scanner(System.in);
int a = <mask>;
int b = sc.nextInt();
int c = a * b;
if(c % 2 == 0){
    System.out.println("Even");
}else{
    System.out.println("Odd");
}"""
    print(masked)


def demo_dfg_masking():
    """演示DFG-based masking"""
    print("\n\n" + "=" * 80)
    print("DFG-based Masking (数据流图)")
    print("=" * 80)
    print("策略: Mask依赖链或依赖子树\n")

    code = """Scanner sc = new Scanner(System.in);
int a = sc.nextInt();
int b = sc.nextInt();
int c = a * b;
if(c % 2 == 0){
    System.out.println("Even");
}else{
    System.out.println("Odd");
}"""

    print("原始代码:")
    print(code)

    print("\n" + "-" * 80)
    print("DFG视角: 识别数据依赖")
    print("-" * 80)
    print("""
依赖链1: sc → a
  S0: Scanner sc = new Scanner(System.in);  [定义 sc]
  S1: int a = sc.nextInt();                  [使用 sc, 定义 a]

依赖链2: sc → b
  S0: Scanner sc = new Scanner(System.in);  [定义 sc]
  S2: int b = sc.nextInt();                  [使用 sc, 定义 b]

依赖子树: a, b → c → if
  S1: int a = sc.nextInt();                  [定义 a]
  S2: int b = sc.nextInt();                  [定义 b]
  S3: int c = a * b;                         [使用 a,b, 定义 c]
  S4: if(c % 2 == 0)...                      [使用 c]
    """)

    print("-" * 80)
    print("DFG Masking示例1: Mask依赖链 (sc → a)")
    print("-" * 80)
    masked = """<mask>
<mask>
int b = sc.nextInt();
int c = a * b;
if(c % 2 == 0){
    System.out.println("Even");
}else{
    System.out.println("Odd");
}"""
    print(masked)
    print("\n解释: Mask了sc的定义和a的定义（因为a依赖sc）")

    print("\n" + "-" * 80)
    print("DFG Masking示例2: Mask依赖子树 (a,b → c)")
    print("-" * 80)
    masked = """Scanner sc = new Scanner(System.in);
<mask>
<mask>
<mask>
if(c % 2 == 0){
    System.out.println("Even");
}else{
    System.out.println("Odd");
}"""
    print(masked)
    print("\n解释: Mask了c的完整依赖链（a, b, c三个语句）")


def compare_strategies():
    """对比两种策略"""
    print("\n\n" + "=" * 80)
    print("核心区别总结")
    print("=" * 80)

    print("""
┌─────────────────┬──────────────────────┬──────────────────────┐
│     特性        │    AST-T5           │    DFG Span          │
├─────────────────┼──────────────────────┼──────────────────────┤
│ 关注点          │ 语法结构             │ 数据流               │
├─────────────────┼──────────────────────┼──────────────────────┤
│ Mask单位        │ 语法子树             │ 依赖链/子树          │
├─────────────────┼──────────────────────┼──────────────────────┤
│ 示例            │ 整个if块             │ 变量定义和使用       │
│                 │ 整个方法调用         │ 完整依赖链           │
├─────────────────┼──────────────────────┼──────────────────────┤
│ 学习目标        │ 语法规则             │ 数据依赖关系         │
│                 │ 代码结构             │ 变量传播             │
├─────────────────┼──────────────────────┼──────────────────────┤
│ 适用任务        │ 代码生成             │ 程序理解             │
│                 │ 语法补全             │ 漏洞检测             │
│                 │                      │ 变量追踪             │
└─────────────────┴──────────────────────┴──────────────────────┘

具体例子对比:

代码: int c = a * b;

AST视角:
  └─ LocalVariableDeclaration
      ├─ type: int
      ├─ declarator: c
      └─ initializer: BinaryOperation
          ├─ left: a
          ├─ operator: *
          └─ right: b

  Mask策略: Mask整个BinaryOperation节点 → "int c = <mask>;"

DFG视角:
  S3: int c = a * b;
      ├─ DEFS: {c}
      ├─ USES: {a, b}
      └─ 依赖: S1(定义a), S2(定义b)

  Mask策略: Mask依赖链 → Mask S1, S2, S3 三条语句

结论:
  - AST-T5: 适合学习"怎么写代码"（语法）
  - DFG Span: 适合学习"数据怎么流动"（语义）
    """)


if __name__ == "__main__":
    demo_ast_masking()
    demo_dfg_masking()
    compare_strategies()
