# DFG-based Span Denoising

## 原理

Mask代码中对应DFG（数据流图）有意义单元的spans，从单个表达式到整个依赖链。

**类比AST-T5**:
- AST-T5: Mask AST子树（语法结构）
- DFG Span: Mask DFG单元（数据流结构）

## DFG单元类型

### 1. 单个语句 (Single Statement)
```java
int a = sc.nextInt();  // 独立的单个语句
```

### 2. 依赖链 (Dependency Chain)
```java
int a = sc.nextInt();      // S0
int b = a + 1;             // S1 依赖 S0
int c = b * 2;             // S2 依赖 S1
// 依赖链: S0 -> S1 -> S2
```

### 3. 依赖子树 (Dependency Subtree)
```java
int a = sc.nextInt();      // S0
int b = sc.nextInt();      // S1
int c = a * b;             // S2 依赖 S0, S1
int d = c + 1;             // S3 依赖 S2
// 子树: {S0, S1, S2} 是 S3 的依赖子树
```

## 算法流程

### 1. 构建DFG
```python
# 分析每个语句的defs和uses
stmt.defs = {'a'}      # 定义的变量
stmt.uses = {'sc'}     # 使用的变量

# 构建依赖图
deps[S1] = {S0}  # S1依赖S0
```

### 2. 识别DFG单元
```python
units = [
    DFGUnit([S0, S1, S2], "chain"),      # 依赖链
    DFGUnit([S0, S1, S2, S3], "subtree"), # 依赖子树
    DFGUnit([S0], "single"),              # 单个语句
]
```

### 3. Masking策略

**参数**:
- `mask_ratio`: 要mask的代码比例（默认15%）
- `theta`: 粒度控制阈值（默认5 tokens）

**策略**:
```python
# 1. 计算mask quota
m = total_tokens * mask_ratio

# 2. 分离大小单元
large_units = [u for u in units if u.size > theta]
small_units = [u for u in units if u.size <= theta]

# 3. 优先mask小单元（细粒度）
for unit in shuffle(small_units):
    if unit.size <= m_remaining:
        mask(unit)
        m_remaining -= unit.size

# 4. 如果还有quota，部分mask大单元
for unit in shuffle(large_units):
    mask_partial(unit, m_remaining)
```

## 示例

### 输入代码
```java
public class Solution {
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
}
```

### 识别的DFG单元
```
Chain[15tok]: [S0->S1->S3]  (Scanner -> a -> c)
Chain[15tok]: [S0->S2->S3]  (Scanner -> b -> c)
Subtree[20tok]: [S0,S1,S2,S3] (c的完整依赖)
Single[5tok]: S0 (Scanner声明)
Single[4tok]: S1 (读a)
Single[4tok]: S2 (读b)
```

### Masked输出
```java
public class Solution {
public static void main(String[] args){
<mask>
int a = sc.nextInt();
<mask>
int c = a * b;
if(c % 2 == 0){
System.out.println("Even");
}else{
System.out.println("Odd");
}
}
}
```

## 对比AST-T5

| 特性 | AST-T5 | DFG Span |
|------|--------|----------|
| 基础结构 | AST（语法树） | DFG（数据流图） |
| Mask单元 | 语法子树 | 依赖链/子树 |
| 关注点 | 语法结构 | 数据流依赖 |
| 粒度 | 表达式 -> 函数体 | 表达式 -> 依赖链 |

## 优势

1. **语义相关**: Mask的是有数据依赖关系的代码单元
2. **灵活粒度**: 从单个语句到整个依赖链
3. **数据流感知**: 模型学习理解变量的定义和使用关系
4. **适合预训练**: 强制模型推理被mask的数据流

## 使用

```bash
python dfg_span_masking.py
```
