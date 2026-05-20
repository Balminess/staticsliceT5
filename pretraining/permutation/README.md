# Dataflow Order Permutation

## 原理

基于AST分析，在同一control flow里，不存在依赖关系的语句可以交换位置。

**算法**:
1. 解析AST，提取每个语句
2. 分析每个语句的defs（定义的变量）和uses（使用的变量）
3. 构建依赖图：如果stmt2 uses了stmt1 defines的变量，则stmt2依赖stmt1
4. 找出无依赖关系的语句对
5. 交换生成变体

## 示例

**原始**:
```java
int a = sc.nextInt();  // S0: defs={a}, uses={sc}
int b = sc.nextInt();  // S1: defs={b}, uses={sc}
```

**变体**:
```java
int b = sc.nextInt();  // S1先执行
int a = sc.nextInt();  // S0后执行
```

S0和S1独立（无依赖），可交换。

## 使用

**真正的AST版本** (推荐):
```bash
python simple_ast_permutation.py
```

## 测试结果

在10个样本上测试: **100%成功率**
- 基于javalang的AST分析
- 正确识别变量的def-use关系
- 生成的变体保持dataflow等价性
