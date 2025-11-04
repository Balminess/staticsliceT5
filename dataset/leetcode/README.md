# LeetCode数据集 / LeetCode Dataset

## 📖 简介 / Introduction

LeetCode数据集来自LLM-slicer项目，专门用于Python程序的静态切片任务。

The LeetCode dataset is from the LLM-slicer project, specifically designed for static slicing tasks on Python programs.

## 📥 下载 / Download

**来源 / Source**: https://github.com/kimixz/ProgramSlicingLLMs/tree/main/src/data

### 下载方法 / Download Methods

#### 方法1: 直接下载文件
访问上述链接，下载以下文件到本文件夹：
- `train.jsonl` - 训练集
- `valid.jsonl` - 验证集
- `test.jsonl` - 测试集

#### 方法2: 使用Git
```bash
cd dataset/leetcode
git clone https://github.com/kimixz/ProgramSlicingLLMs.git temp
cp temp/src/data/*.jsonl .
rm -rf temp
```

## 📊 数据格式 / Data Format

每个JSONL文件包含多条记录，每条记录格式如下：

```json
{
  "code": "完整的Python源代码",
  "variable": "切片变量名称",
  "line_number": 切片行号(整数),
  "backward": "向后切片结果",
  "forward": "向前切片结果"
}
```

### 字段说明 / Field Description

- **code**: 原始Python程序源代码（通常是LeetCode题解）
- **variable**: 要切片的变量名
- **line_number**: 切片准则所在的行号
- **backward**: 向后切片结果（影响该变量的代码）
- **forward**: 向前切片结果（被该变量影响的代码）

## 📈 数据统计 / Statistics

下载数据集后，你可以运行以下命令查看数据统计：

```bash
wc -l *.jsonl
```

## 🔧 使用方式 / Usage

在`inference_cocoslicer.py`中使用此数据集：

```python
from utilities import read_jsonline, add_line_numbers, get_lined_code

# 读取数据集
test_dataset_list = read_jsonline('dataset/leetcode/test.jsonl')

# 准备输入（完整代码模式）
test_inputs = [
    '<code>' + add_line_numbers(obj['code']) + '</code>' +
    '<criterion>' + obj['variable'] + '</criterion>' +
    '<line_number>' + str(obj['line_number']) + '</line_number>'
    for obj in test_dataset_list
]

# 准备输出
test_outputs = [
    '<backward>' + get_lined_code(obj['backward'], obj['code'].splitlines()) +
    '</backward><forward>' + get_lined_code(obj['forward'], obj['code'].splitlines()) +
    '</forward>'
    for obj in test_dataset_list
]
```

### 使用破损代码模式 / Using Corrupt Code Mode

如果需要测试不完整代码的切片能力：

```python
from utilities import corrupt_code

test_inputs = [
    '<code>' + corrupt_code(add_line_numbers(obj['code']), 'class') + '</code>' +
    '<criterion>' + obj['variable'] + '</criterion>' +
    '<line_number>' + str(obj['line_number']) + '</line_number>'
    for obj in test_dataset_list
]
```

## 💡 特点 / Features

- 基于真实的LeetCode题解
- 覆盖多种Python编程模式
- 包含类、函数、循环等多种代码结构
- 适合测试模型在实际编程场景中的表现

## 📝 许可证 / License

请遵循LLM-slicer项目的原始许可证。

Please follow the original license from the LLM-slicer project.

## 🔗 相关链接 / Related Links

- **LLM-slicer GitHub**: https://github.com/kimixz/ProgramSlicingLLMs
- **Paper**: 参见LLM-slicer项目的论文引用
