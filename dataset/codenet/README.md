# CodeNet数据集 / CodeNet Dataset

## 📖 简介 / Introduction

CodeNet数据集来自NS-slicer项目，专门用于C/C++程序的静态切片任务。

The CodeNet dataset is from the NS-slicer project, specifically designed for static slicing tasks on C/C++ programs.

## 📥 下载 / Download

**来源 / Source**: https://github.com/aashishyadavally/ns-slicer/tree/main/data

### 下载方法 / Download Methods

#### 方法1: 直接下载文件
访问上述链接，下载以下文件到本文件夹：
- `train.jsonl` - 训练集
- `valid.jsonl` - 验证集
- `test.jsonl` - 测试集

#### 方法2: 使用Git
```bash
cd dataset/codenet
git clone https://github.com/aashishyadavally/ns-slicer.git temp
cp temp/data/*.jsonl .
rm -rf temp
```

## 📊 数据格式 / Data Format

每个JSONL文件包含多条记录，每条记录格式如下：

```json
{
  "code": "完整的C/C++源代码",
  "variable": "切片变量名称",
  "line_number": 切片行号(整数),
  "back_lined_code": ["向后切片的代码行列表"],
  "forward_lined_code": ["向前切片的代码行列表"]
}
```

### 字段说明 / Field Description

- **code**: 原始程序源代码
- **variable**: 要切片的变量名
- **line_number**: 切片准则所在的行号
- **back_lined_code**: 向后切片结果（影响该变量的代码行）
- **forward_lined_code**: 向前切片结果（被该变量影响的代码行）

## 📈 数据统计 / Statistics

下载数据集后，你可以运行以下命令查看数据统计：

```bash
wc -l *.jsonl
```

## 🔧 使用方式 / Usage

在`inference_cocoslicer.py`中使用此数据集：

```python
from utilities import read_jsonline, add_line_numbers

# 读取数据集
test_dataset_list = read_jsonline('dataset/codenet/test.jsonl')

# 准备输入
test_inputs = [
    'code:' + add_line_numbers(obj['code']) +
    '\ncriterion:' + obj['variable'] +
    '\nline_number:' + str(obj['line_number'])
    for obj in test_dataset_list
]

# 准备输出
test_outputs = [
    '<backward>' + '\n'.join(obj['back_lined_code']) +
    '</backward><forward>' + '\n'.join(obj['forward_lined_code']) +
    '</forward>'
    for obj in test_dataset_list
]
```

## 📝 许可证 / License

请遵循NS-slicer项目的原始许可证。

Please follow the original license from the NS-slicer project.

## 🔗 相关链接 / Related Links

- **NS-slicer GitHub**: https://github.com/aashishyadavally/ns-slicer
- **Paper**: 参见NS-slicer项目的论文引用
