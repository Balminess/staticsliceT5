# 数据集说明 / Dataset Instructions

本文件夹包含CocoSlicer项目所需的数据集。

This folder contains the datasets required for the CocoSlicer project.

## 📁 数据集结构 / Dataset Structure

```
dataset/
├── codenet/          # CodeNet数据集 (from NS-slicer)
│   └── README.md     # CodeNet数据集说明
├── leetcode/         # LeetCode数据集 (from LLM-slicer)
│   └── README.md     # LeetCode数据集说明
└── README.md         # 本文件
```

## 📥 数据集来源 / Dataset Sources

### 1. CodeNet数据集
- **来源**: NS-slicer项目
- **下载链接**: https://github.com/aashishyadavally/ns-slicer/tree/main/data
- **用途**: 用于C/C++程序的静态切片训练和测试
- **存放位置**: `dataset/codenet/`

### 2. LeetCode数据集
- **来源**: LLM-slicer项目
- **下载链接**: https://github.com/kimixz/ProgramSlicingLLMs/tree/main/src/data
- **用途**: 用于Python程序的静态切片训练和测试
- **存放位置**: `dataset/leetcode/`

## 🚀 快速开始 / Quick Start

### 方法1: 手动下载
1. 访问上述数据集链接
2. 下载数据文件到相应文件夹
3. 确保文件格式为`.jsonl`

### 方法2: 使用Git克隆

```bash
# 下载CodeNet数据集
cd dataset/codenet
git clone https://github.com/aashishyadavally/ns-slicer.git temp
mv temp/data/* .
rm -rf temp

# 下载LeetCode数据集
cd ../leetcode
git clone https://github.com/kimixz/ProgramSlicingLLMs.git temp
mv temp/src/data/* .
rm -rf temp
```

## 📊 数据格式 / Data Format

两个数据集都使用JSONL格式（每行一个JSON对象），包含以下字段：

Both datasets use JSONL format (one JSON object per line) with the following fields:

### CodeNet格式:
```json
{
  "code": "程序源代码",
  "variable": "切片变量",
  "line_number": "切片行号",
  "back_lined_code": ["向后切片代码行"],
  "forward_lined_code": ["向前切片代码行"]
}
```

### LeetCode格式:
```json
{
  "code": "程序源代码",
  "variable": "切片变量",
  "line_number": "切片行号",
  "backward": "向后切片",
  "forward": "向前切片"
}
```

## ⚠️ 注意事项 / Notes

1. 数据集文件较大，下载可能需要一些时间
2. 请确保有足够的磁盘空间
3. 数据集遵循原项目的许可证
4. 使用前请阅读各数据集文件夹中的详细说明

## 📖 参考文献 / References

- **NS-slicer**: https://github.com/aashishyadavally/ns-slicer
- **LLM-slicer**: https://github.com/kimixz/ProgramSlicingLLMs
