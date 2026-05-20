"""
评测 CodeT5+ 770M 在 Python 测试集上的性能
计算四个指标：ACC-D, Exact Match, TSED, CodeBLEU
注意：TSED 和 CodeBLEU 只在 backward 部分上计算
"""
import torch
from transformers import T5ForConditionalGeneration, AutoTokenizer
from utils import read_jsonline, add_line_numbers, extract_numbers, calculate_accuracy, calculate_exact_match, remove_line_numbers
from codebleu import calc_codebleu
import TSED
from statistics import mean
import json
from tqdm import tqdm
import re

# 配置
CHECKPOINT_PATH = "/home/pengfei/code/cocoslicer/model/codet5p_770m_both_continual/checkpoint-epoch=00-val_loss=0.0014.ckpt"
TEST_DATA_PATH = "/home/pengfei/code/cocoslicer/data/test-example-codenet-python.jsonl"
OUTPUT_JSONL = "/home/pengfei/code/cocoslicer/results/codet5p_770m_python_results_backward_only.jsonl"
DEVICE = "cuda:0"
MAX_LENGTH = 512

print("=" * 80)
print("CodeT5+ 770M Python 测试集评测")
print("=" * 80)

# 加载模型和tokenizer
print("\n加载模型...")
device = torch.device(DEVICE)

# tokenizer从HuggingFace加载（本地没有tokenizer文件）
# 使用use_fast=False避免下载问题
import os
os.environ["TRANSFORMERS_OFFLINE"] = "0"
tokenizer = AutoTokenizer.from_pretrained("Salesforce/codet5p-770m", use_fast=False, trust_remote_code=True)

# 本地模型路径
BASE_MODEL_PATH = "/home/pengfei/code/cocoslicer/model/codet5p_770m"

# 添加特殊token
special_tokens = ["<code>", "</code>", "<criterion>", "</criterion>",
                  "<line_number>", "</line_number>", "<backward>", "</backward>",
                  "<forward>", "</forward>"]
tokenizer.add_tokens(special_tokens)

# 从PyTorch Lightning checkpoint加载训练后的模型
print(f"从checkpoint加载: {CHECKPOINT_PATH}")
import warnings
warnings.filterwarnings('ignore')

# 先加载基础模型架构（从本地）
model = T5ForConditionalGeneration.from_pretrained(BASE_MODEL_PATH)
model.resize_token_embeddings(len(tokenizer))

# 加载训练后的checkpoint权重
checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)

# 提取模型state_dict (PyTorch Lightning格式)
if 'state_dict' in checkpoint:
    state_dict = checkpoint['state_dict']
    # 移除 'model.' 前缀
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith('model.'):
            new_state_dict[key[6:]] = value  # 去掉 'model.' 前缀
        else:
            new_state_dict[key] = value
    model.load_state_dict(new_state_dict, strict=False)
    print("成功加载训练后的模型权重！")
else:
    model.load_state_dict(checkpoint, strict=False)
    print("成功加载训练后的模型权重！")

model = model.to(device)
model.eval()
print("模型加载完成！")

# 加载测试数据
print(f"\n加载测试数据: {TEST_DATA_PATH}")
test_data = read_jsonline(TEST_DATA_PATH)
print(f"测试样本数量: {len(test_data)}")

# 提取backward部分的函数
def extract_backward_section(text):
    """从包含<backward>...</backward>格式的文本中提取backward部分"""
    match = re.search(r'<backward>(.*?)</backward>', text, re.DOTALL)
    if match:
        return match.group(1)
    return ""

# 推理函数
def generate_slicing(code, variable, line_number):
    input_text = f"<code>{add_line_numbers(code)}</code><criterion>{variable}</criterion><line_number>{line_number}</line_number>"
    input_ids = tokenizer(input_text, return_tensors="pt", max_length=MAX_LENGTH, truncation=True).input_ids.to(device)

    with torch.no_grad():
        output_ids = model.generate(input_ids, max_length=MAX_LENGTH)

    return tokenizer.decode(output_ids[0], skip_special_tokens=True)

# 评测
print("\n开始评测...")
forward_accuracies = []
backward_accuracies = []
f_exact_matches = []
b_exact_matches = []
codebleus = []
tseds = []

results = []

for idx, sample in enumerate(tqdm(test_data, desc="推理中")):
    # 构造输入和标签
    code = sample['code']
    variable = sample['variable']
    line_number = sample['line_number']

    label = f"<backward>{chr(10).join(sample['back_lined_code'])}</backward><forward>{chr(10).join(sample['forward_lined_code'])}</forward>"

    # 生成预测
    actual = generate_slicing(code, variable, line_number)

    # 提取行号
    actual_forward = extract_numbers(actual, forward=True)
    label_forward = extract_numbers(label, forward=True)
    actual_backward = extract_numbers(actual, forward=False)
    label_backward = extract_numbers(label, forward=False)

    # 计算准确率 (ACC-D是backward accuracy)
    forward_acc = calculate_accuracy(actual_forward, label_forward)
    backward_acc = calculate_accuracy(actual_backward, label_backward)

    # 计算精确匹配
    forward_exact_match = calculate_exact_match(actual_forward, label_forward)
    backward_exact_match = calculate_exact_match(actual_backward, label_backward)

    forward_accuracies.append(forward_acc)
    backward_accuracies.append(backward_acc)
    f_exact_matches.append(forward_exact_match)
    b_exact_matches.append(backward_exact_match)

    # 提取backward部分（只计算backward的TSED和CodeBLEU）
    actual_backward_section = extract_backward_section(actual)
    label_backward_section = extract_backward_section(label)

    # 去除行号
    actual_backward_no_lines = remove_line_numbers(actual_backward_section)
    label_backward_no_lines = remove_line_numbers(label_backward_section)

    # 计算 CodeBLEU (只在backward部分)
    cb_dict = calc_codebleu([label_backward_no_lines], [actual_backward_no_lines],
                            lang="python",
                            weights=(0.25, 0.25, 0.25, 0.25),
                            tokenizer=None)
    codebleus.append(cb_dict["codebleu"])

    # 计算 TSED (只在backward部分)
    ts_score = TSED.Calculate("python", actual_backward_no_lines, label_backward_no_lines, 1.0, 0.8, 1.0)
    tseds.append(ts_score)

    # 保存结果
    results.append({
        "index": idx,
        "actual": actual,
        "label": label,
        "backward_acc": backward_acc,
        "forward_acc": forward_acc,
        "backward_exact_match": backward_exact_match,
        "forward_exact_match": forward_exact_match,
        "codebleu": cb_dict["codebleu"],
        "tsed": ts_score
    })

# 保存详细结果
import os
os.makedirs(os.path.dirname(OUTPUT_JSONL), exist_ok=True)
with open(OUTPUT_JSONL, 'w') as f:
    for result in results:
        json.dump(result, f)
        f.write('\n')

# 计算平均指标
avg_forward_acc = sum(forward_accuracies) / len(forward_accuracies) if forward_accuracies else 0.0
avg_backward_acc = sum(backward_accuracies) / len(backward_accuracies) if backward_accuracies else 0.0
f_exact_match_rate = sum(f_exact_matches) / len(f_exact_matches) * 100 if f_exact_matches else 0.0
b_exact_match_rate = sum(b_exact_matches) / len(b_exact_matches) * 100 if b_exact_matches else 0.0
avg_codebleu = mean(codebleus) if codebleus else 0.0
avg_tsed = mean(tseds) if tseds else 0.0

# 打印结果
print("\n" + "=" * 80)
print("评测结果")
print("=" * 80)
print(f"测试样本数: {len(test_data)}")
print(f"\n四个核心指标:")
print(f"  1. ACC-D (Backward Accuracy):    {avg_backward_acc:.2f}%")
print(f"  2. Exact Match (Backward):       {b_exact_match_rate:.2f}%")
print(f"  3. TSED:                          {avg_tsed:.4f}")
print(f"  4. CodeBLEU:                      {avg_codebleu:.4f}")
print(f"\n附加指标:")
print(f"  Forward Accuracy:                {avg_forward_acc:.2f}%")
print(f"  Forward Exact Match:             {f_exact_match_rate:.2f}%")
print(f"\n详细结果已保存至: {OUTPUT_JSONL}")
print("=" * 80)
