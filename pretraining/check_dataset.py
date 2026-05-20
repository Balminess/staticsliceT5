"""
检查Python数据集的正确性
"""

import json
import ast
from collections import defaultdict

def check_dataset(filepath):
    print("=" * 80)
    print(f"检查数据集: {filepath}")
    print("=" * 80)

    # 统计信息
    total = 0
    valid = 0
    errors = defaultdict(list)

    # 字段统计
    field_stats = defaultdict(int)
    variable_types = defaultdict(int)

    with open(filepath, 'r') as f:
        for line_num, line in enumerate(f, 1):
            total += 1

            try:
                data = json.loads(line)

                # 1. 检查必需字段
                required_fields = ['eid', 'code', 'variable', 'variable_loc',
                                   'line_number', 'backward_slice', 'forward_slice',
                                   'back_lined_code', 'forward_lined_code']

                missing_fields = [f for f in required_fields if f not in data]
                if missing_fields:
                    errors['missing_fields'].append((line_num, missing_fields))
                    continue

                for field in required_fields:
                    if field in data:
                        field_stats[field] += 1

                # 2. 检查代码是否是有效的Python
                code = data['code']
                try:
                    ast.parse(code)
                except SyntaxError as e:
                    errors['invalid_python'].append((line_num, str(e)))
                    continue

                # 3. 检查变量类型
                variable = data['variable']
                variable_types[variable] += 1

                # 4. 检查variable_loc格式
                variable_loc = data['variable_loc']
                if not isinstance(variable_loc, list) or len(variable_loc) != 2:
                    errors['invalid_variable_loc'].append((line_num, variable_loc))
                    continue

                # 5. 检查line_number是否在合理范围
                line_number = data['line_number']
                code_lines = code.split('\n')
                if line_number < 0 or line_number >= len(code_lines):
                    errors['invalid_line_number'].append((line_num, f"line_number={line_number}, code_lines={len(code_lines)}"))
                    continue

                # 6. 检查backward_slice和forward_slice
                backward_slice = data['backward_slice']
                forward_slice = data['forward_slice']

                if not isinstance(backward_slice, list) or not isinstance(forward_slice, list):
                    errors['invalid_slice_format'].append((line_num, "slice不是list"))
                    continue

                # 检查slice索引是否在代码范围内
                for idx in backward_slice + forward_slice:
                    if idx < 0 or idx >= len(code_lines):
                        errors['slice_out_of_range'].append((line_num, f"idx={idx}, code_lines={len(code_lines)}"))
                        break

                # 7. 检查back_lined_code和forward_lined_code
                back_lined_code = data['back_lined_code']
                forward_lined_code = data['forward_lined_code']

                if not isinstance(back_lined_code, list) or not isinstance(forward_lined_code, list):
                    errors['invalid_lined_code'].append((line_num, "lined_code不是list"))
                    continue

                # 检查lined_code格式: "行号: 代码"
                for lined in back_lined_code + forward_lined_code:
                    if ':' not in lined:
                        errors['invalid_lined_format'].append((line_num, lined))
                        break

                # 8. 检查backward_slice和back_lined_code的一致性
                if len(backward_slice) != len(back_lined_code):
                    errors['slice_lined_mismatch'].append((line_num, f"backward: {len(backward_slice)} vs {len(back_lined_code)}"))
                    continue

                if len(forward_slice) != len(forward_lined_code):
                    errors['slice_lined_mismatch'].append((line_num, f"forward: {len(forward_slice)} vs {len(forward_lined_code)}"))
                    continue

                # 9. 检查line_number是否在backward_slice中
                if line_number not in backward_slice:
                    errors['line_not_in_backward'].append((line_num, f"line_number={line_number} not in {backward_slice}"))
                    continue

                # 10. 检查line_number是否在forward_slice中
                if line_number not in forward_slice:
                    errors['line_not_in_forward'].append((line_num, f"line_number={line_number} not in {forward_slice}"))
                    continue

                valid += 1

            except json.JSONDecodeError as e:
                errors['json_decode'].append((line_num, str(e)))
            except Exception as e:
                errors['unexpected'].append((line_num, str(e)))

    # 打印结果
    print(f"\n【总体统计】")
    print(f"总样本数: {total}")
    print(f"有效样本: {valid} ({valid/total*100:.2f}%)")
    print(f"无效样本: {total - valid} ({(total-valid)/total*100:.2f}%)")

    print(f"\n【字段统计】")
    for field, count in sorted(field_stats.items()):
        print(f"  {field}: {count}/{total} ({count/total*100:.2f}%)")

    print(f"\n【变量类型统计 (Top 20)】")
    for var, count in sorted(variable_types.items(), key=lambda x: x[1], reverse=True)[:20]:
        print(f"  {var}: {count}")

    if errors:
        print(f"\n【错误统计】")
        for error_type, error_list in sorted(errors.items()):
            print(f"\n{error_type}: {len(error_list)} 个错误")
            # 只显示前5个错误示例
            for line_num, msg in error_list[:5]:
                print(f"  行{line_num}: {msg}")
            if len(error_list) > 5:
                print(f"  ... 还有 {len(error_list) - 5} 个错误")
    else:
        print(f"\n✓ 未发现错误！")

    # 抽样检查
    print(f"\n【抽样检查】")
    with open(filepath, 'r') as f:
        lines = f.readlines()

        # 检查第1个、中间、最后一个样本
        sample_indices = [0, len(lines) // 2, len(lines) - 1]

        for idx in sample_indices:
            data = json.loads(lines[idx])
            print(f"\n样本 {idx + 1}/{len(lines)}: {data['eid']}")
            print(f"  变量: {data['variable']} at line {data['line_number']}")
            print(f"  Backward slice: {len(data['backward_slice'])} 行")
            print(f"  Forward slice: {len(data['forward_slice'])} 行")
            print(f"  代码长度: {len(data['code'].split(chr(10)))} 行")


if __name__ == "__main__":
    import sys

    filepath = "/home/pengfei/code/cocoslicer/data/train-example-codenet-python.jsonl"
    check_dataset(filepath)
