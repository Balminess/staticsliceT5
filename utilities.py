from typing import List, Any
import jsonlines
import re
import json

def show_jsonline(file_name, index):
    with open(file_name, 'r') as infile:
        for i, line in enumerate(infile):
            if i == index:
                record = json.loads(line)  
                for key, value in record.items():
                    print(f"{key}: {value}")  
                break
        else:
            print(f"Index {index} out of range in file {file_name}")

def read_jsonline(file_path: str) -> List[Any]:
    with jsonlines.open(file_path) as reader:
        return list(reader)

def remove_line_numbers(code):
    # Use regex to match and remove patterns like '0:', '1:', etc. at the beginning of lines
    cleaned_code = re.sub(r'^\d+:\s*', '', code, flags=re.MULTILINE)
    special_tokens = [
            "<code>", "</code>", "<criterion>", "</criterion>",
            "<line_number>", "</line_number>",
            "<backward>", "</backward>", "<forward>", "</forward>","<pad>","<s>", "</s>"
        ]
    text= cleaned_code.replace('\n', ' ').strip()
    for token in special_tokens:
        text = text.replace(token, "")
    return text.replace('0:', ' ').strip()
def add_line_numbers(code):
    lines = code.split("\n")
    numbered_lines = [f"{i}: {line}" for i, line in enumerate(lines)]
    return "\n".join(numbered_lines)
def extract_numbers(lined_code: str, forward: bool = True) -> List[int]:
    """
    Extract line numbers from the <forward> or <backward> section of the input string.

    Args:
        lined_code (str): A string containing <forward> or <backward> sections with line numbers.
        forward (bool): If True, extract numbers from the <forward> section. Otherwise, extract from <backward>.

    Returns:
        List[int]: A list of integers representing line numbers.
    """
    # Use re.S to enable matching across multiple lines
    if forward:
        match = re.search(r'<forward>(.*?)</forward>', lined_code, re.S)
    else:
        match = re.search(r'<backward>(.*?)</backward>', lined_code, re.S)
    
    if match:
        # Extract the content between <forward> or <backward> tags
        content = match.group(1)
        # Extract only the numbers before the colon (e.g., "1:" -> "1")
        numbers = [int(num) for num in re.findall(r'(\d+):', content)]
        return numbers
    else:
        # Return an empty list if no matching section is found
        return []
def extract_numbers_actual(lined_code: str, forward: bool = True) -> List[int]:
    """
    Extract line numbers from the <forward> or <backward> section of the input string.

    Args:
        lined_code (str): A string containing <forward> or <backward> sections with line numbers.
        forward (bool): If True, extract numbers from the <forward> section. Otherwise, extract from <backward>.

    Returns:
        List[int]: A list of integers representing line numbers.
    """
    # Use re.S to enable matching across multiple lines
 
        # Extract only the numbers before the colon (e.g., "1:" -> "1")
    numbers = [int(num) for num in re.findall(r'(\d+):', lined_code)]
    return numbers
import re

def calculate_accuracy_from_log(log_file_path):
    """
    Calculate accuracy from a log file containing Actual and Label lists in each log entry.
    Ignores the last log entry when calculating accuracy.

    """
    total_accuracy = 0
    total_records = 0

    with open(log_file_path, 'r') as log_file:
        lines = log_file.readlines()

    # Ignore the last line
    for line in lines[:-1]:
        # Extract Actual and Label lists using regular expressions
        actual_match = re.search(r"Actual: \[(.*?)\]", line)
        label_match = re.search(r"Label: \[(.*?)\]", line)

        if actual_match and label_match:
            actual = set(map(int, actual_match.group(1).split(", "))) if actual_match.group(1) else set()
            label = list(map(int, label_match.group(1).split(", "))) if label_match.group(1) else []

            if not label:  # Skip if Label is empty
                continue

            # Calculate intersection and len(label)
            intersection = actual.intersection(label)
            accuracy = len(intersection) / len(label)
            
            total_accuracy += accuracy
            total_records += 1

    # Calculate overall accuracy
    if total_records == 0:
        return 0.0  # Avoid division by zero

    return total_accuracy / total_records

def calculate_accuracy(predicted: List[int], actual: List[int]) -> float:
    """
    Calculate accuracy as the proportion of correctly matched elements.
    
    Args:
        predicted (List[int]): Predicted list of integers.
        actual (List[int]): Actual list of integers.

    Returns:
        float: Accuracy as a percentage.
    """
    if not actual:
        return 0.0
    correct = sum(1 for x in predicted if x in actual)
    return correct / len(actual) * 100
def calculate_exact_match(predicted: List[int], actual: List[int]) -> bool:
    """
    Determine if the predicted list matches the actual list exactly.
    
    Args:
        predicted (List[int]): Predicted list of integers.
        actual (List[int]): Actual list of integers.

    Returns:
        bool: True if the lists match exactly, False otherwise.
    """
    return sorted(predicted) == sorted(actual)

import json
from statistics import mean
from codebleu import calc_codebleu
import TSED   # <- your wrapper around the original paper’s code
def evaluate_old(file_path: str):
    """
    Calculate forward and backward accuracies and exact matches from a JSON Lines file.
    
    Args:
        file_path (str): Path to the JSON Lines file.
    """
    forward_accuracies = []
    backward_accuracies = []
    f_exact_matches = []
    b_exact_matches = []

    with open(file_path, 'r') as infile:
        for line in infile:
            data = json.loads(line)
            actual = data.get('actual', '')
            label = data.get('label', '')

            # Extract forward and backward numbers
            actual_forward = extract_numbers(actual, forward=True)
            label_forward = extract_numbers(label, forward=True)
            actual_backward = extract_numbers(actual, forward=False)
            label_backward = extract_numbers(label, forward=False)

            # Calculate accuracies
            forward_acc = calculate_accuracy(actual_forward, label_forward)
            backward_acc = calculate_accuracy(actual_backward, label_backward)

            # Calculate exact match
            forward_exact_match = calculate_exact_match(actual_forward, label_forward)
            backward_exact_match=calculate_exact_match(actual_backward, label_backward)

            forward_accuracies.append(forward_acc)
            backward_accuracies.append(backward_acc)
            f_exact_matches.append(forward_exact_match)
            b_exact_matches.append(backward_exact_match)

    avg_forward_acc = sum(forward_accuracies) / len(forward_accuracies) if forward_accuracies else 0.0
    avg_backward_acc = sum(backward_accuracies) / len(backward_accuracies) if backward_accuracies else 0.0
    f_exact_match_rate = sum(f_exact_matches) / len(f_exact_matches) * 100 if f_exact_matches else 0.0
    b_exact_match_rate = sum(b_exact_matches) / len(b_exact_matches) * 100 if b_exact_matches else 0.0
    print(f"Average Backward Accuracy: {avg_backward_acc:.2f}%")
    print(f"Average Forward Accuracy: {avg_forward_acc:.2f}%")
    print(f"Backward Exact Match Rate: {b_exact_match_rate:.2f}%")
    print(f"Forward Exact Match Rate: {f_exact_match_rate:.2f}%")

def evaluate(file_path: str):
    """
    Evaluate forward/backward accuracy, exact‑match,
    CodeBLEU and TSED on a JSON‑Lines file.

    JSON keys required per line:
        - 'actual'
        - 'label'
    """
    forward_accuracies = []
    backward_accuracies = []
    f_exact_matches = []
    b_exact_matches = []

    codebleus, tseds = [], []

    with open(file_path, "r") as infile:
        for line in infile:
            data = json.loads(line)
            actual = data.get('actual', '')
            label = data.get('label', '')

            # Extract forward and backward numbers
            actual_forward = extract_numbers(actual, forward=True)
            label_forward = extract_numbers(label, forward=True)
            actual_backward = extract_numbers(actual, forward=False)
            label_backward = extract_numbers(label, forward=False)
            # Calculate accuracies
            forward_acc = calculate_accuracy(actual_forward, label_forward)
            backward_acc = calculate_accuracy(actual_backward, label_backward)

            # Calculate exact match
            forward_exact_match = calculate_exact_match(actual_forward, label_forward)
            backward_exact_match=calculate_exact_match(actual_backward, label_backward)

            forward_accuracies.append(forward_acc)
            backward_accuracies.append(backward_acc)
            f_exact_matches.append(forward_exact_match)
            b_exact_matches.append(backward_exact_match)

            # ------------- CodeBLEU ----------------------
            actual = remove_line_numbers(actual)
            label = remove_line_numbers(label)
            cb_dict = calc_codebleu([label], [actual],
                                    lang="java",
                                    weights=(0.25,0.25,0.25,0.25),
                                    tokenizer=None)
            codebleus.append(cb_dict["codebleu"])

            # -------------   TSED   ----------------------
            ts_score = TSED.Calculate("java", actual, label,1.0,0.8,1.0)
            tseds.append(ts_score)

    avg_forward_acc = sum(forward_accuracies) / len(forward_accuracies) if forward_accuracies else 0.0
    avg_backward_acc = sum(backward_accuracies) / len(backward_accuracies) if backward_accuracies else 0.0
    f_exact_match_rate = sum(f_exact_matches) / len(f_exact_matches) * 100 if f_exact_matches else 0.0
    b_exact_match_rate = sum(b_exact_matches) / len(b_exact_matches) * 100 if b_exact_matches else 0.0
    print(f"Average Backward Accuracy: {avg_backward_acc:.2f}%")
    print(f"Average Forward Accuracy: {avg_forward_acc:.2f}%")
    print(f"Backward Exact Match Rate: {b_exact_match_rate:.2f}%")
    print(f"Forward Exact Match Rate: {f_exact_match_rate:.2f}%")
    print(f"Mean CodeBLEU             : {mean(codebleus):.4f}")
    print(f"Mean TSED                 : {mean(tseds):.4f}")



import random

def split_lined_code(entry):
    if ":" not in entry:
        return None, entry
    idx, line = entry.split(":", 1)
    return idx, line

def join_lined_code(idx, line):
    return f"{idx}:{line}"

def remove_class_declaration(code_lines):

    if isinstance(code_lines, str):
        lines = code_lines.strip().split('\n')
    else:
        lines = code_lines
    
    result_lines = []
    
    for i, line in enumerate(lines):
        if i == 0:  # 第一行：只保留行号
            if ':' in line:
                line_num = line.split(':', 1)[0].strip()
                result_lines.append(f"{line_num}: ")
            else:
                result_lines.append(line)
        else:  # 其他行：保持原样
            result_lines.append(line)
    
    return result_lines

    
import random

def remove_semicolons(content):
    semicolon_lines = [i for i, line in enumerate(content) if ";" in line]
    lines_to_modify = random.sample(semicolon_lines, min(3, len(semicolon_lines)))
    for i in lines_to_modify:
        content[i] = content[i].replace(";", "")
    return content


def modify_braces(content):
    brace_lines = [i for i, line in enumerate(content) if "{" in line or "}" in line]
    if not brace_lines:
        return content

    lines_to_modify = random.sample(brace_lines, min(3, len(brace_lines)))
    for i in lines_to_modify:
        content[i] = content[i].replace("{", "").replace("}", "")

    return content


def corrupt_code(code: str, mode: str) -> str:
    content = code.splitlines()
    if mode == "class":
        content = remove_class_declaration(code)
    elif mode == "semicolon":
        content = remove_semicolons(content)
    elif mode == "brace":
        content = modify_braces(content)
    else:
        return code
    return "\n".join(content)
def remove_first_class_line(code: str) -> str:
    """
    Removes the first line that contains the word 'class' from a multi-line string.

    Args:
        code (str): Multi-line code string.

    Returns:
        str: Modified code string with the first 'class' line removed.
    """
    lines = code.splitlines()
    removed = False
    result = []

    for line in lines:
        if not removed and 'class' in line:
            removed = True
            continue  # skip this line
        result.append(line)

    return "\n".join(result)


def get_lined_code(line_numbers, code_lines):
    lined_code = []
    
    for line_num in line_numbers:
        if 0 <= line_num < len(code_lines):
            lined_code.append(f"{line_num}: {code_lines[line_num]}")
    
    return "\n".join(lined_code)