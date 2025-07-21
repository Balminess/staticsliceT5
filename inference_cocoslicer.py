
checkpoint='model_path'
result_path = 'result_psth'
from transformers import T5ForConditionalGeneration,LogitsProcessorList
import torch
import sys
import re
from graphmatch import *
import TSED
sys.path.append('folder_path')
from utilies import read_jsonline,add_line_numbers,extract_numbers,remove_line_numbers,corrupt_code,get_lined_code
from transformers import AutoTokenizer, RobertaTokenizer
from model import T5ForConditionalGenerationWithCopyMech
from coco_constraint import ExtractiveLogitsProcessor,TSEDMonotonicConstraint
#set your device
device = torch.device("cuda:2")

#codeT5+ tokenizer
tokenizer = AutoTokenizer.from_pretrained("Salesforce/codet5p-770m")
#codeT5 tokenizer
tokenizer = RobertaTokenizer.from_pretrained('Salesforce/codet5-base')

special_tokens = ["<code>", "</code>", "<criterion>", "</criterion>", 
                  "<line_number>", "</line_number>", 
                  "<backward>", "</backward>", "<forward>", "</forward>"]

# Add special tokens to the tokenizer
special_tokens_dict = {"additional_special_tokens": special_tokens}
tokenizer.add_special_tokens(special_tokens_dict)

#Vanilla model
model = T5ForConditionalGeneration.from_pretrained(checkpoint,device_map={"": device})
#copy-enhanced model
# model = T5ForConditionalGenerationWithCopyMech.from_pretrained(checkpoint,device_map={"": device})

model.resize_token_embeddings(len(tokenizer))



def T5_slicing(code,length=256, logitprocessor=False, beam_search=False):
    """
    Generate code slicing using T5 with conditional logits processor and beam search.

    Args:
        code (str): Input code containing the original code within <code> tags.
        checker (Callable): Checker function that validates slicing correctness.
        length (int): Maximum length of the generated sequence.
        logitprocessor (bool): Whether to use the logits processor for lexing constraint.
        beam_search (bool): Whether to use beam search for syntactic constraint.

    Returns:
        str: The best valid slicing code based on TSED score or the first beam's output if none are valid.
    """
    # Extract original code using regex

    match = re.search(r"<code>(.*?)</code>", code, re.DOTALL)
    if not match:
        raise ValueError("Original code not found in the input with <code> tags.")
    original_code = remove_line_numbers(match.group(1))
    # Tokenize the input
    input_ids = tokenizer(code, return_tensors="pt").input_ids.to(device)

    # Create the logits processor list if logitprocessor is True
    logits_processor = None
    if logitprocessor:
        special_tokens = tokenizer.all_special_ids  # Include all special tokens
        allowed_processor = NoRepeatNGramLogitsProcessor(input_ids, special_tokens)
        logits_processor = LogitsProcessorList([allowed_processor])

    # Generate outputs based on beam_search flag
    if beam_search:
        output_ids = model.generate(
            input_ids=input_ids,
            max_length=length,
            logits_processor=logits_processor if logitprocessor else None,
            constrait=TSEDMonotonicConstraint,
            num_beams=3,
            num_return_sequences=3,
        )

        # Decode outputs and calculate TSED scores
        decoded_outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=False)

        return decoded_outputs[0]
    else:
        output_ids = model.generate(
            input_ids=input_ids,
            max_length=length,
            logits_processor=logits_processor if logitprocessor else None,
            early_stopping=True,
        )
        # Decode the first output (greedy search)
        decoded_output = tokenizer.decode(output_ids[0], skip_special_tokens=False)

        return decoded_output




from datasets import Dataset, DatasetDict

# test_dataset_list = read_jsonline(f'data.jsonl')

# leetcode dataset
# corropt mode to render code incomplete
# test_inputs = [
#     'code:'+add_line_numbers(obj['code'])+'\ncriterion:'+obj['variable'] +'\nline_number:' +str(obj['line_number'])
#     for obj in test_dataset_list
# ]
# complete code
# test_inputs = [
#     '<code>'+corrupt_code(add_line_numbers(obj['code']),'class')+'</code><criterion>'+obj['variable'] +'</criterion><line_number>' +str(obj['line_number'])+ '</line_number>'
#     for obj in test_dataset_list
# ]
# test_outputs = [
#     '<backward>'+get_lined_code(obj['backward'],obj['code'].splitlines()) + '</backward>><forward>' + get_lined_code(obj['forward'],obj['code'].splitlines()) + '</forward>'
#     for obj in test_dataset_list
# ]

### codenet dataset
# corropt mode to render code incomplete
# test_inputs = [
#     '<code>'+corrupt_code(add_line_numbers(obj['code']),'semicolon')+'</code><criterion>'+obj['variable'] +'</criterion><line_number>' +str(obj['line_number'])+ '</line_number>'
#     for obj in test_dataset_list
# ]

# complete code
test_inputs = [
    'code:'+add_line_numbers(obj['code'])+'\ncriterion:'+obj['variable'] +'\nline_number:' +str(obj['line_number'])
    for obj in test_dataset_list
]

test_outputs = [
    '<backward>' + '\n'.join(obj['back_lined_code']) + '</backward>><forward>' + '\n'.join(obj['forward_lined_code']) + '</forward>'
    for obj in test_dataset_list
]


dataset_dict = DatasetDict({
    'test': Dataset.from_dict({'input': test_inputs, 'output':  test_outputs}),
})

max_input_length = 256
max_target_length = 256

import random
import logging

# Configure logging
logging.basicConfig(
    filename=logging_path,  # Log file name
    level=logging.INFO,          # Logging level
    format='%(asctime)s - %(levelname)s - %(message)s'  # Log message format
)

logger = logging.getLogger()

import json
def calculate_metrics_with_logging(dataset_dict, output_file, sample_size=500):
    # sampled_dataset  = dataset_dict['test']
    test_dataset  = dataset_dict['test']
    total_samples = len(test_dataset)
    sample_size = min(sample_size, total_samples)
    sampled_dataset = test_dataset.shuffle(seed=42).select(range(sample_size))


    total_samples = len(sampled_dataset)
    sample_size = min(sample_size, total_samples)


    back_match_count = 0
    forward_match_count = 0
    exact_match_count = 0
    numeric_match_count = 0
    results = []
    checker = CodeSliceChecker()
    for idx, sample in enumerate(sampled_dataset):
        code = sample['input']
        label = sample['output']
        actual = T5_slicing(code, length=256, logitprocessor=False, beam_search=False)
   
        is_exact_match = actual == label
        if is_exact_match:
            exact_match_count += 1
        
        actual_back = extract_numbers(actual, forward=False)
        label_back = extract_numbers(label, forward=False)

        # Extract the <forward> part numbers
        actual_forward = extract_numbers(actual, forward=True)
        label_forward = extract_numbers(label, forward=True)
        actual_total=actual_back + actual_forward
        label_total=label_back + label_forward
        # Calculate exact match for backward and forward parts
        if actual_back == label_back:
            back_match_count += 1
        if actual_forward == label_forward:
            forward_match_count += 1
        if actual_total == label_total:
            numeric_match_count += 1
        is_numeric_match=actual_total == label_total

        logger.info(f"Processing {idx + 1}/{sample_size}: Exact Match: {is_exact_match}, Numeric Match: {is_numeric_match}, Actual: {actual_back}, Label: {label_back}")
        # print(f"Actual: {actual}, Label: {label}")
        # print(f"ActualIdx: {actual_total}, LabelIdx: {label_total}")
        print(f"Processing {idx + 1}/{sample_size}")

        results.append({
            "index": idx,
            "label": label,
            "actual": actual,
            "is_exact_match": is_exact_match,
            "is_numeric_match": is_numeric_match
        })
    
    with open(output_file, 'w') as f:
        for result in results:
            json.dump(result, f)
            f.write('\n')

    exact_match_rate = exact_match_count / sample_size if sample_size > 0 else 0
    numeric_match_rate = numeric_match_count / sample_size if sample_size > 0 else 0
    back_exact_match_rate = back_match_count / sample_size if sample_size > 0 else 0
    forward_exact_match_rate = forward_match_count / sample_size if sample_size > 0 else 0
    return exact_match_rate, numeric_match_rate, back_exact_match_rate, forward_exact_match_rate

output_file= result_path
exact_match_rate, numeric_match_rate, back_exact_match_rate, forward_exact_match_rate = calculate_metrics_with_logging(dataset_dict, output_file, sample_size=1000)
logger.info(f"Exact Match Rate: {exact_match_rate}, Numeric Match Rate: {numeric_match_rate},Exact_Match_Back: {back_exact_match_rate}, Exact_Match_Forward: {forward_exact_match_rate}")

