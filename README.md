# SliceFormer

**SliceFormer** is a neural program slicer based on CodeT5+ 770M. Given a slicing criterion (variable + line number), it performs backward and forward program slicing via sequence-to-sequence generation. SliceFormer enables **lossless code compression** for retrieval-augmented generation (RAG) by extracting only the def-use chain of query-relevant variables.

## Paper

The full paper (ACL 2026) is available on arXiv:  
[https://arxiv.org/abs/2604.26961](https://arxiv.org/abs/2604.26961)

## Pretrained Model

The pretrained SliceFormer model is available on Hugging Face:  
[https://huggingface.co/Balminess/slicerformer](https://huggingface.co/Balminess/slicerformer)

## Model Architecture

- **Base model**: Salesforce/codet5p-770m (encoder-decoder seq2seq)
- **Input format**: `<code>{numbered_code}</code><criterion>{variable}</criterion><line_number>{line_no}</line_number>`
- **Output format**: `<backward>{slice_lines}</backward><forward>{slice_lines}</forward>`

## Metrics

| Language | Acc-D | ExactMatch | CodeBLEU | TSED |
|----------|-------|------------|----------|------|
| Java     | 98.78% | 92.20%    | 93.23%   | 97.68% |
| Python   | 90.85% | 83.15%    | 85.35%   | 89.74% |

- **ExactMatch**: predicted backward slice line numbers match reference exactly
- **Acc-D**: recall of correct backward slice lines
- **CodeBLEU**: code quality of predicted slice
- **TSED**: Tree Similarity Edit Distance

## Baseline Repositories

- **NS-Slicer** (CodeNet Dataset):  
  https://github.com/aashishyadavally/ns-slicer

- **LLM-Slicer** (LeetCode Dataset):  
  https://github.com/kimixz/ProgramSlicingLLMs

## Datasets

- **CodeNet** (from NS-Slicer):  
  https://github.com/aashishyadavally/ns-slicer/tree/main/data

- **LeetCode** (from LLM-Slicer):  
  https://github.com/kimixz/ProgramSlicingLLMs/tree/main/src/data

## TSED Metric

The evaluation metric **TSED** (Tree-based Structural Edit Distance) is adapted from:  
https://github.com/Etamin/TSED

## Directory Structure

```text
├── finetune_slicing.py         # Fine-tuning script (PyTorch Lightning)
├── finetune_unified.py         # Unified fine-tuning for both languages
├── inference_cocoslicer.py     # Inference with optional constraint decoding
├── evaluate.py                 # Evaluation script (TSED, EM, Acc-D, CodeBLEU)
├── utils.py                    # Utility functions (extract_numbers, metrics)
├── TSED.py                     # Tree Similarity Edit Distance implementation
├── coco_constraint.py          # Lexical & syntactic decoding constraints
├── data/                       # Train/val/test JSONL files (Java & Python)
├── model/                      # Model checkpoints (HuggingFace format)
├── result/                     # Prediction outputs and evaluation results
├── paper/                      # LaTeX source for the paper
```

## Citation

```bibtex
@article{sliceformer,
  title={SliceFormer: Neural Program Slicing via Sequence-to-Sequence Generation},
  author={...},
  journal={arXiv preprint arXiv:2604.26961},
  year={2026},
  url={https://arxiv.org/abs/2604.26961}
}
```
