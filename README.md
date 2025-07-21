# CocoSlicer

**CocoSlicer** is a static slicing-specific model based on the T5 architecture. It enhances the slicing quality of generated code by integrating constraint decoding and a copy mechanism. This repository provides training, inference, and evaluation scripts tailored for static code slicing.

## 🔗 Baseline Repositories

- **NS-slicer** (CodeNet Dataset):  
  https://github.com/aashishyadavally/ns-slicer

- **LLM-slicer** (LeetCode Dataset):  
  https://github.com/kimixz/ProgramSlicingLLMs

## 📚 Datasets

- **CodeNet** (from NS-slicer):  
  https://github.com/aashishyadavally/ns-slicer/tree/main/data

- **LeetCode** (from LLM-slicer):  
  https://github.com/kimixz/ProgramSlicingLLMs/tree/main/src/data

## 📏 Metrics

The evaluation metric **TSED** (Tree-based Structural Edit Distance) is adapted from:  
https://github.com/Etamin/TSED

## 🧱 Framework Overview

### 🔧 Fine-tuning

- **File**: `finetune.py`  
  Fine-tune CodeT5(+) models using PyTorch Lightning.

### 🔍 Inference

- **File**: `inference_cocoslicer.py`  
  Inference script for CocoSlicer. Supports constrained decoding with two modes:
  - Lexical constraints (via Hugging Face `LogitsProcessor`)
  - Syntactic constraints (via Hugging Face's `Constraint` utilities)

### 📊 Evaluation

- **File**: `Evaluate.py`  
  Implements four evaluation metrics as defined in the original papers.

### 📁 Result

- Folder to store the slicing predictions and evaluation outputs.

## 🧠 CocoSlicer Model Components

- **Copy Mechanism**  
  Implemented in `model.py`, extending T5 with copying ability for token-level preservation in slices.

- **Constraint Decoding**  
  Implemented in `Coco_Constraint.py`:
  - Lexical constraints: restrict output tokens using token-level rules.
  - Syntactic constraints: enforce structural correctness via generation utilities.

## 📁 Directory Structure
```
├── finetune.py              # Fine-tuning script
├── inference_cocoslicer.py  # Inference with optional constraints
├── evaluate.py              # Evaluation script with TSED, EM, F1
├── model.py                 # Modified T5 with copy mechanism
├── coco_constraint.py       # Lexical & syntactic decoding constraints
├── result/                  # Output folder for prediction results
├── data/  
```

## 📌 Citation
@article{ns-slicer,
  title={NS-Slicer: Neural Program Slicing},
  author={Yadavally, Aashish and others},
  journal={GitHub Repository},
  year={2023},
  url={https://github.com/aashishyadavally/ns-slicer}
}

@article{llm-slicer,
  title={Program Slicing with Large Language Models},
  author={Kim, I. and others},
  journal={GitHub Repository},
  year={2023},
  url={https://github.com/kimixz/ProgramSlicingLLMs}
}

@article{tsed,
  title={Tree-based Structural Edit Distance (TSED)},
  author={Etamin},
  year={2022},
  url={https://github.com/Etamin/TSED}
}