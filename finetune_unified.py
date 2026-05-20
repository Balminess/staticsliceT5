"""
Unified fine-tuning script for CodeT5/CodeT5+ on Java and Python slicing data
支持从头训练或继续训练已有checkpoint
"""

import argparse
import wandb
import torch
import os
from datasets import Dataset, DatasetDict
from torch.utils.data import DataLoader
from transformers import (
    RobertaTokenizer,
    AutoTokenizer,
    T5ForConditionalGeneration,
    get_linear_schedule_with_warmup
)
from torch.optim import AdamW
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint

# Parse command line arguments
parser = argparse.ArgumentParser(description='Unified fine-tuning for CodeT5/CodeT5+ on slicing tasks')
parser.add_argument('--model_type', type=str, choices=['codet5', 'codet5p_220m', 'codet5p_770m'], required=True,
                    help='Model type: codet5, codet5p_220m, or codet5p_770m')
parser.add_argument('--language', type=str, choices=['java', 'python', 'both'], default='both',
                    help='Training language: java, python, or both (default: both for better generalization)')
parser.add_argument('--checkpoint', type=str, default='auto',
                    help='Path to checkpoint (default: auto, use existing Java checkpoint)')
parser.add_argument('--device', type=int, default=0,
                    help='CUDA device number (default: 0)')
parser.add_argument('--batch_size', type=int, default=8,
                    help='Training batch size (default: 8)')
parser.add_argument('--eval_batch_size', type=int, default=4,
                    help='Evaluation batch size (default: 4)')
parser.add_argument('--learning_rate', type=float, default=5e-5,
                    help='Learning rate (default: 5e-5)')
parser.add_argument('--num_epochs', type=int, default=10,
                    help='Number of training epochs (default: 10)')
parser.add_argument('--warmup_steps', type=int, default=500,
                    help='Number of warmup steps (default: 500)')
parser.add_argument('--max_length', type=int, default=512,
                    help='Max sequence length (default: 512)')
parser.add_argument('--output_dir', type=str, default=None,
                    help='Output directory for model (default: auto-generated)')
parser.add_argument('--resume_from_checkpoint', type=str, default=None,
                    help='Path to checkpoint to resume training from (default: None)')
args = parser.parse_args()

# Training hyperparameters
BATCH_SIZE = args.batch_size
EVAL_BATCH_SIZE = args.eval_batch_size
LEARNING_RATE = args.learning_rate
NUM_EPOCHS = args.num_epochs
WARMUP_STEPS = args.warmup_steps
MAX_INPUT_LENGTH = args.max_length
MAX_TARGET_LENGTH = args.max_length
DATA_DIR = '/home/pengfei/code/cocoslicer/data'
ROOT_DIR = '/home/pengfei/code/cocoslicer/training_logs'
WANDB_PROJECT = 'CocoSlicer-Unified'

# Determine checkpoint and save paths
if args.checkpoint == 'auto':
    # 自动使用已有的Java checkpoint（推荐）
    mode = "continual"
    if args.model_type == 'codet5':
        checkpoint = "/home/pengfei/code/CopyCodeT5/zipt5copy"
    elif args.model_type == 'codet5p_220m':
        checkpoint = "/home/pengfei/code/cocoslicer/model/codet5p_220m"
    else:  # codet5p_770m
        checkpoint = "/home/pengfei/code/cocoslicer/model/codet5p_770m"
elif args.checkpoint == 'scratch':
    # 从头训练模式
    mode = "scratch"
    if args.model_type == 'codet5':
        checkpoint = "Salesforce/codet5-base"
    elif args.model_type == 'codet5p_220m':
        checkpoint = "Salesforce/codet5p-220m"
    else:  # codet5p_770m
        checkpoint = "Salesforce/codet5p-770m"
else:
    # 使用指定的checkpoint
    checkpoint = args.checkpoint
    mode = "continual"

# Auto-generate output directory if not specified
if args.output_dir:
    save_dir = args.output_dir
else:
    save_dir = f"/home/pengfei/code/cocoslicer/model/{args.model_type}_{args.language}_{mode}"

# WandB name
wandb_name = f'{args.model_type}-{args.language}-{mode}'

print("=" * 80)
print("Fine-tuning Configuration")
print("=" * 80)
print(f"Model type: {args.model_type}")
print(f"Language: {args.language}")
print(f"Training mode: {mode}")
print(f"Checkpoint: {checkpoint}")
print(f"Save to: {save_dir}")
print(f"Device: cuda:{args.device}")
print(f"Batch size: {BATCH_SIZE}")
print(f"Learning rate: {LEARNING_RATE}")
print(f"Epochs: {NUM_EPOCHS}")
print(f"Max length: {MAX_INPUT_LENGTH}")
print("=" * 80)

# Login to WandB
wandb.login()

# Set device
device = torch.device(f"cuda:{args.device}")

# Helper functions
def read_jsonline(filepath):
    import json
    with open(filepath, 'r') as f:
        return [json.loads(line) for line in f]

def add_line_numbers(code):
    lines = code.split("\n")
    numbered_lines = [f"{i}: {line}" for i, line in enumerate(lines)]
    return "\n".join(numbered_lines)

# Load datasets based on language
print("\nLoading datasets...")
train_dataset_list = []
val_dataset_list = []
test_dataset_list = []

if args.language in ['java', 'both']:
    print("Loading Java datasets...")
    java_train = read_jsonline(f'{DATA_DIR}/train-example-codenet-java.jsonl')
    java_val = read_jsonline(f'{DATA_DIR}/val-example-codenet-java.jsonl')
    java_test = read_jsonline(f'{DATA_DIR}/test-example-codenet-java.jsonl')
    train_dataset_list.extend(java_train)
    val_dataset_list.extend(java_val)
    test_dataset_list.extend(java_test)
    print(f"  Java - Train: {len(java_train)}, Val: {len(java_val)}, Test: {len(java_test)}")

if args.language in ['python', 'both']:
    print("Loading Python datasets...")
    python_train = read_jsonline(f'{DATA_DIR}/train-example-codenet-python.jsonl')
    python_val = read_jsonline(f'{DATA_DIR}/val-example-codenet-python.jsonl')
    python_test = read_jsonline(f'{DATA_DIR}/test-example-codenet-python.jsonl')
    train_dataset_list.extend(python_train)
    val_dataset_list.extend(python_val)
    test_dataset_list.extend(python_test)
    print(f"  Python - Train: {len(python_train)}, Val: {len(python_val)}, Test: {len(python_test)}")

print(f"\nTotal samples - Train: {len(train_dataset_list)}, Val: {len(val_dataset_list)}, Test: {len(test_dataset_list)}")

# Prepare inputs and outputs
train_inputs = [
    '<code>'+add_line_numbers(obj['code'])+'</code><criterion>'+obj['variable'] +'</criterion><line_number>' +str(obj['line_number'])+ '</line_number>'
    for obj in train_dataset_list
]
train_outputs = [
    '<backward>' +  '\n'.join(obj['back_lined_code'])  + '</backward><forward>' + '\n'.join(obj['forward_lined_code']) + '</forward>'
    for obj in train_dataset_list
]

val_inputs = [
    '<code>'+add_line_numbers(obj['code'])+'</code><criterion>'+obj['variable'] +'</criterion><line_number>' +str(obj['line_number'])+ '</line_number>'
    for obj in val_dataset_list
]
val_outputs = [
    '<backward>' +  '\n'.join(obj['back_lined_code']) + '</backward><forward>' + '\n'.join(obj['forward_lined_code']) + '</forward>'
    for obj in val_dataset_list
]

test_inputs = [
    '<code>'+add_line_numbers(obj['code'])+'</code><criterion>'+obj['variable'] +'</criterion><line_number>' +str(obj['line_number'])+ '</line_number>'
    for obj in test_dataset_list
]
test_outputs = [
    '<backward>' + '\n'.join(obj['back_lined_code']) + '</backward><forward>' + '\n'.join(obj['forward_lined_code']) + '</forward>'
    for obj in test_dataset_list
]

# Create dataset
dataset_dict = DatasetDict({
    'train': Dataset.from_dict({'input': train_inputs, 'output': train_outputs}),
    'val': Dataset.from_dict({'input': val_inputs, 'output': val_outputs}),
    'test': Dataset.from_dict({'input': test_inputs, 'output': test_outputs}),
})

example = dataset_dict['train'][0]
print("\nExample input:", example["input"][:150], "...")
print("Example output:", example["output"][:150], "...")

# Initialize tokenizer
print(f"\nLoading tokenizer for {args.model_type}...")
if args.model_type == 'codet5':
    tokenizer = RobertaTokenizer.from_pretrained("Salesforce/codet5-base")
elif args.model_type == 'codet5p_220m':
    tokenizer = AutoTokenizer.from_pretrained("Salesforce/codet5p-220m")
else:  # codet5p_770m
    tokenizer = AutoTokenizer.from_pretrained("Salesforce/codet5p-770m")

# Add special tokens
special_tokens = ["<code>", "</code>", "<criterion>",'</criterion>','<line_number>','</line_number>','<backward>','</backward>','<forward>','</forward>']
num_added = tokenizer.add_tokens(special_tokens)
print(f"Added {num_added} special tokens")

# Preprocess function
def preprocess_examples(examples):
    inputs = examples['input']
    outputs = examples['output']

    model_inputs = tokenizer(inputs, max_length=MAX_INPUT_LENGTH, padding="max_length", truncation=True)
    labels = tokenizer(outputs, max_length=MAX_TARGET_LENGTH, padding="max_length", truncation=True).input_ids

    # Replace padding token index by -100 for CrossEntropyLoss
    labels_with_ignore_index = []
    for labels_example in labels:
        labels_example = [label if label != 0 else -100 for label in labels_example]
        labels_with_ignore_index.append(labels_example)

    model_inputs["labels"] = labels_with_ignore_index
    return model_inputs

# Preprocess dataset
print("\nPreprocessing dataset...")
dataset = dataset_dict.map(preprocess_examples, batched=True)

# Create dataloaders
dataset.set_format(type="torch", columns=['input_ids', 'attention_mask', 'labels'])
train_dataloader = DataLoader(dataset['train'], shuffle=True, batch_size=BATCH_SIZE)
valid_dataloader = DataLoader(dataset['val'], batch_size=EVAL_BATCH_SIZE)
test_dataloader = DataLoader(dataset['test'], batch_size=EVAL_BATCH_SIZE)

print(f"Train batches: {len(train_dataloader)}")
print(f"Val batches: {len(valid_dataloader)}")
print(f"Test batches: {len(test_dataloader)}")
print(f"Tokenizer vocabulary size: {len(tokenizer)}")

# Define PyTorch Lightning model
class CodeT5Slicer(pl.LightningModule):
    def __init__(self, checkpoint, lr, num_train_epochs, warmup_steps):
        super().__init__()
        print(f"\nLoading model from: {checkpoint}")
        self.model = T5ForConditionalGeneration.from_pretrained(checkpoint).to(device)
        self.model.resize_token_embeddings(len(tokenizer))
        self.save_hyperparameters()

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        return outputs

    def common_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss = outputs.loss
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.common_step(batch, batch_idx)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.common_step(batch, batch_idx)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        loss = self.common_step(batch, batch_idx)
        self.log("test_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.hparams.lr)
        num_train_optimization_steps = self.hparams.num_train_epochs * len(train_dataloader)
        lr_scheduler = {
            'scheduler': get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=self.hparams.warmup_steps,
                num_training_steps=num_train_optimization_steps
            ),
            'name': 'learning_rate',
            'interval': 'step',
            'frequency': 1
        }
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}

# Initialize model
print("\nInitializing model...")
model = CodeT5Slicer(
    checkpoint=checkpoint,
    lr=LEARNING_RATE,
    num_train_epochs=NUM_EPOCHS,
    warmup_steps=WARMUP_STEPS
)

# Setup callbacks
wandb_logger = WandbLogger(name=wandb_name, project=WANDB_PROJECT)

early_stop_callback = EarlyStopping(
    monitor='val_loss',
    patience=3,
    strict=False,
    verbose=True,
    mode='min'
)

lr_monitor = LearningRateMonitor(logging_interval='step')

checkpoint_callback = ModelCheckpoint(
    dirpath=save_dir,
    filename='checkpoint-{epoch:02d}-{val_loss:.4f}',
    monitor='val_loss',
    mode='min',
    save_top_k=2,
    save_last=True
)

# Initialize trainer
trainer = Trainer(
    accelerator="gpu",
    devices=[args.device],
    max_epochs=NUM_EPOCHS,
    default_root_dir=ROOT_DIR,
    logger=wandb_logger,
    callbacks=[early_stop_callback, lr_monitor, checkpoint_callback],
    log_every_n_steps=50,
    val_check_interval=0.5,
)

# Train model
print("\n" + "=" * 80)
if args.resume_from_checkpoint:
    print(f"Resuming training from checkpoint: {args.resume_from_checkpoint}")
print(f"Starting training ({mode} mode)...")
print("=" * 80)
trainer.fit(model, train_dataloader, valid_dataloader, ckpt_path=args.resume_from_checkpoint)

# Save final model
print(f"\nSaving final model to {save_dir}...")
os.makedirs(save_dir, exist_ok=True)
model.model.save_pretrained(save_dir)
tokenizer.save_pretrained(save_dir)
print("Training complete!")

# Test
print("\n" + "=" * 80)
print("Running test evaluation...")
print("=" * 80)
trainer.test(model, test_dataloader)

print(f"\nAll done! Model saved to: {save_dir}")
