import argparse
import wandb
import torch
from datasets import Dataset, DatasetDict
from torch.utils.data import DataLoader
from transformers import (
    RobertaTokenizer,
    AutoTokenizer,
    T5ForConditionalGeneration,
    AdamW,
    get_linear_schedule_with_warmup
)
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor

# Parse command line arguments
parser = argparse.ArgumentParser(description='Fine-tune CodeT5 or CodeT5+ for program slicing')
parser.add_argument('--model_type', type=str, choices=['codet5', 'codet5p'], required=True,
                    help='Model type: codet5 or codet5p')
parser.add_argument('--device', type=int, default=0,
                    help='CUDA device number (default: 0)')
args = parser.parse_args()

# Default training hyperparameters
BATCH_SIZE = 8
EVAL_BATCH_SIZE = 4
LEARNING_RATE = 5e-5
NUM_EPOCHS = 15
WARMUP_STEPS = 1000
MAX_INPUT_LENGTH = 256
MAX_TARGET_LENGTH = 256
DATA_DIR = '/home/pengfei/code/LLaMA-Factory/slicing_dataset'
ROOT_DIR = '/home/pengfei/code/codeT5copy'
WANDB_PROJECT = 'CopyCodeT5'

# Set model-specific defaults
if args.model_type == 'codet5':
    checkpoint = "Salesforce/codet5-base-codexglue-refine-medium"
    save_dir = "/home/pengfei/code/CopyCodeT5/zipt5copy"
    wandb_name = 'zipt5copy'
else:  # codet5p
    checkpoint = "Salesforce/codet5p-220m"
    save_dir = "/home/pengfei/code/CopyCodeT5/codet5p_220m"
    wandb_name = 'CopyT5_slicing_org_v3_220m'

# Login to WandB
wandb.login()

# Set device
device = torch.device(f"cuda:{args.device}")
print(f"Using device: {device}")
print(f"Model type: {args.model_type}")
print(f"Checkpoint: {checkpoint}")

# Helper functions
def read_jsonline(filepath):
    import json
    with open(filepath, 'r') as f:
        return [json.loads(line) for line in f]

def add_line_numbers(code):
    lines = code.split("\n")
    numbered_lines = [f"{i}: {line}" for i, line in enumerate(lines)]
    return "\n".join(numbered_lines)

# Load datasets
print("Loading datasets...")
train_dataset_list = read_jsonline(f'{DATA_DIR}/train-example.jsonl')
test_dataset_list = read_jsonline(f'{DATA_DIR}/test-example.jsonl')
val_dataset_list = read_jsonline(f'{DATA_DIR}/val-example.jsonl')

# Prepare inputs and outputs
train_inputs = [
    '<code>'+add_line_numbers(obj['code'])+'</code><criterion>'+obj['variable'] +'</criterion><line_number>' +str(obj['line_number'])+ '</line_number>'
    for obj in train_dataset_list
]
train_outputs = [
    '<backward>' +  '\n'.join(obj['back_lined_code'])  + '</backward><forward>' + '\n'.join(obj['forward_lined_code']) + '</forward>'
    for obj in train_dataset_list
]

test_inputs = [
    '<code>'+add_line_numbers(obj['code'])+'</code><criterion>'+obj['variable'] +'</criterion><line_number>' +str(obj['line_number'])+ '</line_number>'
    for obj in test_dataset_list
]
test_outputs = [
    '<backward>' + '\n'.join(obj['back_lined_code']) + '</backward><forward>' + '\n'.join(obj['forward_lined_code']) + '</forward>'
    for obj in test_dataset_list
]

dev_inputs = [
    '<code>'+add_line_numbers(obj['code'])+'</code><criterion>'+obj['variable'] +'</criterion><line_number>' +str(obj['line_number'])+ '</line_number>'
    for obj in val_dataset_list
]
dev_outputs= [
    '<backward>' +  '\n'.join(obj['back_lined_code']) + '</backward><forward>' + '\n'.join(obj['forward_lined_code']) + '</forward>'
    for obj in val_dataset_list
]

# Create dataset
dataset_dict = DatasetDict({
    'train': Dataset.from_dict({'input': train_inputs, 'output': train_outputs}),
    'dev': Dataset.from_dict({'input': dev_inputs, 'output': dev_outputs}),
    'test': Dataset.from_dict({'input': test_inputs, 'output': test_outputs}),
})

example = dataset_dict['train'][0]
print("Example input:", example["input"][:100], "...")
print("Example output:", example["output"][:100], "...")

# Initialize tokenizer
print(f"Loading tokenizer for {args.model_type}...")
if args.model_type == 'codet5':
    tokenizer = RobertaTokenizer.from_pretrained("Salesforce/codet5-base")
else:  # codet5p
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)

# Add special tokens
special_tokens = ["<code>", "</code>", "<criterion>",'</criterion>','<line_number>','</line_number>','<backward>','</backward>','<forward>','</forward>']
tokenizer.add_tokens(special_tokens)

# Preprocess function
def preprocess_examples(examples):
    inputs = examples['input']
    outputs = examples['output']

    model_inputs = tokenizer(inputs, max_length=MAX_INPUT_LENGTH, padding="max_length", truncation=True)

    # Encode the summaries
    labels = tokenizer(outputs, max_length=MAX_TARGET_LENGTH, padding="max_length", truncation=True).input_ids

    # Replace padding token index by -100 for CrossEntropyLoss
    labels_with_ignore_index = []
    for labels_example in labels:
        labels_example = [label if label != 0 else -100 for label in labels_example]
        labels_with_ignore_index.append(labels_example)

    model_inputs["labels"] = labels_with_ignore_index

    return model_inputs

# Preprocess dataset
print("Preprocessing dataset...")
dataset = dataset_dict.map(preprocess_examples, batched=True)

# Create dataloaders
dataset.set_format(type="torch", columns=['input_ids', 'attention_mask', 'labels'])
train_dataloader = DataLoader(dataset['train'], shuffle=True, batch_size=BATCH_SIZE)
valid_dataloader = DataLoader(dataset['dev'], batch_size=EVAL_BATCH_SIZE)
test_dataloader = DataLoader(dataset['test'], batch_size=EVAL_BATCH_SIZE)

batch = next(iter(train_dataloader))
print("Batch keys:", batch.keys())
print("Tokenizer vocabulary size:", len(tokenizer))

# Define PyTorch Lightning model
class CodeT5(pl.LightningModule):
    def __init__(self, checkpoint, lr, num_train_epochs, warmup_steps):
        super().__init__()
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
        self.log("training_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.common_step(batch, batch_idx)
        self.log("validation_loss", loss, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx):
        loss = self.common_step(batch, batch_idx)
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

    def train_dataloader(self):
        return train_dataloader

    def val_dataloader(self):
        return valid_dataloader

    def test_dataloader(self):
        return test_dataloader

# Initialize model
print("Initializing model...")
model = CodeT5(
    checkpoint=checkpoint,
    lr=LEARNING_RATE,
    num_train_epochs=NUM_EPOCHS,
    warmup_steps=WARMUP_STEPS
)

# Setup callbacks
wandb_logger = WandbLogger(name=wandb_name, project=WANDB_PROJECT)

early_stop_callback = EarlyStopping(
    monitor='validation_loss',
    patience=3,
    strict=False,
    verbose=False,
    mode='min'
)
lr_monitor = LearningRateMonitor(logging_interval='step')

# Initialize trainer
trainer = Trainer(
    accelerator="gpu",
    devices=[args.device],
    default_root_dir=ROOT_DIR,
    logger=wandb_logger,
    callbacks=[early_stop_callback, lr_monitor]
)

# Train model
print("Starting training...")
trainer.fit(model)

# Save model
print(f"Saving model to {save_dir}...")
model.model.save_pretrained(save_dir)
print("Training complete!")
