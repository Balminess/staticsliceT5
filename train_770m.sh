#!/bin/bash
# Train CodeT5+ 770M on Java + Python mixed data
# 从已有的Java checkpoint继续训练

set -e

# 激活conda环境
source /usr/local/anaconda3/etc/profile.d/conda.sh
conda activate codex

echo "=========================================="
echo "CocoSlicer - CodeT5+ 770M Training"
echo "=========================================="
echo "Environment: codex"
echo "Checkpoint: /home/pengfei/code/cocoslicer/model/codet5p_770m"
echo "Data: Java + Python (all data)"
echo "Device: GPU 0"
echo "Batch size: 4"
echo "=========================================="
echo ""

python finetune_unified.py \
  --model_type codet5p_770m \
  --language both \
  --checkpoint /home/pengfei/code/cocoslicer/model/codet5p_770m \
  --device 0 \
  --batch_size 4 \
  --learning_rate 3e-5 \
  --num_epochs 10 \
  --max_length 512

echo ""
echo "=========================================="
echo "Training Complete!"
echo "Model saved to: /home/pengfei/code/cocoslicer/model/codet5p_770m_both_continual/"
echo "=========================================="
