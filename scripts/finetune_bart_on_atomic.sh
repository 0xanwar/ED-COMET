#!/bin/bash
# Complete workflow for finetuning BART on ATOMIC2020

set -e  # Exit on error

echo "=================================================="
echo "BART Finetuning on ATOMIC2020 - Complete Workflow"
echo "=================================================="

# Configuration
ATOMIC_TSV_DIR="./atomic2020_data-feb2021"  # Directory with train.tsv, dev.tsv, test.tsv
ATOMIC_BART_DATA="./data/atomic_bart"        # Output directory for .source/.target files
HF_BART_CHECKPOINT="./checkpoints/bart_hf"   # Your converted HuggingFace BART checkpoint
OUTPUT_DIR="./results/bart_atomic_finetune"  # Output directory for finetuned model

# Training hyperparameters
LEARNING_RATE=1e-5
TRAIN_BATCH_SIZE=32
EVAL_BATCH_SIZE=32
NUM_EPOCHS=10
MAX_SOURCE_LENGTH=48
MAX_TARGET_LENGTH=24
GPUS=1

# ===== STEP 1: Convert ATOMIC TSV to BART format =====
echo ""
echo "Step 1: Converting ATOMIC2020 TSV files to BART format..."
echo "Input: $ATOMIC_TSV_DIR"
echo "Output: $ATOMIC_BART_DATA"

python scripts/prepare_atomic_for_bart.py \
    --input_dir "$ATOMIC_TSV_DIR" \
    --output_dir "$ATOMIC_BART_DATA"

echo "✓ Data conversion complete!"

# ===== STEP 2: Finetune BART on ATOMIC =====
echo ""
echo "Step 2: Finetuning BART on ATOMIC2020..."
echo "Checkpoint: $HF_BART_CHECKPOINT"
echo "Output: $OUTPUT_DIR"

CUDA_VISIBLE_DEVICES=0 python models/comet_atomic2020_bart/finetune.py \
    --task summarization \
    --num_workers 2 \
    --learning_rate=$LEARNING_RATE \
    --gpus $GPUS \
    --do_train \
    --do_predict \
    --n_val 100 \
    --val_check_interval 1.0 \
    --sortish_sampler \
    --max_source_length=$MAX_SOURCE_LENGTH \
    --max_target_length=$MAX_TARGET_LENGTH \
    --val_max_target_length=$MAX_TARGET_LENGTH \
    --test_max_target_length=$MAX_TARGET_LENGTH \
    --data_dir "$ATOMIC_BART_DATA" \
    --train_batch_size=$TRAIN_BATCH_SIZE \
    --eval_batch_size=$EVAL_BATCH_SIZE \
    --output_dir "$OUTPUT_DIR" \
    --num_train_epochs $NUM_EPOCHS \
    --model_name_or_path "$HF_BART_CHECKPOINT" \
    --atomic

echo ""
echo "=================================================="
echo "✓ Finetuning complete!"
echo "Model saved at: $OUTPUT_DIR"
echo "=================================================="

# ===== OPTIONAL: Test the model =====
echo ""
echo "To test your model, you can run:"
echo "  python models/comet_atomic2020_bart/generation_example.py"
echo "  (Make sure to update the model path in the script)"