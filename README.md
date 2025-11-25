# BART Finetuning on ATOMIC2020 - Complete Guide

This guide walks through finetuning pretrained BART model (from fairseq) on the ATOMIC2020 dataset using COMET's training methodology.

## Prerequisites

- BART checkpoint pretrained with fairseq (denoising objective)
- ATOMIC2020 dataset (download from [Google Drive](https://drive.google.com/file/d/1uuY0Y_s8dhxdsoOe8OHRgsqf-9qJIai7/view?usp=drive_link))
- Python 3.x with PyTorch and Transformers

## Step-by-Step Process

### Step 1: Convert Fairseq BART to HuggingFace Format

Since the finetuning script uses HuggingFace Transformers, we need to convert our fairseq checkpoint first:

```bash
# Install transformers if not already installed
pip install transformers torch

# Convert fairseq checkpoint to HuggingFace format
python -m transformers.models.bart.convert_bart_original_pytorch_checkpoint_to_pytorch \
    --fairseq_path /path/to/your/fairseq/checkpoint.pt \
    --pytorch_dump_folder_path ./checkpoints/bart_hf
```

**Note:** Replace `/path/to/your/fairseq/checkpoint.pt` with the actual path to our pretrained checkpoint.

### Step 2: Extract ATOMIC2020 Dataset

Extract the downloaded ATOMIC2020 zip file:

```bash
# Extract the dataset
unzip atomic2020_data-feb2021.zip -d ./atomic2020_data-feb2021

# Check the contents
ls -la ./atomic2020_data-feb2021/
# Should contain: train.tsv, dev.tsv, test.tsv, README.md, LICENSE, etc.
```

The TSV files have the format:
- Column 1: `head` (head event, e.g., "PersonX goes to the store")
- Column 2: `relation` (e.g., "xIntent", "xNeed", "oEffect")
- Column 3: `tail` (tail event, e.g., "to buy groceries")

### Step 3: Convert ATOMIC TSV to BART Training Format

Convert the TSV files to `.source` and `.target` format required by the training script:

```bash
python scripts/prepare_atomic_for_bart.py \
    --input_dir ./atomic2020_data-feb2021 \
    --output_dir ./data/atomic_bart
```

This creates:
- `train.source` / `train.target` (training set)
- `val.source` / `val.target` (validation set, from dev.tsv)
- `test.source` / `test.target` (test set)

**Format:**
- `.source` files: `"head_event relation"` (e.g., "PersonX goes to the store xIntent")
- `.target` files: `"tail_event"` (e.g., "to buy groceries")

**Optional:** Add `--add_gen_token` flag to append `[GEN]` to source (for demo model compatibility):
```bash
python scripts/prepare_atomic_for_bart.py \
    --input_dir ./atomic2020_data-feb2021 \
    --output_dir ./data/atomic_bart \
    --add_gen_token
```

### Step 4: Finetune BART on ATOMIC2020

Run the finetuning script:

```bash
CUDA_VISIBLE_DEVICES=0 python models/comet_atomic2020_bart/finetune.py \
    --task summarization \
    --num_workers 2 \
    --learning_rate=1e-5 \
    --gpus 1 \
    --do_train \
    --do_predict \
    --n_val 100 \
    --val_check_interval 1.0 \
    --sortish_sampler \
    --max_source_length=48 \
    --max_target_length=24 \
    --val_max_target_length=24 \
    --test_max_target_length=24 \
    --data_dir ./data/atomic_bart \
    --train_batch_size=32 \
    --eval_batch_size=32 \
    --output_dir ./results/bart_atomic_finetune \
    --num_train_epochs=10 \
    --model_name_or_path ./checkpoints/bart_hf \
    --atomic
```

**Important Parameters:**
- `--model_name_or_path`: Path to your converted HuggingFace BART checkpoint
- `--data_dir`: Directory containing `.source` and `.target` files
- `--atomic`: **CRITICAL** - Adds ATOMIC relation tokens to the tokenizer
- `--num_train_epochs`: Number of training epochs (adjust as needed)
- `--train_batch_size`: Adjust based on your GPU memory

### Step 5: Use the Complete Workflow Script

Alternatively, you can use the all-in-one script (edit paths first):

```bash
# Edit the script to set your paths
nano scripts/finetune_bart_on_atomic.sh

# Run the complete workflow
bash scripts/finetune_bart_on_atomic.sh
```

## Quick Start Commands

```bash
# 1. Convert fairseq checkpoint
python -m transformers.models.bart.convert_bart_original_pytorch_checkpoint_to_pytorch \
    --fairseq_path <your_checkpoint.pt> \
    --pytorch_dump_folder_path ./checkpoints/bart_hf

# 2. Prepare ATOMIC data
python scripts/prepare_atomic_for_bart.py \
    --input_dir ./atomic2020_data-feb2021 \
    --output_dir ./data/atomic_bart

# 3. Finetune BART
bash scripts/finetune_bart_on_atomic.sh
```
