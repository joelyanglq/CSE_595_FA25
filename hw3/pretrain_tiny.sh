#!/bin/bash
#SBATCH --job-name=pretrain_debug
#SBATCH --mail-user=joelyang@umich.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=32GB
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --account=eecs595f25_class
#SBATCH --partition=standard
#SBATCH --output=logs/debug.log
#SBATCH --error=logs/debug.err
# EECS 595 HW3: Tiny GPT Training Script
# This script trains a very small GPT model for local testing
# Designed to run quickly on CPU for student verification

echo "Starting tiny GPT training for local testing..."
echo "=============================================="

# Training hyperparameters for tiny model (CPU-friendly)
python pretrain_gpt.py \
    --batch_size 4 \
    --learning_rate 1e-3 \
    --max_epochs 1 \
    --emb_dim 32 \
    --n_layers 2 \
    --n_heads 4 \
    --context_length 64 \
    --drop_rate 0.0 \
    --weight_decay 0.01 \
    --max_docs 100 \
    --save_every 50 \
    --eval_every 25 \
    --device cpu \
    --data_path /scratch/eecs595f25_class_root/eecs595f25_class/joelyang/Data/fineweb-edu-sample-1M.jsonl.gz \
    --output_dir /scratch/eecs595f25_class_root/eecs595f25_class/joelyang/models/tiny/ \
    --wandb_run_name "gpt-tiny-test-$(date +%Y%m%d-%H%M%S)"

echo "Tiny training completed!"
echo "This should run in a few minutes on CPU."
echo "Check ./models/tiny/ for the saved model."
