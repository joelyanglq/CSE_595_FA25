#!/bin/bash
#SBATCH --job-name=pretrain
#SBATCH --mail-user=joelyang@umich.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=32GB
#SBATCH --cpus-per-gpu=1
#SBATCH --time=04:00:00
#SBATCH --account=eecs595f25_class
#SBATCH --partition=spgpu,gpu_mig40,gpu
#SBATCH --gres=gpu:1
#SBATCH --output=logs/pretrain.log
#SBATCH --error=logs/pretrain.err
# EECS 595 HW3: Full GPT Training Script
# This script trains the GPT model with production-ready hyperparameters
# Designed for use on Great Lakes cluster with GPU resources

echo "Starting full GPT training..."
echo "=================================="

# Set up environment
export CUDA_VISIBLE_DEVICES=0
export WANDB_PROJECT="eecs595-gpt-pretraining"
export DATA_PATH="/scratch/eecs595f25_class_root/eecs595f25_class/joelyang/Data"
export OUTPUT_DIR="/scratch/eecs595f25_class_root/eecs595f25_class/joelyang/models/pretrained-models/"
export TOKENIZERS_PARALLELISM=false

# Use these hyperparameters for your full pretraining

# Training hyperparameters for full model
python pretrain_gpt.py \
    --batch_size 16 \
    --learning_rate 6e-4 \
    --max_epochs 1 \
    --emb_dim 512 \
    --n_layers 12 \
    --n_heads 8 \
    --context_length 1024 \
    --save_every 1000 \
    --eval_every 1000 \
    --device cuda \
    --data_path $DATA_PATH/fineweb-edu-sample-1B-hf/ \
    --data_format arrow \
    --eval_data_path $DATA_PATH/fineweb-edu-eval-3M.jsonl.gz \
    --eval_data_format jsonl \
    --eval_batch_size 16 \
    --output_dir $OUTPUT_DIR \
    --wandb_project $WANDB_PROJECT \
    --wandb_run_name "gpt-pretraining-$(date +%Y%m%d-%H%M%S)"

echo "Training completed!"
echo "Check the output directory for saved models and logs."
