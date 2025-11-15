#!/bin/bash
#SBATCH --job-name=sft
#SBATCH --mail-user=joelyang@umich.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=32GB
#SBATCH --cpus-per-gpu=1
#SBATCH --time=04:00:00
#SBATCH --account=eecs595f25_class
#SBATCH --partition=spgpu,gpu_mig40
#SBATCH --gres=gpu:1
#SBATCH --output=logs/sft.log
#SBATCH --error=logs/sft.err
# Script to run SFT training on the full dataset.
# Designed for larger-scale training on clusters.

echo "Starting SFT training on full dataset..."

export CUDA_VISIBLE_DEVICES=0
export WANDB_PROJECT="eecs595-gpt-finetuning"
export DATA_PATH="/scratch/eecs595f25_class_root/eecs595f25_class/joelyang/Data"
export MODEL_PATH="/scratch/eecs595f25_class_root/eecs595f25_class/joelyang/models/pretrained-models/"
export OUTPUT_DIR="/scratch/eecs595f25_class_root/eecs595f25_class/joelyang"
export TOKENIZERS_PARALLELISM=false

# Use these hyperparameters for your full SFT training
python sft_gpt_copy.py \
    --train_data_path $DATA_PATH/sft_data_packed.arrow \
    --val_data_path $DATA_PATH/smol-smoltalk-dev.jsonl.gz \
    --train_data_format arrow \
    --model_path $MODEL_PATH/model_epoch_0.pt \
    --context_length 1024 \
    --emb_dim 512 \
    --n_heads 8 \
    --n_layers 12 \
    --drop_rate 0.1 \
    --batch_size 16 \
    --learning_rate 2e-5 \
    --weight_decay 0.01 \
    --max_epochs 2 \
    --gradient_accumulation_steps 4 \
    --warmup_steps 100 \
    --output_dir $OUTPUT_DIR/models/sft-models/ \
    --save_every 1000 \
    --eval_every 1000 \
    --wandb_project "gpt-sft-full" \
    --device "auto" \
    --num_workers 4 \
    --seed 42

echo "SFT training on full dataset finished."
