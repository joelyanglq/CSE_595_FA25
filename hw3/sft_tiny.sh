#!/bin/bash
#SBATCH --job-name=sft_debug
#SBATCH --mail-user=joelyang@umich.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=32GB
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --account=eecs595f25_class
#SBATCH --partition=standard
#SBATCH --output=logs/sft_debug.log
#SBATCH --error=logs/sft_debug.err
# Script to run SFT training on a tiny dataset for local verification.
# This configuration is designed to run quickly on a CPU for a few steps.

echo "Starting SFT training on tiny dataset for local verification..."

export DATA_PATH="/scratch/eecs595f25_class_root/eecs595f25_class/joelyang/Data"
export MODEL_PATH="/scratch/eecs595f25_class_root/eecs595f25_class/joelyang/models"
export OUTPUT_DIR="/scratch/eecs595f25_class_root/eecs595f25_class/joelyang/models/pretrained-models/"

python sft_gpt_copy.py \
    --train_data_path $DATA_PATH/smol-smoltalk-train.jsonl.gz \
    --val_data_path $DATA_PATH/smol-smoltalk-dev.jsonl.gz \
    --model_path $MODEL_PATH/tiny/model_epoch_0.pt \
    --context_length 64 \
    --emb_dim 32 \
    --n_heads 4 \
    --n_layers 2 \
    --drop_rate 0.0 \
    --batch_size 4 \
    --learning_rate 1e-4 \
    --weight_decay 0.01 \
    --max_epochs 1 \
    --gradient_accumulation_steps 2 \
    --warmup_steps 10 \
    --output_dir $MODEL_PATH/tiny-sft/ \
    --save_every 50 \
    --eval_every 25 \
    --wandb_project "gpt-sft-tiny-local" \
    --device "cpu" \
    --num_workers 8 \
    --seed 42

echo "SFT training on tiny dataset finished."
echo "This should run in a few minutes on CPU."
echo "Check models/tiny-sft/ for the saved model."
