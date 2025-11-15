#!/bin/bash
#SBATCH --job-name=eval
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
#SBATCH --output=logs/eval.log
#SBATCH --error=logs/eval.err
# Script to evaluate SFT-trained GPT model on test questions
# This script runs the evaluation with default parameters

echo "🚀 Starting GPT Model Evaluation..."

export MODEL_PATH="/scratch/eecs595f25_class_root/eecs595f25_class/joelyang/models"

python score_gpt.py \
    --model_path "$MODEL_PATH/sft-models/sft_model_step_36000.pth" \
    --questions_file "test_questions.jsonl" \
    --output_file "evaluation_results_$(date +'%Y%m%d_%H%M%S').csv" \
    --vocab_size 50262 \
    --context_length 1024 \
    --emb_dim 512 \
    --n_heads 8 \
    --n_layers 12 \
    --drop_rate 0.1 \
    --max_tokens 200 \
    --temperature 0.3 \
    --device "auto"

echo "✅ Evaluation completed!"
