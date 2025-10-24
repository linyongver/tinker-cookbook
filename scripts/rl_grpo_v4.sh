#!/bin/bash

# DAPO (Dynamic Adaptive Policy Optimization) RL Training Script
# This script enables DAPO for more efficient trajectory collection

# Activate the tinker conda environment
source ~/.bashrc
conda activate tinker

export WANDB_API_KEY="7937eb41f5b281b019c83a5a8a06d9f9656647e5"
export WANDB_ENTITY="linyongver"
export WANDB_DIR="/scratch/gpfs/yl7690/projects/tinker_lean/wandb"

# 配置参数
MODEL_NAME="Qwen/Qwen3-32B"
LOAD_CHECKPOINT_PATH="tinker://dbf08cfc-bf91-4ecf-a9a7-a3ff141258e4/weights/003200"

LORA_RANK=128
# TRAIN_DATASET_PATH="/home/zy7019/tinker/tinker-cookbook/tinker_logs/data/train_a2_860K_first_50K.jsonl"
TRAIN_DATASET_PATH="/scratch/gpfs/CHIJ/ziran/LeanRL/tinker/tinker_logs/data/rl/v9_26_RL_v4_completion_50K.jsonl"
# TEST_DATASET_PATH="/scratch/gpfs/yl7690/projects/DeepSeek-Prover-V1.5/datasets/train_datasets/train_a2_860K.jsonl"
NUM_EPOCHS=1
LEARNING_RATE=1e-5
BATCH_SIZE=16
MAX_TOKENS=$((12 * 1024))
LR_SCHEDULE="constant"
# WARMUP_RATIO="0.1"  # 10% of total steps for warmup
SAVE_EVERY=50
EVAL_EVERY=0
GROUP_SIZE=2
LOG_PATH="tinker-cookbook/tinker_logs"
WANDB_PROJECT="tinker-rl"

# DAPO specific configuration
USE_DAPO=false
DAPO_MIN_ELIGIBLE_GROUPS=32
DAPO_MAX_COLLECTION_STEPS=10
DAPO_ALL_GOOD_THRESHOLD=0.95
DAPO_ALL_BAD_THRESHOLD=0.05
SAVE_TRAJECTORIES=true

# Environment specific configuration
MAX_ROUNDS=1
FORMAT_COEF=0.0


WANDB_NAME="$(date +%m%d_%H%M%S)_rl_debug_grpo_qwen3-32b_RL_v9_19_rank${LORA_RANK}__maxtokens${MAX_TOKENS}_maxrounds${MAX_ROUNDS}_formatcoef${FORMAT_COEF}"

# # Evaluator配置
# EVAL_DATASET_PATH="/home/zy7019/tinker/tinker-cookbook/tinker_logs/data/eval_lean4_problems.jsonl" 
# EVAL_NAME="lean4_eval"
# NUM_EVAL_SAMPLES=1
# EVAL_TEMPERATURE="0.7"
# EVAL_HANDLER_TYPE="dpskcot"

echo "Starting DAPO RL training with the following configuration:"
echo "  Model: $MODEL_NAME"
echo "  DAPO enabled: $USE_DAPO"
echo "  Min eligible groups: $DAPO_MIN_ELIGIBLE_GROUPS"
echo "  Max collection steps: $DAPO_MAX_COLLECTION_STEPS"
echo "  All good threshold: $DAPO_ALL_GOOD_THRESHOLD"
echo "  All bad threshold: $DAPO_ALL_BAD_THRESHOLD"
echo "  Save trajectories: $SAVE_TRAJECTORIES"
echo "  Batch size: $BATCH_SIZE"
echo "  Group size: $GROUP_SIZE"
echo "  Learning rate: $LEARNING_RATE"
echo "  Max tokens: $MAX_TOKENS"
echo "  Max rounds: $MAX_ROUNDS"
echo "  Format coefficient: $FORMAT_COEF"
echo "  Log path: $LOG_PATH"
echo "  WANDB name: $WANDB_NAME"
echo ""

# 运行训练脚本，传入所有配置参数
python3 -m tinker_cookbook.recipes.rl_lean \
    model_name="$MODEL_NAME" \
    lora_rank="$LORA_RANK" \
    train_dataset_path="$TRAIN_DATASET_PATH" \
    test_dataset_path="$TEST_DATASET_PATH" \
    load_checkpoint_path="$LOAD_CHECKPOINT_PATH" \
    learning_rate="$LEARNING_RATE" \
    groups_per_batch="$BATCH_SIZE" \
    group_size="$GROUP_SIZE" \
    max_tokens="$MAX_TOKENS" \
    save_every="$SAVE_EVERY" \
    eval_every="$EVAL_EVERY" \
    log_path="$LOG_PATH" \
    wandb_project="$WANDB_PROJECT" \
    wandb_name="$WANDB_NAME" \
    use_dapo="$USE_DAPO" \
    dapo_min_eligible_groups="$DAPO_MIN_ELIGIBLE_GROUPS" \
    dapo_max_collection_steps="$DAPO_MAX_COLLECTION_STEPS" \
    dapo_all_good_threshold="$DAPO_ALL_GOOD_THRESHOLD" \
    dapo_all_bad_threshold="$DAPO_ALL_BAD_THRESHOLD" \
    save_trajectories="$SAVE_TRAJECTORIES" \
    max_rounds="$MAX_ROUNDS" \
    format_coef="$FORMAT_COEF"

    # warmup_ratio="$WARMUP_RATIO" \

    # eval_dataset_path="$EVAL_DATASET_PATH" \
    # eval_name="$EVAL_NAME" \
    # num_eval_samples="$NUM_EVAL_SAMPLES" \
    # eval_temperature="$EVAL_TEMPERATURE" \
    # eval_handler_type="$EVAL_HANDLER_TYPE" 
