#!/bin/bash

# ---------------------------------------------------------------------------
# Single-machine multi-GPU training script
# bash scripts/sft.sh [task_name] [num_gpus]
# example: bash scripts/sft.sh my_experiment 8
# ---------------------------------------------------------------------------

# Parameters
TASK_NAME=${1:-"sft_experiment"}
NUM_GPUS=${2:-8}

# GPU settings visible (can be modified as needed)
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# ---------------------------------------------------------------------------
# Output configuration
OUTPUT_DIR=#TODO/qwen-vl-finetune/results/${TASK_NAME}
mkdir -p ${OUTPUT_DIR}
cp "$0" "${OUTPUT_DIR}/"

# logs
exec > >(tee -a "$OUTPUT_DIR/job.out") 2> >(tee -a "$OUTPUT_DIR/job.err" >&2)

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

log "Task name: ${TASK_NAME}"
log "Number of GPUs: ${NUM_GPUS}"
log "Output dir: ${OUTPUT_DIR}"

# ---------------------------------------------------------------------------
# WANDB config (optional)
export WANDB_ENTITY="#TODO"
export WANDB_PROJECT="#TODO"
export WANDB_MODE="offline"  # "online"

# ---------------------------------------------------------------------------
# Dataset setting
# Generation datasets
export rh20t_gen_datasets="rh20t_vla_current_box_train%65,rh20t_vla_contact_box_train%30,rh20t_vla_final_box_train%20,rh20t_vla_traj_qa_train,rh20t_vla_traj_init_point_qa_train,rh20t_vla_gripper_det_qa_train%50"
export droid_gen_datasets="droid_vla_contact_box_train%40,droid_vla_current_box_train,droid_vla_final_box_train%20,droid_vla_traj_qa_train,droid_vla_traj_init_point_qa_train,droid_vla_gripper_det_qa_train%50"

# Understanding datasets
export rh20t_und_datasets="rh20t_contact_choice_qa_train,rh20t_graspppose_choice_qa_train,rh20t_grounding_choice_qa_train,rh20t_traj_lang_choice_qa_train,rh20t_traj_direction_choice_qa_train,rh20t_traj_choice_qa_train,rh20t_traj_direction_choice_with_traj_qa_train"
export droid_und_datasets="droid_contact_choice_qa_train,droid_grounding_choice_qa_train,droid_traj_lang_choice_qa_train,droid_traj_direction_choice_qa_train,droid_traj_choice_qa_train,droid_traj_direction_choice_with_traj_qa_train"

# Language datasets
export lang_datasets="manipvqa_train"

DATASETS="${rh20t_gen_datasets},${droid_gen_datasets},${rh20t_und_datasets},${droid_und_datasets},${lang_datasets}"
log "Datasets: ${DATASETS}"

# ---------------------------------------------------------------------------
# Train setting
PRETRAIN_MODEL=#TODO/Qwen2.5-VL-3B-Instruct
DEEPSPEED_CONFIG=./scripts/zero3.json

# Training hyperparameters
LR=5e-6
VIT_LR=1e-6
GRAD_ACCUM_STEPS=2
PER_DEVICE_TRAIN_BATCH_SIZE=4
PER_DEVICE_EVAL_BATCH_SIZE=16

# ---------------------------------------------------------------------------
# Training entry point
cd #TODO/qwen-vl-finetune

log "Starting training..."

torchrun \
    --nproc_per_node ${NUM_GPUS} \
    --master_port $(shuf -i 20000-30000 -n 1) \
    qwenvl/train/train_qwen.py \
    --deepspeed ${DEEPSPEED_CONFIG} \
    --model_name_or_path ${PRETRAIN_MODEL} \
    --dataset_use ${DATASETS} \
    --data_flatten True \
    --tune_mm_vision False \
    --tune_mm_mlp False \
    --tune_mm_llm False \
    --tune_moe True \
    --bf16 \
    --output_dir ${OUTPUT_DIR} \
    --num_train_epochs 1 \
    --per_device_train_batch_size ${PER_DEVICE_TRAIN_BATCH_SIZE} \
    --per_device_eval_batch_size ${PER_DEVICE_EVAL_BATCH_SIZE} \
    --gradient_accumulation_steps ${GRAD_ACCUM_STEPS} \
    --max_pixels 12845056 \
    --min_pixels 3136 \
    --eval_strategy no \
    --save_strategy steps \
    --save_steps 3000 \
    --save_total_limit 2 \
    --learning_rate ${LR} \
    --mm_projector_lr ${LR} \
    --vision_tower_lr ${VIT_LR} \
    --weight_decay 0 \
    --warmup_ratio 0.03 \
    --max_grad_norm 1 \
    --lr_scheduler_type cosine \
    --logging_steps 1 \
    --model_max_length 8192 \
    --gradient_checkpointing True \
    --dataloader_num_workers 8 \
    --run_name ${TASK_NAME} \
    --report_to wandb

log "Training finished!"
