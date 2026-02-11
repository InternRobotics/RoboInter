#!/bin/bash

# ---------------------------------------------------------------------------
# Single-machine multi-GPU training script using torchrun
# ---------------------------------------------------------------------------

export GPUS_PER_NODE=8
export MASTER_PORT=$((RANDOM % 101 + 20000))

# Task and output settings
task_name="llava-onevision-7B-manip-finetune"
output_dir="./results/${task_name}"

# WANDB settings (optional)
export WANDB_ENTITY="TODO: set your wandb entity"
export WANDB_PROJECT="TODO: set your wandb project"
export WANDB_MODE="online"

# ---------------------------------------------------------------------------
# Dataset setting
# ---------------------------------------------------------------------------
export rh20t_Gdatasets="rh20t_vla_current_box_train,rh20t_vla_contact_box_train,rh20t_vla_final_box_train,rh20t_vla_traj_qa_train,rh20t_vla_gripper_det_qa_train,rh20t_vla_traj_init_point_qa_train"
export droid_Gdatasets="droid_vla_contact_box_train,droid_vla_current_box_train,droid_vla_final_box_train,droid_vla_traj_qa_train,droid_vla_gripper_det_qa_train,droid_vla_traj_init_point_qa_train"
export rh20t_Udatasets="rh20t_contact_choice_qa_train,rh20t_graspppose_choice_qa_train,rh20t_grounding_choice_qa_train,rh20t_traj_lang_choice_qa_train,rh20t_traj_lang_sub_choice_qa_train,rh20t_traj_direction_choice_qa_train,rh20t_traj_choice_qa_train,rh20t_traj_direction_choice_with_traj_qa_train"
export droid_Udatasets="droid_contact_choice_qa_train,droid_grounding_choice_qa_train,droid_traj_lang_choice_qa_train,droid_traj_lang_sub_choice_qa_train,droid_traj_direction_choice_qa_train,droid_traj_choice_qa_train,droid_traj_direction_choice_with_traj_qa_train"
export manip_dataset="manipvqa_train"

export datasets="${rh20t_Gdatasets},${droid_Gdatasets},${rh20t_Udatasets},${droid_Udatasets},${manip_dataset}"
echo "Using datasets: ${datasets}"

# ---------------------------------------------------------------------------
# Model settings
# ---------------------------------------------------------------------------
VISION_MODEL_VERSION="google/siglip-so400m-patch14-384"
PROMPT_VERSION="qwen_1_5"
pretrain_model="lmms-lab/llava-onevision-qwen2-7b-ov"
echo "Task name: ${task_name}"

# ---------------------------------------------------------------------------
# Launch training
# ---------------------------------------------------------------------------
torchrun --nproc_per_node=${GPUS_PER_NODE} --master_port=${MASTER_PORT} \
    llava/train/train_mem.py \
    --deepspeed scripts/zero3.json \
    --model_name_or_path ${pretrain_model} \
    --version ${PROMPT_VERSION} \
    --data_path ${datasets} \
    --mm_tunable_parts="mm_vision_tower,mm_mlp_adapter,mm_language_model" \
    --mm_vision_tower_lr=4e-6 \
    --vision_tower ${VISION_MODEL_VERSION} \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --group_by_modality_length True \
    --image_aspect_ratio anyres_max_9 \
    --image_grid_pinpoints  "(1x1),...,(6x6)" \
    --mm_patch_merge_type spatial_unpad \
    --bf16 True \
    --run_name ${task_name} \
    --output_dir ${output_dir} \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 10000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 32768 \
    --gradient_checkpointing True \
    --dataloader_num_workers 8 \
    --lazy_preprocess True \
    --report_to wandb \
    --torch_compile True \
    --torch_compile_backend "inductor" \
    --dataloader_drop_last True \
    --frames_upbound 32
