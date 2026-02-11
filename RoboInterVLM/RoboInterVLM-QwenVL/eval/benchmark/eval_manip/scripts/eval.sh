#!/bin/bash
# Evaluation script for intermediate representations
# Usage: bash eval.sh

export TOKENIZERS_PARALLELISM=True

# Model paths to evaluate
model_path_list=(
    "#TODO/model_checkpoint"
)

model_name_list=(
    "model_name"
)

# Evaluation datasets
json_path_list=(
    "#TODO/eval_data.json"
)

type_list=(
    "contact_box"  # Options: contact_box, current_box, final_box, gripper_det, traj, traj_wo_init_pos
)

image_dirs=(
    "#TODO/image_dir"
)

question_types=(
    "default"  # Options: default, qwen_grounding, refindoor, w2p
)

# Run evaluation
for i in ${!model_path_list[@]}; do
    for j in ${!json_path_list[@]}; do
        model_name=${model_name_list[$i]}
        task_type=${type_list[$j]}
        log_file="logs/${model_name}_${task_type}.log"

        echo "Evaluating ${model_name} with ${task_type}"
        echo "Log output to: ${log_file}"

        mkdir -p logs

        python evaluation_intermediate.py \
            --model_path ${model_path_list[$i]} \
            --json_path ${json_path_list[$j]} \
            --image_dir ${image_dirs[$j]} \
            --type ${type_list[$j]} \
            --question_type ${question_types[$j]} \
            --batch_size 16 \
            --max_new_tokens 512 2>&1 | tee ${log_file}
    done
done
