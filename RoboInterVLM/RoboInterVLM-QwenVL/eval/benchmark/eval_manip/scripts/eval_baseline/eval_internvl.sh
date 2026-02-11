#!/bin/bash
# Baseline evaluation for InternVL models
# Usage: bash eval_internvl.sh

export TOKENIZERS_PARALLELISM=True

# Model paths to evaluate
model_path_list=(
    "#TODO/InternVL3-2B"
    # "#TODO/InternVL3-8B"
)

model_name_list=(
    "InternVL-2B"
    # "InternVL-8B"
)

# Evaluation datasets
json_path_list=(
    "#TODO/eval_data.json"
)

type_list=(
    "contact_box"
)

image_dirs=(
    "#TODO/image_dir"
)

question_types=(
    "default"  # Options: default, qwen_grounding, refindoor
)

# Run evaluation
mkdir -p logs

for i in ${!model_path_list[@]}; do
    for j in ${!json_path_list[@]}; do
        model_name=${model_name_list[$i]}
        task_type=${type_list[$j]}
        log_file="logs/${model_name}_${task_type}.log"

        echo "Evaluating ${model_name} with ${task_type}"
        echo "Log output to: ${log_file}"

        python baselines/evaluation_intermediate_baseline_internvl.py \
            --model_path ${model_path_list[$i]} \
            --json_path ${json_path_list[$j]} \
            --image_dir ${image_dirs[$j]} \
            --type ${type_list[$j]} \
            --question_type ${question_types[$j]} \
            --batch_size 8 \
            --max_new_tokens 512 2>&1 | tee ${log_file}
    done
done
