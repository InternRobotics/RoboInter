#!/bin/bash
# Baseline evaluation for RoboBrain models
# Usage: bash eval_robobrain.sh

export TOKENIZERS_PARALLELISM=True

# Model paths to evaluate
model_path_list=(
    "BAAI/RoboBrain2.0-7B"
    # "BAAI/RoboBrain2.0-32B"
)

model_name_list=(
    "RoboBrain2.0-7B"
    # "RoboBrain2.0-32B"
)

# Output directory (since HuggingFace model IDs can't store results)
OUTPUT_DIR="#TODO/results"

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
    "default"
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

        python baselines/evaluation_intermediate_baseline_robobrain.py \
            --model_path ${model_path_list[$i]} \
            --json_path ${json_path_list[$j]} \
            --image_dir ${image_dirs[$j]} \
            --output_dir ${OUTPUT_DIR}/${model_name} \
            --type ${type_list[$j]} \
            --question_type ${question_types[$j]} \
            --batch_size 8 \
            --max_new_tokens 512 2>&1 | tee ${log_file}
    done
done
