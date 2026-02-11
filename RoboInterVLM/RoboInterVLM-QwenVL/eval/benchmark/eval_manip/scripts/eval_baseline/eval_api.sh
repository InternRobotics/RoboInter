#!/bin/bash
# Baseline evaluation for API-based models (GPT-4o, Gemini, etc.)
# Usage: bash eval_api.sh

export TOKENIZERS_PARALLELISM=True

# Model to evaluate
model_list=(
    "gpt4o-mini"
    # "gemini"
    # "gpt4o"
    # "qwenvl2.5-72B"
)

# API configuration
API_KEY="#TODO"
BASE_URL="#TODO"  # Optional, leave empty for default
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

# Run evaluation
for i in ${!model_list[@]}; do
    for j in ${!json_path_list[@]}; do
        echo "Evaluating ${model_list[$i]} with ${json_path_list[$j]}"

        python baselines/evaluation_intermediate_baseline_api.py \
            --model ${model_list[$i]} \
            --api_key ${API_KEY} \
            --json_path ${json_path_list[$j]} \
            --image_dir ${image_dirs[$j]} \
            --output_dir ${OUTPUT_DIR}/${model_list[$i]} \
            --type ${type_list[$j]} \
            --batch_size 1 \
            --max_new_tokens 512
    done
done
