#!/bin/bash
# Evaluation script for language understanding tasks
# Usage: bash eval_lang.sh

export TOKENIZERS_PARALLELISM=True

# Model paths to evaluate
model_path_list=(
    "#TODO/model_checkpoint"
)

# Evaluation datasets
json_path_list=(
    "#TODO/eval_data.json"
)

type_list=(
    "qa"
)

image_dir="#TODO/image_dir"

# Run evaluation
for i in ${!model_path_list[@]}; do
    for j in ${!json_path_list[@]}; do
        echo "Evaluating ${model_path_list[$i]} with ${json_path_list[$j]}"

        python evaluation_intermediate_lang.py \
            --model_path ${model_path_list[$i]} \
            --json_path ${json_path_list[$j]} \
            --image_dir ${image_dir} \
            --type ${type_list[$j]} \
            --batch_size 16 \
            --max_new_tokens 512
    done
done
