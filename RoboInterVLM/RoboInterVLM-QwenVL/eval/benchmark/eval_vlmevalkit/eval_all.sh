# Download the evaluation data and put it in the same directory as this script, and set the name to LMU_for_VLMEVALKIT
export LMUData=LMU_for_VLMEVALKIT

echo LMUData: $LMUData

MODEL_NAME=(
  "Qwen2.5-VL-7B-Instruct"
  "llava_onevision_qwen2_7b_ov"
)

VLMEVALKIT_PATH="eval_vlmevalkit/VLMEvalKit"
CONFIG_PATH="${VLMEVALKIT_PATH}/vlmeval/config.py"

MODEL_PATH_LIST=(
    # "Qwen2.5-VL-3B-Instruct"
    # "Qwen2.5-VL-7B-Instruct"
    # "RoboBrain2.0-3B"
    # "InternVL3-1B"
    # "InternVL3-2B"
    # "InternVL3-8B"
    "manip_sys2_qwen25_7b_gdata_udata_manipvqa_generaldata". # qwen25 must in model name
    "llava-one-vision-7B_gdata_udata_manipvqa_generaldata"
)

# find the line number of model_path in config.py and put it in the array
motified_lines=(
  # TODO
)

cd "$VLMEVALKIT_PATH" || exit 1


for i in "${!MODEL_PATH_LIST[@]}"; do
  MODEL_PATH=${MODEL_PATH_LIST[$i]}
  echo "🔧 Current Model Path: $MODEL_PATH"

  OUTPUT_DIR="$MODEL_PATH/eval_rlt/vlmevalkit"
  mkdir -p "$OUTPUT_DIR"
  
  echo "🔁 Motified config.py model_path in line ${motified_lines[$i]}"
  sed -i "${motified_lines[$i]}s|model_path=\"[^\"]*\"|model_path=\"$MODEL_PATH\"|" "$CONFIG_PATH"

  sed -n "${motified_lines[$i]}p" "$CONFIG_PATH"

  echo "🚀 Start Eval: $MODEL_PATH"

  torchrun --nproc-per-node=8 run.py --data MMVet COCO_VAL TextVQA_VAL OCRBench POPE \
    --model "${MODEL_NAME[$i]}" \
    --work-dir "$OUTPUT_DIR" \
    --verbose \
    --reuse

  echo "✅ Finish Eval：$MODEL_PATH"
  echo "-----------------------------------------"
done
