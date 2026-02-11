#!/bin/bash

export PYTHONPATH=$(pwd)
output_path="/mnt/petrelfs/wangziqin/project/System2VLA/qwen-vl-finetune/results/llava-one-vision-7B_gdata_udata_manipvqa_generaldata"
BBOX_TYPT="llava"  

find "$output_path" -type f -name "*.json" | while read -r json_path; do
    echo "Processing: $json_path"

    if [[ "$json_path" == *"refcoco"* ]]; then
        echo "-> Running refcoco evaluator"
        python refcoco/s2_eval.py \
            --answer_file "$json_path" --bbox_output_type $BBOX_TYPT

    # if [[ "$json_path" == *"roboreflt"* ]]; then
    #     echo "-> Running roboreflt evaluator"
    #     python roboreflt/s2_eval.py \
    #      --answer_file "$json_path" \
    #      --bbox_output_type $BBOX_TYPT \
    #      --data_dir /mnt/inspurfs/efm_t/sys2_data/roboreflt/

    # elif [[ "$json_path" == *"where2"* ]]; then
    #     echo "-> Running where2place evaluator"
    #     python where2place/s2_eval.py \
    #      --answer_file "$json_path" \
    #      --data_dir /mnt/petrelfs/share/efm_p/sys2_data/where2place/where2place \
    #      --bbox_output_type $BBOX_TYPT \

    # # elif [[ "$json_path" == *"ocr_real_data"* ]]; then
    # #     echo "-> Running OCR demo evaluator"
    # #     python demo_desktop/s2_eval.py --answer_file "$json_path" --parse_func_version yk_demo_250514 # qwen_demo_250514 # yk_demo_250514 # jh_cot_250514
    
    # elif [[ "$json_path" == *"bench200"* ]]; then
    #     echo "-> Running Bench200 evaluator"
    #     python genmanip_200/s2_eval.py \
    #      --answer_file "$json_path" \
    #      --bbox_output_type internvl # qwen_demo_250514 # yk_demo_250514 # jh_cot_250514

    # # elif [[ "$json_path" == *"erqa"* ]]; then
    # #     echo "-> Running ERQA evaluator"
    # #     python erqa/s2_eval.py \
    # #      --answer_file "$json_path"

    # elif [[ "$json_path" == *"ManipInterface"* ]]; then
    #     echo "-> Running ManipInterface evaluator"
    #     python oxe_300/s2_eval.py \
    #      --answer_file "$json_path" --bbox_output_type internvl \
    #      --data_dir /mnt/inspurfs/efm_t/sys2_data/oxe_sys2_bench/
         
    # # else
    # #     echo "-> Skipped (no matching pattern)"
    fi

done
