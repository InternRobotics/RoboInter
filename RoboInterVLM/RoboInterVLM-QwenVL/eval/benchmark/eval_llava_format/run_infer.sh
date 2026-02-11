#!/bin/bash

# ========================
# Global Configuration
# ========================
export PYTHONPATH=$(pwd)
# export CUDA_VISIBLE_DEVICES=3,4 # ,1  #,4 #,5
nproc_per_node=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
master_port=$((RANDOM % 101 + 20000))
debug=0

output_path_base=./log_0715_1

# Prompt template version # qwen_pickplace_demo_250515 # yk_demo_250514
# for demo
demo_valdata_prompt_version="yk_demo_250514"

# ========================
# Model List
# ========================
model_list=(
    # "/ssd/home/zhuyangkun/tmp/jinhui_model_0514/Qwen2.5-VL-3B-Instruct"
    # "/ssd/home/zhuyangkun/tmp/sys2_ocr_pic_bucket"
    # "/ssd/home/zhuyangkun/tmp/qwen_fruit_ocr_v2"
    # "/mnt/petrelfs/share/yejinhui/Models/Pretrained_models/Qwen2.5-VL-3B-Instruct"

    # "/mnt/petrelfs/share/efm_p/zhuyangkun/share_model/Qwen2.5-VL-3B-Instruct"
    # "/mnt/petrelfs/zhuyangkun/exp_output/local_debug_genmanip_data2/checkpoint-6000"
    # "/mnt/petrelfs/zhuyangkun/exp_output/local_run_genmanip_1M_0630/qwenvl25_data_20_percent"
    # "/mnt/petrelfs/zhuyangkun/exp_output/local_run_genmanip_1M_0630/checkpoint-26000"

    # "/mnt/petrelfs/zhuyangkun/exp_output/manip_sys2_genmanipdata_only_0702/checkpoint-3000"
    # "/mnt/petrelfs/zhuyangkun/exp_output/manip_sys2_genmanipdata_only_0702/checkpoint-6000"
    # "/mnt/petrelfs/zhuyangkun/exp_output/manip_sys2_genmanipdata_only_0702/checkpoint-9000"
    # "/mnt/petrelfs/zhuyangkun/exp_output/manip_sys2_genmanipdata_only_0702/checkpoint-12000"
    # "/mnt/petrelfs/zhuyangkun/exp_output/manip_sys2_genmanipdata_only_0702/checkpoint-15000"
    # "/mnt/petrelfs/zhuyangkun/exp_output/manip_sys2_genmanipdata_coco_0702/checkpoint-3000"
    # "/mnt/petrelfs/zhuyangkun/exp_output/manip_sys2_genmanipdata_coco_0703/checkpoint-3000/"
    # "/mnt/petrelfs/zhuyangkun/exp_output/manip_sys2_genmanipdata_coco_0703/checkpoint-6000/"
    # "/mnt/petrelfs/zhuyangkun/exp_output/manip_sys2_genmanipdata_coco_0703/checkpoint-9000/"
    # "/mnt/petrelfs/zhuyangkun/exp_output/manip_sys2_genmanipdata_coco_0703/"

    # /mnt/petrelfs/share/efm_p/zhuyangkun/share_model/Qwen2.5-VL-3B-Instruct
    /mnt/petrelfs/zhuyangkun/exp_output/Qwen2.5-VL-3B-Instruct
    /mnt/petrelfs/zhuyangkun/exp_output/manip_sys2_genmanipdata_coco_llavasubset_0707
    /mnt/petrelfs/share/efm_p/zhuyangkun/share_model/release_model/manip_sys2_genmanipdata_coco_0703
    /mnt/petrelfs/zhuyangkun/exp_output/manip_sys2_genmanipdata_only_0702_qwen25
    /mnt/petrelfs/zhuyangkun/exp_output/manip_sys2_cocoonly_qwen25
)

# ========================
# Dataset Toggles & Batch Sizes
# ========================
bs=32

run_refcoco=false
bs_refcoco=$bs
sys_p_refcoco=false # system prompt

run_refindoor=false
bs_refindoor=$bs
sys_p_refindoor=false

run_where2place=false
bs_where2place=$bs
sys_p_where2place=false

run_demo_eval=false
bs_demo_eval=$bs
sys_p_demo_eval=true

run_erqa=false
bs_erqa=1
sys_p_erqa=false

run_bench200=false
bs_bench200=$bs
sys_p_bench200=false

run_oxe300=true
bs_oxe300=$bs
sys_p_oxe300=false
# ========================
# Inference Function
# ========================
run_infer() {
    local model_path="$1"
    local dataset_path="$2"
    local data_root="$3"
    local question_template_version="$4"
    local batch_size="$5"
    local system_prompt="$6"

    if [ ! -d "$model_path" ]; then
        echo "[WARN] Skipping non-existent model path: $model_path"
        return
    fi

    local output_path_name=$(basename "$model_path")
    local output_path="$output_path_base/$output_path_name/$question_template_version"

    mkdir -p "$output_path"

    echo "====================================="
    echo "Running inference:"
    echo "  Model:     $model_path"
    echo "  Dataset:   $dataset_path"
    echo "  Template:  $question_template_version"
    echo "  Output:    $output_path"
    echo "====================================="
    echo torchrun --nproc_per_node=$nproc_per_node --master_port=$master_port \
        project/benchmark/eval_llava_format/qwen_dist_infer.py \
        --model_path "$model_path" \
        --dataset_path "$dataset_path" \
        --data_root "$data_root"\
        --batch_size "$batch_size" \
        --output_path "$output_path" \
        --question_template_version "$question_template_version" \
        --debug "$debug" \
        --system_prompt "$system_prompt"
    echo "====================================="

    torchrun --nproc_per_node=$nproc_per_node --master_port=$master_port \
        project/benchmark/eval_llava_format/qwen_dist_infer.py \
        --model_path "$model_path" \
        --dataset_path "$dataset_path" \
        --data_root "$data_root"\
        --batch_size "$batch_size" \
        --output_path "$output_path" \
        --question_template_version "$question_template_version" \
        --debug "$debug" \
        --system_prompt "$system_prompt"
}


# ========================
# Loop over Models
# ========================
for model_path in "${model_list[@]}"; do

    # ----- oxe200 -----
    if [ "$run_oxe300" = true ]; then
        # # v0
        # run_infer "$model_path" \
        # "/mnt/petrelfs/share/efm_p/sys2_data/oxe_sys2_bench/oxe_300_raw_0709.jsonl" \
        # "/mnt/petrelfs/share/efm_p/sys2_data/oxe_sys2_bench/" \
        # "default" $bs_oxe300 $sys_p_oxe300

        # v0.1
        # run_infer "$model_path" \
        # "/mnt/petrelfs/share/efm_p/sys2_data/oxe_sys2_bench/ManipInterface_bbox500_0710.jsonl" \
        # "/mnt/petrelfs/share/efm_p/sys2_data/oxe_sys2_bench/" \
        # "default" $bs_oxe300 $sys_p_oxe300

        # v1.0
        # run_infer "$model_path" \
        # "/mnt/petrelfs/share/efm_p/sys2_data/oxe_sys2_bench/ManipInterface_bbox500_0710v1.jsonl" \
        # "/mnt/petrelfs/share/efm_p/sys2_data/oxe_sys2_bench/" \
        # "default" $bs_oxe300 $sys_p_oxe300

        # v1.1
        run_infer "$model_path" \
        "/mnt/petrelfs/share/efm_p/sys2_data/oxe_sys2_bench/ManipInterface_bbox500_instruction.jsonl" \
        "/mnt/petrelfs/share/efm_p/sys2_data/oxe_sys2_bench/" \
        "default" $bs_oxe300 $sys_p_oxe300
    fi

    # ----- bench200 -----
    if [ "$run_bench200" = true ]; then
        # v1
        # run_infer "$model_path" \
        # "/mnt/petrelfs/share/efm_p/zhuyangkun/sys2_data/genmanip_sim_data/bench200_v1/json/bench200_gd.jsonl" \
        # "/mnt/petrelfs/share/efm_p/zhuyangkun/sys2_data/genmanip_sim_data/bench200_v1/" \
        # "default" $bs_bench200v1 $sys_p_bench200v1

        # v2.2
        run_infer "$model_path" \
        "/mnt/petrelfs/share/efm_p/sys2_data/genmanip_sim_data/bench200_v2/json/bench200_gd_v2.3.jsonl" \
        "/mnt/petrelfs/share/efm_p/sys2_data/genmanip_sim_data/bench200_v2/" \
        "default" $bs_bench200 $sys_p_bench200
    fi

    # ----- Where2Place -----
    if [ "$run_where2place" = true ]; then
        run_infer "$model_path" \
            "/mnt/petrelfs/share/efm_p/sys2_data/where2place/where2place/where2place_llava/bbox_questions_wo_format.jsonl" \
            "/mnt/petrelfs/share/efm_p/sys2_data/where2place/where2place/" \
            "default" $bs_where2place $sys_p_where2place
    fi

    # ----- ERQA -----
    if [ "$run_erqa" = true ]; then
        run_infer "$model_path" \
        "/mnt/petrelfs/share/efm_p/sys2_data/erqa/erqa_llava_format.jsonl" \
        "/mnt/petrelfs/share/efm_p/sys2_data/erqa/image" \
        "default" $bs_erqa $sys_p_erqa
    fi 

    # ----- RefIndoor -----
    if [ "$run_refindoor" = true ]; then
        run_infer "$model_path" \
            "/mnt/petrelfs/share/efm_p/sys2_data/roboreflt/roboreflt_llava/roborefit_testA.json" \
            "/mnt/petrelfs/share/efm_p/sys2_data/roboreflt" \
            "refindoor" $bs_refindoor $sys_p_refindoor
    fi
    # ----- RefCOCO -----
    if [ "$run_refcoco" = true ]; then
        run_infer "$model_path" \
            "/mnt/petrelfs/share/efm_p/sys2_data/coco/refcoco_llava/refcoco+_val.jsonl" \
            "/mnt/petrelfs/share/efm_p/sys2_data/coco/" \
            "qwen_grounding" $bs_refcoco $sys_p_refcoco

        run_infer "$model_path" \
            "/mnt/petrelfs/share/efm_p/sys2_data/coco/refcoco_llava/refcoco_val.jsonl" \
            "/mnt/petrelfs/share/efm_p/sys2_data/coco/" \
            "qwen_grounding" $bs_refcoco $sys_p_refcoco

        run_infer "$model_path" \
            "/mnt/petrelfs/share/efm_p/sys2_data/coco/refcoco_llava/refcocog_val.jsonl" \
            "/mnt/petrelfs/share/efm_p/sys2_data/coco/" \
            "qwen_grounding" $bs_refcoco $sys_p_refcoco
    fi


    # ----- Demo Evaluation -----
    if [ "$run_demo_eval" = true ]; then
        # run_infer "$model_path" "/ssd/home/zhuyangkun/data/process_data/sys2_data_0409/dataset/json_llava/t2_sys2_data_0409_rn103_rr0.8_tysimple_2025041414.json" "$demo_valdata_prompt_version" $bs_demo_eval $sys_p_demo_eval
        
        # ----- OCR Demo Evaluation -----
        run_infer "$model_path" "/ssd/home/zhuyangkun/data/process_data/benchmark_ocr_real_data_250515/benchmark_ocr_real_data_250515.jsonl" "$demo_valdata_prompt_version" $bs_demo_eval $sys_p_demo_eval
    fi
done
