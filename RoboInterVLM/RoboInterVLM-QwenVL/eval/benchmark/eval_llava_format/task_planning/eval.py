#!/usr/bin/env python
import os
import re
import json
import numpy as np
from tqdm import tqdm
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from pycocoevalcap.bleu.bleu import Bleu

    
    
def get_bleu_score(prediction, target):
    bleu1, bleu2, bleu3, bleu4 = 0, 0, 0, 0
    candidate = list(prediction.split(" "))
    reference = [list(target.split(" "))]
    smooth = SmoothingFunction().method1
    if len(reference[0]) <= 1:
        bleu1 = sentence_bleu(reference, candidate, weights=(1.00, 0.00, 0.00, 0.00), smoothing_function=smooth)
        bleu2 = sentence_bleu(reference, candidate, weights=(1.00, 0.00, 0.00, 0.00), smoothing_function=smooth)
        bleu3 = sentence_bleu(reference, candidate, weights=(1.00, 0.00, 0.00, 0.00), smoothing_function=smooth)
        bleu4 = sentence_bleu(reference, candidate, weights=(1.00, 0.00, 0.00, 0.00), smoothing_function=smooth)
    elif len(reference[0]) == 2:
        bleu1 = sentence_bleu(reference, candidate, weights=(1.00, 0.00, 0.00, 0.00), smoothing_function=smooth)
        bleu2 = sentence_bleu(reference, candidate, weights=(0.50, 0.50, 0.00, 0.00), smoothing_function=smooth)
        bleu3 = sentence_bleu(reference, candidate, weights=(0.50, 0.50, 0.00, 0.00), smoothing_function=smooth)
        bleu4 = sentence_bleu(reference, candidate, weights=(0.50, 0.50, 0.00, 0.00), smoothing_function=smooth)
    elif len(reference[0]) == 3:
        bleu1 = sentence_bleu(reference, candidate, weights=(1.00, 0.00, 0.00, 0.00), smoothing_function=smooth)
        bleu2 = sentence_bleu(reference, candidate, weights=(0.50, 0.50, 0.00, 0.00), smoothing_function=smooth)
        bleu3 = sentence_bleu(reference, candidate, weights=(0.33, 0.33, 0.33, 0.00), smoothing_function=smooth)
        bleu4 = sentence_bleu(reference, candidate, weights=(0.33, 0.33, 0.33, 0.00), smoothing_function=smooth)
    else:
        bleu1 = sentence_bleu(reference, candidate, weights=(1.00, 0.00, 0.00, 0.00), smoothing_function=smooth)
        bleu2 = sentence_bleu(reference, candidate, weights=(0.50, 0.50, 0.00, 0.00), smoothing_function=smooth)
        bleu3 = sentence_bleu(reference, candidate, weights=(0.33, 0.33, 0.33, 0.00), smoothing_function=smooth)
        bleu4 = sentence_bleu(reference, candidate, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smooth)
    
    score = (bleu1 + bleu2 + bleu3 + bleu4) / 4
    return {
        'score': score,
        'bleu1': bleu1,
        'bleu2': bleu2,
        'bleu3': bleu3,
        'bleu4': bleu4
    }
    
  
task_name_list = [
    "eval_details_all_plan_VQA_selection_judgement_val.json"
]

choice_task_name = [
    "Scene_Understanding", "Past_Multi_task_Selection", "Past_Primitive_Selection", "Future_Multi_task_Selection", "Future_Primitive_Selection", "Temporal_Understanding"
]

decide_task_name = [
    "Success_Negative_Task", "Success_Positive_Task", "Discriminative_Affordance_Negative_Task", "Discriminative_Affordance_Positive_Task"
]

planning_task_name = [
    "Planning_Task", "Past_Description_Task", "Planning_with_Context_Task", "Planning_Remaining_Steps_Task", "Generative_Affordance_Task", "Future_Prediction_Task",
]

def decide_task_type(s):
    for i in choice_task_name:
        if i in s:
            return "choice"
    for i in decide_task_name:
        if i in s:
            return "decide"
    for i in planning_task_name:
        if i in s:
            return "planning"
    raise ValueError("Unknown task type")

if __name__ == "__main__":
    dir_list = [
        # "/mnt/petrelfs/wangziqin/project/System2VLA/playground/Pretrained_models/gpt4o-mini",
        # "/mnt/petrelfs/wangziqin/project/System2VLA/playground/Pretrained_models/gemini",
        "/mnt/petrelfs/wangziqin/project/System2VLA/playground/Pretrained_models/Qwen2.5-VL-3B-Instruct",
        "/mnt/petrelfs/wangziqin/project/System2VLA/playground/Pretrained_models/Qwen2.5-VL-7B-Instruct",
        "/mnt/petrelfs/wangziqin/project/System2VLA/playground/Pretrained_models/InternVL3-1B",
        "/mnt/petrelfs/wangziqin/project/System2VLA/playground/Pretrained_models/InternVL3-2B",
        "/mnt/petrelfs/wangziqin/project/System2VLA/playground/Pretrained_models/InternVL3-8B",
        "/mnt/petrelfs/wangziqin/project/System2VLA/playground/Pretrained_models/RoboBrain2.0-3B",
        "/mnt/petrelfs/wangziqin/project/System2VLA/playground/Pretrained_models/RoboBrain2.0-7B",
        "/mnt/petrelfs/wangziqin/project/System2VLA/qwen-vl-finetune/results/manip_sys2_qwen25_3b_gdata_udata_manipvqa_generaldata",
        "/mnt/petrelfs/wangziqin/project/System2VLA/qwen-vl-finetune/results/manip_sys2_qwen25_7b_gdata_udata_manipvqa_generaldata",
        "/mnt/petrelfs/wangziqin/project/System2VLA/qwen-vl-finetune/results/llava-one-vision-7B_gdata_udata_manipvqa_generaldata"
    ]
    for dir in dir_list:
        for file_name in os.listdir(dir):
            if file_name not in task_name_list:
                continue
            print(f"Processing {file_name}...")
            with open(os.path.join(dir, file_name), "r") as f:
                results = json.load(f)
            
            all_scores = {
                "choice": [],
                "decide": [],
                "planning": []
            }
            for item in tqdm(results):
                pred_answer = item["pred"].split('\n')[-1].lower()
                gt_answer = item["gt"].lower()
                task_type = decide_task_type(item["id"])
                
                if task_type == "choice":
                    if gt_answer in pred_answer or pred_answer in gt_answer:
                        all_scores['choice'].append(1)
                    else:
                        all_scores['choice'].append(0)
                    continue
                
                if task_type == "decide":
                    if gt_answer in ['yes', 'no']:
                        if 'yes' in pred_answer:
                            pred_answer = 'yes'
                        elif 'no' in pred_answer:
                            pred_answer = 'no'
                    if gt_answer == pred_answer:
                        all_scores['decide'].append(1)
                    else:
                        all_scores['decide'].append(0)
                    continue
                
                if task_type == "planning":
                    all_scores['planning'].append(get_bleu_score(pred_answer, gt_answer))
                    continue
            
            print(len(all_scores['choice']), len(all_scores['decide']), len(all_scores['planning']))
            
            # Calculate final scores
            final_scores = {}
            for k, v in all_scores.items():
                if k in ['choice', 'decide']:
                    final_scores[k] = np.mean(v)
                else:
                    avg_score = {
                        'score': np.mean([x['score'] for x in v]),
                        'bleu1': np.mean([x['bleu1'] for x in v]),
                        'bleu2': np.mean([x['bleu2'] for x in v]),
                        'bleu3': np.mean([x['bleu3'] for x in v]),
                        'bleu4': np.mean([x['bleu4'] for x in v]),
                    }
                    final_scores[k] = avg_score
            print(f"============={dir.split('/')[-1]}==================")
            print(final_scores)
