import os
import re
import json
import math
import argparse
import numpy as np

json_name_list = [
    "eval_details_contact_box.json",
    "eval_details_current_box.json",
    "eval_details_final_box.json",
    "eval_details_gripper_det.json",
    "eval_details_traj.json",
    "eval_details_traj_wo_init_pos.json",
    "eval_details_contact_decide.json",
    "eval_details_grasppose_choice.json",
    "eval_details_grounding_choice.json",
    "eval_details_traj_choice.json",
    "eval_details_traj_direction_choice.json",
    "eval_details_traj_direction_choice_with_traj.json",
    "eval_details_trajlang_choice.json",
    "eval_details_trajlang_sub_choice.json",
]

task_type = [
    "box",
    "box",
    "box",
    "box",
    "traj",
    "traj",
    "decide",
    "decide",
    "decide",
    "decide",
    "decide",
    "decide",
    "decide",
    "decide"
]

root_dir = "/mnt/petrelfs/wangziqin/project/System2VLA/qwen-vl-finetune/results"


def extract_bbox_answer(content):
    pattern = r"[\[\(]?\s*(-?\d*\.?\d+)\s*,\s*(-?\d*\.?\d+)\s*,\s*(-?\d*\.?\d+)\s*,\s*(-?\d*\.?\d+)\s*[\]\)]?"
    match = re.search(pattern, content)
    if match:
        try:
            res = np.array(json.loads(match.group())).reshape(-1)
            return res.tolist()
        except:
            pass
    pattern = r'\[\s*\[\s*\d+\s*,\s*\d+\s*\]\s*,\s*\[\s*\d+\s*,\s*\d+\s*\]\s*\]'
    match = re.search(pattern, content)
    if match:
        try:
            res = np.array(json.loads(match.group())).reshape(-1)
            return res.tolist()
        except:
            pass
    res = extract_number(content)
    if len(res) < 4:
        return [0,0,0,0]
    else:
        return res[:4]


def extract_traj_answer(content, remove_idx=False):
    if remove_idx:
        for i in range(10):
            content = content.replace(f"{i}.", "")
    res = extract_number(content)
    if len(res) % 2 != 0:
        res = res[:-1]
    return res


def compute_iou(b1,b2):
    x1,y1 = max(b1[0],b2[0]), max(b1[1],b2[1])
    x2,y2 = min(b1[2],b2[2]), min(b1[3],b2[3])
    inter = max(0,x2-x1)*max(0,y2-y1)
    a1 = (b1[2]-b1[0])*(b1[3]-b1[1])
    a2 = (b2[2]-b2[0])*(b2[3]-b2[1])
    uni = a1+a2-inter
    return inter/uni if uni>0 else 0.0


def dtw(P, Q):
    n, m = len(P), len(Q)
    dtw_matrix = np.full((n + 1, m + 1), np.inf)
    dtw_matrix[0, 0] = 0

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = np.linalg.norm(P[i - 1] - Q[j - 1])
            dtw_matrix[i, j] = cost + min(dtw_matrix[i - 1, j],
                                          dtw_matrix[i, j - 1],
                                          dtw_matrix[i - 1, j - 1])
    return dtw_matrix[n, m]


def extract_number(content):
    pattern = r'\d+\.?\d*|\.\d+'
    numbers = re.findall(pattern, content)
    numbers = [float(num) for num in numbers]
    return numbers


def smart_resize(
    height: int,
    width: int,
    factor: int = 28,
    min_pixels: int = 56 * 56,
    max_pixels: int = 14 * 14 * 4 * 1280,
):
    """Rescales the image so that the following conditions are met:
    1. Both dimensions (height and width) are divisible by 'factor'.
    2. The total number of pixels is within the range ['min_pixels', 'max_pixels'].
    3. The aspect ratio of the image is maintained as closely as possible.
    """
    if height < factor or width < factor:
        raise ValueError(
            f"height:{height} or width:{width} must be larger than factor:{factor}"
        )
    elif max(height, width) / min(height, width) > 200:
        raise ValueError(
            f"absolute aspect ratio must be smaller than 200, got {max(height, width) / min(height, width)}"
        )
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = math.floor(height / beta / factor) * factor
        w_bar = math.floor(width / beta / factor) * factor
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


def convert_intern_data(data, image_size):
    data = np.array(data).reshape(-1, 2)
    data[:, 0] = data[:, 0] * image_size[0] / 1000
    data[:, 1] = data[:, 1] * image_size[1] / 1000
    return data


def convert_qwen_data(data, image_size):
    new_h, new_w = smart_resize(
        image_size[0], image_size[1]
    )
    data = np.array(data).reshape(-1, 2)
    data[:, 0] = data[:, 0] * image_size[0] / new_h
    data[:, 1] = data[:, 1] * image_size[1] / new_w
    return data


def convert_llava_data(data, image_size):
    data = np.array(data).reshape(-1, 2)
    data[:, 0] = data[:, 0] * image_size[0]
    data[:, 1] = data[:, 1] * image_size[1]
    return data

def eval(s, task_type, source):
    # find all number
    image_size = s['image_size'] if 'image_size' in s else None
    if image_size is None:
        if 'RH20T' in s['id']:
            image_size = (640, 480)
        else:
            image_size = (320, 180)
    
    if task_type == "box":
        pred_box = np.array(extract_bbox_answer(s['pred'])[:4])
        gt_box = np.array(extract_number(s['gt'])[:4])
        if source == "intern":
            pred_box = convert_intern_data(pred_box, image_size).reshape(-1).tolist()
            gt_box = convert_intern_data(gt_box, image_size).reshape(-1).tolist()
        elif source == "qwen":
            pred_box = convert_qwen_data(pred_box, image_size).reshape(-1).tolist()
            gt_box = convert_qwen_data(gt_box, image_size).reshape(-1).tolist()
        elif source == "api":
            pred_box = pred_box.reshape(-1).tolist()
            gt_box = convert_qwen_data(gt_box, image_size).reshape(-1).tolist()
        elif source == "llava":
            pred_box = convert_llava_data(pred_box, image_size).reshape(-1).tolist()
            gt_box = convert_llava_data(gt_box, image_size).reshape(-1).tolist()
        
        res = compute_iou(pred_box, gt_box)
    
    elif task_type == "traj":
        pred_traj = np.array(extract_traj_answer(s['pred'])).reshape(-1, 2)
        gt_traj = np.array(extract_traj_answer(s['gt'])).reshape(-1, 2)
        if source == "intern":
            pred_traj = convert_intern_data(pred_traj, image_size)
            gt_traj = convert_intern_data(gt_traj, image_size)
        elif source == "qwen":
            pred_traj = convert_qwen_data(pred_traj, image_size)
            gt_traj = convert_qwen_data(gt_traj, image_size)
        elif source == "api":
            pred_traj = pred_traj
            gt_traj = convert_qwen_data(gt_traj, image_size)
        elif source == "llava":
            pred_traj = convert_llava_data(pred_traj, image_size)
            gt_traj = convert_llava_data(gt_traj, image_size)
            
        res = dtw(pred_traj, gt_traj)
        if math.isinf(res):
            res = 0
    
    else:
        pred = s['pred'].strip()
        gt = s['gt'].strip()

        if gt in pred[-100:] or pred in gt:
            # print("========================================")
            # print(f"Correct Prediction: {pred} | GT: {gt}")
            res = 1
        else:
            res = 0
    return res

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="internvl")
    args = parser.parse_args()

    model = args.model
    if "intern" in model.lower():
        source = "intern"
    elif 'qwen25' in model.lower() or 'brain' in model.lower():
        source = "qwen"
    elif 'llava' in model.lower():
        source = "llava"
    else:
        source = "api"

    final_res = {}
    for json_name, t in zip(json_name_list, task_type):
        json_path = os.path.join(root_dir, model, json_name)
        # json_path = os.path.join("/mnt/petrelfs/wangziqin/project/System2VLA/qwen-vl-finetune/results/results/manip_sys2_qwen25_3b_gdata_udata_manipvqa_cot_generaldata/checkpoint-42401", json_name)
        if not os.path.exists(json_path):
            print(f"{json_path} not exists!")
            continue
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        res_list = []
        for s in data:
            # try:
            res = eval(s, t, source)
            res_list.append(res)
            # except:
            #     continue
        
        if t == "box":
            final_res[json_name] = sum(res_list) / len(res_list)
            accuracy = sum([1 if r > 0.1 else 0 for r in res_list]) / len(res_list)
            print(f"{json_name} : Mean IOU: {final_res[json_name]} Accuracy (IOU > 0.1): {accuracy}")
        elif t == "traj":
            # remove 0
            res_list = [r for r in res_list if r > 0]
            final_res[json_name] = sum(res_list) / len(res_list)
            print(f"{json_name} :Mean DTW: {final_res[json_name]}")
        else:
            final_res[json_name] = sum(res_list) / len(res_list)
            print(f"{json_name} : Accuracy: {final_res[json_name]}")

    
    # for k, v in final_res.items():
    #     print(f"{k}: {v:.4f}")
    
    avg_score = sum(final_res.values()) / len(final_res)
    print(f"Average Score: {avg_score:.4f}")