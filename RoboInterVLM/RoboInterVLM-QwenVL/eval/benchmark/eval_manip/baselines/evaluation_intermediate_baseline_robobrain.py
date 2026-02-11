#!/usr/bin/env python
"""
Baseline evaluation script for RoboBrain models.
"""
import argparse
import os
import json
import sys
import torch
from tqdm import tqdm
from pathlib import Path
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.append(str(project_root))

from eval.benchmark.eval_manip.image_reader import get_reader

reader = get_reader()

TASK_TYPE_MAP = {
    "contact_box": "grounding",
    "final_box": "grounding",
    "current_box": "grounding",
    "gripper_det": "grounding",
    "traj": "trajectory",
    "traj_wo_init_pos": "trajectory",
    "roborefit_testB": "default",
    "point_questions_wo_format": "pointing",
    "refcocog_val": "grounding",
    "refcoco_val": "grounding",
    "refcoco+_val": "grounding",
    "contact_decide": "default",
    "grasppose_choice": "default",
    "grounding_choice": "default",
    "traj_choice": "default",
    "traj_direction_choice_with_traj": "default",
    "traj_direction_choice": "default",
    "trajlang_choice": "default",
    "trajlang_sub_choice": "default",
    "manip_qa": "default",
    "robovqa": "default",
}

NEED_PARSE_PROMPT = ["contact_box", "current_box", "traj", "traj_wo_init_pos"]


def get_prompt(question_template_version):
    if question_template_version == "qwen_grounding":
        return "Locate the {Question}, output the bbox coordinates."
    elif question_template_version == "refindoor":
        return "Give the box coordinates according to the instruction, Instruction: {Question}, output the bbox coordinates in the format of [x1, y1, x2, y2]."
    else:
        return "{Question}"


def parse_input(s):
    s = s.split('\'')[3].split(',')[0]
    return s


def collate_batch(examples, processor, device, task, enable_thinking, prompt, task_name):
    """Collate batch for RoboBrain evaluation."""
    messages = []

    for ex in examples:
        is_local = ex.get("is_local", False)

        if isinstance(ex["image_path"], list):
            if '#' in ex["image_path"][0] and not is_local:
                images = [reader.get_images_for_VQA(fp) for fp in ex["image_path"]]
            else:
                images = [fp for fp in ex["image_path"]]
        else:
            if '#' in ex["image_path"] and not is_local:
                images = [reader.get_images_for_VQA(ex["image_path"])]
            else:
                images = [ex["image_path"]]

        # Parse task description
        if task_name in NEED_PARSE_PROMPT:
            task_desc = parse_input(ex['prompt'])
        elif task_name == "gripper_det":
            task_desc = "The gripper"
        else:
            task_desc = ex['prompt'].replace("<image>\n", "")

        task_desc = prompt.replace("{Question}", task_desc)

        # Build text based on task type
        if task == "pointing":
            text = f"{task_desc}. Your answer should be formatted as a list of tuples, i.e. [(x1, y1), (x2, y2), ...], where each tuple contains the x and y coordinates of a point satisfying the conditions above. The coordinates should indicate the normalized pixel locations of the points in the image."
        elif task == "affordance":
            text = f"You are a robot using the joint control. \"{task_desc}\". Please predict a possible affordance area of the end effector."
        elif task == "trajectory":
            text = f"You are a robot using the joint control. \"{task_desc}\". Please predict up to 10 key trajectory points to complete the task. Your answer should be formatted as a list of tuples, i.e. [[x1, y1], [x2, y2], ...], where each tuple contains the x and y coordinates of a point."
        elif task == "grounding":
            text = f"Please provide the bounding box coordinate of the region this sentence describes: {task_desc}."
        else:
            text = task_desc

        ex["parsed_prompt"] = text

        messages.append([{
            "role": "user",
            "content": [
                *[{"type": "image", "image": image} for image in images],
                {"type": "text", "text": f"{text}"},
            ],
        }])

    texts = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)

    if enable_thinking:
        texts = [f"{text}<think>" for text in texts]
    else:
        texts = [f"{text}<think></think><answer>" for text in texts]

    inputs = processor(
        text=texts,
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(device, dtype=torch.bfloat16)

    return inputs, examples


def evaluate(records, model, processor, device, batch_size=8, max_new_tokens=512,
             task="grounding", task_name="", enable_thinking=False, question_template_version=""):
    """Run evaluation."""
    details = []
    prompt = get_prompt(question_template_version)

    for i in tqdm(range(0, len(records), batch_size), desc="Evaluating"):
        batch = records[i:i+batch_size]
        inputs, metas = collate_batch(batch, processor, device, task, enable_thinking, prompt, task_name)

        with torch.inference_mode():
            generated_ids = model.generate(**inputs, max_new_tokens=768, do_sample=True, temperature=0.7)

        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_texts = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        for ex, pred in zip(metas, output_texts):
            if enable_thinking:
                thinking_text = pred.split("</think>")[0].replace("<think>", "").strip()
                answer_text = pred.split("</think>")[1].replace("<answer>", "").replace("</answer>", "").strip() if "</think>" in pred else pred
            else:
                answer_text = pred.replace("<answer>", "").replace("</answer>", "").strip()

            details.append({
                "id": ex["id"],
                "pred": answer_text,
                "gt": ex["ground_truth"],
                "image_path": ex["image_path"],
                "input": ex["parsed_prompt"]
            })

    return details


def main():
    parser = argparse.ArgumentParser(description="RoboBrain baseline evaluation")
    parser.add_argument("--model_path", type=str, required=True, help="Path to RoboBrain model (e.g., BAAI/RoboBrain2.0-7B)")
    parser.add_argument("--json_path", type=str, required=True, help="Path to evaluation JSON")
    parser.add_argument("--image_dir", type=str, default="", help="Image directory prefix")
    parser.add_argument("--output_dir", type=str, default="", help="Output directory (defaults to model_path)")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--type", type=str, default='')
    parser.add_argument("--question_type", type=str, default="")
    parser.add_argument("--enable_thinking", action="store_true", help="Enable thinking mode")
    args = parser.parse_args()

    if args.image_dir == "1":
        args.image_dir = ""

    name = args.json_path.split("/")[-1].split(".")[0]
    outd = args.output_dir if args.output_dir else args.model_path

    # Check if already evaluated
    output_file = f"eval_details_{args.type}.json" if args.type else f"eval_details_{name}.json"
    output_path = os.path.join(outd, output_file)
    if os.path.exists(output_path):
        print(f"Eval details already exists, skip eval: {output_path}")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype="auto",
        device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(args.model_path)
    processor.tokenizer.padding_side = 'left'

    # Load JSON
    if any(x in args.json_path for x in ['roborefit', 'where2place', 'coco']):
        raw = []
        with open(args.json_path, "r", encoding="utf-8") as f:
            for line in f:
                raw.append(json.loads(line))
    else:
        with open(args.json_path, "r", encoding="utf-8") as f:
            raw = json.load(f)

    # Build eval records
    records = []
    for item in raw:
        fp = item["images"]
        if isinstance(fp, list):
            fp = [os.path.join(args.image_dir, i) for i in fp]
        else:
            fp = os.path.join(args.image_dir, fp)
        records.append({
            "id": item["id"],
            "image_path": fp,
            "is_local": item.get('is_local', False),
            "prompt": item["conversations"][0]["value"],
            "ground_truth": item["conversations"][1]["value"]
        })

    # Determine task type
    task = TASK_TYPE_MAP.get(args.type, "default")

    print(f"Evaluating {len(records)} samples...")
    details = evaluate(
        records, model, processor, device,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        task=task,
        enable_thinking=args.enable_thinking,
        task_name=args.type,
        question_template_version=args.question_type
    )

    # Save results
    os.makedirs(outd, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(details, f, indent=2, ensure_ascii=False)
    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
