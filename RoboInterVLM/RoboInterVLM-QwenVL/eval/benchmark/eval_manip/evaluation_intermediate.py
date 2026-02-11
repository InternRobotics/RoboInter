#!/usr/bin/env python
"""
Evaluation script for intermediate representations (bbox, trajectory, contact point).
"""
import argparse
import os
import json
import sys
import torch
from tqdm import tqdm
from pathlib import Path
from transformers import AutoTokenizer, AutoProcessor

project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / "qwen-vl-finetune"))

from eval.benchmark.eval_manip.image_reader import get_reader
from qwenvl.train.modeling_qwen2_5_moe_vl import (
    Qwen2_5_VLMoeForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration,
)
from qwen_vl_utils import process_vision_info

reader = get_reader()


def get_prompt(question_template_version):
    if question_template_version == "qwen_grounding":
        return "Please provide the bounding box coordinate of the region this sentence describes: <ref>{Question}</ref>"
    elif question_template_version == "refindoor":
        return "Give the box coordinates according to the instruction, Instruction: {Question}, output the bbox coordinates."
    else:
        return "{Question}"


def collate_batch(examples, processor, device, prompt):
    """
    Collate batch for evaluation.

    Args:
        examples: list of dicts with keys: 'id', 'image_path', 'prompt', 'ground_truth'
        processor: model processor
        device: torch device
        prompt: prompt template
    """
    messages = []
    for ex in examples:
        is_local = ex.get("is_local", False)

        # Process images
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

        messages.append([{
            "role": "user",
            "content": [
                *[{"type": "image", "image": image} for image in images],
                {"type": "text", "text": prompt.replace("{Question}", ex["prompt"].replace('<image>\n', ''))},
            ],
        }])

    # Apply chat template
    texts = [
        processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
        for msg in messages
    ]

    image_inputs, video_inputs = process_vision_info(messages)
    batch = processor(
        text=texts,
        images=image_inputs,
        videos=None,
        padding=True,
        return_tensors="pt"
    )
    return batch.to(device), examples


def evaluate(records, model, processor, device, batch_size=8, max_new_tokens=512,
             dataset_name='', question_template_version=""):
    """Run evaluation on records."""
    details = []
    model.eval()
    prompt = get_prompt(question_template_version)

    for i in tqdm(range(0, len(records), batch_size), desc="Evaluating"):
        batch = records[i:i+batch_size]
        inputs, metas = collate_batch(batch, processor, device, prompt)

        try:
            with torch.no_grad(), torch.autocast(device.type):
                out_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        except Exception as e:
            print(f"Error: {e}")
            json.dump(metas, open(f"broken_metas_{dataset_name}_{i}.json", "w"), indent=2, ensure_ascii=False)
            continue

        outs = processor.tokenizer.batch_decode(out_ids, skip_special_tokens=True)

        for ex, pred in zip(metas, outs):
            details.append({
                "id": ex["id"],
                "pred": pred.strip(),
                "gt": ex["ground_truth"],
                "image_path": ex["image_path"],
            })

    return details


def main():
    parser = argparse.ArgumentParser(description="Evaluate intermediate representations")
    parser.add_argument("--model_path", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--json_path", type=str, required=True, help="Path to evaluation JSON file")
    parser.add_argument("--image_dir", type=str, default="", help="Image directory prefix")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--question_type", type=str, default="", help="Question template version")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    name = args.json_path.split("/")[-1].split(".")[0]

    # Check if already evaluated
    output_path = os.path.join(args.model_path, f"eval_details_{name}.json")
    if os.path.exists(output_path):
        print(f"Eval details for {name} already exists, skip eval.")
        return

    # Load model & processor
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True)
    processor = AutoProcessor.from_pretrained(args.model_path)
    processor.tokenizer = tokenizer
    processor.tokenizer.padding_side = 'left'

    if args.image_dir == "1":
        args.image_dir = ""

    # Load model
    if 'moe' in args.model_path.lower():
        model = Qwen2_5_VLMoeForConditionalGeneration.from_pretrained(
            args.model_path, torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="auto"
        )
    else:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            args.model_path, torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="auto"
        )
    model = model.half().cuda()

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
            "is_local": item.get("is_local", False),
            "prompt": item["conversations"][0]["value"],
            "ground_truth": item["conversations"][1]["value"]
        })

    print(f"Evaluating {len(records)} samples...")
    details = evaluate(
        records, model, processor, device,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        dataset_name=name,
        question_template_version=args.question_type,
    )

    # Save results
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(details, f, indent=2, ensure_ascii=False)
    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
