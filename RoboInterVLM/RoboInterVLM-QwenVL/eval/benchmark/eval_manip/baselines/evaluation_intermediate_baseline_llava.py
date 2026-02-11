#!/usr/bin/env python
"""
Baseline evaluation script for LLaVA models.
"""
import argparse
import os
import json
import copy
import sys
import torch
from tqdm import tqdm
from pathlib import Path
from PIL import Image
from llava.model.builder import load_pretrained_model
from llava.mm_utils import process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX
from llava.conversation import conv_templates

project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.append(str(project_root))

from eval.benchmark.eval_manip.image_reader import get_reader

reader = get_reader()


def get_prompt(question_template_version):
    if question_template_version == "qwen_grounding":
        return "<image>\nPlease provide the bounding box coordinate of the region this sentence describes: <ref>{Question}</ref>"
    elif question_template_version == "refindoor":
        return "<image>\nGive the box coordinates according to the instruction, Instruction: {Question}, output the bbox coordinates."
    elif question_template_version == "w2p":
        return "<image>\n{Question}"
    else:
        return "{Question}"


def collate_batch(examples, image_processor, config, tokenizer, prompt, device):
    """
    Collate batch for LLaVA evaluation.
    Note: LLaVA only supports batch_size=1
    """
    assert len(examples) == 1, "LLaVA only supports batch_size=1"

    is_local = examples[0].get("is_local", False)

    # Process images
    if isinstance(examples[0]["image_path"], list):
        if not is_local and '#' in examples[0]["image_path"][0]:
            images = [reader.get_images_for_VQA(fp) for fp in examples[0]["image_path"]]
        else:
            images = [Image.open(fp).convert("RGB") for fp in examples[0]["image_path"]]
    else:
        if not is_local and '#' in examples[0]["image_path"]:
            images = [reader.get_images_for_VQA(examples[0]["image_path"])]
        else:
            images = [Image.open(examples[0]["image_path"]).convert("RGB")]

    image_tensor = process_images(images, image_processor, config)
    image_tensor = [_image.to(dtype=torch.float16, device=device) for _image in image_tensor]

    # Build conversation
    conv_template = "qwen_1_5"
    question = prompt.replace("{Question}", examples[0]["prompt"])
    examples[0]["parsed_prompt"] = question

    conv = copy.deepcopy(conv_templates[conv_template])
    conv.append_message(conv.roles[0], question)
    conv.append_message(conv.roles[1], None)
    prompt_question = conv.get_prompt()

    input_ids = tokenizer_image_token(
        prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
    ).unsqueeze(0).to(device)

    image_sizes = [image.size for image in images]

    return input_ids, image_tensor, image_sizes, examples


def evaluate(records, model, image_processor, config, tokenizer, device,
             batch_size=1, max_new_tokens=512, question_template_version=""):
    """Run evaluation."""
    details = []
    assert batch_size == 1, "LLaVA only supports batch_size=1"

    prompt = get_prompt(question_template_version)

    for i in tqdm(range(0, len(records), batch_size), desc="Evaluating"):
        batch = records[i:i+batch_size]
        input_ids, image_tensor, image_sizes, metas = collate_batch(
            batch, image_processor, config, tokenizer, prompt, device
        )

        output_ids = model.generate(
            input_ids,
            images=image_tensor,
            image_sizes=image_sizes,
            do_sample=False,
            temperature=0,
            max_new_tokens=max_new_tokens,
        )
        text_outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)

        for ex, pred in zip(metas, text_outputs):
            details.append({
                "id": ex["id"],
                "pred": pred,
                "gt": ex["ground_truth"],
                "image_path": ex["image_path"],
                "prompt": ex["parsed_prompt"],
            })

    return details


def main():
    parser = argparse.ArgumentParser(description="LLaVA baseline evaluation")
    parser.add_argument("--model_path", type=str, required=True, help="Path to LLaVA model")
    parser.add_argument("--json_path", type=str, required=True, help="Path to evaluation JSON")
    parser.add_argument("--image_dir", type=str, default="", help="Image directory prefix")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--question_type", type=str, default="", help="Question template version")
    args = parser.parse_args()

    if args.image_dir == "1":
        args.image_dir = ""

    device = "cuda"
    name = args.json_path.split("/")[-1].split(".")[0]

    # Load model & processor
    model_name = "llava_qwen"
    tokenizer, model, image_processor, max_length = load_pretrained_model(
        args.model_path, None, model_name, device_map="auto"
    )
    model.eval()

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
        records, model, image_processor, model.config, tokenizer, device,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        question_template_version=args.question_type,
    )

    # Save results
    output_path = os.path.join(args.model_path, f"eval_details_{name}.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(details, f, indent=2, ensure_ascii=False)
    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
