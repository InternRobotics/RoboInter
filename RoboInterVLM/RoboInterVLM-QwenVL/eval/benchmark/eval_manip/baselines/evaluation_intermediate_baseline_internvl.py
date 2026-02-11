#!/usr/bin/env python
"""
Baseline evaluation script for InternVL models.
"""
import argparse
import os
import json
import sys
import torch
from tqdm import tqdm
from pathlib import Path
from transformers import AutoTokenizer, AutoModel
from PIL import Image
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode

project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.append(str(project_root))

from eval.benchmark.eval_manip.image_reader import get_reader

reader = get_reader()

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def get_prompt(question_template_version):
    if question_template_version == "qwen_grounding":
        return "Please provide the bounding box coordinate of the region this sentence describes: <ref>{Question}</ref>"
    elif question_template_version == "refindoor":
        return "Please provide the bounding box coordinate of the region this sentence describes: <ref>{Question}</ref>"
    else:
        return "{Question}"


def build_transform(input_size):
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])
    return transform


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1)
        for i in range(1, n + 1) for j in range(1, n + 1)
        if i * j <= max_num and i * j >= min_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size
    )

    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        split_img = resized_img.crop(box)
        processed_images.append(split_img)

    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)

    return processed_images


def load_image(image, input_size=448, max_num=12):
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(img) for img in images]
    pixel_values = torch.stack(pixel_values).to(torch.bfloat16)
    return pixel_values


def collate_batch(examples, device, prompt):
    """Collate batch for InternVL evaluation."""
    all_images, all_question, all_patches_list, all_image_size, all_eval_type = [], [], [], [], []

    for ex in examples:
        is_local = ex.get("is_local", False)

        if isinstance(ex["image_path"], list):
            if '#' in ex["image_path"][0] and not is_local:
                images = [reader.get_images_for_VQA(fp) for fp in ex["image_path"]]
            else:
                images = [Image.open(fp) for fp in ex["image_path"]]
        else:
            if '#' in ex["image_path"] and not is_local:
                images = [reader.get_images_for_VQA(ex["image_path"])]
            else:
                images = [Image.open(ex["image_path"])]

        all_images.append(torch.cat([load_image(i) for i in images]))
        all_question.append(prompt.replace("{Question}", ex['prompt'].replace('<image>\n', '')))
        all_patches_list.append(all_images[-1].size(0))
        all_image_size.append(images[0].size)
        all_eval_type.append(ex.get('eval_type'))

    all_images = torch.cat(all_images).to(torch.bfloat16).to(device)
    return all_images, all_question, all_patches_list, all_image_size, all_eval_type


def evaluate(records, model, tokenizer, device, batch_size=8, max_new_tokens=512,
             question_template_version=""):
    """Run evaluation."""
    details = []
    generation_config = dict(max_new_tokens=1024, do_sample=True)
    prompt = get_prompt(question_template_version)

    for i in tqdm(range(0, len(records), batch_size), desc="Evaluating"):
        batch = records[i:i+batch_size]
        all_images, all_question, all_patches_list, all_image_size, all_eval_type = collate_batch(
            batch, device, prompt
        )

        responses = model.batch_chat(
            tokenizer,
            all_images,
            num_patches_list=all_patches_list,
            questions=all_question,
            generation_config=generation_config
        )

        for ex, pred, image_size, eval_type in zip(batch, responses, all_image_size, all_eval_type):
            details.append({
                "id": ex["id"],
                "pred": pred,
                "image_path": ex["image_path"],
                "gt": ex["ground_truth"],
                "image_size": image_size,
                "eval_type": eval_type
            })

    return details


def main():
    parser = argparse.ArgumentParser(description="InternVL baseline evaluation")
    parser.add_argument("--model_path", type=str, required=True, help="Path to InternVL model")
    parser.add_argument("--json_path", type=str, required=True, help="Path to evaluation JSON")
    parser.add_argument("--image_dir", type=str, default="", help="Image directory prefix")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--type", type=str, default='')
    parser.add_argument("--question_type", type=str, default="")
    args = parser.parse_args()

    if args.image_dir == "1":
        args.image_dir = ""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    name = args.json_path.split("/")[-1].split(".")[0]

    # Load model
    model = AutoModel.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        load_in_8bit=False,
        low_cpu_mem_usage=True,
        use_flash_attn=True,
        trust_remote_code=True,
        device_map=device
    ).eval()

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True, use_fast=False)
    tokenizer.padding_side = 'left'

    # Load JSON
    if any(x in args.json_path for x in ['roborefit', 'where2place', 'bench200', 'ManipInterface', 'ref']):
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
            "id": item.get("id"),
            "image_path": fp,
            "prompt": item["conversations"][0]["value"],
            "ground_truth": item["conversations"][1]["value"],
            "is_local": item.get('is_local', False),
            "eval_type": item.get('raw_data', {}).get('eval_type')
        })

    print(f"Evaluating {len(records)} samples...")
    details = evaluate(
        records, model, tokenizer, device,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        question_template_version=args.question_type
    )

    # Save results
    output_file = f"eval_details_{args.type}.json" if args.type else f"eval_details_{name}.json"
    output_path = os.path.join(args.model_path, output_file)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(details, f, indent=2, ensure_ascii=False)
    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
