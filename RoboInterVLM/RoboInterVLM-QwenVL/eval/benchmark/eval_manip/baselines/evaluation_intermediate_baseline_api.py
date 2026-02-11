#!/usr/bin/env python
"""
Baseline evaluation script for API-based models (GPT-4o, Gemini, etc).
"""
import argparse
import os
import json
import sys
import torch
from tqdm import tqdm
from pathlib import Path

project_root = Path(__file__).parent
sys.path.append(str(project_root))

from query_model import ClientMap


def collate_batch(examples, image_root):
    """
    Collate batch for API evaluation.
    """
    all_images, all_question = [], []
    for ex in examples:
        if isinstance(ex["image_path"], list):
            if '#' in ex["image_path"][0]:
                images = [os.path.join(image_root, fp + '.jpg') for fp in ex["image_path"]]
            else:
                images = [fp for fp in ex["image_path"]]
        else:
            if '#' in ex["image_path"]:
                images = [os.path.join(image_root, ex["image_path"] + '.jpg')]
            else:
                images = [ex["image_path"]]

        all_images.append(images)
        all_question.append(ex["prompt"].replace("<image>\n", ""))

    return all_images, all_question, examples


def infer_api(client, prompt_qry, img_path=None):
    if img_path is not None:
        response_text = client(query=prompt_qry, img_path=img_path)
    else:
        response_text = client(query=prompt_qry)
    return response_text


def evaluate(records, client, image_root, batch_size=1):
    """Run evaluation with API client."""
    details = []

    for i in tqdm(range(0, len(records), batch_size), desc="Evaluating"):
        batch = records[i:i+batch_size]
        all_images, all_question, examples = collate_batch(batch, image_root)

        res = []
        for images, question in zip(all_images, all_question):
            api_ret = infer_api(client, question, img_path=images[0] if images else None)
            res.append(api_ret)

        for ex, pred in zip(examples, res):
            details.append({
                "id": ex["id"],
                "pred": pred,
                "gt": ex["ground_truth"],
                "image_path": ex["image_path"],
            })

    return details


def main():
    parser = argparse.ArgumentParser(description="API-based model evaluation")
    parser.add_argument("--model", type=str, default="gpt4o-mini",
                        choices=["gpt4o", "gpt4o-mini", "gemini", "qwenvl2.5-72B"])
    parser.add_argument("--api_key", type=str, required=True, help="API key")
    parser.add_argument("--base_url", type=str, default="", help="API base URL")
    parser.add_argument("--json_path", type=str, required=True, help="Path to evaluation JSON")
    parser.add_argument("--image_dir", type=str, default="", help="Image directory prefix")
    parser.add_argument("--image_root", type=str, default="#TODO", help="Root for LMDB images")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--type", type=str, default='')
    args = parser.parse_args()

    name = args.json_path.split("/")[-1].split(".")[0]

    # Initialize client
    client_cls = ClientMap[args.model]
    if args.base_url:
        client = client_cls(api_key=args.api_key, base_url=args.base_url)
    else:
        client = client_cls(api_key=args.api_key)

    # Load JSON
    if any(x in args.json_path for x in ['roborefit', 'where2place']):
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
            "prompt": item["conversations"][0]["value"],
            "ground_truth": item["conversations"][1]["value"]
        })

    print(f"Evaluating {len(records)} samples with {args.model}...")
    details = evaluate(records, client, args.image_root, batch_size=args.batch_size)

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    output_file = f"eval_details_{args.type}.json" if args.type else f"eval_details_{name}.json"
    output_path = os.path.join(args.output_dir, output_file)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(details, f, indent=2, ensure_ascii=False)
    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
