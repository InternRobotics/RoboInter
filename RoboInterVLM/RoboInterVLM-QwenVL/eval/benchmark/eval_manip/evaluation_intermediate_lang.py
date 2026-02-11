#!/usr/bin/env python
"""
Evaluation script for language understanding tasks.
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

from eval.utils.metric_utils import (
    compute_seg_metric,
    compute_trajectory_metric,
    compute_contact_point_metric,
    compute_language_metric,
)
from eval.benchmark.eval_manip.image_reader import get_reader
from qwenvl.train.modeling_qwen2_5_moe_vl import (
    Qwen2_5_VLMoeForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration,
)
from qwen_vl_utils import process_vision_info

reader = get_reader()


def metric_factory(type='contact_bbox'):
    if type == 'qa':
        return compute_language_metric
    elif 'bbox' in type:
        return compute_seg_metric
    elif 'traj' in type:
        return compute_trajectory_metric
    elif 'point' in type:
        return compute_contact_point_metric
    else:
        raise NotImplementedError(f"Metric {type} not implemented")


def collate_batch(examples, processor, device):
    """
    Collate batch for evaluation.
    """
    messages = []
    for ex in examples:
        content_image = []
        for image_item in ex['image_path']:
            content_image.append({"type": "image", "image": f"{image_item}"})
        messages.append([{
            "role": "user",
            "content": content_image + [{"type": "text", "text": ex["prompt"].replace('<image>', '')}],
        }])

    # Apply template
    texts = [
        processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
        for msg in messages
    ]

    # Extract vision inputs - handle LMDB images
    for message in messages:
        for msg in message:
            if isinstance(msg["content"], list):
                for ele in msg["content"]:
                    if ele['type'] == 'image':
                        if isinstance(ele['image'], list):
                            for idx in range(len(ele['image'])):
                                if '#' in ele['image'][idx]:
                                    ele['image'][idx] = reader.get_images_for_VQA(ele['image'][idx])
                        elif isinstance(ele['image'], str) and '#' in ele['image']:
                            ele['image'] = reader.get_images_for_VQA(ele['image'])

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
             eval_type='qa', dataset_name=None):
    """Run evaluation with metrics."""
    details, results = [], {}
    model.eval()

    for i in tqdm(range(0, len(records), batch_size), desc="Evaluating"):
        batch = records[i:i+batch_size]
        inputs, metas = collate_batch(batch, processor, device)

        try:
            with torch.no_grad(), torch.autocast(device.type):
                out_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
        except Exception as e:
            print(f"Error: {e}")
            json.dump(metas, open(f"broken_metas_{dataset_name}_{i}.json", "w"), indent=2, ensure_ascii=False)
            continue

        outs = processor.tokenizer.batch_decode(out_ids, skip_special_tokens=True)

        for ex, pred in zip(metas, outs):
            gt = ex["ground_truth"]
            p = pred.split("assistant\n")[-1]
            t = gt

            metric = metric_factory(eval_type)
            try:
                result = metric(p, t)
            except Exception as e:
                print(f"Metric error: {e}")
                continue

            res_dict = {
                "id": ex["id"],
                "pred": pred,
                "gt": gt,
                "p": p,
                "t": t,
                "metric": {}
            }

            for k, v in result.items():
                v = float(v)
                if k not in results:
                    results[k] = []
                if k not in res_dict["metric"]:
                    res_dict["metric"][k] = []

                if v != -1:
                    results[k].append(v)
                    res_dict["metric"][k].append(v)
                else:
                    res_dict["metric"][k] = v

            details.append(res_dict)

        # Print progress
        if i % 20 == 0 and results:
            resf = f"{i+batch_size}/{len(records)} ▶ "
            for k, v in results.items():
                if v:
                    resf += f"{k}={float(sum(v)/len(v)):.3f}, \t"
            tqdm.write(resf)

    results_avg = {k: sum(v)/len(v) for k, v in results.items() if v}
    return results_avg, details


def main():
    parser = argparse.ArgumentParser(description="Evaluate language understanding")
    parser.add_argument("--model_path", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--json_path", type=str, required=True, help="Path to evaluation JSON file")
    parser.add_argument("--image_dir", type=str, default="", help="Image directory prefix")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--type", type=str, default='qa', help="Evaluation type: qa, bbox, traj, point")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    name = args.json_path.split("/")[-1].split(".")[0]

    # Load model & processor
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True)
    processor = AutoProcessor.from_pretrained(args.model_path)
    processor.tokenizer = tokenizer
    processor.tokenizer.padding_side = 'left'

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

    # Load JSON
    with open(args.json_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    # Build eval records
    records = []
    for item in raw:
        fp = item["images"] if isinstance(item["images"], list) else [item["images"]]
        fp = [os.path.join(args.image_dir, i) if '#' not in i else i for i in fp]
        records.append({
            "id": item["id"],
            "image_path": fp,
            "prompt": item["conversations"][0]["value"],
            "ground_truth": item["conversations"][1]["value"]
        })

    print(f"Evaluating {len(records)} samples...")
    results_avg, details = evaluate(
        records, model, processor, device,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        eval_type=args.type,
        dataset_name=name,
    )

    # Print results
    log_s = ""
    for k, v in results_avg.items():
        log_s += f"{k}={float(v):.4f}, "
    print(log_s)

    # Save results
    outd = args.model_path
    with open(os.path.join(outd, f"lang_eval_results_{name}.txt"), "w") as f:
        f.write(log_s)
    with open(os.path.join(outd, f"lang_eval_details_{name}.json"), "w", encoding="utf-8") as f:
        json.dump(details, f, indent=2, ensure_ascii=False)
    print(f"Results saved to {outd}")


if __name__ == "__main__":
    main()
