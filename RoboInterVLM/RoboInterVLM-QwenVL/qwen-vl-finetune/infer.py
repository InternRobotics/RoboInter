#!/usr/bin/env python
"""
Unified inference script for Qwen-VL and LLaVA models.

Usage:
    # Qwen-VL model
    python infer.py --model_type qwen --model_path /path/to/qwen/model \
        --prompt "Describe this image" --images /path/to/img1.jpg /path/to/img2.jpg

    # LLaVA model
    python infer.py --model_type llava --model_path /path/to/llava/model \
        --prompt "What's in this image?" --images /path/to/img.jpg
"""
import argparse
import sys
import copy
from pathlib import Path
from typing import List, Union
import torch
from PIL import Image

project_root = Path(__file__).parent
sys.path.append(str(project_root))


class QwenVLInference:
    """Qwen-VL model inference wrapper."""

    def __init__(self, model_path: str, device: str = "cuda"):
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor

        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.processor.tokenizer = self.tokenizer
        self.processor.tokenizer.padding_side = 'left'

        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="auto",
        )
        self.model.eval()

    def infer(self, prompt: str, images: List[Union[str, Image.Image]],
              max_new_tokens: int = 512) -> str:
        """
        Run inference with Qwen-VL model.

        Args:
            prompt: Text prompt for the model
            images: List of image paths or PIL Image objects
            max_new_tokens: Maximum number of tokens to generate

        Returns:
            Generated text response
        """
        from qwen_vl_utils import process_vision_info
        from eval.benchmark.eval_manip.image_reader import get_reader

        reader = get_reader()

        # Process images
        processed_images = []
        for img in images:
            if isinstance(img, Image.Image):
                processed_images.append(img)
            elif isinstance(img, str):
                if '#' in img:
                    # LMDB format
                    processed_images.append(reader.get_images_for_VQA(img))
                else:
                    # Local file path
                    processed_images.append(img)

        # Build message
        messages = [[
            {
                "role": "user",
                "content": [
                    *[{"type": "image", "image": img} for img in processed_images],
                    {"type": "text", "text": prompt.replace('<image>\n', '')},
                ],
            }
        ]]

        # Apply chat template
        texts = [
            self.processor.apply_chat_template(
                msg, tokenize=False, add_generation_prompt=True
            )
            for msg in messages
        ]

        image_inputs, video_inputs = process_vision_info(messages)
        batch = self.processor(
            text=texts,
            images=image_inputs,
            videos=None,
            padding=True,
            return_tensors="pt"
        ).to(self.device)

        # Generate
        with torch.no_grad(), torch.autocast(self.device.type):
            out_ids = self.model.generate(**batch, max_new_tokens=max_new_tokens)

        output = self.processor.tokenizer.batch_decode(out_ids, skip_special_tokens=True)[0]
        # Remove the prompt from output
        output = output.replace(prompt.replace("<image>\n", ""), "").strip()

        return output


class LLaVAInference:
    """LLaVA model inference wrapper."""

    def __init__(self, model_path: str, device: str = "cuda"):
        from llava.model.builder import load_pretrained_model

        self.device = device
        model_name = "llava_qwen"
        self.tokenizer, self.model, self.image_processor, _ = load_pretrained_model(
            model_path, None, model_name, device_map="auto"
        )
        self.model.eval()
        self.config = self.model.config

    def infer(self, prompt: str, images: List[Union[str, Image.Image]],
              max_new_tokens: int = 512) -> str:
        """
        Run inference with LLaVA model.

        Args:
            prompt: Text prompt for the model
            images: List of image paths or PIL Image objects
            max_new_tokens: Maximum number of tokens to generate

        Returns:
            Generated text response
        """
        from llava.mm_utils import process_images, tokenizer_image_token
        from llava.constants import IMAGE_TOKEN_INDEX
        from llava.conversation import conv_templates
        from eval.benchmark.eval_manip.image_reader import get_reader

        reader = get_reader()

        # Process images
        processed_images = []
        for img in images:
            if isinstance(img, Image.Image):
                processed_images.append(img)
            elif isinstance(img, str):
                if '#' in img:
                    # LMDB format
                    processed_images.append(reader.get_images_for_VQA(img))
                else:
                    # Local file path
                    processed_images.append(Image.open(img).convert("RGB"))

        # Process images for model
        image_tensor = process_images(processed_images, self.image_processor, self.config)
        image_tensor = [_image.to(dtype=torch.float16, device=self.device) for _image in image_tensor]

        # Build conversation
        conv_template = "qwen_1_5"
        conv = copy.deepcopy(conv_templates[conv_template])
        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], None)
        prompt_question = conv.get_prompt()

        input_ids = tokenizer_image_token(
            prompt_question, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
        ).unsqueeze(0).to(self.device)

        image_sizes = [img.size for img in processed_images]

        # Generate
        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids,
                images=image_tensor,
                image_sizes=image_sizes,
                do_sample=False,
                temperature=0,
                max_new_tokens=max_new_tokens,
            )

        output = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
        return output


def create_model(model_type: str, model_path: str, device: str = "cuda"):
    """
    Factory function to create model inference wrapper.

    Args:
        model_type: 'qwen' or 'llava'
        model_path: Path to the model checkpoint
        device: Device to run inference on

    Returns:
        Model inference wrapper instance
    """
    if model_type.lower() == "qwen":
        return QwenVLInference(model_path, device)
    elif model_type.lower() == "llava":
        return LLaVAInference(model_path, device)
    else:
        raise ValueError(f"Unsupported model type: {model_type}. Use 'qwen' or 'llava'.")


def main():
    parser = argparse.ArgumentParser(description="Unified inference for Qwen-VL and LLaVA models")
    parser.add_argument("--model_type", type=str, required=True, choices=["qwen", "llava"],
                        help="Model type: 'qwen' or 'llava'")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the model checkpoint")
    parser.add_argument("--prompt", type=str, required=True,
                        help="Text prompt for inference")
    parser.add_argument("--images", type=str, nargs="+", required=True,
                        help="Image paths (local files or LMDB format with #)")
    parser.add_argument("--max_new_tokens", type=int, default=512,
                        help="Maximum number of tokens to generate")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to run on")

    args = parser.parse_args()

    print(f"Loading {args.model_type} model from {args.model_path}...")
    model = create_model(args.model_type, args.model_path, args.device)

    print(f"Running inference with {len(args.images)} image(s)...")
    result = model.infer(args.prompt, args.images, args.max_new_tokens)

    print("\n" + "=" * 50)
    print("Result:")
    print("=" * 50)
    print(result)

    return result


if __name__ == "__main__":
    main()
