# RoboInterVLM-LLaVAOV

[English](README.md) | [中文](README_zh.md)

Manipulation-oriented Visual Question Answering based on [LLaVA-OneVision](https://github.com/LLaVA-VL/LLaVA-NeXT). This codebase fine-tunes LLaVA-OneVision for robotic manipulation understanding tasks, including grounding (bounding box, trajectory, contact point prediction) and understanding (choice-based QA) on real-world robot datasets (RH20T, DROID) and ManipVQA. Official model weight is in [https://huggingface.co/InternRobotics/RoboInter-VLM_llavaov_7B](https://huggingface.co/InternRobotics/RoboInter-VLM_llavaov_7B).

## Installation

```bash
conda env create -f environment.yaml
conda activate llava
pip install -e .
```

## Data Preparation

Before training, you need to set the dataset paths in `llava/data.py`. Search for all `TODO` entries and replace them with your local paths.

The following datasets are used:

| Dataset | Type | Description |
|---------|------|-------------|
| RH20T grounding | VLA grounding | Contact box, current box, final box, trajectory, gripper detection |
| DROID grounding | VLA grounding | Contact box, current box, final box, trajectory, gripper detection |
| RH20T understanding | Choice QA | Contact, grasp pose, grounding, trajectory language, trajectory direction |
| DROID understanding | Choice QA | Contact, grounding, trajectory language, trajectory direction |
| ManipVQA | Task planning | Manipulation task planning VQA |

Each dataset entry in `llava/data.py` has the following format:

```python
dataset_name = {
    "annotation_path": "/path/to/annotation.json",  # LLaVA-format JSON
    "data_path": "/path/to/images/",                 # Image root directory
}
```

The annotation JSON files should follow the LLaVA conversation format:

```json
[
    {
        "id": "unique_id",
        "images": ["relative/path/to/image.jpg"],
        "conversations": [
            {"from": "human", "value": "<image>\nYour question here"},
            {"from": "gpt", "value": "Answer here"}
        ]
    }
]
```

> **Note: Coordinate Normalization**
>
> In the LLaVA-OneVision version, all spatial coordinates (bounding boxes, trajectory points, contact points, etc.) are **normalized to [0, 1]** relative to the original image dimensions. This is different from Qwen-VL series models which use smart resize and represent coordinates in absolute pixel values based on the resized image. Make sure your annotation data follows this convention when preparing datasets.

## Training

### Single-machine multi-GPU (8 GPUs)

```bash
bash scripts/train/finetune_manip.sh
```

Before running, update the following in the script:

1. `GPUS_PER_NODE` - Number of GPUs (default: 8)
2. `pretrain_model` - Base model path or HuggingFace model ID
3. `VISION_MODEL_VERSION` - Vision encoder path or HuggingFace model ID
4. `output_dir` - Where to save checkpoints
5. `WANDB_ENTITY` / `WANDB_PROJECT` - Your W&B settings (or set `WANDB_MODE=disabled`)
6. `datasets` - Comma-separated dataset names registered in `llava/data.py`

### Key training arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--per_device_train_batch_size` | 2 | Batch size per GPU |
| `--gradient_accumulation_steps` | 2 | Gradient accumulation steps |
| `--learning_rate` | 2e-5 | Learning rate for language model |
| `--mm_vision_tower_lr` | 4e-6 | Learning rate for vision tower |
| `--num_train_epochs` | 1 | Number of training epochs |
| `--model_max_length` | 32768 | Maximum sequence length |
| `--deepspeed` | `scripts/zero3.json` | DeepSpeed config |

## Project Structure

```
.
├── llava/
│   ├── data.py                    # Dataset registry and configuration
│   ├── train/
│   │   ├── train.py               # Main training logic
│   │   ├── train_mem.py           # Training entry point (with flash attention)
│   │   └── llava_trainer.py       # Custom trainer with MeZO support
│   ├── model/
│   │   ├── llava_arch.py          # LLaVA architecture (multimodal fusion)
│   │   ├── language_model/        # Language model backends (Qwen, LLaMA, etc.)
│   │   ├── multimodal_encoder/    # Vision encoders (SigLIP, CLIP, MLCD, etc.)
│   │   ├── multimodal_projector/  # Vision-language projectors
│   │   └── multimodal_resampler/  # Vision token resamplers
│   ├── conversation.py            # Chat templates
│   ├── constants.py               # Constants
│   ├── mm_utils.py                # Multimodal utilities
│   └── utils.py                   # General utilities
├── scripts/
│   ├── train/
│   │   └── finetune_manip.sh      # Training launch script
│   └── zero3.json                 # DeepSpeed ZeRO-3 config
├── pyproject.toml                 # Package configuration
└── LICENSE
```

## Acknowledgements

This project is built upon [LLaVA-NeXT / LLaVA-OneVision](https://github.com/LLaVA-VL/LLaVA-NeXT). We thank the LLaVA team for their open-source contributions.

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.
