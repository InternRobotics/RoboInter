# RoboInterVLM

**English** | [简体中文](./README_CN.md)

---

## Overview

RoboInterVLM is a vision-language-model (VLM) model framework for robotic manipulation tasks, built on the Qwen2.5-VL foundation model. This repository provides complete training and evaluation pipelines for fine-tuning vision-language models on robotic manipulation datasets.

## Key Features

- **Multi-task Learning**: Supports generation, understanding, and language-based robotic manipulation tasks
- **Multiple Datasets**: Integration with RH20T, DROID, and ManipVQA datasets
- **Comprehensive Evaluation**: Evaluation scripts for multiple intermediate representations including:
  - Contact point detection
  - Current/Final bounding box prediction
  - Gripper detection
  - Trajectory prediction
  - Planning tasks
  - General leaderboard testing

## Repository Structure

```
RoboInterVLM/
├── qwen-vl-finetune/          # Training code
│   ├── scripts/               # Training scripts
│   │   ├── sft.sh            # Main training script
│   │   ├── zero2.json        # DeepSpeed ZeRO-2 config
│   │   ├── zero3.json        # DeepSpeed ZeRO-3 config
│   │   └── zero3_offload.json # DeepSpeed ZeRO-3 with offload
│   ├── qwenvl/               # Model and data modules
│   │   ├── train/            # Training modules
│   │   └── data/             # Data processing
│   └── infer.py              # Inference script
├── eval/                      # Evaluation code
│   ├── benchmark/
│   │   ├── eval_manip/       # Manipulation task evaluation
│   │   │   ├── scripts/      # Evaluation scripts
│   │   │   ├── evaluation_intermediate.py      # Main eval script
│   │   │   └── evaluation_intermediate_lang.py # Language eval script
│   │   ├── eval_llava_format/  # LLaVA format evaluation
│   │   └── eval_vlmevalkit/    # VLMEvalKit benchmarks
│   └── utils/                # Evaluation utilities
├── data_process/             # Data processing scripts
└── playground/               # Playground for experiments
```

---

## Training

### Prerequisites

1. **Environment Setup**
```bash
# Install dependencies
pip install torch transformers deepspeed wandb
pip install qwen-vl-utils
```

2. **Data Preparation**
   - Refer [here](./data_process/README_DATA.md) to prepare dataset
   - Prepare your datasets in the required format
   - Update dataset paths in the training script

3. **Model Preparation**
   - Download [Qwen2.5-VL-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct) pretrained model
   - Update model path in the training script

### Configuration

Edit `qwen-vl-finetune/scripts/sft.sh` and update the following:

```bash
# Line 18: Output directory
OUTPUT_DIR=/your/output/path/qwen-vl-finetune/results/${TASK_NAME}

# Line 35-36: Weights & Biases config (optional)
export WANDB_ENTITY="your-wandb-entity"
export WANDB_PROJECT="your-wandb-project"

# Line 57: Pretrained model path
PRETRAIN_MODEL=/path/to/Qwen2.5-VL-3B-Instruct

# Line 69: Project root directory
cd /your/project/path/qwen-vl-finetune
```

### Training Script

The training script supports multiple dataset types:

**Generation Datasets:**
- `rh20t_vla_current_box_train`: Current object bounding box prediction
- `rh20t_vla_contact_box_train`: Contact point bounding box prediction
- `rh20t_vla_final_box_train`: Final object bounding box prediction
- `rh20t_vla_traj_qa_train`: Trajectory Q&A
- `rh20t_vla_gripper_det_qa_train`: Gripper detection Q&A
- Similar datasets for DROID

**Understanding Datasets:**
- `rh20t_contact_choice_qa_train`: Contact point choice Q&A
- `rh20t_grounding_choice_qa_train`: Grounding choice Q&A
- `rh20t_traj_lang_choice_qa_train`: Trajectory language choice Q&A
- Similar datasets for DROID

**Language Datasets:**
- `manipvqa_train`: Manipulation vision Q&A

### Run Training

```bash
# Single-node multi-GPU training (default: 8 GPUs)
cd qwen-vl-finetune
bash scripts/sft.sh my_experiment 8

# Specify custom GPU count
bash scripts/sft.sh my_experiment 4

# The script will:
# 1. Create output directory and backup the script
# 2. Start training with DeepSpeed ZeRO-3
# 3. Save checkpoints every 3000 steps
# 4. Log to both console and WandB (if configured)
```

### Training Parameters

Key hyperparameters in the training script:

- **Learning Rate**: 5e-6 (LLM), 1e-6 (Vision Tower)
- **Batch Size**: 4 per device × 2 gradient accumulation = effective batch size of 8 per GPU
- **Training Strategy**:
  - Freeze vision tower and MLP (`tune_mm_vision=False`, `tune_mm_mlp=False`)
- **Max Sequence Length**: 8192 tokens
- **Image Resolution**: Min 3136 pixels, Max 12845056 pixels
- **Mixed Precision**: BF16
- **Optimizer**: AdamW with cosine learning rate schedule

---

## Evaluation

### Manipulation Task Evaluation

The evaluation scripts are located in `eval/benchmark/eval_manip/scripts/`. Official model weights is in [https://huggingface.co/InternRobotics/RoboInter-VLM_llavaov_7B](https://huggingface.co/InternRobotics/RoboInter-VLM_llavaov_7B)

#### 1. Intermediate Representation Evaluation (`eval.sh`)

Evaluates various manipulation-related predictions:

```bash
cd eval/benchmark/eval_manip/scripts
bash eval.sh
```

**Configuration:**

Edit `eval.sh` to set:

```bash
# Line 8-10: Model configuration
model_path_list=(
    "/path/to/your/model/checkpoint"
)

model_name_list=(
    "your_model_name"
)

# Line 16-19: Evaluation data
json_path_list=(
    "/path/to/eval_data.json"
)

# Line 21-23: Evaluation type
type_list=(
    "contact_box"  # Options: contact_box, current_box, final_box,
                   #          gripper_det, traj, traj_wo_init_pos
)

# Line 25-27: Image directory
image_dirs=(
    "/path/to/images"
)

# Line 29-31: Question type
question_types=(
    "default"  # Options: default, qwen_grounding, refindoor, w2p
)
```

**Evaluation Types:**
- `contact_box`: Contact point bounding box prediction
- `current_box`: Current object bounding box
- `final_box`: Final object position bounding box
- `gripper_det`: Gripper detection
- `traj`: Full trajectory with initial position
- `traj_wo_init_pos`: Trajectory without initial position

**Evaluation Script Parameters:**
```bash
python evaluation_intermediate.py \
    --model_path /path/to/model \
    --json_path /path/to/eval_data.json \
    --image_dir /path/to/images \
    --type contact_box \
    --question_type default \
    --batch_size 16 \
    --max_new_tokens 512
```

#### 2. Language Understanding Evaluation (`eval_lang.sh`)

Evaluates language understanding capabilities:

```bash
cd eval/benchmark/eval_manip/scripts
bash eval_lang.sh
```

**Configuration:**

Edit `eval_lang.sh`:

```bash
# Line 8-10: Model paths
model_path_list=(
    "/path/to/your/model/checkpoint"
)

# Line 13-15: Evaluation data
json_path_list=(
    "/path/to/eval_data.json"
)

# Line 21: Image directory
image_dir="/path/to/images"
```

**Evaluation Script Parameters:**
```bash
python evaluation_intermediate_lang.py \
    --model_path /path/to/model \
    --json_path /path/to/eval_data.json \
    --image_dir /path/to/images \
    --type qa \
    --batch_size 16 \
    --max_new_tokens 512
```

#### 3. Baseline Model Evaluation

Scripts for evaluating baseline models are in `scripts/eval_baseline/`:

- `eval_api.sh`: Evaluate API-based models (GPT-4V, etc.)
- `eval_internvl.sh`: Evaluate InternVL models
- `eval_llava.sh`: Evaluate LLaVA models
- `eval_robobrain.sh`: Evaluate RoboBrain models

### Evaluation Output

Evaluation results are saved to:
- **Logs**: `eval/benchmark/eval_manip/scripts/logs/`
- **Format**: `{model_name}_{task_type}.log`

Each log contains:
- Inference results for each sample
- Aggregate metrics (accuracy, IoU, etc.)
- Performance statistics

---

## Model Architecture

The framework uses a modified Qwen2.5-VL model

---

## Data Format

### Training Data Format

Training data should be in JSON format with the following structure:

```json
{
    "id": "sample_id",
    "image": "path/to/image.jpg",
    "conversations": [
        {
            "from": "human",
            "value": "<image>\nQuestion about the image"
        },
        {
            "from": "gpt",
            "value": "Answer or prediction"
        }
    ]
}
```

### Evaluation Data Format

Evaluation data format:

```json
{
    "id": "sample_id",
    "image_path": "path/to/image.jpg",
    "question": "Evaluation question",
    "ground_truth": "Ground truth answer or coordinates"
}
```

---

## Citation

If you use this code in your research, please cite:

```bibtex
@article{system2vla2024,
    title={RoboInterVLM},
}
```

---

## License

This project is licensed under the Apache 2.0 License - see the LICENSE file for details.

---

## Acknowledgments

- Built on [Qwen-VL](https://github.com/QwenLM/Qwen-VL) by Alibaba Cloud

---
