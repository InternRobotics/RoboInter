# RoboInterVLM-LLaVAOV

[English](README.md) | [中文](README_zh.md)

基于 [LLaVA-OneVision](https://github.com/LLaVA-VL/LLaVA-NeXT) 的面向操作的视觉问答系统。本代码库在 LLaVA-OneVision 基础上微调，用于机器人操作理解任务，包括定位（边界框、轨迹、接触点预测）和理解（选择式问答），支持真实机器人数据集（RH20T、DROID）和 ManipVQA。

## 安装

```bash
conda env create -f environment.yaml
conda activate llava
pip install -e .
```

## 数据准备

训练前需要在 `llava/data.py` 中设置数据集路径。搜索所有 `TODO` 条目并替换为你的本地路径。

使用的数据集如下：

| 数据集 | 类型 | 描述 |
|--------|------|------|
| RH20T grounding | VLA 定位 | 接触框、当前框、目标框、轨迹、夹爪检测 |
| DROID grounding | VLA 定位 | 接触框、当前框、目标框、轨迹、夹爪检测 |
| RH20T understanding | 选择题 QA | 接触、抓取姿态、定位、轨迹语言、轨迹方向 |
| DROID understanding | 选择题 QA | 接触、定位、轨迹语言、轨迹方向 |
| ManipVQA | 任务规划 | 操作任务规划 VQA |

`llava/data.py` 中每个数据集的格式如下：

```python
dataset_name = {
    "annotation_path": "/path/to/annotation.json",  # LLaVA 格式的 JSON 标注文件
    "data_path": "/path/to/images/",                 # 图片根目录
}
```

标注 JSON 文件需遵循 LLaVA 对话格式：

```json
[
    {
        "id": "unique_id",
        "images": ["relative/path/to/image.jpg"],
        "conversations": [
            {"from": "human", "value": "<image>\n你的问题"},
            {"from": "gpt", "value": "回答内容"}
        ]
    }
]
```

> **注意：坐标归一化**
>
> 在 LLaVA-OneVision 版本中，所有空间坐标（边界框、轨迹点、接触点等）均**归一化到 [0, 1]** 区间，相对于原始图像尺寸。这与 Qwen-VL 系列模型不同——Qwen-VL 使用 smart resize，坐标以 resize 后图像的绝对像素值表示。请确保准备数据时标注遵循此约定。

## 训练

### 单机多卡训练（8 GPUs）

```bash
bash scripts/train/finetune_manip.sh
```

运行前需修改脚本中的以下配置：

1. `GPUS_PER_NODE` - GPU 数量（默认：8）
2. `pretrain_model` - 基座模型路径或 HuggingFace 模型 ID
3. `VISION_MODEL_VERSION` - 视觉编码器路径或 HuggingFace 模型 ID
4. `output_dir` - 检查点保存路径
5. `WANDB_ENTITY` / `WANDB_PROJECT` - W&B 配置（或设置 `WANDB_MODE=disabled` 关闭）
6. `datasets` - 在 `llava/data.py` 中注册的数据集名称，逗号分隔

### 关键训练参数

| 参数 | 默认值 | 描述 |
|------|--------|------|
| `--per_device_train_batch_size` | 2 | 每张 GPU 的 batch size |
| `--gradient_accumulation_steps` | 2 | 梯度累积步数 |
| `--learning_rate` | 2e-5 | 语言模型学习率 |
| `--mm_vision_tower_lr` | 4e-6 | 视觉塔学习率 |
| `--num_train_epochs` | 1 | 训练轮数 |
| `--model_max_length` | 32768 | 最大序列长度 |
| `--deepspeed` | `scripts/zero3.json` | DeepSpeed 配置文件 |

## 项目结构

```
.
├── llava/
│   ├── data.py                    # 数据集注册与配置
│   ├── train/
│   │   ├── train.py               # 主训练逻辑
│   │   ├── train_mem.py           # 训练入口（使用 flash attention）
│   │   └── llava_trainer.py       # 自定义 Trainer（支持 MeZO）
│   ├── model/
│   │   ├── llava_arch.py          # LLaVA 架构（多模态融合）
│   │   ├── language_model/        # 语言模型后端（Qwen、LLaMA 等）
│   │   ├── multimodal_encoder/    # 视觉编码器（SigLIP、CLIP、MLCD 等）
│   │   ├── multimodal_projector/  # 视觉-语言投影器
│   │   └── multimodal_resampler/  # 视觉 token 重采样器
│   ├── conversation.py            # 对话模板
│   ├── constants.py               # 常量定义
│   ├── mm_utils.py                # 多模态工具函数
│   └── utils.py                   # 通用工具函数
├── scripts/
│   ├── train/
│   │   └── finetune_manip.sh      # 训练启动脚本
│   └── zero3.json                 # DeepSpeed ZeRO-3 配置
├── pyproject.toml                 # 包配置
└── LICENSE
```

## 致谢

本项目基于 [LLaVA-NeXT / LLaVA-OneVision](https://github.com/LLaVA-VL/LLaVA-NeXT) 构建，感谢 LLaVA 团队的开源贡献。

## 许可证

本项目采用 Apache License 2.0 许可 - 详见 [LICENSE](LICENSE) 文件。
