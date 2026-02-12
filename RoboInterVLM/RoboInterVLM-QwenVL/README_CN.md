# RoboInterVLM

[English](./README.md) | **简体中文**

---

## 项目概述

RoboInterVLM 是一个面向机器人操作任务的视觉-语言（VLM）模型框架，基于 Qwen2.5-VL 基础模型，采用混合专家（MoE）架构。本仓库提供了完整的训练和评测流程，用于在机器人操作数据集上微调视觉语言模型。

## 核心特性

- **多任务学习**：支持生成、理解和基于语言的机器人操作任务
- **多数据集支持**：集成 RH20T、DROID 和 ManipVQA 数据集
- **全面评测**：提供多种中间表示的评测脚本，包括：
  - 接触点检测
  - 当前/最终边界框预测
  - 抓手检测
  - 轨迹预测
  - 规划任务
  - 通用榜单测试

## 目录结构

```
RoboInterVLM/
├── qwen-vl-finetune/          # 训练代码
│   ├── scripts/               # 训练脚本
│   │   ├── sft.sh            # 主训练脚本
│   │   ├── zero2.json        # DeepSpeed ZeRO-2 配置
│   │   ├── zero3.json        # DeepSpeed ZeRO-3 配置
│   │   └── zero3_offload.json # DeepSpeed ZeRO-3 带卸载
│   ├── qwenvl/               # 模型和数据模块
│   │   ├── train/            # 训练模块
│   │   └── data/             # 数据处理
│   └── infer.py              # 推理脚本
├── eval/                      # 评测代码
│   ├── benchmark/
│   │   ├── eval_manip/       # 操作任务评测
│   │   │   ├── scripts/      # 评测脚本
│   │   │   ├── evaluation_intermediate.py      # 主评测脚本
│   │   │   └── evaluation_intermediate_lang.py # 语言评测脚本
│   │   ├── eval_llava_format/  # LLaVA 格式评测
│   │   └── eval_vlmevalkit/    # VLMEvalKit 基准测试
│   └── utils/                # 评测工具
├── data_process/             # 数据处理脚本
└── playground/               # 实验环境
```

---

## 训练

### 环境准备

1. **环境配置**
```bash
# 安装依赖
pip install torch transformers deepspeed wandb
pip install qwen-vl-utils
```

2. **数据准备**
   - 参考[here](./data_process/README_DATA_zh.md)准备数据集
   - 按照要求的格式准备数据集
   - 在训练脚本中更新数据集路径

3. **模型准备**
   - 下载 [Qwen2.5-VL-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct) 预训练模型
   - 在训练脚本中更新模型路径

### 配置说明

编辑 `qwen-vl-finetune/scripts/sft.sh` 并更新以下配置：

```bash
# 第18行: 输出目录
OUTPUT_DIR=/your/output/path/qwen-vl-finetune/results/${TASK_NAME}

# 第35-36行: Weights & Biases 配置（可选）
export WANDB_ENTITY="your-wandb-entity"
export WANDB_PROJECT="your-wandb-project"

# 第57行: 预训练模型路径
PRETRAIN_MODEL=/path/to/Qwen2.5-VL-3B-Instruct

# 第69行: 项目根目录
cd /your/project/path/qwen-vl-finetune
```

### 训练脚本说明

训练脚本支持多种数据集类型：

**生成类数据集：**
- `rh20t_vla_current_box_train`: 当前物体边界框预测
- `rh20t_vla_contact_box_train`: 接触点边界框预测
- `rh20t_vla_final_box_train`: 最终物体边界框预测
- `rh20t_vla_traj_qa_train`: 轨迹问答
- `rh20t_vla_gripper_det_qa_train`: 抓手检测问答
- DROID 数据集的类似任务

**理解类数据集：**
- `rh20t_contact_choice_qa_train`: 接触点选择问答
- `rh20t_grounding_choice_qa_train`: 定位选择问答
- `rh20t_traj_lang_choice_qa_train`: 轨迹语言选择问答
- DROID 数据集的类似任务

**语言类数据集：**
- `manipvqa_train`: 操作视觉问答

### 运行训练

```bash
# 单机多卡训练（默认：8块GPU）
cd qwen-vl-finetune
bash scripts/sft.sh my_experiment 8

# 指定自定义GPU数量
bash scripts/sft.sh my_experiment 4

# 脚本会自动：
# 1. 创建输出目录并备份脚本
# 2. 使用 DeepSpeed ZeRO-3 启动训练
# 3. 每3000步保存一次检查点
# 4. 记录日志到控制台和 WandB（如已配置）
```

### 训练参数

训练脚本中的关键超参数：

- **学习率**: 5e-6（LLM），1e-6（视觉编码器）
- **批次大小**: 每设备4 × 2梯度累积步数 = 每GPU有效批次大小8
- **训练策略**:
  - 仅训练 MoE 层（`tune_moe=True`）
  - 冻结视觉编码器和MLP（`tune_mm_vision=False`, `tune_mm_mlp=False`）
- **最大序列长度**: 8192 tokens
- **图像分辨率**: 最小3136像素，最大12845056像素
- **混合精度**: BF16
- **优化器**: AdamW，余弦学习率调度

---

## 评测

### 操作任务评测

评测脚本位于 `eval/benchmark/eval_manip/scripts/` 目录。我们的预训练权重在 [https://huggingface.co/InternRobotics/RoboInter-VLM](https://huggingface.co/InternRobotics/RoboInter-VLM) 和 [https://huggingface.co/InternRobotics/RoboInter-VLM_qwenvl25_3b](https://huggingface.co/InternRobotics/RoboInter-VLM_qwenvl25_3b).

#### 1. 中间表示评测 (`eval.sh`)

评测各种操作相关的预测任务：

```bash
cd eval/benchmark/eval_manip/scripts
bash eval.sh
```

**配置说明：**

编辑 `eval.sh` 设置：

```bash
# 第8-10行: 模型配置
model_path_list=(
    "/path/to/your/model/checkpoint"
)

model_name_list=(
    "your_model_name"
)

# 第16-19行: 评测数据
json_path_list=(
    "/path/to/eval_data.json"
)

# 第21-23行: 评测类型
type_list=(
    "contact_box"  # 可选: contact_box, current_box, final_box,
                   #       gripper_det, traj, traj_wo_init_pos
)

# 第25-27行: 图像目录
image_dirs=(
    "/path/to/images"
)

# 第29-31行: 问题类型
question_types=(
    "default"  # 可选: default, qwen_grounding, refindoor, w2p
)
```

**评测类型说明：**
- `contact_box`: 接触点边界框预测
- `current_box`: 当前物体边界框
- `final_box`: 最终物体位置边界框
- `gripper_det`: 抓手检测
- `traj`: 包含初始位置的完整轨迹
- `traj_wo_init_pos`: 不含初始位置的轨迹

**评测脚本参数：**
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

#### 2. 语言理解评测 (`eval_lang.sh`)

评测语言理解能力：

```bash
cd eval/benchmark/eval_manip/scripts
bash eval_lang.sh
```

**配置说明：**

编辑 `eval_lang.sh`:

```bash
# 第8-10行: 模型路径
model_path_list=(
    "/path/to/your/model/checkpoint"
)

# 第13-15行: 评测数据
json_path_list=(
    "/path/to/eval_data.json"
)

# 第21行: 图像目录
image_dir="/path/to/images"
```

**评测脚本参数：**
```bash
python evaluation_intermediate_lang.py \
    --model_path /path/to/model \
    --json_path /path/to/eval_data.json \
    --image_dir /path/to/images \
    --type qa \
    --batch_size 16 \
    --max_new_tokens 512
```

#### 3. 基线模型评测

基线模型评测脚本位于 `scripts/eval_baseline/`:

- `eval_api.sh`: 评测基于API的模型（如 GPT-4V）
- `eval_internvl.sh`: 评测 InternVL 模型
- `eval_llava.sh`: 评测 LLaVA 模型
- `eval_robobrain.sh`: 评测 RoboBrain 模型

### 评测输出

评测结果保存在：
- **日志目录**: `eval/benchmark/eval_manip/scripts/logs/`
- **文件格式**: `{模型名称}_{任务类型}.log`

每个日志文件包含：
- 每个样本的推理结果
- 聚合指标（准确率、IoU等）
- 性能统计信息

---

## 模型架构

本框架使用改进的 Qwen2.5-VL 模型，集成了 MoE 层：

- **基础模型**: Qwen2.5-VL-3B-Instruct
- **视觉编码器**: 基于 ViT 的视觉塔
- **MoE 集成**: 稀疏混合专家层，实现高效扩展
- **自定义改进**:
  - 用于空间理解的 2D RoPE
  - 视觉-语言对齐的自定义注意力机制

---

## 数据格式

### 训练数据格式

训练数据应为 JSON 格式，结构如下：

```json
{
    "id": "样本ID",
    "image": "图片路径/image.jpg",
    "conversations": [
        {
            "from": "human",
            "value": "<image>\n关于图像的问题"
        },
        {
            "from": "gpt",
            "value": "答案或预测结果"
        }
    ]
}
```

### 评测数据格式

评测数据格式：

```json
{
    "id": "样本ID",
    "image_path": "图片路径/image.jpg",
    "question": "评测问题",
    "ground_truth": "真实答案或坐标"
}
```

---

## 引用

如果您在研究中使用了本代码，请引用：

```bibtex
@article{li2026robointer,
  title={RoboInter: A Holistic Intermediate Representation Suite Towards Robotic Manipulation},
  author={Li, Hao and Wang, Ziqin and Ding, Zi-han and Yang, Shuai and Chen, Yilun and Tian, Yang and Hu, Xiaolin and Wang, Tai and Lin, Dahua and Zhao, Feng and Liu, Si and Pang, Jiangmiao},
  journal={arXiv preprint arXiv:2602.09973},
  year={2025}
}
```

---

## 许可证

本项目采用 Apache 2.0 许可证 - 详见 LICENSE 文件。

---

## 致谢

- 基于阿里云的 [Qwen-VL](https://github.com/QwenLM/Qwen-VL) 构建
- 训练框架改编自 [FastChat](https://github.com/lm-sys/FastChat)
- 评测工具受 [LLaVA](https://github.com/haotian-liu/LLaVA) 启发

---
