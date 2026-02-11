# RobotInterVQA: 机器人交互理解与生成视觉问答数据集

[English](./README_DATA.md) | **简体中文**

[RoboInter](https://github.com/InternRobotics/RoboInter)中的大规模抓取操作VQA数据集，涵盖**中间表征生成**、**中间表征理解**和**任务规划**三大方向。基于 [DROID](https://droid-dataset.github.io/) 和 [RH20T](https://rh20t.github.io/) 机器人数据集构建。

## 目录结构

```
robotinter/
├── Generation/          # 交互生成任务（定位、轨迹、接触）
│   ├── image/           # 图像文件（zip 压缩包，使用前需解压）
│   │   ├── train/{droid,rh20t}/
│   │   └── val/
│   └── meta/            # QA 标注 JSON 文件
│       ├── train/{droid,rh20t}/{origin_format,llava_format,smart_resize_format}/
│       └── val/{origin_format,llava_format,smart_resize_format}/
├── Understanding/       # 交互理解任务（选择题）
│   ├── image/
│   │   ├── train/{droid,rh20t}/
│   │   └── val/
│   └── meta/
│       ├── train/{droid,rh20t}/
│       └── val/
└── Task_planning/       # 任务规划与动作识别
    ├── image/
    │   ├── train/manipvqa/
    │   └── val/{planning,choice,decide}/
    └── meta/
        ├── train/manipvqa/
        └── val/{planning,choice,decide}/
```

## 快速开始

1. **解压图像**：所有图像以 `.zip` 格式存储，原地解压即可使用：
   ```bash
   cd RoboInter-VQA/Task_planning/image/train/manipvqa
   cat task_planning.zip.* > task_planning_full.zip
   cd ../../../../
   find . -name "*.zip" -execdir unzip -o {} \;
   ```

2. **加载标注**：配合[RoboInterVLM](https://github.com/InternRobotics/RoboInter/blob/main/RoboInterVLM/RoboInterVLM-QwenVL/qwen-vl-finetune/qwenvl/data/__init__.py).


## 坐标格式（仅 Generation 部分）

Generation 的标注提供了**三种坐标格式**。底层数据和图像完全相同，仅答案中的坐标表示方式不同：

| 格式 | 说明 | 示例 |
|---|---|---|
| `origin_format` | 原始图像分辨率下的像素坐标（`h` x `w`） | `[[72, 102], [192, 179]]` |
| `llava_format` | 归一化到 `[0, 1]` 范围的相对坐标, 例如 LLaVA-based 模型 | `[[0.22, 0.57], [0.60, 0.99]]` |
| `smart_resize_format` | 缩放后图像分辨率下的像素坐标（`new_h` x `new_w`），例如 Qwen-based 模型 | `[[69, 95], [184, 167]]` |

## JSON 文件说明

### Generation（交互生成，7 种任务）

每个条目的基本结构：
```json
{
  "id": "样本唯一标识",
  "task": "任务类型",
  "conversations": [{"from": "human", "value": "..."}, {"from": "gpt", "value": "..."}],
  "images": "图像相对路径",
  "gt": "真实标注值",
  "h": 180, "w": 320,
  "new_h": 168, "new_w": 308
}
```

| JSON 文件 | 任务名称 | 说明 | 输出格式 |
|---|---|---|---|
| `*_traj_qa.json` | 轨迹预测 | 给定任务描述和**起始位置**，预测夹爪未来的 K 个轨迹路点。 | `{"future_traj": [[x1,y1], ...]}` |
| `*_traj_qa_wo_init_pos.json` | 轨迹预测（无起始位置） | 与上述相同，但提示中**不提供**起始位置。 | `{"future_traj": [[x1,y1], ...]}` |
| `*_gripper_det_qa.json` | 夹爪检测 | 检测场景中机器人夹爪的当前边界框。 | `{"gripper_det_bbox": [[x1,y1],[x2,y2]]}` |
| `*_contact_point_qa.json` | 接触点预测 | 预测夹爪两指与被操作物体的两个接触点。 | `{"contact_point": [[x1,y1],[x2,y2]]}` |
| `*_contact_box_qa.json` | 接触框预测 | 预测夹爪与物体接触瞬间的夹爪边界框。 | `{"contact_bbox": [[x1,y1],[x2,y2]]}` |
| `*_current_box_qa.json` | 当前物体框 | 预测被操作物体的当前边界框。 | `{"current_bbox": [[x1,y1],[x2,y2]]}` |
| `*_final_box_qa.json` | 最终物体框 | 预测操作完成后被操作物体的最终边界框。 | `{"final_bbox": [[x1,y1],[x2,y2]]}` |

### Understanding（交互理解，6 种任务）

选择题形式的视觉问答任务，评估模型对机器人交互场景的视觉理解能力。使用单图或拼接的多选图像。

| JSON 文件 | 任务名称 | 说明 | 答案格式 |
|---|---|---|---|
| `contact_decide.json` | 接触判断 | 判断当前场景中夹爪是否已经接触/到达目标物体。 | `Yes` / `No` |
| `grasppose_choice.json` | 抓取姿态选择 | 从 4 张候选图片（A/B/C/D）中选择正确的抓取姿态，图中橙色叉形图案表示可能的夹爪姿态。 | `A`/`B`/`C`/`D` |
| `grounding_choice.json` | 物体定位选择 | 从 4 张候选图片中选择哪张正确标注了被操作物体的边界框。 | `A`/`B`/`C`/`D` |
| `traj_choice.json` | 轨迹选择 | 从 4 张候选图片中选择正确的夹爪轨迹，轨迹用渐变色表示（绿色=起点，红色=终点）。 | `A`/`B`/`C`/`D` |
| `trajlang_choice.json` | 轨迹-语言匹配 | 给定一个轨迹可视化图像，从 4 个语言描述中选择对应的任务。 | `A`/`B`/`C`/`D` |
| `traj_direction_choice.json` | 轨迹方向选择 | 给定夹爪周围的多色箭头，选择哪个颜色代表实际运动方向。 | `A`/`B`/`C`/`D` |

### Task Planning（任务规划，4 种任务）

多图（视频帧）视觉问答任务，用于高层级任务理解与规划。每个条目使用 8 帧采样图像作为输入。

| JSON 文件 | 任务名称 | 说明 | 答案格式 |
|---|---|---|---|
| `train/manipvqa/task_planning.json` | 下一步规划（训练集） | 给定 N 帧视频和目标任务，预测接下来应执行的子任务。 | 自由文本 |
| `val/planning/task_planning.json` | 下一步规划（验证集） | 与训练集相同任务，使用独立的验证数据。 | 自由文本 |
| `val/choice/task_planning.json` | 动作原语选择 | 给定 8 帧视频，从 4 个选项中选择刚执行的动作原语。 | `A`/`B`/`C`/`D` |
| `val/decide/task_planning.json` | 成功判断 | 给定 8 帧视频和子任务描述，判断该子任务是否成功完成。 | `Yes` / `No` |

## 数据统计

### Generation（交互生成）

| 数据源 | traj_qa | gripper_det | contact_point | contact_box | current_box | final_box |
|---|---|---|---|---|---|---|
| DROID（训练） | 31,282 | 84,777 | 78,004 | 78,004 | 149,671 | 145,996 |
| RH20T（训练） | 33,803 | 120,747 | 115,266 | 115,266 | 225,055 | 224,944 |
| 验证集 | 2,000 | 2,000 | 2,000 | 2,000 | 2,000 | 2,000 |

### Understanding（交互理解）

| 数据源 | contact_decide | grasppose_choice | grounding_choice | traj_choice | trajlang_choice | traj_direction |
|---|---|---|---|---|---|---|
| RH20T（训练） | 15,060 | 9,835 | 8,158 | 3,610 | 3,610 | 3,729 |
| DROID（训练） | 18,184 | - | 57,572 | 8,245 | 8,245 | 6,500 |
| 验证集 | 15,514 | 2,702 | 6,108 | 787 | 1,474 | 266 |

### Task Planning（任务规划）

| 数据划分 | 条目数 |
|---|---|
| 训练集（manipvqa） | 928,819 |
| 验证集 - planning | 10,806 |
| 验证集 - choice | 15,059 |
| 验证集 - decide | 10,629 |

## 许可证

请参阅 [DROID](https://droid-dataset.github.io/) 和 [RH20T](https://rh20t.github.io/) 原始数据集的许可协议。
