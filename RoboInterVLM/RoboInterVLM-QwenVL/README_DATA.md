# RobotInterVQA: Robot Interaction Understanding & Generation VQA Dataset

**English** | [简体中文](./README_DATA_zh.md)

A Visual Question Answering dataset for robotic manipulation, derived from [RoboInter](https://github.com/InternRobotics/RoboInter), covering **intermediate representation generation**, **intermediate representation understanding**, and **task planning**. Built on top of [DROID](https://droid-dataset.github.io/) and [RH20T](https://rh20t.github.io/) robot datasets.

## Dataset Structure

```
robotinter/
├── Generation/          # Interaction generation tasks (grounding, trajectory, contact)
│   ├── image/           # Images (zip archives, extract before use)
│   │   ├── train/{droid,rh20t}/
│   │   └── val/
│   └── meta/            # QA annotations in JSON
│       ├── train/{droid,rh20t}/{origin_format,llava_format,smart_resize_format}/
│       └── val/{origin_format,llava_format,smart_resize_format}/
├── Understanding/       # Interaction understanding tasks (multiple-choice)
│   ├── image/
│   │   ├── train/{droid,rh20t}/
│   │   └── val/
│   └── meta/
│       ├── train/{droid,rh20t}/
│       └── val/
└── Task_planning/       # Task planning & primitive recognition
    ├── image/
    │   ├── train/manipvqa/
    │   └── val/{planning,choice,decide}/
    └── meta/
        ├── train/manipvqa/
        └── val/{planning,choice,decide}/
```

## Quick Start

1. **Extract images**: All images are stored as `.zip` files. Extract them in place:
   ```bash
   cd RoboInter-VQA/Task_planning/image/train/manipvqa
   cat task_planning.zip.* > task_planning_full.zip
   cd ../../../../
   find . -name "*.zip" -execdir unzip -o {} \;
   ```

2. **Load VQA data**: 
Refer to [RoboInterVLM](https://github.com/InternRobotics/RoboInter/blob/main/RoboInterVLM/RoboInterVLM-QwenVL/qwen-vl-finetune/qwenvl/data/__init__.py).


## Coordinate Formats (Generation only)

The Generation annotations are provided in **three coordinate formats**. The underlying data and images are identical; only the coordinate representation in the answers differs:

| Format | Description | Example |
|---|---|---|
| `origin_format` | Pixel coordinates in original image resolution (`h` x `w`) | `[[72, 102], [192, 179]]` |
| `llava_format` | Normalized coordinates in `[0, 1]` range as LLaVA-based Model | `[[0.22, 0.57], [0.60, 0.99]]` |
| `smart_resize_format` | Pixel coordinates in resized image resolution as Qwen-based Model (`new_h` x `new_w`) | `[[69, 95], [184, 167]]` |

## JSON Descriptions

### Generation (7 task types)

Each entry follows this schema:
```json
{
  "id": "unique_sample_id",
  "task": "task_type",
  "conversations": [{"from": "human", "value": "..."}, {"from": "gpt", "value": "..."}],
  "images": "relative/path/to/image.jpg",
  "gt": "ground_truth_value",
  "h": 180, "w": 320,
  "new_h": 168, "new_w": 308
}
```

| JSON File | Task | Description | Output Format |
|---|---|---|---|
| `*_traj_qa.json` | Trajectory Prediction | Given a task description and a **starting position**, predict 10 future trajectory waypoints for the gripper. | `{"future_traj": [[x1,y1], ...]}` |
| `*_traj_qa_wo_init_pos.json` | Trajectory Prediction (no init pos) | Same as above but **without** providing the starting position in the prompt. | `{"future_traj": [[x1,y1], ...]}` |
| `*_gripper_det_qa.json` | Gripper Detection | Detect the current bounding box of the robot gripper in the scene. | `{"gripper_det_bbox": [[x1,y1],[x2,y2]]}` |
| `*_contact_point_qa.json` | Contact Point Prediction | Predict the two contact points where the gripper fingers touch the manipulated object. | `{"contact_point": [[x1,y1],[x2,y2]]}` |
| `*_contact_box_qa.json` | Contact Box Prediction | Predict the bounding box of the gripper at the moment of contact with the object. | `{"contact_bbox": [[x1,y1],[x2,y2]]}` |
| `*_current_box_qa.json` | Current Object Box | Predict the current bounding box of the manipulated object. | `{"current_bbox": [[x1,y1],[x2,y2]]}` |
| `*_final_box_qa.json` | Final Object Box | Predict the final bounding box of the manipulated object (at the end of the action). | `{"final_bbox": [[x1,y1],[x2,y2]]}` |

### Understanding (6 task types)

Multiple-choice VQA tasks that evaluate visual understanding of robot interactions. Each entry uses single-image or multi-choice concatenated images.

| JSON File | Task | Description | Answer |
|---|---|---|---|
| `contact_decide.json` | Contact Decision | Given a scene, determine whether the gripper has reached/contacted the object. | `Yes` / `No` |
| `grasppose_choice.json` | Grasp Pose Choice | Select the correct grasping pose from 4 candidate images (A/B/C/D), where orange fork-like patterns represent possible gripper poses. | `A`/`B`/`C`/`D` |
| `grounding_choice.json` | Grounding Choice | Select which image correctly depicts the bounding box (purple box) of the manipulated object from 4 candidates. | `A`/`B`/`C`/`D` |
| `traj_choice.json` | Trajectory Choice | Select the correct gripper trajectory from 4 candidate images with gradient-colored paths (green=start, red=end). | `A`/`B`/`C`/`D` |
| `trajlang_choice.json` | Trajectory-Language Choice | Given a trajectory visualization, select the correct task description from 4 language options. | `A`/`B`/`C`/`D` |
| `traj_direction_choice.json` | Trajectory Direction Choice | Given colored arrows around the gripper, select which color represents the actual movement direction. | `A`/`B`/`C`/`D` |

### Task Planning (4 task types)

Multi-image (video frame) VQA tasks for high-level task understanding. Each entry uses 8 sampled frames as input.

| JSON File | Task | Description | Answer |
|---|---|---|---|
| `train/manipvqa/task_planning.json` | Next Step Planning (train) | Given 8 video frames and a goal, predict the next sub-task to perform. | Free-form text |
| `val/planning/task_planning.json` | Next Step Planning (val) | Same as training but on held-out validation data. | Free-form text |
| `val/choice/task_planning.json` | Primitive Selection | Given 8 video frames, select which action primitive was just executed from 4 options. | `A`/`B`/`C`/`D` |
| `val/decide/task_planning.json` | Success Decision | Given 8 video frames and a sub-task description, determine whether the sub-task was successfully completed. | `Yes` / `No` |

## Data Statistics

### Generation

| Source | traj_qa | gripper_det | contact_point | contact_box | current_box | final_box |
|---|---|---|---|---|---|---|
| DROID (train) | 31,282 | 84,777 | 78,004 | 78,004 | 149,671 | 145,996 |
| RH20T (train) | 33,803 | 120,747 | 115,266 | 115,266 | 225,055 | 224,944 |
| Val | 2,000 | 2,000 | 2,000 | 2,000 | 2,000 | 2,000 |

### Understanding

| Source | contact_decide | grasppose_choice | grounding_choice | traj_choice | trajlang_choice | traj_direction |
|---|---|---|---|---|---|---|
| RH20T (train) | 15,060 | 9,835 | 8,158 | 3,610 | 3,610 | 3,729 |
| DROID (train) | 18,184 | - | 57,572 | 8,245 | 8,245 | 6,500 |
| Val | 15,514 | 2,702 | 6,108 | 787 | 1,474 | 266 |

### Task Planning

| Split | Entries |
|---|---|
| Train (manipvqa) | 928,819 |
| Val - planning | 10,806 |
| Val - choice | 15,059 |
| Val - decide | 10,629 |

## License

Please refer to the original dataset licenses for [DROID](https://droid-dataset.github.io/) and [RH20T](https://rh20t.github.io/).
