[English](README.md) | [中文](README_zh.md)

# RoboInter Demo 可视化工具

基于 Gradio 的机器人交互标注可视化工具，支持语言指令、视觉标注、抓取姿态等多种标注的可视化。

## 在线 Demo

我们在 HuggingFace Spaces 上部署了在线演示版本。由于存储限制，在线版本仅包含 **20 个采样视频**。

> **[点击查看在线 Demo](https://huggingface.co/spaces/wz7in/robointer-demo)**

如需查看完整数据集（120 个视频），请按以下步骤在本地运行。

## 可视化

![Demo 截图](assets/demo.png)

## 本地部署

### 1. 克隆仓库

```bash
git clone https://github.com/InternRobotics/RoboInter
cd RoboInter/RoboInterData-Demo
```

### 2. 下载数据

从 HuggingFace 数据集下载完整的标注数据和视频：

```bash
# 安装 huggingface_hub（如果没有的话）
pip install huggingface_hub

# 下载数据集
python -c "
from huggingface_hub import snapshot_download
snapshot_download(repo_id='wz7in/robointer-demo-data', repo_type='dataset', local_dir='.')
"
# or
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="InternRobotics/RoboInter-Data",
    repo_type="dataset",
    local_dir="/your/local/path/Annotation_demo_app",
    allow_patterns="Annotation_demo_app/*"
)
```

下载内容包括：
- `demo_data/` — LMDB 标注数据库
- `videos/` — 视频文件（`.mp4`）

### 3. 安装依赖

```bash
pip install -r requirements.txt
```

### 4. 启动

```bash
python app.py
```

浏览器访问 http://localhost:7860 即可使用。

## 标注类型

### 语言标注
| 类型 | 说明 |
|------|------|
| **Instruction** | 每帧的任务指令 |
| **Subtask** | 子任务描述 |
| **Primitive Skill** | 底层技能标签 |

### 视觉标注
| 类型 | 颜色 | 说明 |
|------|------|------|
| **Object Mask** | 青色 | 物体分割 mask + 轮廓边缘 |
| **Placement Proposal** | 绿色 | 目标放置区域框 |
| **Gripper Box** | 黄色 | 夹爪边界框 |
| **Trace** | 紫色 | 运动轨迹（渐变光流效果） |

### 接触与可操作性
| 类型 | 颜色 | 说明 |
|------|------|------|
| **Contact Points** | 橙色 | 物体上的接触关键点 |
| **Affordance Box** | 红色 | 可操作区域框 |
| **Grasp Pose** | 绿色 | 6 关键点抓取姿态 |

## 数据格式

标注数据存储在 LMDB 数据库中，每个视频对应一组逐帧标注：

```python
{
    "video_name": {
        frame_idx: {
            "instruction_add": str,
            "substask": str,
            "primitive_skill": str,
            "segmentation": np.ndarray (H, W),   # 二值 mask
            "object_box": [[x1, y1], [x2, y2]],
            "placement_proposal": [[x1, y1], [x2, y2]],
            "trace": [[x, y], ...],
            "gripper_box": [[x1, y1], [x2, y2]],
            "contact_frame": int,
            "affordance_box": [[x1, y1], [x2, y2]],
            "contact_points": [[x, y], ...],
            "grasp_pose": [[x, y], ...],          # 6 个关键点
        },
    },
}
```

## 项目结构

```
ICLR_VIS/
├── app.py              # 主应用程序
├── config.py           # 配置与数据加载
├── README.md           # 英文文档
├── README_zh.md        # 中文文档
├── demo_data/          # LMDB 标注数据库
│   ├── data.mdb
│   └── lock.mdb
└── videos/             # 视频文件
    └── *.mp4
```

## 许可

本项目仅用于研究和演示目的。
