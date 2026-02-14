# DataLoader of RoboInter-Data Annotation_with_action_lerobotv21

A lightweight, portable dataloader for LeRobot v2.1 format datasets with rich annotation support for robot manipulation.


## Download & Extract
Dataset [link](https://huggingface.co/datasets/InternRobotics/RoboInter-Data/tree/main/Annotation_with_action_lerobotv21).
The `data/` and `videos/` directories are distributed as `.tar` archives (one per chunk) to reduce the number of files during transfer. After downloading, extract them in place:

```bash
cd Annotation_with_action_lerobotv21

for dataset in lerobot_droid_anno lerobot_rh20t_anno; do
  for subdir in data videos; do
    cd ${dataset}/${subdir}
    for f in *.tar; do tar xf "$f" && rm "$f"; done
    cd ../..
  done
done
```

After extraction, each `data/` will contain `chunk-000/`, `chunk-001/`, ... with `.parquet` files, and each `videos/` will contain `chunk-000/`, `chunk-001/`, ... with `.mp4` files. The `meta/` directories are ready to use without extraction.


## Features

- Compatible with LeRobot v2.1 format (parquet + video)
- Support for single or multiple datasets
- Action horizon (action chunking) for policy training
- Rich annotation fields support (object_box, trace, gripper_box, etc.)
- **Frame range filtering** - Remove idle frames at episode start/end
- **Q_annotation filtering** - Select episodes by annotation quality
- Standard PyTorch DataLoader interface
- Easy to integrate into existing training pipelines

## Installation

```bash
pip install numpy torch pyarrow av opencv-python
pip install lerobot
lerobot-info
```

## Dataset Structure (LeRobot v2.1 Format)

### Directory Layout

```
lerobot_dataset/
├── meta/
│   ├── info.json           # Dataset metadata (fps, features, etc.)
│   ├── episodes.jsonl      # Episode information (one JSON per line)
│   └── tasks.jsonl         # Task/instruction mapping
├── data/
│   └── chunk-000/          # Data chunks
│       ├── episode_000000.parquet
│       ├── episode_000001.parquet
│       └── ...
└── videos/
    └── chunk-000/
        ├── observation.images.primary/
        │   ├── episode_000000.mp4
        │   └── ...
        └── observation.images.wrist/
            ├── episode_000000.mp4
            └── ...
```

### meta/info.json

```json
{
  "fps": 10,
  "robot_type": "franka_robotiq",
  "chunks_size": 1000,
  "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
  "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
  "features": {
    "action": {"dtype": "float64", "shape": [7]},
    "state": {"dtype": "float64", "shape": [7]},
    ...
  }
}
```
### Core Fields of data

| Field | Shape | Type | Description |
|-------|-------|------|-------------|
| `action` | (7,) | float64 | Delta EEF Action:[delta_x, delta_y, delta_z, delta_rx, delta_ry, delta_rz, gripper_command] |
| `state` | (7,) | float64 | EEF State:[x, y, z, rx, ry, rz, gripper_state] |
| `observation.images.primary` | (H, W, 3) | uint8 | Image (RGB) |
| `observation.images.wrist` | (H, W, 3) | uint8 | Image (RGB) |


### Metadata Fields of Data

| Field | Type | Description |
|-------|------|-------------|
| `episode_name` | string | Episode unique identifier, like "3072_exterior_image_1_left" |
| `camera_view` | string | Camera perspective name, like "exterior_image_1_left" |
| `task` | string | Task description |
| `episode_index` | int | Episode index in dataset |
| `frame_index` | int | The index of the current frame in the episode |



## Annotation Fields

All annotation fields are prefixed with `annotation.` and stored as JSON strings (empty string = no annotation, -1 = no contact_frame).

| Field | Format | Description |
|-------|--------|-------------|
| `annotation.time_clip` | `[[start, end], ...]` | Valid action segments (frame ranges) |
| `annotation.instruction_add` | string | Detailed instruction supplement |
| `annotation.substask` | string | Current subtask description |
| `annotation.primitive_skill` | string | Primitive skill label (pick, place, push, twist, etc.) |
| `annotation.segmentation` | string | Path to segmentation file |
| `annotation.object_box` | `[[x1, y1], [x2, y2]]` | Target object bounding box |
| `annotation.placement_proposal` | `[[x1, y1], [x2, y2]]` | Placement location proposal |
| `annotation.trace` | `[[x, y], ...]` | Future 10-frame gripper trajectory |
| `annotation.gripper_box` | `[[x1, y1], [x2, y2]]` | Gripper bounding box |
| `annotation.contact_frame` | int | Frame index when contact occurs |
| `annotation.state_affordance` | `[x, y, z, rx, ry, rz]` | 6D robot state at contact frame |
| `annotation.affordance_box` | `[[x1, y1], [x2, y2]]` | Gripper box at contact frame |
| `annotation.contact_points` | `[x, y]` | Contact point in pixel coordinates |

### Bounding Box Format

All bounding boxes use pixel coordinates with origin at top-left:
```json
[[x1, y1], [x2, y2]]  // [top-left, bottom-right]
```

### Trace Format

10 future waypoints for gripper trajectory prediction:
```json
[[110, 66], [112, 68], [115, 70], [118, 72], [120, 75], [122, 78], [125, 80], [128, 82], [130, 85], [132, 88]]
```

### range_nop.json (for frame range filtering)

For filting invalid frames at the beginning and end of the episode:：

```json
{
  "3072_exterior_image_1_left": [12, 217, 206],
  "50453_exterior_image_1_left": [9, 623, 615]
}
```

Format：`episode_name: [start_frame, end_frame, total_length]`
- `start_frame`: Valid action start frame
- `end_frame`: Valid Action End Frame
- `total_length`: Total valid frames of episode

---

## Q_Annotation Fields (Quality Indicators)

Each annotation has a corresponding quality indicator prefixed with `Q_annotation.`:

| Field | Values |
|-------|--------|
| `Q_annotation.instruction_add` | "Primary" / "Secondary" / "" |
| `Q_annotation.substask` | "Primary" / "Secondary" / "" |
| `Q_annotation.primitive_skill` | "Primary" / "Secondary" / "" |
| `Q_annotation.segmentation` | "Primary" / "Secondary" / "" |
| `Q_annotation.object_box` | "Primary" / "Secondary" / "" |
| `Q_annotation.placement_proposal` | "Primary" / "Secondary" / "" |
| `Q_annotation.trace` | "Primary" / "Secondary" / "" |
| `Q_annotation.gripper_box` | "Primary" / "Secondary" / "" |
| `Q_annotation.contact_frame` | "Primary" / "Secondary" / "" |
| `Q_annotation.state_affordance` | "Primary" / "Secondary" / "" |
| `Q_annotation.affordance_box` | "Primary" / "Secondary" / "" |
| `Q_annotation.contact_points` | "Primary" / "Secondary" / "" |

- **Primary**: More stable quality
- **Secondary**: Acceptable quality, may have minor errors
- **""** (empty): No annotation available

### RoboInter_Data_Qsheet_v1.json

Episode levels of quality callout sources:

```json
{
  "3072_exterior_image_1_left": {
    "ori_path": "gs://...",
    "view": "exterior_image_1_left",
    "range_nop": [12, 217, 206],
    "instruction_ori": true,
    "Q_instruction_add": "Primary",
    "Q_substask": "Primary",
    "Q_primitive_skill": "Primary",
    "Q_segmentation": null,
    "Q_object_box": null,
    "Q_placement_proposal": null,
    "Q_trace": "Secondary",
    "Q_gripper_box": "Secondary",
    "Q_contact_frame": null,
    "Q_state_affordance": null,
    "Q_affordance_box": null,
    "Q_contact_points": null
  }
}
```

---

## Quick Start

```python
from lerobot_dataloader import create_dataloader

dataloader = create_dataloader(
    "/path/to/lerobot_dataset",
    batch_size=32,
    action_horizon=16,
)

for batch in dataloader:
    images = batch["observation.images.primary"]  # (B, H, W, 3)
    actions = batch["action"]                     # (B, 16, 7)
    trace = batch["annotation.trace"]             # Parsed JSON lists
    skill = batch["annotation.primitive_skill"]   # List of strings
    break
```

---


## Data Filtering

### Frame Range Filtering

Remove idle frames at episode start/end using `range_nop.json`:

```python
dataloader = create_dataloader(
    "/path/to/dataset",
    range_nop_path="/path/to/range_nop.json",
)
```

Format of `range_nop.json`:
```json
{
  "episode_name": [start_frame, end_frame, total_length]
}
```

### Q_Annotation Filtering

Select episodes by annotation quality:

```python
from lerobot_dataloader import create_dataloader, QAnnotationFilter

# Only Primary quality
dataloader = create_dataloader(
    "/path/to/dataset",
    q_filters=[
        QAnnotationFilter("Q_annotation.instruction_add", ["Primary"]),
        QAnnotationFilter("Q_annotation.gripper_box", ["Primary"]),
    ]
)

# Any non-empty annotation
dataloader = create_dataloader(
    "/path/to/dataset",
    q_filters=[
        QAnnotationFilter("Q_annotation.trace", ["not_empty"])
    ]
)
```

### Combined Filtering

```python
from lerobot_dataloader import FilterConfig, QAnnotationFilter

config = FilterConfig(
    range_nop_path="/path/to/range_nop.json",
    q_filters=[
        QAnnotationFilter("Q_annotation.trace", ["Primary", "Secondary"]),
    ],
    q_filter_mode="all",  # "all" = AND, "any" = OR
)

dataloader = create_dataloader("/path/to/dataset", filter_config=config)
```

### FilterConfig from Dict/YAML

```python
config = FilterConfig.from_dict({
    "range_nop_path": "/path/to/range_nop.json",
    "q_filters": [
        {"field": "Q_annotation.trace", "values": ["Primary"]}
    ]
})
```

---

## Multiple Datasets

```python
dataloader = create_dataloader(
    ["/path/to/droid_lerobot", "/path/to/rh20t_lerobot"],
    batch_size=32,
)

for batch in dataloader:
    print(batch["dataset_name"])  # Source dataset identifier
    break
```

---

## Transforms

```python
from lerobot_dataloader import Compose, Normalize, ResizeImages, ToTensorImages
from lerobot_dataloader.transforms import compute_stats

# Compute normalization stats
dataset = LeRobotDataset("/path/to/dataset", load_videos=False)
stats = compute_stats(dataset)

# Create transform pipeline
transform = Compose([
    ResizeImages(height=224, width=224),
    ToTensorImages(),
    Normalize(stats),
])

dataloader = create_dataloader("/path/to/dataset", transform=transform)
```

---

## API Reference

### create_dataloader

```python
create_dataloader(
    root,                    # Path or list of paths
    batch_size=32,
    shuffle=True,
    num_workers=4,
    action_horizon=1,        # 1 = single action, >1 = action chunk
    load_videos=True,
    transform=None,
    pin_memory=True,
    drop_last=True,
    filter_config=None,      # FilterConfig object
    range_nop_path=None,     # Path to range_nop.json
    q_filters=None,          # List of QAnnotationFilter
)
```

### LeRobotDataset

```python
LeRobotDataset(
    root,                    # Path to dataset
    split="train",
    delta_timestamps=None,   # For action horizon
    image_keys=None,         # Auto-detect from features
    load_videos=True,
    transform=None,
    filter_config=None,
    range_nop_path=None,
    q_filters=None,
)
```

### FilterConfig

```python
FilterConfig(
    range_nop_path=None,     # Path to range_nop.json
    use_range_filter=True,
    q_filters=[],            # List of QAnnotationFilter
    q_filter_mode="all",     # "all" (AND) or "any" (OR)
)

# From dict/YAML
config = FilterConfig.from_dict({
    "range_nop_path": "/path/to/range_nop.json",
    "q_filters": [
        {"field": "Q_annotation.trace", "values": ["Primary"]}
    ]
})
```

### QAnnotationFilter

```python
QAnnotationFilter(
    field,                   # e.g., "Q_annotation.instruction_add"
    values=["not_empty"],    # ["Primary"], ["Secondary"], or ["not_empty"]
    require_all_frames=False
)
```

## File Structure

```
lerobot_dataloader/
├── __init__.py      # Module exports
├── dataset.py       # LeRobotDataset, MultiDataset, FilterConfig, etc.
├── transforms.py    # Compose, Normalize, ResizeImages, etc.
├── example_usage.py # Usage examples
└── README.md        # This documentation
```


---

## License

MIT
