# RoboInterData

Toolkit for building, converting, and loading the **RoboInter** annotated robot manipulation dataset. It covers the full pipeline from raw data (DROID / RH20T) to training-ready PyTorch dataloaders, with rich per-frame annotation support.

---

## Directory Structure

```
RoboInterData/
├── convert_to_lerobot/    # Convert raw LMDB data to LeRobot v2.1 / v3.0 format
├── hr_video_reader/       # Download high-resolution DROID videos from Google Cloud Storage
├── lmdb_tool/             # Convert annotation pkl files to frame-level LMDB
├── lerobot_dataloader/    # PyTorch dataloader for LeRobot v2.1 with annotation & filtering
└── README.md
```

---

## Modules

### 1. `convert_to_lerobot/` — Format Conversion

Converts DROID and RH20T episodes (stored in LMDB) into LeRobot format with full RoboInter annotations. Supports distributed SLURM-based sharding for large-scale processing.

**Pipeline:**

```
LMDB Raw Data ──> LeRobot v2.1 (sharded) ──> merge shards ──> LeRobot v3.0 (optional)
```

**Key scripts:**

| Script | Description |
|--------|-------------|
| `convert_droid_to_lerobot_anno_fast.py` | DROID LMDB → LeRobot v2.1 (multi-threaded) |
| `convert_rh20t_to_lerobot_anno_fast.py` | RH20T LMDB → LeRobot v2.1 (multi-threaded) |
| `merge_lerobot_shards.py` | Merge v2.1 shards into a single dataset |
| `convert_v21_to_v30.py` | LeRobot v2.1 → v3.0 format upgrade |
| `merge_v30_shards.py` | Merge v3.0 shards into a single dataset |
| `scripts/submit_all_shards.sh` | SLURM batch submission for LMDB → v2.1 |
| `scripts/submit_v21_to_v30.sh` | SLURM batch submission for v2.1 → v3.0 |

**Quick start:**

```bash
# Single-node conversion (DROID)
python convert_to_lerobot/convert_droid_to_lerobot_anno_fast.py \
    --input_dir /path/to/droid_lmdb \
    --output_dir /path/to/lerobot_droid_anno \
    --annotation_lmdb /path/to/annotation_lmdb \
    --qsheet_path /path/to/RoboInter_Data_Qsheet_v1.json

# Distributed (10 SLURM shards)
cd convert_to_lerobot && ./scripts/submit_all_shards.sh droid 10

# Merge after all shards finish
python convert_to_lerobot/merge_lerobot_shards.py \
    --shard_pattern '/path/to/lerobot_droid_anno_shard*' \
    --output_dir /path/to/lerobot_droid_anno
```

See [`convert_to_lerobot/README.md`](convert_to_lerobot/README.md) for full documentation.

---

### 2. `hr_video_reader/` — High-Resolution Video Download

Tools for selectively downloading high-resolution DROID videos from Google Cloud Storage (`gs://gresearch/robotics/droid_raw`).

| Script | Description |
|--------|-------------|
| `droid_hr_reader.py` | Download HR videos for specific episodes (filtered by quality sheet) |
| `order_video_view.py` | Select specific camera angles from downloaded videos |

```bash
python hr_video_reader/droid_hr_reader.py \
    --key "3072_exterior_image_1_left" \
    --out_dir /path/to/output \
    --sheet_path /path/to/RoboInter_Data_Qsheet_v1.json
```

See [`hr_video_reader/README.md`](hr_video_reader/README.md) for GCS setup instructions.

---

### 3. `lmdb_tool/` — Annotation LMDB Management

Converts merged annotation pkl files (from the annotation pipeline) into a frame-level LMDB database, and provides tools for inspection and validation.

| Script | Description |
|--------|-------------|
| `convert_pkl_to_lmdb.py` | Convert DROID/RH20T annotation pkl → frame-level LMDB |
| `read_lmdb.py` | Inspect, validate, and print statistics of LMDB files |

```bash
# Convert
python lmdb_tool/convert_pkl_to_lmdb.py \
    --droid_pkl /path/to/droid_annotation.pkl \
    --rh20t_pkl /path/to/rh20t_annotation.pkl \
    --output_lmdb /path/to/annotation_lmdb \
    --data_lmdb_path /path/to/episode_directories/

# Inspect
python lmdb_tool/read_lmdb.py --lmdb_path /path/to/annotation_lmdb --action summary
```

See [`lmdb_tool/README.md`](lmdb_tool/README.md) for LMDB data format details.

---

### 4. `lerobot_dataloader/` — PyTorch DataLoader

Lightweight, portable PyTorch dataloader for LeRobot v2.1 format with:

- Action horizon (action chunking) for policy training
- Frame range filtering (`range_nop.json`) to remove idle frames
- Quality-based episode filtering (`Q_annotation`)
- Multi-dataset loading with weighted sampling
- Built-in transforms (resize, normalize, to-tensor)

```python
from lerobot_dataloader import create_dataloader, QAnnotationFilter

dataloader = create_dataloader(
    "/path/to/lerobot_dataset",
    batch_size=32,
    action_horizon=16,
    q_filters=[
        QAnnotationFilter("Q_annotation.trace", ["Primary"]),
    ],
)

for batch in dataloader:
    images = batch["observation.images.primary"]  # (B, H, W, 3)
    actions = batch["action"]                     # (B, 16, 7)
    trace = batch["annotation.trace"]             # Parsed annotation
    break
```

See [`lerobot_dataloader/README.md`](lerobot_dataloader/README.md) for full API reference.

---

## Annotation Fields

All 13 per-frame annotation fields included in the converted dataset:

| Field | Format | Description |
|-------|--------|-------------|
| `annotation.time_clip` | `[[start, end], ...]` | Temporal action segment boundaries |
| `annotation.instruction_add` | string | Language instruction |
| `annotation.substask` | string | Subtask description |
| `annotation.primitive_skill` | string | Skill label (pick, place, push, twist, ...) |
| `annotation.segmentation` | string | Path to segmentation file |
| `annotation.object_box` | `[[x1,y1],[x2,y2]]` | Target object bounding box |
| `annotation.placement_proposal` | `[[x1,y1],[x2,y2]]` | Placement location proposal |
| `annotation.trace` | `[[x,y], ...]` | Future 10-frame gripper trajectory |
| `annotation.gripper_box` | `[[x1,y1],[x2,y2]]` | Gripper bounding box |
| `annotation.contact_frame` | int | Contact event frame index |
| `annotation.state_affordance` | `[x,y,z,rx,ry,rz]` | 6D robot state at contact |
| `annotation.affordance_box` | `[[x1,y1],[x2,y2]]` | Gripper box at contact frame |
| `annotation.contact_points` | `[x,y]` | Contact point pixel coordinates |

Each field has a corresponding quality indicator (`Q_annotation.*`) rated as `"Primary"`, `"Secondary"`, or `""` (empty).

---

## End-to-End Workflow

```
                        ┌─────────────────────┐
                        │  Raw DROID / RH20T   │
                        │  (LMDB episodes)     │
                        └─────────┬───────────┘
                                  │
          ┌───────────────────────┼───────────────────────┐
          │                       │                       │
          ▼                       ▼                       ▼
   ┌─────────────┐     ┌──────────────────┐     ┌──────────────────┐
   │ hr_video_    │     │ lmdb_tool/       │     │ convert_to_      │
   │ reader/      │     │ convert_pkl_to_  │     │ lerobot/         │
   │              │     │ lmdb.py          │     │ (LMDB → LeRobot) │
   │ Download HR  │     │                  │     │                  │
   │ videos from  │     │ Annotation pkl   │     │ Uses annotation  │
   │ GCS          │     │ → frame LMDB     │──── │ LMDB as input    │
   └─────────────┘     └──────────────────┘     └────────┬─────────┘
                                                         │
                                                         ▼
                                                ┌──────────────────┐
                                                │ LeRobot v2.1     │
                                                │ (parquet + mp4   │
                                                │  + annotations)  │
                                                └────────┬─────────┘
                                                         │
                                              ┌──────────┴──────────┐
                                              │                     │
                                              ▼                     ▼
                                     ┌────────────────┐   ┌─────────────────┐
                                     │ lerobot_       │   │ convert_v21_to_ │
                                     │ dataloader/    │   │ v30.py          │
                                     │                │   │                 │
                                     │ PyTorch        │   │ LeRobot v2.1    │
                                     │ DataLoader     │   │ → v3.0          │
                                     └────────────────┘   └─────────────────┘
```

---

## Dependencies

- Python 3.8+
- PyTorch, torchvision
- lerobot
- lmdb, pyarrow, numpy, pandas
- opencv-python, imageio, av
- tqdm

---

## License

MIT
