# Convert RoboInter Data to LeRobot Format

This directory provides tools for converting DROID and RH20T datasets (stored in LMDB format) into the [LeRobot](https://github.com/huggingface/lerobot) dataset format, with full support for RoboInter annotations.

---

## Overview

The conversion pipeline has three stages:

```
LMDB Raw Data ──(Step 1)──> LeRobot v2.1 ──(Step 2)──> LeRobot v3.0
                                 │                           │
                          merge shards (v2.1)          merge shards (v3.0)
```

All stages support **distributed sharding** via SLURM for large-scale parallel processing.

---

## Directory Structure

```
convert_to_lerobot/
├── convert_droid_to_lerobot_anno_fast.py   # DROID LMDB -> LeRobot v2.1
├── convert_rh20t_to_lerobot_anno_fast.py   # RH20T LMDB -> LeRobot v2.1
├── convert_v21_to_v30.py                   # LeRobot v2.1 -> v3.0
├── merge_lerobot_shards.py                 # Merge v2.1 shards
├── merge_v30_shards.py                     # Merge v3.0 shards
├── scripts/
│   ├── submit_all_shards.sh                # SLURM job submission (LMDB -> v2.1)
│   ├── submit_v21_to_v30.sh                # SLURM job submission (v2.1 -> v3.0)
│   └── merge_data.sh                       # Example merge command
└── README.md
```

---

## Step 1: Convert LMDB to LeRobot v2.1

Converts raw LMDB episode data into LeRobot v2.1 format with RoboInter annotations. Uses `ThreadPoolExecutor` for multi-threaded parallel processing and generates MP4 videos from image frames.

### Single-node Usage

**DROID:**

```bash
python convert_droid_to_lerobot_anno_fast.py \
    --input_dir /path/to/droid_lmdb_episodes \
    --output_dir /path/to/lerobot_droid_anno \
    --annotation_lmdb /path/to/annotation_lmdb \
    --qsheet_path /path/to/RoboInter_Data_Qsheet_v1.json \
    --num_threads 64 \
    --fps 10
```

**RH20T:**

```bash
python convert_rh20t_to_lerobot_anno_fast.py \
    --input_dir /path/to/rh20t_lmdb_episodes \
    --output_dir /path/to/lerobot_rh20t_anno \
    --annotation_lmdb /path/to/annotation_lmdb \
    --qsheet_path /path/to/RoboInter_Data_Qsheet_v1.json \
    --num_threads 256 \
    --fps 10
```

### Distributed Sharded Processing (SLURM)

For large datasets, split the work across multiple SLURM jobs:

```bash
# Submit 10 shards for DROID conversion
./scripts/submit_all_shards.sh droid 10

# Submit 5 shards for RH20T conversion
./scripts/submit_all_shards.sh rh20t 5

# Resume from shard 5 (submit shards 5-9)
./scripts/submit_all_shards.sh droid 10 5

# Submit only shards 0-4
./scripts/submit_all_shards.sh droid 10 0 5
```

After all shards complete, merge them:

```bash
python merge_lerobot_shards.py \
    --shard_pattern '/path/to/lerobot_droid_anno_shard*' \
    --output_dir /path/to/lerobot_droid_anno \
    --num_threads 32
```

### Arguments

| Argument | Description | Default |
|---|---|---|
| `--input_dir` | Directory containing LMDB episode subdirectories | Required |
| `--output_dir` | Output directory for the LeRobot dataset | Required |
| `--annotation_lmdb` | Path to the RoboInter annotation LMDB | Required |
| `--qsheet_path` | Path to the quality sheet JSON | Required |
| `--num_threads` | Number of threads for parallel processing | 64 (DROID) / 256 (RH20T) |
| `--fps` | Frames per second for video encoding | 10 |
| `--max_episodes` | Limit the number of episodes to convert | None (all) |
| `--keep_temp_videos` | Keep temporary video files after conversion | False |
| `--shard_id` | Shard ID for distributed processing (0-indexed) | None |
| `--num_shards` | Total number of shards | None |
| `--no_resize` | (RH20T only) Do not resize images to half size | False |

---

## Step 2: Convert LeRobot v2.1 to v3.0

Converts the LeRobot v2.1 dataset to v3.0 format, which reorganizes files from episode-based to file-based layout:

| Component | v2.1 | v3.0 |
|---|---|---|
| Data | `data/chunk-XXX/episode_XXXXXX.parquet` | `data/chunk-XXX/file_XXX.parquet` |
| Videos | `videos/chunk-XXX/CAMERA/episode_XXXXXX.mp4` | `videos/CAMERA/chunk-XXX/file_XXX.mp4` |
| Episodes | `meta/episodes.jsonl` | `meta/episodes/chunk-XXX/episodes_XXX.parquet` |
| Tasks | `meta/tasks.jsonl` | `meta/tasks/chunk-XXX/file_XXX.parquet` |
| Stats | `meta/episodes_stats.jsonl` | `meta/episodes_stats/chunk-XXX/file_XXX.parquet` |

### Usage

```bash
python convert_v21_to_v30.py \
    --input_dir /path/to/v21/dataset \
    --output_dir /path/to/v30/dataset
```

### Distributed Sharded Processing (SLURM)

```bash
# Submit 10 shards for DROID v2.1->v3.0 conversion
./scripts/submit_v21_to_v30.sh droid 10

# Submit 5 shards for RH20T v2.1->v3.0 conversion
./scripts/submit_v21_to_v30.sh rh20t 5
```

After all shards complete, merge them:

```bash
python merge_v30_shards.py \
    --shard_pattern '/path/to/lerobot_droid_anno_v30_shard*' \
    --output_dir /path/to/lerobot_droid_anno_v30 \
    --num_threads 32
```

---

## Annotation Fields

The following RoboInter annotation fields are included in the converted dataset, along with their corresponding quality scores (`Q_annotation.*`):

| Field | Type | Description |
|---|---|---|
| `annotation.time_clip` | JSON | Temporal clip boundaries |
| `annotation.instruction_add` | String | Additional language instruction |
| `annotation.substask` | String | Subtask description |
| `annotation.primitive_skill` | String | Primitive skill label |
| `annotation.segmentation` | String | Segmentation annotation |
| `annotation.object_box` | JSON | Object bounding boxes |
| `annotation.placement_proposal` | JSON | Placement proposal coordinates |
| `annotation.trace` | JSON | Motion trace |
| `annotation.gripper_box` | JSON | Gripper bounding box |
| `annotation.contact_frame` | JSON | Contact frame information |
| `annotation.state_affordance` | JSON | State affordance |
| `annotation.affordance_box` | JSON | Affordance bounding box |
| `annotation.contact_points` | JSON | Contact point coordinates |
| `annotation.origin_shape` | JSON | Original image shape |

Quality scores (`Q_annotation.*`) are rated as `"Primary"` or `"Secondary"`.

---

## Dependencies

- Python 3.8+
- [lerobot](https://github.com/huggingface/lerobot)
- lmdb
- opencv-python
- imageio
- numpy
- pandas
- pyarrow
- torch, torchvision
- tqdm
