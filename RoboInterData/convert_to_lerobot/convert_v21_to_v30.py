#!/usr/bin/env python
"""
LeRobot v2.1 to v3.0 Dataset Conversion Script

This script converts LeRobot datasets from codebase version 2.1 to 3.0.
It works for both DROID and RH20T datasets that have been converted to LeRobot v2.1 format.

Main changes from v2.1 to v3.0:
- Data files: data/chunk-XXX/episode_XXXXXX.parquet -> data/chunk-XXX/file_XXX.parquet
- Video files: videos/chunk-XXX/CAMERA/episode_XXXXXX.mp4 -> videos/CAMERA/chunk-XXX/file_XXX.mp4
- Episodes metadata: episodes.jsonl -> meta/episodes/chunk-XXX/episodes_XXX.parquet
- Tasks: tasks.jsonl -> meta/tasks/chunk-XXX/file_XXX.parquet
- Stats: episodes_stats.jsonl -> meta/episodes_stats/chunk-XXX/file_XXX.parquet
- Updated meta/info.json with codebase_version = "v3.0"

Usage:
    # Convert a single dataset
    python convert_v21_to_v30.py \
        --input_dir /path/to/v21/dataset \
        --output_dir /path/to/v30/dataset

    # Convert with sharding for large datasets
    python convert_v21_to_v30.py \
        --input_dir /path/to/v21/dataset \
        --output_dir /path/to/v30/dataset \
        --shard_id 0 \
        --num_shards 10

Conda environment: openpai_new
"""

import argparse
import json
import logging
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import jsonlines
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from datasets import Dataset, Features, Image

# Try to import from lerobot - handle different package structures
try:
    from lerobot.datasets.compute_stats import aggregate_stats
    from lerobot.datasets.lerobot_dataset import CODEBASE_VERSION, LeRobotDataset
    from lerobot.datasets.utils import (
        DEFAULT_CHUNK_SIZE,
        DEFAULT_DATA_FILE_SIZE_IN_MB,
        DEFAULT_DATA_PATH,
        DEFAULT_VIDEO_FILE_SIZE_IN_MB,
        DEFAULT_VIDEO_PATH,
        LEGACY_EPISODES_PATH,
        LEGACY_EPISODES_STATS_PATH,
        LEGACY_TASKS_PATH,
        cast_stats_to_numpy,
        flatten_dict,
        get_file_size_in_mb,
        get_parquet_file_size_in_mb,
        get_parquet_num_frames,
        load_info,
        update_chunk_file_indices,
        write_episodes,
        write_info,
        write_stats,
        write_tasks,
    )
    from lerobot.datasets.video_utils import concatenate_video_files, get_video_duration_in_s
except ImportError:
    # Fallback for older lerobot versions
    from lerobot.common.datasets.compute_stats import aggregate_stats
    from lerobot.common.datasets.lerobot_dataset import CODEBASE_VERSION, LeRobotDataset
    from lerobot.common.datasets.utils import (
        DEFAULT_CHUNK_SIZE,
        DEFAULT_DATA_FILE_SIZE_IN_MB,
        DEFAULT_DATA_PATH,
        DEFAULT_VIDEO_FILE_SIZE_IN_MB,
        DEFAULT_VIDEO_PATH,
        LEGACY_EPISODES_PATH,
        LEGACY_EPISODES_STATS_PATH,
        LEGACY_TASKS_PATH,
        cast_stats_to_numpy,
        flatten_dict,
        get_file_size_in_mb,
        get_parquet_file_size_in_mb,
        get_parquet_num_frames,
        load_info,
        update_chunk_file_indices,
        write_episodes,
        write_info,
        write_stats,
        write_tasks,
    )
    from lerobot.common.datasets.video_utils import concatenate_video_files, get_video_duration_in_s

# Constants
V21 = "v2.1"
V30 = "v3.0"

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


def load_jsonlines(fpath: Path) -> List[Any]:
    """Load a JSONL file and return its contents as a list."""
    with jsonlines.open(fpath, "r") as reader:
        return list(reader)


def legacy_load_episodes(local_dir: Path) -> Dict:
    """Load episodes from legacy episodes.jsonl file."""
    episodes_path = local_dir / LEGACY_EPISODES_PATH
    if not episodes_path.exists():
        # Try alternative path
        episodes_path = local_dir / "meta" / "episodes.jsonl"

    episodes = load_jsonlines(episodes_path)
    return {item["episode_index"]: item for item in sorted(episodes, key=lambda x: x["episode_index"])}


def legacy_load_episodes_stats(local_dir: Path) -> Dict:
    """Load episode statistics from legacy episodes_stats.jsonl file."""
    stats_path = local_dir / LEGACY_EPISODES_STATS_PATH
    if not stats_path.exists():
        # Try alternative path
        stats_path = local_dir / "meta" / "episodes_stats.jsonl"

    episodes_stats = load_jsonlines(stats_path)
    return {
        item["episode_index"]: cast_stats_to_numpy(item["stats"])
        for item in sorted(episodes_stats, key=lambda x: x["episode_index"])
    }


def legacy_load_tasks(local_dir: Path) -> Tuple[Dict, Dict]:
    """Load tasks from legacy tasks.jsonl file."""
    tasks_path = local_dir / LEGACY_TASKS_PATH
    if not tasks_path.exists():
        # Try alternative path
        tasks_path = local_dir / "meta" / "tasks.jsonl"

    tasks = load_jsonlines(tasks_path)
    tasks = {item["task_index"]: item["task"] for item in sorted(tasks, key=lambda x: x["task_index"])}
    task_to_task_index = {task: task_index for task_index, task in tasks.items()}
    return tasks, task_to_task_index


def validate_local_dataset_version(local_path: Path) -> None:
    """Validate that the local dataset has the expected v2.1 version."""
    info = load_info(local_path)
    dataset_version = info.get("codebase_version", "unknown")
    if dataset_version != V21:
        raise ValueError(
            f"Local dataset has codebase version '{dataset_version}', expected '{V21}'. "
            f"This script is specifically for converting v2.1 datasets to v3.0."
        )


def get_video_keys(root: Path) -> List[str]:
    """Get list of video feature keys from the dataset."""
    info = load_info(root)
    features = info["features"]
    video_keys = [key for key, ft in features.items() if ft["dtype"] == "video"]
    return video_keys


def get_image_keys(root: Path) -> List[str]:
    """Get list of image feature keys from the dataset."""
    info = load_info(root)
    features = info["features"]
    image_keys = [key for key, ft in features.items() if ft["dtype"] == "image"]
    return image_keys


def convert_tasks(root: Path, new_root: Path) -> None:
    """Convert tasks from JSONL to parquet format."""
    logging.info(f"Converting tasks from {root} to {new_root}")
    tasks, _ = legacy_load_tasks(root)
    task_indices = list(tasks.keys())
    task_strings = list(tasks.values())
    df_tasks = pd.DataFrame({"task_index": task_indices}, index=task_strings)
    write_tasks(df_tasks, new_root)


def concat_data_files(
    paths_to_cat: List[Path],
    new_root: Path,
    chunk_idx: int,
    file_idx: int,
    image_keys: List[str]
) -> None:
    """Concatenate multiple parquet files into one with proper schema."""
    # Read all tables
    tables = [pq.read_table(f) for f in paths_to_cat]

    # Concatenate with type promotion
    table = pa.concat_tables(tables, promote_options="default")

    # Build HF Features from arrow schema
    features = Features.from_arrow_schema(table.schema)

    # Override image columns to be HF Image()
    for key in image_keys:
        if key in features:
            features[key] = Image()

    # Convert back to arrow schema with updated metadata
    arrow_schema = features.arrow_schema

    # Write parquet with correct schema
    path = new_root / DEFAULT_DATA_PATH.format(chunk_index=chunk_idx, file_index=file_idx)
    path.parent.mkdir(parents=True, exist_ok=True)

    pq.write_table(table.cast(arrow_schema), path)


def convert_data(
    root: Path,
    new_root: Path,
    data_file_size_in_mb: int,
    episode_indices: Optional[List[int]] = None
) -> List[Dict]:
    """Convert data files from episode-based to file-based format."""
    data_dir = root / "data"
    ep_paths = sorted(data_dir.glob("*/*.parquet"))

    # Filter episodes if sharding
    if episode_indices is not None:
        ep_indices_set = set(episode_indices)
        ep_paths = [
            p for p in ep_paths
            if int(p.stem.split("_")[1]) in ep_indices_set
        ]

    image_keys = get_image_keys(root)

    ep_idx = 0
    chunk_idx = 0
    file_idx = 0
    size_in_mb = 0
    num_frames = 0
    paths_to_cat = []
    episodes_metadata = []

    logging.info(f"Converting data files from {len(ep_paths)} episodes")

    for ep_path in tqdm.tqdm(ep_paths, desc="Converting data files"):
        ep_size_in_mb = get_parquet_file_size_in_mb(ep_path)
        ep_num_frames = get_parquet_num_frames(ep_path)

        # Extract original episode index from filename
        orig_ep_idx = int(ep_path.stem.split("_")[1])

        ep_metadata = {
            "episode_index": orig_ep_idx,
            "data/chunk_index": chunk_idx,
            "data/file_index": file_idx,
            "dataset_from_index": num_frames,
            "dataset_to_index": num_frames + ep_num_frames,
        }
        size_in_mb += ep_size_in_mb
        num_frames += ep_num_frames
        episodes_metadata.append(ep_metadata)
        ep_idx += 1

        if size_in_mb < data_file_size_in_mb:
            paths_to_cat.append(ep_path)
            continue

        if paths_to_cat:
            concat_data_files(paths_to_cat, new_root, chunk_idx, file_idx, image_keys)

        # Reset for the next file
        size_in_mb = ep_size_in_mb
        paths_to_cat = [ep_path]

        chunk_idx, file_idx = update_chunk_file_indices(chunk_idx, file_idx, DEFAULT_CHUNK_SIZE)

    # Write remaining data if any
    if paths_to_cat:
        concat_data_files(paths_to_cat, new_root, chunk_idx, file_idx, image_keys)

    return episodes_metadata


def convert_videos_of_camera(
    root: Path,
    new_root: Path,
    video_key: str,
    video_file_size_in_mb: int,
    episode_indices: Optional[List[int]] = None
) -> List[Dict]:
    """Convert videos of a single camera from episode-based to file-based format."""
    videos_dir = root / "videos"
    ep_paths = sorted(videos_dir.glob(f"*/{video_key}/*.mp4"))

    # Filter episodes if sharding
    if episode_indices is not None:
        ep_indices_set = set(episode_indices)
        ep_paths = [
            p for p in ep_paths
            if int(p.stem.split("_")[1]) in ep_indices_set
        ]

    ep_idx = 0
    chunk_idx = 0
    file_idx = 0
    size_in_mb = 0
    duration_in_s = 0.0
    paths_to_cat = []
    episodes_metadata = []

    for ep_path in tqdm.tqdm(ep_paths, desc=f"Converting videos of {video_key}"):
        ep_size_in_mb = get_file_size_in_mb(ep_path)
        ep_duration_in_s = get_video_duration_in_s(ep_path)

        # Extract original episode index from filename
        orig_ep_idx = int(ep_path.stem.split("_")[1])

        # Check if adding this episode would exceed the limit
        if size_in_mb + ep_size_in_mb >= video_file_size_in_mb and len(paths_to_cat) > 0:
            # Size limit would be exceeded, save current accumulation WITHOUT this episode
            output_path = new_root / DEFAULT_VIDEO_PATH.format(
                video_key=video_key, chunk_index=chunk_idx, file_index=file_idx
            )
            concatenate_video_files(paths_to_cat, output_path)

            # Update episodes metadata for the file we just saved
            for i, _ in enumerate(paths_to_cat):
                past_ep_idx = ep_idx - len(paths_to_cat) + i
                episodes_metadata[past_ep_idx][f"videos/{video_key}/chunk_index"] = chunk_idx
                episodes_metadata[past_ep_idx][f"videos/{video_key}/file_index"] = file_idx

            # Move to next file and start fresh with current episode
            chunk_idx, file_idx = update_chunk_file_indices(chunk_idx, file_idx, DEFAULT_CHUNK_SIZE)
            size_in_mb = 0
            duration_in_s = 0.0
            paths_to_cat = []

        # Add current episode metadata
        ep_metadata = {
            "episode_index": orig_ep_idx,
            f"videos/{video_key}/chunk_index": chunk_idx,
            f"videos/{video_key}/file_index": file_idx,
            f"videos/{video_key}/from_timestamp": duration_in_s,
            f"videos/{video_key}/to_timestamp": duration_in_s + ep_duration_in_s,
        }
        episodes_metadata.append(ep_metadata)

        # Add current episode to accumulation
        paths_to_cat.append(ep_path)
        size_in_mb += ep_size_in_mb
        duration_in_s += ep_duration_in_s
        ep_idx += 1

    # Write remaining videos if any
    if paths_to_cat:
        output_path = new_root / DEFAULT_VIDEO_PATH.format(
            video_key=video_key, chunk_index=chunk_idx, file_index=file_idx
        )
        concatenate_video_files(paths_to_cat, output_path)

        # Update episodes metadata for the final file
        for i, _ in enumerate(paths_to_cat):
            past_ep_idx = ep_idx - len(paths_to_cat) + i
            episodes_metadata[past_ep_idx][f"videos/{video_key}/chunk_index"] = chunk_idx
            episodes_metadata[past_ep_idx][f"videos/{video_key}/file_index"] = file_idx

    return episodes_metadata


def convert_videos(
    root: Path,
    new_root: Path,
    video_file_size_in_mb: int,
    episode_indices: Optional[List[int]] = None
) -> Optional[List[Dict]]:
    """Convert all videos from episode-based to file-based format."""
    logging.info(f"Converting videos from {root} to {new_root}")

    video_keys = get_video_keys(root)
    if len(video_keys) == 0:
        return None

    video_keys = sorted(video_keys)

    eps_metadata_per_cam = []
    for camera in video_keys:
        eps_metadata = convert_videos_of_camera(
            root, new_root, camera, video_file_size_in_mb, episode_indices
        )
        eps_metadata_per_cam.append(eps_metadata)

    num_eps_per_cam = [len(eps_cam_map) for eps_cam_map in eps_metadata_per_cam]
    if len(set(num_eps_per_cam)) != 1:
        raise ValueError(f"All cameras don't have same number of episodes ({num_eps_per_cam}).")

    episodes_metadata = []
    num_cameras = len(video_keys)
    num_episodes = num_eps_per_cam[0]

    for ep_idx in tqdm.tqdm(range(num_episodes), desc="Merging video metadata"):
        # Sanity check
        ep_ids = [eps_metadata_per_cam[cam_idx][ep_idx]["episode_index"] for cam_idx in range(num_cameras)]
        if len(set(ep_ids)) != 1:
            raise ValueError(f"All episode indices need to match ({ep_ids}).")

        ep_dict = {}
        for cam_idx in range(num_cameras):
            ep_dict.update(eps_metadata_per_cam[cam_idx][ep_idx])
        episodes_metadata.append(ep_dict)

    return episodes_metadata


def generate_episode_metadata_dict(
    episodes_legacy_metadata: Dict,
    episodes_metadata: List[Dict],
    episodes_stats: Dict,
    episodes_videos: Optional[List[Dict]] = None
):
    """Generator function to yield episode metadata dictionaries."""
    num_episodes = len(episodes_metadata)

    # Create mapping from episode_index to metadata
    ep_metadata_map = {ep["episode_index"]: ep for ep in episodes_metadata}
    ep_videos_map = {}
    if episodes_videos is not None:
        ep_videos_map = {ep["episode_index"]: ep for ep in episodes_videos}

    for ep_idx in sorted(episodes_legacy_metadata.keys()):
        ep_legacy_metadata = episodes_legacy_metadata[ep_idx]

        if ep_idx not in ep_metadata_map:
            continue

        ep_metadata = ep_metadata_map[ep_idx]
        ep_stats = episodes_stats.get(ep_idx, {})

        ep_video = ep_videos_map.get(ep_idx, {})

        ep_dict = {**ep_metadata, **ep_video, **ep_legacy_metadata, **flatten_dict({"stats": ep_stats})}
        ep_dict["meta/episodes/chunk_index"] = 0
        ep_dict["meta/episodes/file_index"] = 0
        yield ep_dict


def convert_episodes_metadata(
    root: Path,
    new_root: Path,
    episodes_metadata: List[Dict],
    episodes_video_metadata: Optional[List[Dict]] = None
) -> None:
    """Convert episodes metadata from JSONL to parquet format."""
    logging.info(f"Converting episodes metadata from {root} to {new_root}")

    episodes_legacy_metadata = legacy_load_episodes(root)
    episodes_stats = legacy_load_episodes_stats(root)

    ds_episodes = Dataset.from_generator(
        lambda: generate_episode_metadata_dict(
            episodes_legacy_metadata, episodes_metadata, episodes_stats, episodes_video_metadata
        )
    )
    write_episodes(ds_episodes, new_root)

    stats = aggregate_stats(list(episodes_stats.values()))
    write_stats(stats, new_root)


def convert_info(
    root: Path,
    new_root: Path,
    data_file_size_in_mb: int,
    video_file_size_in_mb: int
) -> None:
    """Convert and update info.json for v3.0."""
    info = load_info(root)
    info["codebase_version"] = V30

    # Remove deprecated fields if they exist
    info.pop("total_chunks", None)
    info.pop("total_videos", None)

    info["data_files_size_in_mb"] = data_file_size_in_mb
    info["video_files_size_in_mb"] = video_file_size_in_mb
    info["data_path"] = DEFAULT_DATA_PATH
    info["video_path"] = DEFAULT_VIDEO_PATH if info.get("video_path") is not None else None
    info["fps"] = int(info["fps"])

    logging.info(f"Converting info from {root} to {new_root}")

    # Add fps to features that don't have video_info
    for key in info["features"]:
        if info["features"][key]["dtype"] == "video":
            # Already has fps in video_info
            continue
        info["features"][key]["fps"] = info["fps"]

    write_info(info, new_root)


def convert_dataset(
    input_dir: str,
    output_dir: str,
    data_file_size_in_mb: Optional[int] = None,
    video_file_size_in_mb: Optional[int] = None,
    force_overwrite: bool = False,
    shard_id: Optional[int] = None,
    num_shards: Optional[int] = None,
) -> None:
    """
    Convert a LeRobot v2.1 dataset to v3.0 format.

    Args:
        input_dir: Path to v2.1 dataset
        output_dir: Path to output v3.0 dataset
        data_file_size_in_mb: Target size for data files (default: 100MB)
        video_file_size_in_mb: Target size for video files (default: 500MB)
        force_overwrite: Overwrite existing output directory
        shard_id: Shard ID for distributed processing (0-indexed)
        num_shards: Total number of shards
    """
    if data_file_size_in_mb is None:
        data_file_size_in_mb = DEFAULT_DATA_FILE_SIZE_IN_MB
    if video_file_size_in_mb is None:
        video_file_size_in_mb = DEFAULT_VIDEO_FILE_SIZE_IN_MB

    root = Path(input_dir)
    new_root = Path(output_dir)

    # Handle sharding suffix
    if shard_id is not None and num_shards is not None:
        new_root = Path(f"{output_dir}_shard{shard_id:02d}of{num_shards:02d}")

    # Validate input dataset
    if not root.exists():
        raise ValueError(f"Input directory does not exist: {root}")

    validate_local_dataset_version(root)
    logging.info(f"Using local dataset at {root}")

    # Handle output directory
    if new_root.exists():
        if force_overwrite:
            logging.info(f"Removing existing output directory: {new_root}")
            shutil.rmtree(new_root)
        else:
            logging.info(f"Output directory already exists, skipping: {new_root}")
            return

    # Determine episode indices for sharding
    episode_indices = None
    if shard_id is not None and num_shards is not None:
        episodes_legacy_metadata = legacy_load_episodes(root)
        all_episode_indices = sorted(episodes_legacy_metadata.keys())
        total_episodes = len(all_episode_indices)

        shard_size = (total_episodes + num_shards - 1) // num_shards
        start_idx = shard_id * shard_size
        end_idx = min(start_idx + shard_size, total_episodes)
        episode_indices = all_episode_indices[start_idx:end_idx]

        logging.info(f"Shard {shard_id}/{num_shards}: Processing episodes {start_idx} to {end_idx-1} "
                    f"({len(episode_indices)} episodes)")

    try:
        # Convert info.json
        convert_info(root, new_root, data_file_size_in_mb, video_file_size_in_mb)

        # Convert tasks
        convert_tasks(root, new_root)

        # Convert data files
        episodes_metadata = convert_data(root, new_root, data_file_size_in_mb, episode_indices)

        # Convert videos
        episodes_videos_metadata = convert_videos(root, new_root, video_file_size_in_mb, episode_indices)

        # Convert episodes metadata
        convert_episodes_metadata(root, new_root, episodes_metadata, episodes_videos_metadata)

        # Save shard metadata if sharding
        if shard_id is not None and num_shards is not None:
            shard_meta = {
                "shard_id": shard_id,
                "num_shards": num_shards,
                "total_episodes": len(episode_indices) if episode_indices else 0,
            }
            shard_meta_path = new_root / "meta" / "shard_info.json"
            shard_meta_path.parent.mkdir(parents=True, exist_ok=True)
            with open(shard_meta_path, "w") as f:
                json.dump(shard_meta, f, indent=2)
            logging.info(f"Shard metadata saved to: {shard_meta_path}")

        logging.info(f"Conversion completed successfully! Output: {new_root}")

    except Exception as e:
        logging.error(f"Conversion failed: {e}")
        if new_root.exists():
            shutil.rmtree(new_root)
        raise


def main():
    parser = argparse.ArgumentParser(
        description="Convert LeRobot v2.1 dataset to v3.0 format"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Path to input v2.1 dataset directory"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Path to output v3.0 dataset directory"
    )
    parser.add_argument(
        "--data_file_size_in_mb",
        type=int,
        default=None,
        help=f"Target size for data files in MB (default: {DEFAULT_DATA_FILE_SIZE_IN_MB})"
    )
    parser.add_argument(
        "--video_file_size_in_mb",
        type=int,
        default=None,
        help=f"Target size for video files in MB (default: {DEFAULT_VIDEO_FILE_SIZE_IN_MB})"
    )
    parser.add_argument(
        "--force_overwrite",
        action="store_true",
        help="Force overwrite existing output directory"
    )
    parser.add_argument(
        "--shard_id",
        type=int,
        default=None,
        help="Shard ID (0-indexed) for distributed processing"
    )
    parser.add_argument(
        "--num_shards",
        type=int,
        default=None,
        help="Total number of shards for distributed processing"
    )

    args = parser.parse_args()

    # Validate shard arguments
    if (args.shard_id is None) != (args.num_shards is None):
        parser.error("--shard_id and --num_shards must be used together")
    if args.shard_id is not None and (args.shard_id < 0 or args.shard_id >= args.num_shards):
        parser.error(f"--shard_id must be in range [0, {args.num_shards - 1}]")

    convert_dataset(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        data_file_size_in_mb=args.data_file_size_in_mb,
        video_file_size_in_mb=args.video_file_size_in_mb,
        force_overwrite=args.force_overwrite,
        shard_id=args.shard_id,
        num_shards=args.num_shards,
    )


if __name__ == "__main__":
    main()
