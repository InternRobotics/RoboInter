"""
Merge multiple LeRobot v3.0 shard directories into a single dataset.

LeRobot v3.0 uses file-based organization instead of episode-based:
- data/chunk-XXX/file_XXX.parquet
- videos/CAMERA/chunk-XXX/file_XXX.mp4
- meta/episodes/chunk-XXX/episodes_XXX.parquet
- meta/tasks/chunk-XXX/file_XXX.parquet
- meta/episodes_stats/chunk-XXX/file_XXX.parquet

Usage:
    python merge_v30_shards.py \
        --shard_pattern "/path/to/v30_shard*" \
        --output_dir /path/to/merged_output \
        --num_threads 32

Example:
    # After running sharded v2.1->v3.0 conversion:
    # python convert_v21_to_v30.py --shard_id 0 --num_shards 5 ...
    # python convert_v21_to_v30.py --shard_id 1 --num_shards 5 ...
    # ...

    # Merge all shards:
    python merge_v30_shards.py \
        --shard_pattern "/path/to/lerobot_droid_anno_v30_shard*" \
        --output_dir /path/to/lerobot_droid_anno_v30
"""

import os
import json
import shutil
import argparse
import re
from pathlib import Path
from glob import glob
from typing import List, Dict, Tuple, Optional, Generator
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

# Try to import from lerobot
try:
    from lerobot.datasets.utils import (
        DEFAULT_CHUNK_SIZE,
        DEFAULT_DATA_PATH,
        DEFAULT_VIDEO_PATH,
        load_info,
        write_info,
    )
    from lerobot.datasets.compute_stats import aggregate_stats
except ImportError:
    from lerobot.common.datasets.utils import (
        DEFAULT_CHUNK_SIZE,
        DEFAULT_DATA_PATH,
        DEFAULT_VIDEO_PATH,
        load_info,
        write_info,
    )
    from lerobot.common.datasets.compute_stats import aggregate_stats


def find_shard_dirs(shard_pattern: str) -> List[Path]:
    """Find all shard directories matching the pattern, sorted by shard_id."""
    shard_dirs = []
    for path in glob(shard_pattern):
        if os.path.isdir(path):
            shard_dirs.append(Path(path))

    # Sort by shard ID extracted from directory name
    def extract_shard_id(path: Path) -> int:
        match = re.search(r'_shard(\d+)of\d+$', path.name)
        if match:
            return int(match.group(1))
        return 0

    return sorted(shard_dirs, key=extract_shard_id)


def load_shard_info(shard_dir: Path) -> Dict:
    """Load shard metadata."""
    info = load_info(shard_dir)

    shard_info = {}
    shard_info_path = shard_dir / "meta" / "shard_info.json"
    if shard_info_path.exists():
        with open(shard_info_path, "r") as f:
            shard_info = json.load(f)

    return {
        "info": info,
        "shard_info": shard_info,
        "dir": shard_dir,
    }


def get_parquet_files(directory: Path, pattern: str = "*.parquet") -> List[Path]:
    """Get all parquet files in directory tree."""
    return sorted(directory.glob(f"**/{pattern}"))


def get_video_files(directory: Path, pattern: str = "*.mp4") -> List[Path]:
    """Get all video files in directory tree."""
    return sorted(directory.glob(f"**/{pattern}"))


def read_parquet_episodes(shard_dir: Path) -> pd.DataFrame:
    """Read episodes metadata from parquet files."""
    episodes_dir = shard_dir / "meta" / "episodes"
    if not episodes_dir.exists():
        return pd.DataFrame()

    parquet_files = get_parquet_files(episodes_dir)
    if not parquet_files:
        return pd.DataFrame()

    dfs = [pd.read_parquet(f) for f in parquet_files]
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


def read_parquet_tasks(shard_dir: Path) -> pd.DataFrame:
    """Read tasks from parquet files."""
    tasks_dir = shard_dir / "meta" / "tasks"
    if not tasks_dir.exists():
        return pd.DataFrame()

    parquet_files = get_parquet_files(tasks_dir)
    if not parquet_files:
        return pd.DataFrame()

    dfs = [pd.read_parquet(f) for f in parquet_files]
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


def read_parquet_stats(shard_dir: Path) -> pd.DataFrame:
    """Read episode stats from parquet files."""
    stats_dir = shard_dir / "meta" / "episodes_stats"
    if not stats_dir.exists():
        return pd.DataFrame()

    parquet_files = get_parquet_files(stats_dir)
    if not parquet_files:
        return pd.DataFrame()

    dfs = [pd.read_parquet(f) for f in parquet_files]
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


def copy_and_update_data_file(
    src_path: Path,
    dst_path: Path,
    episode_offset: int,
    frame_offset: int,
    task_remap: Dict[int, int],
) -> None:
    """Copy data parquet file and update indices."""
    df = pd.read_parquet(src_path)

    # Update indices
    if "episode_index" in df.columns:
        df["episode_index"] = df["episode_index"] + episode_offset
    if "index" in df.columns:
        df["index"] = df["index"] + frame_offset
    if "task_index" in df.columns and task_remap:
        df["task_index"] = df["task_index"].map(lambda x: task_remap.get(x, x))

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(dst_path, index=False)


def copy_video_file(src_path: Path, dst_path: Path) -> None:
    """Copy a video file."""
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_path, dst_path)


def merge_shards(
    shard_pattern: str,
    output_dir: str,
    num_threads: int = 32,
    delete_shards: bool = False,
):
    """Merge multiple LeRobot v3.0 shard directories into a single dataset."""
    output_path = Path(output_dir)

    # Find shard directories
    print("Finding shard directories...")
    shard_dirs = find_shard_dirs(shard_pattern)

    if not shard_dirs:
        print(f"No shard directories found matching pattern: {shard_pattern}")
        return

    print(f"Found {len(shard_dirs)} shards:")
    for sd in shard_dirs:
        print(f"  - {sd}")

    # Load all shard metadata
    print("\nLoading shard metadata...")
    shard_data = []
    for shard_dir in shard_dirs:
        data = load_shard_info(shard_dir)
        data["episodes"] = read_parquet_episodes(shard_dir)
        data["tasks"] = read_parquet_tasks(shard_dir)
        data["stats"] = read_parquet_stats(shard_dir)
        shard_data.append(data)

        num_eps = len(data["episodes"]) if not data["episodes"].empty else 0
        total_frames = data["info"].get("total_frames", 0)
        print(f"  {shard_dir.name}: {num_eps} episodes, {total_frames} frames")

    # Build global task mapping
    print("\nBuilding global task mapping...")
    global_tasks = {}  # task_text -> global_task_index
    task_remaps = []   # Per-shard: local_task_index -> global_task_index

    for shard in shard_data:
        task_remap = {}
        if not shard["tasks"].empty:
            for _, row in shard["tasks"].iterrows():
                task_text = row.name if isinstance(row.name, str) else str(row.get("task", ""))
                local_idx = row.get("task_index", 0)

                if task_text and task_text not in global_tasks:
                    global_tasks[task_text] = len(global_tasks)

                if task_text:
                    task_remap[local_idx] = global_tasks[task_text]

        task_remaps.append(task_remap)

    print(f"  Total unique tasks: {len(global_tasks)}")

    # Calculate episode and frame offsets
    episode_offsets = [0]
    frame_offsets = [0]
    for shard in shard_data[:-1]:
        num_eps = len(shard["episodes"]) if not shard["episodes"].empty else 0
        episode_offsets.append(episode_offsets[-1] + num_eps)
        frame_offsets.append(frame_offsets[-1] + shard["info"].get("total_frames", 0))

    last_shard = shard_data[-1]
    last_num_eps = len(last_shard["episodes"]) if not last_shard["episodes"].empty else 0
    total_episodes = episode_offsets[-1] + last_num_eps
    total_frames = frame_offsets[-1] + last_shard["info"].get("total_frames", 0)

    print(f"\nTotal episodes: {total_episodes}")
    print(f"Total frames: {total_frames}")

    # Create output directory
    if output_path.exists():
        print(f"\nRemoving existing output directory: {output_path}")
        shutil.rmtree(output_path)
    output_path.mkdir(parents=True)

    # Collect all data files to process
    print("\nCollecting files to process...")
    data_tasks = []
    video_tasks = []

    for shard_idx, shard in enumerate(shard_data):
        shard_dir = shard["dir"]

        # Collect data files
        data_dir = shard_dir / "data"
        if data_dir.exists():
            for parquet_file in get_parquet_files(data_dir):
                rel_path = parquet_file.relative_to(data_dir)
                dst_path = output_path / "data" / rel_path
                data_tasks.append({
                    "src": parquet_file,
                    "dst": dst_path,
                    "episode_offset": episode_offsets[shard_idx],
                    "frame_offset": frame_offsets[shard_idx],
                    "task_remap": task_remaps[shard_idx],
                })

        # Collect video files
        videos_dir = shard_dir / "videos"
        if videos_dir.exists():
            for video_file in get_video_files(videos_dir):
                rel_path = video_file.relative_to(videos_dir)
                dst_path = output_path / "videos" / rel_path
                video_tasks.append({
                    "src": video_file,
                    "dst": dst_path,
                })

    print(f"  Data files: {len(data_tasks)}")
    print(f"  Video files: {len(video_tasks)}")

    # Process data files
    print(f"\nProcessing data files with {num_threads} threads...")
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = []
        for task in data_tasks:
            futures.append(
                executor.submit(
                    copy_and_update_data_file,
                    task["src"],
                    task["dst"],
                    task["episode_offset"],
                    task["frame_offset"],
                    task["task_remap"],
                )
            )

        for future in tqdm(as_completed(futures), total=len(futures), desc="Copying data"):
            future.result()

    # Process video files
    print(f"\nProcessing video files with {num_threads} threads...")
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = []
        for task in video_tasks:
            futures.append(
                executor.submit(
                    copy_video_file,
                    task["src"],
                    task["dst"],
                )
            )

        for future in tqdm(as_completed(futures), total=len(futures), desc="Copying videos"):
            future.result()

    # Merge and write metadata
    print("\nMerging metadata...")

    # Merge episodes
    all_episodes = []
    for shard_idx, shard in enumerate(shard_data):
        if not shard["episodes"].empty:
            df = shard["episodes"].copy()
            if "episode_index" in df.columns:
                df["episode_index"] = df["episode_index"] + episode_offsets[shard_idx]
            # Update data indices
            if "dataset_from_index" in df.columns:
                df["dataset_from_index"] = df["dataset_from_index"] + frame_offsets[shard_idx]
            if "dataset_to_index" in df.columns:
                df["dataset_to_index"] = df["dataset_to_index"] + frame_offsets[shard_idx]
            all_episodes.append(df)

    if all_episodes:
        merged_episodes = pd.concat(all_episodes, ignore_index=True)
        episodes_out_dir = output_path / "meta" / "episodes" / "chunk-000"
        episodes_out_dir.mkdir(parents=True, exist_ok=True)
        merged_episodes.to_parquet(episodes_out_dir / "episodes_000.parquet", index=False)

    # Merge tasks
    if global_tasks:
        tasks_df = pd.DataFrame({
            "task_index": list(global_tasks.values()),
        }, index=list(global_tasks.keys()))
        tasks_out_dir = output_path / "meta" / "tasks" / "chunk-000"
        tasks_out_dir.mkdir(parents=True, exist_ok=True)
        tasks_df.to_parquet(tasks_out_dir / "file_000.parquet")

    # Merge stats
    all_stats = []
    for shard_idx, shard in enumerate(shard_data):
        if not shard["stats"].empty:
            df = shard["stats"].copy()
            if "episode_index" in df.columns:
                df["episode_index"] = df["episode_index"] + episode_offsets[shard_idx]
            all_stats.append(df)

    if all_stats:
        merged_stats = pd.concat(all_stats, ignore_index=True)
        stats_out_dir = output_path / "meta" / "episodes_stats" / "chunk-000"
        stats_out_dir.mkdir(parents=True, exist_ok=True)
        merged_stats.to_parquet(stats_out_dir / "file_000.parquet", index=False)

    # Write aggregated stats.json
    # TODO: Aggregate stats from all shards if needed

    # Write merged info.json
    merged_info = shard_data[0]["info"].copy()
    merged_info["total_episodes"] = total_episodes
    merged_info["total_frames"] = total_frames
    merged_info["total_tasks"] = len(global_tasks)

    # Update splits
    merged_info["splits"] = {
        "train": f"0:{total_episodes}"
    }

    info_path = output_path / "meta" / "info.json"
    info_path.parent.mkdir(parents=True, exist_ok=True)
    with open(info_path, "w") as f:
        json.dump(merged_info, f, indent=4)

    print(f"\n{'='*60}")
    print("Merge completed!")
    print(f"  Total shards: {len(shard_dirs)}")
    print(f"  Total episodes: {total_episodes}")
    print(f"  Total frames: {total_frames}")
    print(f"  Total tasks: {len(global_tasks)}")
    print(f"  Output directory: {output_path}")
    print(f"{'='*60}")

    # Optionally delete shard directories
    if delete_shards:
        print("\nDeleting shard directories...")
        for shard_dir in shard_dirs:
            shutil.rmtree(shard_dir)
            print(f"  Deleted: {shard_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Merge multiple LeRobot v3.0 shard directories into a single dataset"
    )
    parser.add_argument(
        "--shard_pattern", type=str, required=True,
        help="Glob pattern to find shard directories (e.g., '/path/to/output_shard*')"
    )
    parser.add_argument(
        "--output_dir", type=str, required=True,
        help="Output directory for merged dataset"
    )
    parser.add_argument(
        "--num_threads", type=int, default=32,
        help="Number of threads for parallel file copying"
    )
    parser.add_argument(
        "--delete_shards", action="store_true",
        help="Delete shard directories after successful merge"
    )

    args = parser.parse_args()

    merge_shards(
        shard_pattern=args.shard_pattern,
        output_dir=args.output_dir,
        num_threads=args.num_threads,
        delete_shards=args.delete_shards,
    )


if __name__ == "__main__":
    main()
