"""
Merge multiple LeRobot shard directories into a single dataset.

Usage:
    python merge_lerobot_shards.py \
        --shard_pattern "/path/to/output_shard*" \
        --output_dir /path/to/merged_output \
        --num_threads 32

Example:
    # After running sharded conversion:
    # python convert_droid_to_lerobot_anno_fast.py --shard_id 0 --num_shards 5
    # python convert_droid_to_lerobot_anno_fast.py --shard_id 1 --num_shards 5
    # ...

    # Merge all shards:
    python merge_lerobot_shards.py \
        --shard_pattern "/path/to/lerobot_droid_anno_shard*" \
        --output_dir /path/to/lerobot_droid_anno
"""

import os
import json
import shutil
import argparse
import re
from pathlib import Path
from glob import glob
from typing import List, Dict, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import pandas as pd
import numpy as np


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
    info_path = shard_dir / "meta" / "info.json"
    shard_info_path = shard_dir / "meta" / "shard_info.json"

    with open(info_path, "r") as f:
        info = json.load(f)

    shard_info = {}
    if shard_info_path.exists():
        with open(shard_info_path, "r") as f:
            shard_info = json.load(f)

    return {
        "info": info,
        "shard_info": shard_info,
        "dir": shard_dir,
    }


def load_episodes_jsonl(shard_dir: Path) -> List[Dict]:
    """Load episodes.jsonl from a shard."""
    episodes_path = shard_dir / "meta" / "episodes.jsonl"
    episodes = []
    with open(episodes_path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                episodes.append(json.loads(line))
    return episodes


def load_tasks_jsonl(shard_dir: Path) -> List[Dict]:
    """Load tasks.jsonl from a shard."""
    tasks_path = shard_dir / "meta" / "tasks.jsonl"
    tasks = []
    if tasks_path.exists():
        with open(tasks_path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    tasks.append(json.loads(line))
    return tasks


def load_episodes_stats_jsonl(shard_dir: Path) -> List[Dict]:
    """Load episodes_stats.jsonl from a shard."""
    stats_path = shard_dir / "meta" / "episodes_stats.jsonl"
    stats = []
    if stats_path.exists():
        with open(stats_path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    stats.append(json.loads(line))
    return stats


def copy_and_remap_parquet(
    src_parquet: Path,
    dst_parquet: Path,
    episode_offset: int,
    frame_offset: int,
    task_remap: Dict[int, int],
) -> int:
    """Copy parquet file and remap indices. Returns number of frames."""
    df = pd.read_parquet(src_parquet)

    # Remap indices
    df["episode_index"] = df["episode_index"] + episode_offset
    df["index"] = df["index"] + frame_offset

    # Remap task_index using the task remap dictionary
    if "task_index" in df.columns and task_remap:
        df["task_index"] = df["task_index"].map(lambda x: task_remap.get(x, x))

    # Save to destination
    dst_parquet.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(dst_parquet, index=False)

    return len(df)


def copy_video_file(src_video: Path, dst_video: Path):
    """Copy a video file."""
    dst_video.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_video, dst_video)


def process_episode_files(
    shard_dir: Path,
    output_dir: Path,
    old_episode_idx: int,
    new_episode_idx: int,
    frame_offset: int,
    task_remap: Dict[int, int],
    info: Dict,
) -> Tuple[int, List[str]]:
    """Process all files for a single episode. Returns (num_frames, video_keys)."""
    # Determine chunk indices
    chunks_size = info.get("chunks_size", 1000)
    old_chunk = old_episode_idx // chunks_size
    new_chunk = new_episode_idx // chunks_size

    # Copy parquet file
    src_parquet = shard_dir / "data" / f"chunk-{old_chunk:03d}" / f"episode_{old_episode_idx:06d}.parquet"
    dst_parquet = output_dir / "data" / f"chunk-{new_chunk:03d}" / f"episode_{new_episode_idx:06d}.parquet"

    num_frames = 0
    if src_parquet.exists():
        num_frames = copy_and_remap_parquet(src_parquet, dst_parquet, new_episode_idx - old_episode_idx, frame_offset, task_remap)

    # Copy video files
    video_keys = []
    videos_dir = shard_dir / "videos" / f"chunk-{old_chunk:03d}"
    if videos_dir.exists():
        for video_key_dir in videos_dir.iterdir():
            if video_key_dir.is_dir():
                video_key = video_key_dir.name
                src_video = video_key_dir / f"episode_{old_episode_idx:06d}.mp4"
                if src_video.exists():
                    dst_video = output_dir / "videos" / f"chunk-{new_chunk:03d}" / video_key / f"episode_{new_episode_idx:06d}.mp4"
                    copy_video_file(src_video, dst_video)
                    video_keys.append(video_key)

    return num_frames, video_keys


def merge_shards(
    shard_pattern: str,
    output_dir: str,
    num_threads: int = 32,
    delete_shards: bool = False,
):
    """Merge multiple LeRobot shard directories into a single dataset."""
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
        data["episodes"] = load_episodes_jsonl(shard_dir)
        data["tasks"] = load_tasks_jsonl(shard_dir)
        data["stats"] = load_episodes_stats_jsonl(shard_dir)
        shard_data.append(data)
        print(f"  {shard_dir.name}: {len(data['episodes'])} episodes, {data['info']['total_frames']} frames")

    # Build global task mapping
    print("\nBuilding global task mapping...")
    global_tasks = {}  # task_text -> global_task_index
    task_remaps = []   # Per-shard: local_task_index -> global_task_index

    for shard in shard_data:
        task_remap = {}
        for task_entry in shard["tasks"]:
            task_text = task_entry["task"]
            local_idx = task_entry["task_index"]

            if task_text not in global_tasks:
                global_tasks[task_text] = len(global_tasks)

            task_remap[local_idx] = global_tasks[task_text]

        task_remaps.append(task_remap)

    print(f"  Total unique tasks: {len(global_tasks)}")

    # Calculate episode and frame offsets
    episode_offsets = [0]
    frame_offsets = [0]
    for shard in shard_data[:-1]:
        episode_offsets.append(episode_offsets[-1] + len(shard["episodes"]))
        frame_offsets.append(frame_offsets[-1] + shard["info"]["total_frames"])

    total_episodes = episode_offsets[-1] + len(shard_data[-1]["episodes"])
    total_frames = frame_offsets[-1] + shard_data[-1]["info"]["total_frames"]

    print(f"\nTotal episodes: {total_episodes}")
    print(f"Total frames: {total_frames}")

    # Create output directory
    if output_path.exists():
        print(f"\nRemoving existing output directory: {output_path}")
        shutil.rmtree(output_path)
    output_path.mkdir(parents=True)
    (output_path / "meta").mkdir()
    (output_path / "data").mkdir()
    (output_path / "videos").mkdir()

    # Process all episodes
    print(f"\nProcessing episodes with {num_threads} threads...")

    # Build list of all episode processing tasks
    episode_tasks = []
    for shard_idx, shard in enumerate(shard_data):
        for local_ep_idx, episode in enumerate(shard["episodes"]):
            global_ep_idx = episode_offsets[shard_idx] + local_ep_idx
            episode_tasks.append({
                "shard_dir": shard["dir"],
                "old_episode_idx": local_ep_idx,
                "new_episode_idx": global_ep_idx,
                "frame_offset": frame_offsets[shard_idx],
                "task_remap": task_remaps[shard_idx],
                "info": shard["info"],
            })

    # Process episodes in parallel
    video_keys_set = set()
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = []
        for task in episode_tasks:
            futures.append(
                executor.submit(
                    process_episode_files,
                    task["shard_dir"],
                    output_path,
                    task["old_episode_idx"],
                    task["new_episode_idx"],
                    task["frame_offset"],
                    task["task_remap"],
                    task["info"],
                )
            )

        for future in tqdm(as_completed(futures), total=len(futures), desc="Copying episodes"):
            num_frames, video_keys = future.result()
            video_keys_set.update(video_keys)

    # Write merged metadata files
    print("\nWriting merged metadata...")

    # Write episodes.jsonl
    episodes_path = output_path / "meta" / "episodes.jsonl"
    with open(episodes_path, "w") as f:
        for shard_idx, shard in enumerate(shard_data):
            for local_ep_idx, episode in enumerate(shard["episodes"]):
                new_episode = {
                    "episode_index": episode_offsets[shard_idx] + local_ep_idx,
                    "tasks": episode["tasks"],
                    "length": episode["length"],
                }
                f.write(json.dumps(new_episode) + "\n")

    # Write tasks.jsonl
    tasks_path = output_path / "meta" / "tasks.jsonl"
    with open(tasks_path, "w") as f:
        for task_text, task_idx in sorted(global_tasks.items(), key=lambda x: x[1]):
            f.write(json.dumps({"task_index": task_idx, "task": task_text}) + "\n")

    # Write episodes_stats.jsonl (merge all stats with updated episode indices)
    stats_path = output_path / "meta" / "episodes_stats.jsonl"
    with open(stats_path, "w") as f:
        for shard_idx, shard in enumerate(shard_data):
            for local_ep_idx, stat in enumerate(shard["stats"]):
                # Update episode_index in stats if present
                new_stat = stat.copy()
                if "episode_index" in new_stat:
                    new_stat["episode_index"] = episode_offsets[shard_idx] + local_ep_idx
                f.write(json.dumps(new_stat) + "\n")

    # Write info.json
    # Use first shard's info as template
    merged_info = shard_data[0]["info"].copy()
    merged_info["total_episodes"] = total_episodes
    merged_info["total_frames"] = total_frames
    merged_info["total_tasks"] = len(global_tasks)
    merged_info["total_videos"] = total_episodes * len(video_keys_set)
    merged_info["total_chunks"] = (total_episodes + merged_info.get("chunks_size", 1000) - 1) // merged_info.get("chunks_size", 1000)

    # Update splits
    merged_info["splits"] = {
        "train": f"0:{total_episodes}"
    }

    info_path = output_path / "meta" / "info.json"
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
        description="Merge multiple LeRobot shard directories into a single dataset"
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
