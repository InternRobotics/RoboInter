"""
LeRobot-style Dataset and DataLoader implementation.

Compatible with LeRobot v2.1 format (parquet + video).

Features:
- Frame range filtering via range_nop.json (remove idle frames at start/end)
- Episode filtering via Q_annotation fields (quality-based selection)
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, List, Optional, Sequence, Union, Callable
from dataclasses import dataclass, field
import numpy as np
import pyarrow.parquet as pq
from torch.utils.data import Dataset, ConcatDataset, DataLoader
import torch


def parse_annotation_value(value):
    """Parse annotation value from JSON string to Python object."""
    if value is None:
        return None
    if isinstance(value, str):
        if value == "":
            return None
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value  # Return as-is if not valid JSON
    return value


# Coordinate-based annotation fields that need scaling
COORDINATE_ANNOTATION_FIELDS = [
    "annotation.object_box",        # [[x1,y1],[x2,y2]]
    "annotation.placement_proposal", # [[x1,y1],[x2,y2]]
    "annotation.trace",             # [[x,y], ...]
    "annotation.gripper_box",       # [[x1,y1],[x2,y2]]
    "annotation.affordance_box",    # [[x1,y1],[x2,y2]]
    "annotation.contact_points",    # [x, y] or [[x,y], ...]
]


def scale_coordinates(value, scale_x: float, scale_y: float):
    """
    Scale coordinate values by given scale factors.

    Handles various formats:
    - Single point: [x, y] -> [x*scale_x, y*scale_y]
    - Box: [[x1,y1], [x2,y2]] -> [[x1*scale_x, y1*scale_y], [x2*scale_x, y2*scale_y]]
    - Trajectory: [[x1,y1], [x2,y2], ...] -> scaled points
    - Nested structures: recursively scale
    """
    if value is None:
        return None

    if isinstance(value, (list, tuple)):
        if len(value) == 0:
            return value

        # Check if it's a point [x, y] (two numbers)
        if len(value) == 2 and all(isinstance(v, (int, float)) for v in value):
            return [int(value[0] * scale_x), int(value[1] * scale_y)]

        # Check if it's a list of points or nested structure
        if isinstance(value[0], (list, tuple)):
            return [scale_coordinates(v, scale_x, scale_y) for v in value]

        # It might be a flat list of coordinates, try to handle pairs
        if all(isinstance(v, (int, float)) for v in value):
            # Could be [x1, y1, x2, y2, ...] format
            if len(value) % 2 == 0:
                result = []
                for i in range(0, len(value), 2):
                    result.extend([int(value[i] * scale_x), int(value[i+1] * scale_y)])
                return result

    elif isinstance(value, np.ndarray):
        # Convert to list, scale, and return as list
        return scale_coordinates(value.tolist(), scale_x, scale_y)

    # Return as-is if not a recognized format
    return value


@dataclass
class QAnnotationFilter:
    """
    Filter configuration for a single Q_annotation field.

    Args:
        field: Q_annotation field name (e.g., "Q_annotation.instruction_add")
        values: Allowed values. Use ["Primary"], ["Secondary"], ["Primary", "Secondary"],
                or ["not_empty"] for any non-empty value
        require_all_frames: If True, all frames must match; if False, any frame matching is enough
    """
    field: str
    values: List[str] = field(default_factory=lambda: ["not_empty"])
    require_all_frames: bool = False  # Usually we check episode-level, so first frame is enough

    def matches(self, value: str) -> bool:
        """Check if a value matches this filter."""
        if "not_empty" in self.values:
            return value is not None and value != ""
        return value in self.values


@dataclass
class FilterConfig:
    """
    Configuration for dataset filtering.

    Args:
        range_nop_path: Path to range_nop.json for frame range filtering.
                       If provided, only frames within valid ranges are included.
        use_range_filter: Whether to apply range filtering (default: True if range_nop_path provided)
        q_filters: List of QAnnotationFilter for episode-level filtering.
                  Episodes must match ALL filters (AND logic).
        q_filter_mode: "all" = episode must match all filters, "any" = match any filter

    Example:
        # Only episodes with Primary quality instruction_add
        config = FilterConfig(
            q_filters=[
                QAnnotationFilter("Q_annotation.instruction_add", ["Primary"])
            ]
        )

        # Episodes with Primary gripper_box AND Primary state_affordance
        config = FilterConfig(
            q_filters=[
                QAnnotationFilter("Q_annotation.gripper_box", ["Primary"]),
                QAnnotationFilter("Q_annotation.state_affordance", ["Primary"]),
            ]
        )

        # Episodes with any non-empty trace annotation
        config = FilterConfig(
            q_filters=[
                QAnnotationFilter("Q_annotation.trace", ["not_empty"])
            ]
        )
    """
    range_nop_path: Optional[str] = None
    use_range_filter: bool = True
    q_filters: List[QAnnotationFilter] = field(default_factory=list)
    q_filter_mode: str = "all"  # "all" or "any"

    @classmethod
    def from_dict(cls, config_dict: Dict) -> "FilterConfig":
        """Create FilterConfig from a dictionary (e.g., loaded from YAML/JSON)."""
        q_filters = []
        for qf in config_dict.get("q_filters", []):
            q_filters.append(QAnnotationFilter(
                field=qf["field"],
                values=qf.get("values", ["not_empty"]),
                require_all_frames=qf.get("require_all_frames", False),
            ))
        return cls(
            range_nop_path=config_dict.get("range_nop_path"),
            use_range_filter=config_dict.get("use_range_filter", True),
            q_filters=q_filters,
            q_filter_mode=config_dict.get("q_filter_mode", "all"),
        )


def load_range_nop(path: str) -> Dict[str, List[int]]:
    """
    Load range_nop.json file.

    Returns dict mapping episode_name to [start_frame, end_frame, total_length].
    """
    if path and os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return {}


class LeRobotDataset(Dataset):
    """
    Dataset for LeRobot v2.1 format.

    Loads data from parquet files and video frames.
    Compatible with standard PyTorch DataLoader.

    Args:
        root: Path to LeRobot dataset directory
        split: Data split to load ("train" or specific range like "0:100")
        delta_timestamps: Dict mapping action keys to list of time deltas for action horizon
        image_keys: List of image observation keys to load
        load_videos: Whether to load video frames
        transform: Optional transform to apply to each sample
        filter_config: FilterConfig for frame/episode filtering
        range_nop_path: Shortcut for frame range filtering (alternative to filter_config)
        q_filters: Shortcut for Q_annotation filtering (alternative to filter_config)

    Example with filtering:
        # Filter by frame range only
        dataset = LeRobotDataset(
            "/path/to/dataset",
            range_nop_path="/path/to/range_nop.json"
        )

        # Filter by Q_annotation
        dataset = LeRobotDataset(
            "/path/to/dataset",
            q_filters=[
                QAnnotationFilter("Q_annotation.instruction_add", ["Primary"]),
                QAnnotationFilter("Q_annotation.gripper_box", ["Primary", "Secondary"]),
            ]
        )

        # Full filter config
        config = FilterConfig(
            range_nop_path="/path/to/range_nop.json",
            q_filters=[QAnnotationFilter("Q_annotation.trace", ["not_empty"])]
        )
        dataset = LeRobotDataset("/path/to/dataset", filter_config=config)
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        delta_timestamps: Optional[Dict[str, List[float]]] = None,
        image_keys: Optional[List[str]] = None,
        load_videos: bool = True,
        transform: Optional[Callable] = None,
        filter_config: Optional[FilterConfig] = None,
        range_nop_path: Optional[str] = None,
        q_filters: Optional[List[QAnnotationFilter]] = None,
    ):
        self.root = Path(root)
        self.split = split
        self.delta_timestamps = delta_timestamps or {}
        self.load_videos = load_videos
        self.transform = transform

        # Build filter config from shortcuts if not provided directly
        if filter_config is None:
            filter_config = FilterConfig(
                range_nop_path=range_nop_path,
                q_filters=q_filters or [],
            )
        self.filter_config = filter_config

        # Load range_nop data if path provided
        self.range_nop_data = {}
        if self.filter_config.range_nop_path and self.filter_config.use_range_filter:
            self.range_nop_data = load_range_nop(self.filter_config.range_nop_path)
            print(f"Loaded range_nop data for {len(self.range_nop_data)} episodes")

        # Load metadata
        self._load_metadata()

        # Default image keys from features
        if image_keys is None:
            self.image_keys = [k for k in self.features if k.startswith("observation.images.")]
        else:
            self.image_keys = image_keys

        # Build frame index (with filtering)
        self._build_index()

        # Video cache
        self._video_cache: Dict[tuple, np.ndarray] = {}
        self._episode_cache: Dict[int, Dict] = {}

    def _load_metadata(self):
        """Load dataset metadata from meta/ directory."""
        meta_dir = self.root / "meta"

        # Load info.json
        with open(meta_dir / "info.json", "r") as f:
            self.info = json.load(f)

        self.fps = self.info.get("fps", 10)
        self.features = self.info.get("features", {})
        self.chunks_size = self.info.get("chunks_size", 1000)

        # Load episodes.jsonl
        self.episodes = []
        episodes_path = meta_dir / "episodes.jsonl"
        if episodes_path.exists():
            with open(episodes_path, "r") as f:
                for line in f:
                    if line.strip():
                        self.episodes.append(json.loads(line))

        # Load tasks.jsonl
        self.tasks = []
        tasks_path = meta_dir / "tasks.jsonl"
        if tasks_path.exists():
            with open(tasks_path, "r") as f:
                for line in f:
                    if line.strip():
                        self.tasks.append(json.loads(line))

    def _check_episode_q_filters(self, episode_idx: int) -> bool:
        """
        Check if an episode passes Q_annotation filters.

        Returns True if episode should be included.
        """
        if not self.filter_config.q_filters:
            return True

        # Load episode data to check Q_annotation values
        episode_data = self._load_episode_for_filter(episode_idx)
        if episode_data is None:
            return False

        results = []
        for qf in self.filter_config.q_filters:
            field_name = qf.field
            if field_name not in episode_data:
                # Field doesn't exist, filter fails
                results.append(False)
                continue

            values = episode_data[field_name]
            if qf.require_all_frames:
                # All frames must match
                match = all(qf.matches(v) for v in values)
            else:
                # Any frame matching is enough (check first frame for episode-level)
                match = qf.matches(values[0]) if len(values) > 0 else False
            results.append(match)

        if self.filter_config.q_filter_mode == "all":
            return all(results)
        else:  # "any"
            return any(results)

    def _load_episode_for_filter(self, episode_idx: int) -> Optional[Dict[str, Any]]:
        """Load episode data for filtering (lightweight, no video)."""
        parquet_path = self._get_parquet_path(episode_idx)
        if not parquet_path.exists():
            return None
        try:
            table = pq.read_table(parquet_path)
            df = table.to_pandas()
            data = {}
            for col in df.columns:
                data[col] = df[col].values
            return data
        except Exception:
            return None

    def _get_episode_name(self, episode_idx: int) -> Optional[str]:
        """Get episode_name for an episode by loading first frame."""
        episode_data = self._load_episode_for_filter(episode_idx)
        if episode_data is None:
            return None
        if "episode_name" in episode_data and len(episode_data["episode_name"]) > 0:
            return episode_data["episode_name"][0]
        return None

    def _build_index(self):
        """Build index mapping global frame index to (episode, local_frame)."""
        self.frame_index = []  # List of (episode_idx, local_frame_idx)
        self.episode_starts = []
        self.episode_lengths = []
        self.filtered_episode_indices = []  # Track which episodes passed filters

        # Stats for filtering
        total_episodes = len(self.episodes)
        filtered_by_q = 0
        total_frames_before = 0
        total_frames_after = 0

        for ep_idx, ep in enumerate(self.episodes):
            length = ep.get("length", 0)
            total_frames_before += length

            # Check Q_annotation filters
            if self.filter_config.q_filters:
                if not self._check_episode_q_filters(ep_idx):
                    filtered_by_q += 1
                    continue

            # Determine valid frame range
            start_frame = 0
            end_frame = length  # exclusive

            # Apply range_nop filtering if available
            if self.range_nop_data:
                episode_name = self._get_episode_name(ep_idx)
                if episode_name and episode_name in self.range_nop_data:
                    range_info = self.range_nop_data[episode_name]
                    # range_info = [start, end, total_length]
                    start_frame = range_info[0]
                    end_frame = range_info[1] + 1  # Convert to exclusive end

            # Clamp to valid range
            start_frame = max(0, start_frame)
            end_frame = min(length, end_frame)

            if start_frame >= end_frame:
                continue  # No valid frames

            self.filtered_episode_indices.append(ep_idx)
            self.episode_starts.append(len(self.frame_index))
            self.episode_lengths.append(length)  # Original length for action horizon

            for frame_idx in range(start_frame, end_frame):
                self.frame_index.append((ep_idx, frame_idx))
                total_frames_after += 1

        # Print filtering stats
        if self.filter_config.q_filters or self.range_nop_data:
            print(f"Dataset filtering stats:")
            print(f"  Episodes: {total_episodes} -> {len(self.filtered_episode_indices)} "
                  f"(filtered {filtered_by_q} by Q_annotation)")
            print(f"  Frames: {total_frames_before} -> {total_frames_after} "
                  f"(removed {total_frames_before - total_frames_after})")

    def __len__(self) -> int:
        return len(self.frame_index)

    def _get_parquet_path(self, episode_idx: int) -> Path:
        """Get parquet file path for an episode."""
        chunk_idx = episode_idx // self.chunks_size
        template = self.info.get(
            "data_path",
            "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
        )
        return self.root / template.format(
            episode_chunk=chunk_idx,
            episode_index=episode_idx,
        )

    def _get_video_path(self, episode_idx: int, video_key: str) -> Path:
        """Get video file path for an episode."""
        chunk_idx = episode_idx // self.chunks_size
        template = self.info.get(
            "video_path",
            "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
        )
        return self.root / template.format(
            episode_chunk=chunk_idx,
            episode_index=episode_idx,
            video_key=video_key,
        )

    def _load_episode(self, episode_idx: int) -> Dict[str, Any]:
        """Load episode data from parquet with caching."""
        if episode_idx in self._episode_cache:
            return self._episode_cache[episode_idx]

        parquet_path = self._get_parquet_path(episode_idx)
        table = pq.read_table(parquet_path)
        df = table.to_pandas()

        data = {}
        for col in df.columns:
            values = df[col].values
            if len(values) > 0 and isinstance(values[0], (list, np.ndarray)):
                data[col] = np.array([np.array(v) for v in values])
            else:
                data[col] = values

        # Cache (limit size)
        if len(self._episode_cache) > 10:
            self._episode_cache.pop(next(iter(self._episode_cache)))
        self._episode_cache[episode_idx] = data

        return data

    def _load_video_frames(self, episode_idx: int, video_key: str) -> Optional[np.ndarray]:
        """Load video frames with caching."""
        cache_key = (episode_idx, video_key)
        if cache_key in self._video_cache:
            return self._video_cache[cache_key]

        video_path = self._get_video_path(episode_idx, video_key)
        if not video_path.exists():
            return None

        try:
            import av
            container = av.open(str(video_path))
            frames = []
            for frame in container.decode(video=0):
                frames.append(frame.to_ndarray(format="rgb24"))
            container.close()

            frames_arr = np.array(frames)
            self._video_cache[cache_key] = frames_arr
            return frames_arr
        except Exception as e:
            print(f"Error loading video {video_path}: {e}")
            return None

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a single frame's data."""
        episode_idx, frame_idx = self.frame_index[idx]
        episode_data = self._load_episode(episode_idx)
        episode_length = self.episode_lengths[episode_idx]

        sample = {}

        # Get action (with optional action horizon via delta_timestamps)
        if "action" in episode_data:
            if self.delta_timestamps.get("action"):
                # Compute action horizon from delta_timestamps
                action_horizon = len(self.delta_timestamps["action"])
                actions = []
                for t in range(action_horizon):
                    action_idx = min(frame_idx + t, episode_length - 1)
                    actions.append(episode_data["action"][action_idx])
                sample["action"] = np.stack(actions, axis=0)
            else:
                sample["action"] = episode_data["action"][frame_idx]

        # Get state
        if "state" in episode_data:
            sample["state"] = episode_data["state"][frame_idx]

        # Get images
        if self.load_videos:
            for img_key in self.image_keys:
                video_frames = self._load_video_frames(episode_idx, img_key)
                if video_frames is not None and frame_idx < len(video_frames):
                    sample[img_key] = video_frames[frame_idx]

        # Get task/language instruction
        if "task_index" in episode_data:
            task_idx = episode_data["task_index"][frame_idx]
            if hasattr(task_idx, 'item'):
                task_idx = task_idx.item()
            if isinstance(task_idx, np.ndarray):
                task_idx = int(task_idx[0]) if task_idx.size > 0 else 0
            if task_idx < len(self.tasks):
                sample["task"] = self.tasks[task_idx].get("task", "")

        # Get annotation fields (all fields starting with "annotation.") and parse JSON
        for key in episode_data:
            if key.startswith("annotation."):
                raw_value = episode_data[key][frame_idx]
                sample[key] = parse_annotation_value(raw_value)

        # Scale coordinate-based annotations based on origin_shape vs actual image size
        origin_shape = sample.get("annotation.origin_shape")
        if origin_shape is not None and len(origin_shape) >= 2:
            # origin_shape is [width, height] of the original annotation coordinates
            origin_width, origin_height = origin_shape[0], origin_shape[1]

            # Get actual image size from loaded video frame
            actual_width, actual_height = None, None
            for img_key in self.image_keys:
                if img_key in sample and sample[img_key] is not None:
                    img = sample[img_key]
                    if isinstance(img, np.ndarray) and len(img.shape) >= 2:
                        # Image shape is (H, W, C) or (H, W)
                        actual_height, actual_width = img.shape[0], img.shape[1]
                        break

            # Apply scaling if we have both origin and actual sizes
            if actual_width is not None and actual_height is not None:
                if origin_width > 0 and origin_height > 0:
                    scale_x = actual_width / origin_width
                    scale_y = actual_height / origin_height

                    # Scale coordinate-based annotation fields
                    for coord_key in COORDINATE_ANNOTATION_FIELDS:
                        if coord_key in sample and sample[coord_key] is not None:
                            sample[coord_key] = scale_coordinates(
                                sample[coord_key], scale_x, scale_y
                            )

        # Get Q_annotation fields (quality indicators)
        for key in episode_data:
            if key.startswith("Q_annotation."):
                sample[key] = episode_data[key][frame_idx]

        # Get metadata fields
        for key in ["episode_name", "camera_view"]:
            if key in episode_data:
                sample[key] = episode_data[key][frame_idx]

        # Add indices
        sample["episode_index"] = episode_idx
        sample["frame_index"] = frame_idx

        # Apply transform
        if self.transform is not None:
            sample = self.transform(sample)

        return sample

    @property
    def num_episodes(self) -> int:
        return len(self.episodes)

    @property
    def num_frames(self) -> int:
        return len(self.frame_index)


class MultiDataset(Dataset):
    """
    Combines multiple LeRobotDataset instances.

    Args:
        datasets: List of dataset paths or LeRobotDataset instances
        weights: Optional sampling weights for each dataset
        transform: Optional transform to apply to all samples
        filter_config: FilterConfig to apply to all datasets (if creating from paths)
        range_nop_path: Shortcut for frame range filtering
        q_filters: Shortcut for Q_annotation filtering
    """

    def __init__(
        self,
        datasets: List[Union[str, LeRobotDataset]],
        weights: Optional[List[float]] = None,
        transform: Optional[Callable] = None,
        filter_config: Optional[FilterConfig] = None,
        range_nop_path: Optional[str] = None,
        q_filters: Optional[List[QAnnotationFilter]] = None,
        **kwargs,
    ):
        self.datasets = []
        self.dataset_names = []

        # Build filter config for new datasets
        if filter_config is None and (range_nop_path or q_filters):
            filter_config = FilterConfig(
                range_nop_path=range_nop_path,
                q_filters=q_filters or [],
            )

        for ds in datasets:
            if isinstance(ds, str):
                name = Path(ds).name
                ds = LeRobotDataset(
                    ds,
                    transform=None,
                    filter_config=filter_config,
                    **kwargs
                )
            else:
                name = getattr(ds, 'name', f'dataset_{len(self.datasets)}')
            self.datasets.append(ds)
            self.dataset_names.append(name)

        self.weights = weights or [1.0] * len(self.datasets)
        self.transform = transform

        # Build combined index
        self._offsets = []
        offset = 0
        for ds in self.datasets:
            self._offsets.append(offset)
            offset += len(ds)
        self._total_len = offset

        print(f"MultiDataset: {len(self.datasets)} datasets, {self._total_len} total frames")

    def __len__(self) -> int:
        return self._total_len

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        # Find which dataset this index belongs to
        ds_idx = 0
        for i, offset in enumerate(self._offsets):
            if idx >= offset:
                ds_idx = i
            else:
                break

        local_idx = idx - self._offsets[ds_idx]
        sample = self.datasets[ds_idx][local_idx]
        sample["dataset_name"] = self.dataset_names[ds_idx]

        if self.transform is not None:
            sample = self.transform(sample)

        return sample


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Collate batch into stacked arrays/tensors."""
    if len(batch) == 0:
        return {}

    result = {}
    keys = batch[0].keys()

    for key in keys:
        values = [item[key] for item in batch if key in item]
        if len(values) == 0:
            continue

        first = values[0]
        if isinstance(first, np.ndarray):
            try:
                result[key] = np.stack(values, axis=0)
            except ValueError:
                result[key] = values  # Keep as list if shapes don't match
        elif isinstance(first, torch.Tensor):
            result[key] = torch.stack(values, dim=0)
        elif isinstance(first, (int, float, bool)):
            result[key] = np.array(values)
        else:
            result[key] = values  # Keep strings, etc. as list

    return result


def create_dataloader(
    root: Union[str, List[str]],
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 4,
    action_horizon: int = 1,
    load_videos: bool = True,
    transform: Optional[Callable] = None,
    pin_memory: bool = True,
    drop_last: bool = True,
    filter_config: Optional[FilterConfig] = None,
    range_nop_path: Optional[str] = None,
    q_filters: Optional[List[QAnnotationFilter]] = None,
    **kwargs,
) -> DataLoader:
    """
    Create a DataLoader for LeRobot format datasets.

    Args:
        root: Path to dataset or list of paths for multiple datasets
        batch_size: Batch size
        shuffle: Whether to shuffle data
        num_workers: Number of data loading workers
        action_horizon: Number of future actions to include (1 = single action)
        load_videos: Whether to load video frames
        transform: Optional transform to apply
        pin_memory: Pin memory for faster GPU transfer
        drop_last: Drop last incomplete batch
        filter_config: FilterConfig for frame/episode filtering
        range_nop_path: Path to range_nop.json for frame range filtering
        q_filters: List of QAnnotationFilter for episode filtering

    Returns:
        PyTorch DataLoader

    Example with filtering:
        # Filter by frame range
        dataloader = create_dataloader(
            "/path/to/dataset",
            range_nop_path="/path/to/range_nop.json"
        )

        # Filter by Q_annotation (Primary quality only)
        dataloader = create_dataloader(
            "/path/to/dataset",
            q_filters=[
                QAnnotationFilter("Q_annotation.instruction_add", ["Primary"]),
            ]
        )

        # Combined filtering
        config = FilterConfig(
            range_nop_path="/path/to/range_nop.json",
            q_filters=[
                QAnnotationFilter("Q_annotation.trace", ["Primary", "Secondary"]),
            ]
        )
        dataloader = create_dataloader("/path/to/dataset", filter_config=config)
    """
    # Handle single or multiple datasets
    if isinstance(root, str):
        roots = [root]
    else:
        roots = root

    # Build filter config from shortcuts if not provided directly
    if filter_config is None and (range_nop_path or q_filters):
        filter_config = FilterConfig(
            range_nop_path=range_nop_path,
            q_filters=q_filters or [],
        )

    # Build delta_timestamps for action horizon
    delta_timestamps = None
    if action_horizon > 1:
        # Will be adjusted per-dataset based on FPS
        delta_timestamps = {"action": list(range(action_horizon))}

    # Create dataset(s)
    if len(roots) == 1:
        # Adjust delta_timestamps based on actual FPS
        temp_ds = LeRobotDataset(roots[0], load_videos=False)
        fps = temp_ds.fps
        if action_horizon > 1:
            delta_timestamps = {"action": [t / fps for t in range(action_horizon)]}

        dataset = LeRobotDataset(
            roots[0],
            delta_timestamps=delta_timestamps,
            load_videos=load_videos,
            transform=transform,
            filter_config=filter_config,
            **kwargs,
        )
    else:
        datasets = []
        for r in roots:
            temp_ds = LeRobotDataset(r, load_videos=False)
            fps = temp_ds.fps
            dt = {"action": [t / fps for t in range(action_horizon)]} if action_horizon > 1 else None

            datasets.append(LeRobotDataset(
                r,
                delta_timestamps=dt,
                load_videos=load_videos,
                filter_config=filter_config,
                **kwargs,
            ))
        dataset = MultiDataset(datasets, transform=transform)

    # Create dataloader
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
        drop_last=drop_last,
        persistent_workers=num_workers > 0,
    )
