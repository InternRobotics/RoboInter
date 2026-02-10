"""
DROID LMDB to LeRobot v2.1 Conversion Script (with Annotation Support) - Fast Version

Key optimizations:
1. Use ThreadPoolExecutor for batch parallel processing
2. Generate temporary videos in workers, then copy directly
3. Custom Dataset class that copies videos instead of re-encoding

Usage:
    python convert_droid_to_lerobot_anno_fast.py \
        --input_dir /path/to/episodes \
        --output_dir /path/to/output \
        --annotation_lmdb /path/to/annotation_lmdb \
        --num_threads 64
"""

import os
import pickle
import lmdb
import numpy as np
import cv2
import argparse
import json
import gc
import shutil
import imageio
from pathlib import Path
from tqdm import tqdm
from typing import List, Dict, Optional, Tuple
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.datasets.utils import validate_episode_buffer, get_episode_data_index, check_timestamps_sync
from lerobot.common.datasets.compute_stats import get_feature_stats, sample_indices, auto_downsample_height_width
import torchvision
import torch


# Annotation fields
ANNOTATION_FIELDS = [
    'time_clip', 'instruction_add', 'substask', 'primitive_skill',
    'segmentation', 'object_box', 'placement_proposal', 'trace',
    'gripper_box', 'contact_frame', 'state_affordance',
    'affordance_box', 'contact_points'
]

ANNO_TO_Q_FIELD = {
    'instruction_add': 'Q_instruction_add',
    'substask': 'Q_substask',
    'primitive_skill': 'Q_primitive_skill',
    'segmentation': 'Q_segmentation',
    'object_box': 'Q_object_box',
    'placement_proposal': 'Q_placement_proposal',
    'trace': 'Q_trace',
    'gripper_box': 'Q_gripper_box',
    'contact_frame': 'Q_contact_frame',
    'state_affordance': 'Q_state_affordance',
    'affordance_box': 'Q_affordance_box',
    'contact_points': 'Q_contact_points',
}


def get_droid_features_config_with_anno() -> dict:
    """Get LeRobot v2.1 features configuration for DROID dataset with annotations."""
    return {
        "action": {"dtype": "float64", "shape": (7,), "names": ["delta_x", "delta_y", "delta_z", "delta_rx", "delta_ry", "delta_rz", "gripper_command"]},
        "state": {"dtype": "float64", "shape": (7,), "names": ["x", "y", "z", "rx", "ry", "rz", "gripper_state"]},
        "observation.images.primary": {"dtype": "video", "shape": (320, 180, 3), "names": ["height", "width", "channel"]},
        "observation.images.wrist": {"dtype": "video", "shape": (320, 180, 3), "names": ["height", "width", "channel"]},
        "episode_name": {"dtype": "string", "shape": (1,), "names": ["id"]},
        "camera_view": {"dtype": "string", "shape": (1,), "names": ["view"]},
        "other_information.language_instruction_2": {"dtype": "string", "shape": (1,), "names": ["text"]},
        "other_information.language_instruction_3": {"dtype": "string", "shape": (1,), "names": ["text"]},
        "other_information.action_delta_tcp_pose": {"dtype": "float64", "shape": (7,), "names": ["delta_x", "delta_y", "delta_z", "delta_rx", "delta_ry", "delta_rz", "gripper"]},
        "other_information.action_delta_wrist_pose": {"dtype": "float64", "shape": (7,), "names": ["delta_x", "delta_y", "delta_z", "delta_rx", "delta_ry", "delta_rz", "gripper"]},
        "other_information.action_tcp_pose": {"dtype": "float64", "shape": (7,), "names": ["x", "y", "z", "rx", "ry", "rz", "gripper"]},
        "other_information.action_wrist_pose": {"dtype": "float64", "shape": (7,), "names": ["x", "y", "z", "rx", "ry", "rz", "gripper"]},
        "other_information.action_gripper_velocity": {"dtype": "float64", "shape": (1,), "names": ["velocity"]},
        "other_information.action_joint_position": {"dtype": "float64", "shape": (7,), "names": ["j1", "j2", "j3", "j4", "j5", "j6", "j7"]},
        "other_information.action_joint_velocity": {"dtype": "float64", "shape": (7,), "names": ["j1", "j2", "j3", "j4", "j5", "j6", "j7"]},
        "other_information.action_cartesian_velocity": {"dtype": "float64", "shape": (6,), "names": ["vx", "vy", "vz", "wx", "wy", "wz"]},
        "other_information.observation_joint_position": {"dtype": "float64", "shape": (7,), "names": ["j1", "j2", "j3", "j4", "j5", "j6", "j7"]},
        "other_information.observation_gripper_position": {"dtype": "float64", "shape": (1,), "names": ["position"]},
        "other_information.observation_gripper_open_state": {"dtype": "float64", "shape": (1,), "names": ["open_state"]},
        "other_information.observation_gripper_pose6d": {"dtype": "float64", "shape": (6,), "names": ["x", "y", "z", "rx", "ry", "rz"]},
        "other_information.observation_tcp_pose6d": {"dtype": "float64", "shape": (6,), "names": ["x", "y", "z", "rx", "ry", "rz"]},
        "other_information.is_first": {"dtype": "bool", "shape": (1,), "names": ["flag"]},
        "other_information.is_last": {"dtype": "bool", "shape": (1,), "names": ["flag"]},
        "other_information.is_terminal": {"dtype": "bool", "shape": (1,), "names": ["flag"]},
        "annotation.time_clip": {"dtype": "string", "shape": (1,), "names": ["json"]},
        "annotation.instruction_add": {"dtype": "string", "shape": (1,), "names": ["text"]},
        "annotation.substask": {"dtype": "string", "shape": (1,), "names": ["text"]},
        "annotation.primitive_skill": {"dtype": "string", "shape": (1,), "names": ["text"]},
        "annotation.segmentation": {"dtype": "string", "shape": (1,), "names": ["text"]},
        "annotation.object_box": {"dtype": "string", "shape": (1,), "names": ["json"]},
        "annotation.placement_proposal": {"dtype": "string", "shape": (1,), "names": ["json"]},
        "annotation.trace": {"dtype": "string", "shape": (1,), "names": ["json"]},
        "annotation.gripper_box": {"dtype": "string", "shape": (1,), "names": ["json"]},
        "annotation.contact_frame": {"dtype": "string", "shape": (1,), "names": ["json"]},
        "annotation.state_affordance": {"dtype": "string", "shape": (1,), "names": ["json"]},
        "annotation.affordance_box": {"dtype": "string", "shape": (1,), "names": ["json"]},
        "annotation.contact_points": {"dtype": "string", "shape": (1,), "names": ["json"]},
        "annotation.origin_shape": {"dtype": "string", "shape": (1,), "names": ["json"]},
        "Q_annotation.instruction_add": {"dtype": "string", "shape": (1,), "names": ["quality"]},
        "Q_annotation.substask": {"dtype": "string", "shape": (1,), "names": ["quality"]},
        "Q_annotation.primitive_skill": {"dtype": "string", "shape": (1,), "names": ["quality"]},
        "Q_annotation.segmentation": {"dtype": "string", "shape": (1,), "names": ["quality"]},
        "Q_annotation.object_box": {"dtype": "string", "shape": (1,), "names": ["quality"]},
        "Q_annotation.placement_proposal": {"dtype": "string", "shape": (1,), "names": ["quality"]},
        "Q_annotation.trace": {"dtype": "string", "shape": (1,), "names": ["quality"]},
        "Q_annotation.gripper_box": {"dtype": "string", "shape": (1,), "names": ["quality"]},
        "Q_annotation.contact_frame": {"dtype": "string", "shape": (1,), "names": ["quality"]},
        "Q_annotation.state_affordance": {"dtype": "string", "shape": (1,), "names": ["quality"]},
        "Q_annotation.affordance_box": {"dtype": "string", "shape": (1,), "names": ["quality"]},
        "Q_annotation.contact_points": {"dtype": "string", "shape": (1,), "names": ["quality"]},
    }


def serialize_to_json(obj) -> str:
    if obj is None:
        return ""
    if isinstance(obj, np.ndarray):
        obj = obj.tolist()
    try:
        return json.dumps(obj)
    except:
        return ""


def load_qsheet(qsheet_path: str) -> Dict:
    assert qsheet_path and os.path.exists(qsheet_path)
    with open(qsheet_path, "r") as f:
        return json.load(f)


def load_annotation_for_episode(annotation_lmdb: str, episode_name: str) -> Optional[Dict]:
    if not annotation_lmdb or not os.path.exists(annotation_lmdb):
        return None
    try:
        env = lmdb.open(annotation_lmdb, readonly=True, lock=False, readahead=False, max_readers=256)
        result = None
        with env.begin() as txn:
            value = txn.get(episode_name.encode("utf-8"))
            if value:
                result = pickle.loads(value)
        env.close()
        return result
    except Exception:
        pass
    return None


def get_frame_annotation(anno_data: Optional[Dict], frame_id: int, episode_name: str, qsheet_data: Dict) -> Dict:
    default_anno = {
        "time_clip": "", "instruction_add": "", "substask": "", "primitive_skill": "",
        "segmentation": "", "object_box": "", "placement_proposal": "", "trace": "",
        "gripper_box": "", "contact_frame": "", "state_affordance": "", "affordance_box": "", "contact_points": "",
        "origin_shape": "",
    }
    default_q_anno = {
        "Q_instruction_add": "", "Q_substask": "", "Q_primitive_skill": "", "Q_segmentation": "",
        "Q_object_box": "", "Q_placement_proposal": "", "Q_trace": "", "Q_gripper_box": "",
        "Q_contact_frame": "", "Q_state_affordance": "", "Q_affordance_box": "", "Q_contact_points": "",
    }

    if anno_data is None or frame_id not in anno_data:
        return {**default_anno, **default_q_anno}

    frame_anno = anno_data[frame_id]
    ep_qsheet = qsheet_data.get(episode_name, {})

    result = {
        "time_clip": serialize_to_json(frame_anno.get("time_clip")),
        "instruction_add": frame_anno.get("instruction_add") or "",
        "substask": frame_anno.get("substask") or "",
        "primitive_skill": frame_anno.get("primitive_skill") or "",
        "segmentation": frame_anno.get("segmentation") or "",
        "object_box": serialize_to_json(frame_anno.get("object_box")),
        "placement_proposal": serialize_to_json(frame_anno.get("placement_proposal")),
        "trace": serialize_to_json(frame_anno.get("trace")),
        "gripper_box": serialize_to_json(frame_anno.get("gripper_box")),
        "affordance_box": serialize_to_json(frame_anno.get("affordance_box")),
        "contact_points": serialize_to_json(frame_anno.get("contact_points")),
        "contact_frame": serialize_to_json(frame_anno.get("contact_frame")),
        "state_affordance": serialize_to_json(frame_anno.get("state_affordance")),
        "origin_shape": serialize_to_json(frame_anno.get("origin_shape")),
    }

    for anno_field, q_field in ANNO_TO_Q_FIELD.items():
        anno_value = result.get(anno_field, "")
        if not anno_value:
            result[q_field] = ""
        else:
            q_value = ep_qsheet.get(q_field)
            result[q_field] = q_value if q_value in ("Primary", "Secondary") else ""

    return result


def sample_images_for_stats(video_path: str):
    """Sample images from video for computing stats."""
    reader = torchvision.io.VideoReader(video_path, stream="video")
    frames = [frame["data"] for frame in reader]
    frames_array = torch.stack(frames).numpy()
    sampled_indices = sample_indices(len(frames_array))
    images = None
    for i, idx in enumerate(sampled_indices):
        img = frames_array[idx]
        img = auto_downsample_height_width(img)
        if images is None:
            images = np.empty((len(sampled_indices), *img.shape), dtype=np.uint8)
        images[i] = img
    return images


def compute_episode_stats(episode_buffer: Dict, features: Dict) -> Dict:
    """Compute statistics for an episode."""
    ep_stats = {}
    for key, data in episode_buffer.items():
        if key not in features:
            continue
        if features[key]["dtype"] == "string":
            continue
        elif features[key]["dtype"] in ["image", "video"]:
            if isinstance(data, str):
                ep_ft_array = sample_images_for_stats(data)
            else:
                continue
            axes_to_reduce = (0, 2, 3)
            keepdims = True
        else:
            ep_ft_array = data
            axes_to_reduce = 0
            keepdims = data.ndim == 1

        if ep_ft_array is None:
            continue

        # Convert to numpy array for validation and use
        ep_ft_array = np.asanyarray(ep_ft_array)

        if ep_ft_array.size == 0 or ep_ft_array.ndim == 0:
            print(f"Skipping key '{key}' due to invalid shape: ndim={ep_ft_array.ndim}, size={ep_ft_array.size}")
            continue

        if len(ep_ft_array) == 0:
            continue

        ep_stats[key] = get_feature_stats(ep_ft_array, axis=axes_to_reduce, keepdims=keepdims)
        if features[key]["dtype"] in ["image", "video"]:
            ep_stats[key] = {
                k: v if k == "count" else np.squeeze(v / 255.0, axis=0) for k, v in ep_stats[key].items()
            }
    return ep_stats


class DROIDFastDataset(LeRobotDataset):
    """Custom dataset that copies pre-generated videos instead of re-encoding."""

    def add_frame(self, frame: dict) -> None:
        """Add a frame without requiring video features."""
        if self.episode_buffer is None:
            self.episode_buffer = self.create_episode_buffer()

        # Convert torch tensors to numpy
        for name in frame:
            if isinstance(frame[name], torch.Tensor):
                frame[name] = frame[name].numpy()

        frame_index = self.episode_buffer["size"]
        timestamp = frame.pop("timestamp") if "timestamp" in frame else frame_index / self.fps
        self.episode_buffer["frame_index"].append(frame_index)
        self.episode_buffer["timestamp"].append(timestamp)

        for key in frame:
            if key == "task":
                self.episode_buffer["task"].append(frame["task"])
                continue
            if key not in self.features:
                continue  # Skip unknown keys
            self.episode_buffer[key].append(frame[key])

        self.episode_buffer["size"] += 1

    def save_episode(self, episode_data: dict | None = None, videos: dict | None = None) -> None:
        if not episode_data:
            episode_buffer = self.episode_buffer

        validate_episode_buffer(episode_buffer, self.meta.total_episodes, self.features)
        episode_length = episode_buffer.pop("size")
        tasks = episode_buffer.pop("task")
        episode_tasks = list(set(tasks))
        episode_index = episode_buffer["episode_index"]

        episode_buffer["index"] = np.arange(self.meta.total_frames, self.meta.total_frames + episode_length)
        episode_buffer["episode_index"] = np.full((episode_length,), episode_index)

        for task in episode_tasks:
            task_index = self.meta.get_task_index(task)
            if task_index is None:
                self.meta.add_task(task)

        episode_buffer["task_index"] = np.array([self.meta.get_task_index(task) for task in tasks])

        for key, ft in self.features.items():
            if key in ["index", "episode_index", "task_index"] or ft["dtype"] in ["video"]:
                continue
            stacked = np.stack(episode_buffer[key])
            # Only squeeze if result won't be 0-dimensional
            # Also ensure at least 1D for HuggingFace datasets compatibility
            if stacked.ndim > 1:
                stacked = stacked.squeeze()
                # Ensure at least 1D after squeeze
                if stacked.ndim == 0:
                    stacked = stacked.reshape(1)
            episode_buffer[key] = stacked

        # Copy pre-generated videos instead of re-encoding
        for key in self.meta.video_keys:
            video_path = self.root / self.meta.get_video_file_path(episode_index, key)
            episode_buffer[key] = str(video_path)
            video_path.parent.mkdir(parents=True, exist_ok=True)
            if videos and key in videos:
                shutil.copyfile(videos[key], video_path)

        ep_stats = compute_episode_stats(episode_buffer, self.features)
        self._save_episode_table(episode_buffer, episode_index)
        self.meta.save_episode(episode_index, episode_length, episode_tasks, ep_stats)

        ep_data_index = get_episode_data_index(self.meta.episodes, [episode_index])
        ep_data_index_np = {k: t.numpy() for k, t in ep_data_index.items()}
        check_timestamps_sync(
            episode_buffer["timestamp"],
            episode_buffer["episode_index"],
            ep_data_index_np,
            self.fps,
            self.tolerance_s,
        )

        if not episode_data:
            self.episode_buffer = self.create_episode_buffer()


def load_episode_and_generate_videos(
    episode_dir: str,
    annotation_lmdb: str,
    qsheet_data: Dict,
    temp_video_dir: str,
    fps: int = 10,
) -> Optional[Dict]:
    """
    Load episode data and generate temporary video files.
    This function runs in a thread worker.
    """
    episode_path = Path(episode_dir)
    lmdb_path = episode_path / "lmdb"
    meta_path = episode_path / "meta_info.pkl"

    if not lmdb_path.exists() or not meta_path.exists():
        return None

    try:
        with open(meta_path, "rb") as f:
            meta_info = pickle.load(f)

        env = lmdb.open(str(lmdb_path), readonly=True, lock=False)
        txn = env.begin()

        num_steps = meta_info["num_steps"]
        episode_name = meta_info["episode_name"]
        camera_view = meta_info["camera_view"]
        language_instruction = meta_info["language_instruction"]

        # Load annotation
        anno_data = load_annotation_for_episode(annotation_lmdb, episode_name)

        # Collect images for video generation
        primary_images = []
        wrist_images = []
        frames = []

        for step_idx in range(num_steps):
            step_id = str(step_idx).zfill(4)

            # Load action & state
            action_data = pickle.loads(txn.get(f"action/{step_id}".encode("utf-8")))
            state_data = pickle.loads(txn.get(f"state/{step_id}".encode("utf-8")))

            # Load images
            primary_data = pickle.loads(txn.get(f"primary_image/{step_id}".encode("utf-8")))
            primary_img = cv2.imdecode(primary_data, cv2.IMREAD_COLOR)
            primary_images.append(primary_img)

            wrist_raw = txn.get(f"wrist_image/{step_id}".encode("utf-8"))
            if wrist_raw:
                wrist_data = pickle.loads(wrist_raw)
                wrist_img = cv2.imdecode(wrist_data, cv2.IMREAD_COLOR)
            else:
                h, w = primary_img.shape[:2]
                wrist_img = np.zeros((h, w, 3), dtype=np.uint8)
            wrist_images.append(wrist_img)

            # Load other_info
            other_data = pickle.loads(txn.get(f"other_info/{step_id}".encode("utf-8")))

            # Get annotation
            frame_anno = get_frame_annotation(anno_data, step_idx, episode_name, qsheet_data)

            frame_data = {
                "action": np.array(action_data, dtype=np.float64),
                "state": np.array(state_data, dtype=np.float64),
                "episode_name": episode_name,
                "camera_view": camera_view,
                "task": language_instruction,
                "other_information.language_instruction_2": other_data.get("language_instruction_2", ""),
                "other_information.language_instruction_3": other_data.get("language_instruction_3", ""),
                "other_information.action_delta_tcp_pose": np.array(other_data.get("action_delta_tcp_pose", np.zeros(7)), dtype=np.float64),
                "other_information.action_delta_wrist_pose": np.array(other_data.get("action_delta_wrist_pose", np.zeros(7)), dtype=np.float64),
                "other_information.action_tcp_pose": np.array(other_data.get("action_tcp_pose", np.zeros(7)), dtype=np.float64),
                "other_information.action_wrist_pose": np.array(other_data.get("action_wrist_pose", np.zeros(7)), dtype=np.float64),
                "other_information.action_gripper_velocity": np.atleast_1d(other_data.get("action_gripper_velocity", 0.0)).astype(np.float64),
                "other_information.action_joint_position": np.array(other_data.get("action_joint_position", np.zeros(7)), dtype=np.float64),
                "other_information.action_joint_velocity": np.array(other_data.get("action_joint_velocity", np.zeros(7)), dtype=np.float64),
                "other_information.action_cartesian_velocity": np.array(other_data.get("action_cartesian_velocity", np.zeros(6)), dtype=np.float64),
                "other_information.observation_joint_position": np.array(other_data.get("observation_joint_position", np.zeros(7)), dtype=np.float64),
                "other_information.observation_gripper_position": np.atleast_1d(other_data.get("observation_gripper_position", 0.0)).astype(np.float64),
                "other_information.observation_gripper_open_state": np.atleast_1d(other_data.get("observation_gripper_open_state", 0.0)).astype(np.float64),
                "other_information.observation_gripper_pose6d": np.array(other_data.get("observation_gripper_pose6d", np.zeros(6)), dtype=np.float64),
                "other_information.observation_tcp_pose6d": np.array(other_data.get("observation_tcp_pose6d", np.zeros(6)), dtype=np.float64),
                "other_information.is_first": np.atleast_1d(bool(other_data.get("is_first", False))),
                "other_information.is_last": np.atleast_1d(bool(other_data.get("is_last", False))),
                "other_information.is_terminal": np.atleast_1d(bool(other_data.get("is_terminal", False))),
                "annotation.time_clip": frame_anno["time_clip"],
                "annotation.instruction_add": frame_anno["instruction_add"],
                "annotation.substask": frame_anno["substask"],
                "annotation.primitive_skill": frame_anno["primitive_skill"],
                "annotation.segmentation": frame_anno["segmentation"],
                "annotation.object_box": frame_anno["object_box"],
                "annotation.placement_proposal": frame_anno["placement_proposal"],
                "annotation.trace": frame_anno["trace"],
                "annotation.gripper_box": frame_anno["gripper_box"],
                "annotation.contact_frame": frame_anno["contact_frame"],
                "annotation.state_affordance": frame_anno["state_affordance"],
                "annotation.affordance_box": frame_anno["affordance_box"],
                "annotation.contact_points": frame_anno["contact_points"],
                "annotation.origin_shape": frame_anno["origin_shape"],
                "Q_annotation.instruction_add": frame_anno["Q_instruction_add"],
                "Q_annotation.substask": frame_anno["Q_substask"],
                "Q_annotation.primitive_skill": frame_anno["Q_primitive_skill"],
                "Q_annotation.segmentation": frame_anno["Q_segmentation"],
                "Q_annotation.object_box": frame_anno["Q_object_box"],
                "Q_annotation.placement_proposal": frame_anno["Q_placement_proposal"],
                "Q_annotation.trace": frame_anno["Q_trace"],
                "Q_annotation.gripper_box": frame_anno["Q_gripper_box"],
                "Q_annotation.contact_frame": frame_anno["Q_contact_frame"],
                "Q_annotation.state_affordance": frame_anno["Q_state_affordance"],
                "Q_annotation.affordance_box": frame_anno["Q_affordance_box"],
                "Q_annotation.contact_points": frame_anno["Q_contact_points"],
            }
            frames.append(frame_data)

        env.close()

        if not frames:
            return None

        # Generate temporary videos
        temp_ep_dir = Path(temp_video_dir) / episode_name
        temp_ep_dir.mkdir(parents=True, exist_ok=True)

        h, w = primary_images[0].shape[:2]
        video_paths = {}

        # Primary video
        primary_video_path = temp_ep_dir / "primary.mp4"
        imageio.mimsave(
            str(primary_video_path),
            primary_images,
            fps=fps,
            codec="libx264",
            ffmpeg_params=["-crf", "23", "-preset", "fast", "-pix_fmt", "yuv420p", "-y"],
            macro_block_size=1,  # Avoid resizing
        )
        video_paths["observation.images.primary"] = str(primary_video_path)

        # Wrist video
        wrist_video_path = temp_ep_dir / "wrist.mp4"
        imageio.mimsave(
            str(wrist_video_path),
            wrist_images,
            fps=fps,
            codec="libx264",
            ffmpeg_params=["-crf", "23", "-preset", "fast", "-pix_fmt", "yuv420p", "-y"],
            macro_block_size=1,  # Avoid resizing
        )
        video_paths["observation.images.wrist"] = str(wrist_video_path)

        return {
            "episode_name": episode_name,
            "frames": frames,
            "videos": video_paths,
            "has_annotation": anno_data is not None,
            "img_shape": (h, w, 3),
        }

    except Exception as e:
        import traceback
        print(f"Error processing {episode_dir}: {e}")
        traceback.print_exc()
        return None


def get_droid_episode_list(input_dir: Path) -> List[Path]:
    """Get list of DROID episode directories."""
    episodes_dir = input_dir / "episodes" if (input_dir / "episodes").exists() else input_dir
    episode_dirs = []
    for ep_dir in episodes_dir.iterdir():
        if ep_dir.is_dir() and not ep_dir.name.startswith("RH20T"):
            if (ep_dir / "lmdb").exists() and (ep_dir / "meta_info.pkl").exists():
                episode_dirs.append(ep_dir)
    return sorted(episode_dirs)


def convert_droid_to_lerobot_fast(
    input_dir: str,
    output_dir: str,
    annotation_lmdb: str,
    qsheet_path: str,
    num_threads: int = 64,
    fps: int = 10,
    max_episodes: Optional[int] = None,
    delete_temp_videos: bool = True,
    shard_id: Optional[int] = None,
    num_shards: Optional[int] = None,
):
    """Convert DROID dataset using fast parallel processing.

    Args:
        shard_id: If set, only process episodes for this shard (0-indexed)
        num_shards: Total number of shards to split the data into
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    # Append shard suffix if sharding is enabled
    if shard_id is not None and num_shards is not None:
        output_path = Path(str(output_path) + f"_shard{shard_id:02d}of{num_shards:02d}")

    # Setup error log directory and file
    error_log_dir = Path("./convert_to_lerobot/error")
    error_log_dir.mkdir(parents=True, exist_ok=True)
    shard_suffix = f"_shard{shard_id:02d}of{num_shards:02d}" if shard_id is not None else ""
    error_log_file = error_log_dir / f"droid_skipped_episodes{shard_suffix}.txt"
    skipped_episodes = []

    # Use /tmp for temp videos to avoid conflicts with output directory
    temp_video_dir = Path("/tmp") / f"lerobot_temp_{output_path.name}"

    print("Scanning for DROID episodes...")
    episode_dirs = get_droid_episode_list(input_path)
    total_episodes = len(episode_dirs)
    print(f"Found {total_episodes} DROID episodes")

    if max_episodes:
        episode_dirs = episode_dirs[:max_episodes]
        print(f"Limiting to {max_episodes} episodes")

    # Apply sharding if enabled
    if shard_id is not None and num_shards is not None:
        total_to_process = len(episode_dirs)
        shard_size = (total_to_process + num_shards - 1) // num_shards  # Ceiling division
        start_idx = shard_id * shard_size
        end_idx = min(start_idx + shard_size, total_to_process)
        episode_dirs = episode_dirs[start_idx:end_idx]
        print(f"Shard {shard_id}/{num_shards}: Processing episodes {start_idx} to {end_idx-1} ({len(episode_dirs)} episodes)")

    # Load Qsheet
    print(f"Loading Qsheet from: {qsheet_path}")
    qsheet_data = load_qsheet(qsheet_path)
    print(f"  Loaded {len(qsheet_data)} episodes from Qsheet")

    if len(episode_dirs) == 0:
        print("No episodes to process")
        return

    # Clean output directory
    if output_path.exists():
        print(f"Removing existing output directory: {output_path}")
        shutil.rmtree(output_path)

    # Create temp video directory
    temp_video_dir.mkdir(parents=True, exist_ok=True)

    # Get image shape from first episode
    print("Processing first episode to get image shape...")
    first_result = load_episode_and_generate_videos(
        str(episode_dirs[0]), annotation_lmdb, qsheet_data, str(temp_video_dir), fps
    )
    if first_result is None:
        raise ValueError("Failed to load first episode")

    img_shape = first_result["img_shape"]
    print(f"Image shape: {img_shape}")

    # Update features config
    features = get_droid_features_config_with_anno()
    features["observation.images.primary"]["shape"] = list(img_shape)
    features["observation.images.wrist"]["shape"] = list(img_shape)

    # Create dataset
    print("Creating LeRobot dataset...")
    dataset = DROIDFastDataset.create(
        repo_id=output_path.name,
        fps=fps,
        robot_type="franka_robotiq",
        features=features,
        root=output_path,
        use_videos=True,
    )

    # Add first episode
    try:
        for frame_data in first_result["frames"]:
            dataset.add_frame(frame_data)
        dataset.save_episode(videos=first_result["videos"])
    except Exception as e:
        print(f"Error saving first episode {first_result['episode_name']}: {e}")
        skipped_episodes.append(f"{first_result['episode_name']}: {str(e)}")
        dataset.episode_buffer = dataset.create_episode_buffer()  # Reset buffer

    # Clean temp video for first episode
    if delete_temp_videos:
        temp_ep_dir = temp_video_dir / first_result["episode_name"]
        if temp_ep_dir.exists():
            shutil.rmtree(temp_ep_dir)

    # Process remaining episodes in batches
    remaining_dirs = episode_dirs[1:]
    print(f"\nProcessing {len(remaining_dirs)} remaining episodes with {num_threads} threads...")
    print(f"Annotation LMDB: {annotation_lmdb}")

    start_time = time.time()
    processed_count = 1
    failed_count = 0
    anno_found_count = 1 if first_result["has_annotation"] else 0

    # Batch processing
    for batch_start in tqdm(range(0, len(remaining_dirs), num_threads), desc="Batches"):
        batch_dirs = remaining_dirs[batch_start:batch_start + num_threads]

        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = []
            for ep_dir in batch_dirs:
                futures.append(
                    executor.submit(
                        load_episode_and_generate_videos,
                        str(ep_dir), annotation_lmdb, qsheet_data, str(temp_video_dir), fps
                    )
                )

            for future in as_completed(futures):
                result = future.result()
                if result is None:
                    failed_count += 1
                    continue

                # Add frames to dataset
                try:
                    for frame_data in result["frames"]:
                        dataset.add_frame(frame_data)
                    dataset.save_episode(videos=result["videos"])

                    processed_count += 1
                    if result["has_annotation"]:
                        anno_found_count += 1
                except Exception as e:
                    print(f"Error saving episode {result['episode_name']}: {e}")
                    skipped_episodes.append(f"{result['episode_name']}: {str(e)}")
                    dataset.episode_buffer = dataset.create_episode_buffer()  # Reset buffer
                    failed_count += 1

                # Clean temp video
                if delete_temp_videos:
                    temp_ep_dir = temp_video_dir / result["episode_name"]
                    if temp_ep_dir.exists():
                        shutil.rmtree(temp_ep_dir)

        gc.collect()

        if processed_count % 500 == 0:
            elapsed = time.time() - start_time
            eps_per_sec = processed_count / elapsed
            eta = (len(episode_dirs) - processed_count) / eps_per_sec if eps_per_sec > 0 else 0
            print(f"\nProcessed {processed_count}/{len(episode_dirs)} episodes "
                  f"({eps_per_sec:.2f} ep/s, ETA: {eta/60:.1f} min)")
            print(f"  Episodes with annotations: {anno_found_count}/{processed_count}")

    # Clean up temp directory
    if delete_temp_videos and temp_video_dir.exists():
        shutil.rmtree(temp_video_dir)

    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"Conversion completed!")
    print(f"  Total episodes: {len(episode_dirs)}")
    print(f"  Processed: {processed_count}")
    print(f"  Failed: {failed_count}")
    print(f"  Skipped (errors): {len(skipped_episodes)}")
    print(f"  Episodes with annotations: {anno_found_count}")
    print(f"  Time elapsed: {elapsed/60:.1f} minutes")
    print(f"  Speed: {processed_count/elapsed:.2f} episodes/sec")
    print(f"  Output directory: {output_path}")
    print(f"{'='*60}")

    # Save skipped episodes to error log
    if skipped_episodes:
        with open(error_log_file, "w") as f:
            f.write(f"# DROID Skipped Episodes{shard_suffix}\n")
            f.write(f"# Total skipped: {len(skipped_episodes)}\n\n")
            for ep_info in skipped_episodes:
                f.write(f"{ep_info}\n")
        print(f"  Skipped episodes logged to: {error_log_file}")

    # Save shard metadata for later merging
    if shard_id is not None and num_shards is not None:
        shard_meta = {
            "shard_id": shard_id,
            "num_shards": num_shards,
            "total_episodes": processed_count,
            "total_frames": dataset.meta.total_frames,
            "total_tasks": dataset.meta.total_tasks,
            "fps": fps,
            "robot_type": "franka_robotiq",
        }
        shard_meta_path = output_path / "meta" / "shard_info.json"
        with open(shard_meta_path, "w") as f:
            json.dump(shard_meta, f, indent=2)
        print(f"  Shard metadata saved to: {shard_meta_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert DROID dataset to LeRobot v2.1 format (fast version)"
    )
    parser.add_argument("--input_dir", type=str,
                        default="")
    parser.add_argument("--output_dir", type=str,
                        default="")
    parser.add_argument("--annotation_lmdb", type=str,
                        default="")
    parser.add_argument("--qsheet_path", type=str,
                        default="")
    parser.add_argument("--num_threads", type=int, default=64, help="Number of threads for parallel processing")
    parser.add_argument("--fps", type=int, default=10, help="Frames per second")
    parser.add_argument("--max_episodes", type=int, default=None, help="Maximum episodes to convert")
    parser.add_argument("--keep_temp_videos", action="store_true", help="Keep temporary video files")
    parser.add_argument("--shard_id", type=int, default=None, help="Shard ID (0-indexed) for distributed processing")
    parser.add_argument("--num_shards", type=int, default=None, help="Total number of shards for distributed processing")

    args = parser.parse_args()

    # Validate shard arguments
    if (args.shard_id is None) != (args.num_shards is None):
        parser.error("--shard_id and --num_shards must be used together")
    if args.shard_id is not None and (args.shard_id < 0 or args.shard_id >= args.num_shards):
        parser.error(f"--shard_id must be in range [0, {args.num_shards - 1}]")

    convert_droid_to_lerobot_fast(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        annotation_lmdb=args.annotation_lmdb,
        qsheet_path=args.qsheet_path,
        num_threads=args.num_threads,
        fps=args.fps,
        max_episodes=args.max_episodes,
        delete_temp_videos=not args.keep_temp_videos,
        shard_id=args.shard_id,
        num_shards=args.num_shards,
    )


if __name__ == "__main__":
    main()
