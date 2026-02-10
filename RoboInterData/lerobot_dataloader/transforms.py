"""
Data transforms for LeRobot datasets.

Compatible with the transform interface used in LeRobot/OpenPI.
"""

import json
from typing import Dict, Any, List, Optional, Sequence, Callable
from dataclasses import dataclass
import numpy as np
import cv2


class Compose:
    """Compose multiple transforms."""

    def __init__(self, transforms: List[Callable]):
        self.transforms = transforms

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        for t in self.transforms:
            data = t(data)
        return data


@dataclass
class Normalize:
    """
    Normalize actions and/or states.

    Args:
        stats: Dict with 'action' and/or 'state' keys, each containing 'mean' and 'std'
        keys: List of keys to normalize (default: ['action', 'state'])
    """
    stats: Dict[str, Dict[str, np.ndarray]]
    keys: List[str] = None
    eps: float = 1e-6

    def __post_init__(self):
        if self.keys is None:
            self.keys = list(self.stats.keys())

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        for key in self.keys:
            if key in data and key in self.stats:
                x = np.asarray(data[key], dtype=np.float32)
                mean = self.stats[key]["mean"]
                std = self.stats[key]["std"]
                data[key] = (x - mean) / (std + self.eps)
        return data


@dataclass
class Denormalize:
    """Inverse of Normalize."""
    stats: Dict[str, Dict[str, np.ndarray]]
    keys: List[str] = None
    eps: float = 1e-6

    def __post_init__(self):
        if self.keys is None:
            self.keys = list(self.stats.keys())

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        for key in self.keys:
            if key in data and key in self.stats:
                x = np.asarray(data[key], dtype=np.float32)
                mean = self.stats[key]["mean"]
                std = self.stats[key]["std"]
                data[key] = x * (std + self.eps) + mean
        return data


@dataclass
class ResizeImages:
    """
    Resize images to target size.

    Args:
        height: Target height
        width: Target width
        keys: Image keys to resize (default: all 'observation.images.*' keys)
    """
    height: int = 224
    width: int = 224
    keys: Optional[List[str]] = None

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        keys = self.keys or [k for k in data if k.startswith("observation.images.")]

        for key in keys:
            if key in data and data[key] is not None:
                img = data[key]
                if isinstance(img, np.ndarray) and img.ndim == 3:
                    # Handle (C, H, W) format
                    if img.shape[0] == 3:
                        img = np.transpose(img, (1, 2, 0))
                        transposed = True
                    else:
                        transposed = False

                    img = cv2.resize(img, (self.width, self.height), interpolation=cv2.INTER_AREA)

                    if transposed:
                        img = np.transpose(img, (2, 0, 1))

                    data[key] = img
        return data


@dataclass
class ToTensorImages:
    """
    Convert images from (H, W, C) uint8 to (C, H, W) float32 [0, 1].

    Args:
        keys: Image keys to convert (default: all 'observation.images.*' keys)
    """
    keys: Optional[List[str]] = None

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        keys = self.keys or [k for k in data if k.startswith("observation.images.")]

        for key in keys:
            if key in data and data[key] is not None:
                img = data[key]
                if isinstance(img, np.ndarray):
                    # Convert to float [0, 1]
                    img = img.astype(np.float32) / 255.0
                    # (H, W, C) -> (C, H, W)
                    if img.ndim == 3 and img.shape[-1] in [1, 3, 4]:
                        img = np.transpose(img, (2, 0, 1))
                    data[key] = img
        return data


@dataclass
class ParseAnnotations:
    """
    Parse JSON string annotation fields into Python objects.

    Args:
        keys: Annotation keys to parse (default: common JSON annotation fields)
    """
    keys: Optional[List[str]] = None

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        default_keys = [
            "annotation.time_clip",
            "annotation.object_box",
            "annotation.placement_proposal",
            "annotation.trace",
            "annotation.gripper_box",
            "annotation.affordance_box",
            "annotation.contact_points",
        ]
        keys = self.keys or default_keys

        for key in keys:
            if key in data:
                value = data[key]
                if isinstance(value, str) and value:
                    try:
                        data[key] = json.loads(value)
                    except json.JSONDecodeError:
                        pass
        return data


@dataclass
class DeltaActions:
    """
    Convert absolute actions to delta (relative to current state).

    Args:
        mask: Boolean mask indicating which dims to convert (True = convert to delta)
    """
    mask: Sequence[bool]

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        if "action" not in data or "state" not in data:
            return data

        action = np.asarray(data["action"], dtype=np.float32)
        state = np.asarray(data["state"], dtype=np.float32)
        mask = np.asarray(self.mask)

        dims = min(len(mask), action.shape[-1], state.shape[-1])

        if action.ndim == 1:
            action[:dims] -= np.where(mask[:dims], state[:dims], 0)
        else:
            # (T, action_dim)
            state_broadcast = np.where(mask[:dims], state[:dims], 0)
            action[..., :dims] -= state_broadcast

        data["action"] = action
        return data


@dataclass
class AbsoluteActions:
    """
    Convert delta actions back to absolute.

    Args:
        mask: Boolean mask indicating which dims to convert
    """
    mask: Sequence[bool]

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        if "action" not in data or "state" not in data:
            return data

        action = np.asarray(data["action"], dtype=np.float32)
        state = np.asarray(data["state"], dtype=np.float32)
        mask = np.asarray(self.mask)

        dims = min(len(mask), action.shape[-1], state.shape[-1])

        if action.ndim == 1:
            action[:dims] += np.where(mask[:dims], state[:dims], 0)
        else:
            state_broadcast = np.where(mask[:dims], state[:dims], 0)
            action[..., :dims] += state_broadcast

        data["action"] = action
        return data


def compute_stats(
    dataset,
    keys: List[str] = ["action", "state"],
    num_samples: int = 10000,
    seed: int = 42,
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Compute normalization statistics from dataset.

    Args:
        dataset: Dataset to compute stats from
        keys: Keys to compute stats for
        num_samples: Number of samples to use
        seed: Random seed

    Returns:
        Dict with mean and std for each key
    """
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(dataset), size=min(num_samples, len(dataset)), replace=False)

    data_lists = {key: [] for key in keys}

    for idx in indices:
        sample = dataset[int(idx)]
        for key in keys:
            if key in sample:
                val = sample[key]
                if isinstance(val, np.ndarray):
                    if val.ndim > 1:
                        val = val[0]  # Take first if action horizon
                    data_lists[key].append(val)

    stats = {}
    for key in keys:
        if data_lists[key]:
            arr = np.array(data_lists[key])
            stats[key] = {
                "mean": np.mean(arr, axis=0).astype(np.float32),
                "std": np.std(arr, axis=0).astype(np.float32),
            }

    return stats
