"""
LeRobot-style DataLoader for Robot Manipulation Datasets

A lightweight, portable dataloader compatible with LeRobot v2.1 format.
Supports images, actions, states, and custom annotation fields.

Features:
- Frame range filtering via range_nop.json (remove idle frames)
- Episode filtering via Q_annotation fields (quality-based selection)

Usage:
    from lerobot_dataloader import LeRobotDataset, create_dataloader

    # Single dataset
    dataloader = create_dataloader("/path/to/lerobot_dataset", batch_size=32)

    # Multiple datasets
    dataloader = create_dataloader(
        ["/path/to/dataset1", "/path/to/dataset2"],
        batch_size=32,
    )

    # With filtering
    from lerobot_dataloader import FilterConfig, QAnnotationFilter

    dataloader = create_dataloader(
        "/path/to/dataset",
        range_nop_path="/path/to/range_nop.json",  # Frame range filtering
        q_filters=[  # Q_annotation filtering
            QAnnotationFilter("Q_annotation.instruction_add", ["Primary"]),
        ]
    )
"""

from .dataset import (
    LeRobotDataset,
    MultiDataset,
    create_dataloader,
    FilterConfig,
    QAnnotationFilter,
    load_range_nop,
)
from .transforms import (
    Compose,
    Normalize,
    ResizeImages,
    ToTensorImages,
    ParseAnnotations,
)

__all__ = [
    # Dataset classes
    "LeRobotDataset",
    "MultiDataset",
    "create_dataloader",
    # Filtering
    "FilterConfig",
    "QAnnotationFilter",
    "load_range_nop",
    # Transforms
    "Compose",
    "Normalize",
    "ResizeImages",
    "ToTensorImages",
    "ParseAnnotations",
]

__version__ = "0.2.0"
