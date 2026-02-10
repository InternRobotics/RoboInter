"""
Usage examples for the LeRobot DataLoader.

This dataloader is compatible with LeRobot v2.1 format and can be used
with any dataset in this format, including DROID and RH20T.
"""

from lerobot_dataloader import (
    LeRobotDataset,
    MultiDataset,
    create_dataloader,
    Compose,
    Normalize,
    ResizeImages,
    ToTensorImages,
    ParseAnnotations,
)
from lerobot_dataloader.transforms import compute_stats


# ============================================================
# Example 1: Basic Usage - Single Dataset
# ============================================================
def example_basic():
    """Load a single dataset with default settings."""
    print("\n=== Example 1: Basic Usage ===")

    dataloader = create_dataloader(
        "/path/to/lerobot_dataset",
        batch_size=32,
        shuffle=True,
        num_workers=4,
    )

    for batch in dataloader:
        print(f"Batch keys: {list(batch.keys())}")
        print(f"Action shape: {batch['action'].shape}")
        print(f"State shape: {batch['state'].shape}")
        break


# ============================================================
# Example 2: Action Horizon (Action Chunking)
# ============================================================
def example_action_horizon():
    """Load with action horizon for action prediction."""
    print("\n=== Example 2: Action Horizon ===")

    dataloader = create_dataloader(
        "/path/to/lerobot_dataset",
        batch_size=32,
        action_horizon=16,  # Predict next 16 actions
    )

    for batch in dataloader:
        # action shape: (batch_size, action_horizon, action_dim)
        print(f"Action shape: {batch['action'].shape}")
        break


# ============================================================
# Example 3: Multiple Datasets (e.g., DROID + RH20T)
# ============================================================
def example_multi_dataset():
    """Load multiple datasets together."""
    print("\n=== Example 3: Multiple Datasets ===")

    # Both DROID and RH20T can be loaded together
    # They share the same structure (action, state, images, annotations)
    dataloader = create_dataloader(
        [
            "/path/to/droid_lerobot",
            "/path/to/rh20t_lerobot",
        ],
        batch_size=32,
        action_horizon=16,
    )

    for batch in dataloader:
        print(f"Batch keys: {list(batch.keys())}")
        # dataset_name tells which dataset each sample came from
        print(f"Dataset sources: {set(batch['dataset_name'])}")
        break


# ============================================================
# Example 4: Custom Transforms
# ============================================================
def example_transforms():
    """Use custom transforms for preprocessing."""
    print("\n=== Example 4: Custom Transforms ===")

    # First, create dataset without transform to compute stats
    dataset = LeRobotDataset("/path/to/lerobot_dataset", load_videos=False)
    stats = compute_stats(dataset, keys=["action", "state"], num_samples=5000)

    # Create transform pipeline
    transform = Compose([
        ResizeImages(height=224, width=224),
        ToTensorImages(),  # (H,W,C) uint8 -> (C,H,W) float32
        Normalize(stats),
        ParseAnnotations(),  # Parse JSON annotation fields
    ])

    # Create dataloader with transforms
    dataloader = create_dataloader(
        "/path/to/lerobot_dataset",
        batch_size=32,
        transform=transform,
    )

    for batch in dataloader:
        print(f"Image shape: {batch['observation.images.primary'].shape}")
        print(f"Action (normalized) range: [{batch['action'].min():.2f}, {batch['action'].max():.2f}]")
        break


# ============================================================
# Example 5: Working with Annotations
# ============================================================
def example_annotations():
    """Access annotation fields for training with auxiliary tasks."""
    print("\n=== Example 5: Annotations ===")

    dataset = LeRobotDataset(
        "/path/to/lerobot_dataset",
        transform=ParseAnnotations(),  # Parse JSON fields
    )

    sample = dataset[0]

    # Available annotation fields:
    annotation_keys = [k for k in sample.keys() if k.startswith("annotation.")]
    print(f"Annotation fields: {annotation_keys}")

    # Common annotation fields:
    # - annotation.primitive_skill: skill label
    # - annotation.substask: subtask description
    # - annotation.object_box: object bounding box
    # - annotation.gripper_box: gripper bounding box
    # - annotation.trace: future trajectory points
    # - annotation.contact_frame: contact event frame
    # - annotation.state_affordance: 6D state at contact


# ============================================================
# Example 6: Training Loop
# ============================================================
def example_training_loop():
    """Complete training loop example."""
    print("\n=== Example 6: Training Loop ===")

    import torch

    # Create dataloader
    dataloader = create_dataloader(
        "/path/to/lerobot_dataset",
        batch_size=32,
        action_horizon=16,
        shuffle=True,
        num_workers=4,
    )

    # Training loop
    for epoch in range(3):
        for step, batch in enumerate(dataloader):
            # Get data
            images = batch["observation.images.primary"]  # (B, H, W, C)
            states = batch["state"]                       # (B, state_dim)
            actions = batch["action"]                     # (B, horizon, action_dim)

            # Convert to torch tensors
            images = torch.from_numpy(images).permute(0, 3, 1, 2).float() / 255.0
            states = torch.from_numpy(states).float()
            actions = torch.from_numpy(actions).float()

            # Your model forward pass here
            # loss = model(images, states, actions)
            # loss.backward()
            # optimizer.step()

            if step >= 10:
                break

        print(f"Epoch {epoch + 1} done")


# ============================================================
# Example 7: Direct Dataset Access (without DataLoader)
# ============================================================
def example_direct_access():
    """Access dataset directly for debugging or analysis."""
    print("\n=== Example 7: Direct Dataset Access ===")

    dataset = LeRobotDataset("/path/to/lerobot_dataset")

    print(f"Total frames: {len(dataset)}")
    print(f"Total episodes: {dataset.num_episodes}")
    print(f"FPS: {dataset.fps}")
    print(f"Features: {list(dataset.features.keys())[:5]}...")

    # Access single sample
    sample = dataset[0]
    print(f"\nSample keys: {list(sample.keys())}")


# ============================================================
# Run examples
# ============================================================
if __name__ == "__main__":
    print("LeRobot DataLoader Usage Examples")
    print("=" * 50)
    print("\nNote: Update paths before running.\n")

    # These examples show the API but won't run without real data
    # Uncomment the example you want to run:

    # example_basic()
    # example_action_horizon()
    # example_multi_dataset()
    # example_transforms()
    # example_annotations()
    # example_training_loop()
    # example_direct_access()

    print("\nSee function docstrings for details.")
