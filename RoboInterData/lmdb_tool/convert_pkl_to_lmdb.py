"""
RoboInter Data Conversion Script - Convert merged pkl files to frame-level LMDB

Input:
- droid_annotation.pkl: DROID pkl containing intermediate representations and language data
- rh20t_annotation.pkl: RH20T pkl containing intermediate representations and language data

Output LMDB format (per-frame storage following formulation.py):
{
    "video_id": {
        0: {
            "time_clip": [[0,132],[132,197],[198,224]],
            "instruction_add": "make a burger",
            "substask": "pick up the burger",
            "primitive_skill": "pick",
            "segmentation": None,
            "object_box": [[x1,y1],[x2,y2]],
            "placement_proposal": [[x1,y1],[x2,y2]],
            "trace": [[x,y],...],  # next 10 steps
            "gripper_box": [[x1,y1],[x2,y2]],
            "contact_frame": 101,
            "state_affordance": [...],
            "affordance_box": [[x1,y1],[x2,y2]],
            "contact_points": [101, 102],
        },
        1: {...},
        ...
    }
}
"""

import os
import pickle
import lmdb
import numpy as np
from tqdm import tqdm
import argparse


def load_pkl(pkl_path):
    """Load a pkl file."""
    print(f"Loading pkl: {pkl_path}")
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    print(f"  Loaded {len(data)} items")
    return data


def check_posi_available(trajs, min_value, max_value):
    """Check whether a trajectory is valid."""
    if trajs is None or len(trajs) == 0:
        return False
    for traj in trajs:
        if traj[0] <= min_value or traj[1] <= min_value:
            return False
        if traj[0] > max_value[0] or traj[1] > max_value[1]:
            return False
    return True

def get_total_length(episode_dir):
    meta_path = os.path.join(episode_dir, "meta_info.pkl")

    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"{meta_path} not found")

    # Load meta info
    with open(meta_path, "rb") as f:
        meta_info = pickle.load(f)

    if "state_all" not in meta_info:
        raise KeyError("state_all not found in meta_info.pkl")

    ee_pose_state = meta_info["state_all"]  # shape: (T, D)

    return len(ee_pose_state)

def get_frame_state(episode_dir, frame_id):
    """
    Read the ee pose state (first 6 dimensions) for a specific frame
    from episode_dir/meta_info.pkl.

    Args:
        episode_dir (str): Path to the episode directory.
        frame_id (int): Frame index to read.

    Returns:
        np.ndarray: shape (6,)
    """

    meta_path = os.path.join(episode_dir, "meta_info.pkl")

    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"{meta_path} not found")

    # Load meta info
    with open(meta_path, "rb") as f:
        meta_info = pickle.load(f)

    if "state_all" not in meta_info:
        raise KeyError("state_all not found in meta_info.pkl")

    ee_pose_state = meta_info["state_all"]  # shape: (T, D)

    # Keep only the first 6 dimensions (x, y, z, rx, ry, rz)
    ee_pose_state = ee_pose_state[:, :6]

    if frame_id >= len(ee_pose_state):
        raise IndexError(f"frame_id {frame_id} out of range, max {len(ee_pose_state)-1}")

    cot_content = ee_pose_state[frame_id].astype(np.float64)
    cot_content = np.round(cot_content, 5)

    return cot_content


def clip_traj(trajs, min_value, max_value):
    """Clip trajectory to valid range, pad to 10 steps, and ensure int output."""
    if trajs is None or len(trajs) == 0:
        return None
    clip_res = []
    for traj in trajs:
        new_traj = [max(min_value, int(round(traj[0]))), max(min_value, int(round(traj[1])))]
        new_traj = [min(max_value[0], new_traj[0]), min(max_value[1], new_traj[1])]
        clip_res.append(new_traj)
    if len(clip_res) < 10:
        clip_res = clip_res + [list(clip_res[-1]) for _ in range(10 - len(clip_res))]
    return clip_res


def get_contact_info_for_frame(frame_id, contact_point):
    """Get contact information for the current frame."""
    if contact_point is None:
        return None, None, None, None

    for (s, e) in contact_point:
        contact_frame = contact_point[(s, e)]['contact_frame']
        contact_pt = contact_point[(s, e)]['contact_pt']
        if contact_pt is not None:
            assert len(contact_pt) == 2, f"contact_point={contact_point}, len={len(contact_point)}"
        contact_box = contact_point[(s, e)]['contact_box']
        if frame_id >= s and frame_id <= e:
            if frame_id <= contact_frame:
                return contact_frame, (s, e), contact_pt, contact_box
            else:
                return -1, (s, e), [], []
    return None, None, None, None


def numpy_to_list(obj):
    """Recursively convert numpy arrays to Python lists."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: numpy_to_list(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [numpy_to_list(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(numpy_to_list(item) for item in obj)
    else:
        return obj


def to_int_list(arr):
    """Convert an array to a list of int values."""
    if arr is None:
        return None
    if hasattr(arr, 'tolist'):
        arr = arr.tolist()
    # Recursively ensure all values are int
    if isinstance(arr, list):
        return [[int(round(x)) for x in row] if isinstance(row, list) else int(round(row)) for row in arr]
    return int(round(arr))


def convert_single_item(key, inter_data, data_lmdb_path):
    """
    Convert a single item to per-frame storage format.

    Args:
        key: Data ID (episode name).
        inter_data: Raw intermediate representation data.

    Returns:
        dict: Per-frame data dictionary.
    """
    result = {}

    # Get language data (may be None)
    all_language = inter_data.get('all_language')

    # Get intermediate representation data
    all_gripper_box = inter_data.get('all_gripper_box')
    all_grounding_box = inter_data.get('all_grounding_box')
    all_contact_point = inter_data.get('all_contact_point')
    all_traj = inter_data.get('all_traj')

    # Get total frame count: prefer from language data, otherwise infer from other sources
    total_frames = None
    if all_language is not None:
        total_frames = all_language.get('frames')

    if total_frames is None:
        total_frames = get_total_length(data_lmdb_path+key)

    if total_frames is None:
        return None

    # Language-related fields (may be None)
    raw_time_clip = all_language.get('time_clip', []) if all_language else None
    instruction = all_language.get('instruction', '') if all_language else None
    task_steps = all_language.get('task_steps', []) if all_language else None
    action_steps = all_language.get('action_steps', []) if all_language else None

    # Get origin_shape
    origin_shape = inter_data.get('origin_shape')
    if origin_shape is None:
        origin_shape = (224, 224)

    # Build time_clip to step mapping
    frame_to_step = {}
    if raw_time_clip:
        time_clip = [list(tc) for tc in raw_time_clip]
        time_clip[0][0] = 0
        time_clip[-1][-1] = total_frames - 1

        for step_i, (start, end) in enumerate(time_clip):
            for frame_id in range(start, end + 1):
                frame_to_step[frame_id] = step_i
    else:
        time_clip = raw_time_clip


    # Iterate over each frame
    for frame_id in range(total_frames):
        frame_data = {
            'origin_shape': origin_shape if origin_shape else None,
            'time_clip': time_clip if time_clip else None,
            'instruction_add': instruction if instruction else None,
            'substask': None,
            'primitive_skill': None,
            'segmentation': None,  # not stored for now
            'object_box': None,
            'placement_proposal': None,
            'trace': None,
            'gripper_box': None,
            'contact_frame': None,
            'state_affordance': None,
            'affordance_box': None,
            'contact_points': None,
        }

        # Get the step index corresponding to the current frame
        step_i = frame_to_step.get(frame_id, 0)
        if task_steps and step_i < len(task_steps):
            frame_data['substask'] = task_steps[step_i]
        if action_steps and step_i < len(action_steps):
            frame_data['primitive_skill'] = action_steps[step_i]

        # Get gripper_box
        if all_gripper_box is not None and frame_id in all_gripper_box:
            gripper_box = all_gripper_box[frame_id].get('gripper_box')
            if gripper_box is not None:
                gripper_box = np.clip(gripper_box, 0, None)
                max_value = (origin_shape[0]-1, origin_shape[1]-1)
                if check_posi_available(gripper_box, 0, max_value):
                    frame_data['gripper_box'] = to_int_list(gripper_box)

        # Get grounding_box (object_box and placement_proposal)
        if all_grounding_box is not None and frame_id in all_grounding_box:
            obj_box = all_grounding_box[frame_id].get('obj_box')
            if obj_box is not None:
                frame_data['object_box'] = to_int_list(np.clip(obj_box, 0, None))
            # placement_proposal: requires time_clip to be available
            if time_clip and step_i < len(time_clip):
                placement_proposal_id = time_clip[step_i][-1]
                if placement_proposal_id in all_grounding_box:
                    obj_box_placement_proposal = all_grounding_box[placement_proposal_id].get('obj_box')
                    if obj_box_placement_proposal is not None:
                        frame_data['placement_proposal'] = to_int_list(np.clip(obj_box_placement_proposal, 0, None))

        # Get contact information
        grasp_pose_frame, subtask_range, contact_pt, contact_box = get_contact_info_for_frame(frame_id, all_contact_point)

        if grasp_pose_frame is not None and  grasp_pose_frame != -1:
            frame_data['contact_frame'] = grasp_pose_frame

            state_aff = get_frame_state(data_lmdb_path+key, grasp_pose_frame)
            frame_data['state_affordance'] = state_aff.tolist() if hasattr(state_aff, 'tolist') else list(state_aff)

            if contact_pt is not None and len(contact_pt) > 0:
                contact_pt = np.clip(contact_pt, 0, None)
                frame_data['contact_points'] = to_int_list(contact_pt)

            # Get affordance_box (gripper box at the contact frame)
            if all_gripper_box is not None and grasp_pose_frame in all_gripper_box:
                aff_box = all_gripper_box[grasp_pose_frame].get('gripper_box')
                if aff_box is not None:
                    aff_box = np.clip(aff_box, 0, None)
                    if contact_box is not None:
                        contact_box = np.clip(contact_box, 0, None)
                        assert np.array_equal(aff_box, contact_box)
                    frame_data['affordance_box'] = to_int_list(aff_box)

        elif grasp_pose_frame == -1:
            frame_data['contact_frame'] = -1
            frame_data['affordance_box'] = []
            frame_data['contact_points'] = []
            frame_data['state_affordance'] = []


        # Get trajectory (next 10 steps)
        if all_traj is not None and all_traj.get('trace') is not None:
            end_frame = min(total_frames, frame_id + 10)
            traj = all_traj['trace'][frame_id:end_frame]

            if traj is not None and len(traj) > 0:
                max_value = (origin_shape[0] + 9, origin_shape[1] + 9)
                if check_posi_available(traj, 0, max_value):
                    frame_data['trace'] = clip_traj(traj, 0, (origin_shape[0] - 1, origin_shape[1] - 1))

        result[frame_id] = frame_data



    return result


def convert_to_lmdb(droid_pkl_path, rh20t_pkl_path, output_lmdb_path, map_size=100 * 1024**3, data_lmdb_path=None, dry_run=False):
    """
    Convert merged pkl files to frame-level LMDB storage.

    Args:
        droid_pkl_path: Path to the merged DROID pkl file.
        rh20t_pkl_path: Path to the merged RH20T pkl file.
        output_lmdb_path: Output LMDB path.
        map_size: LMDB map size in bytes.
        data_lmdb_path: Path to the action dataset (episode directories).
        dry_run: Debug mode - process data without writing to LMDB.
    """
    if dry_run:
        print("\n" + "="*60)
        print("DRY RUN MODE - data will not be written to LMDB")
        print("="*60)

    # Load data
    droid_data = load_pkl(droid_pkl_path)
    rh20t_data = load_pkl(rh20t_pkl_path)

    # Statistics
    stats = {
        'droid_total': len(droid_data),
        'rh20t_total': len(rh20t_data),
        'droid_converted': 0,
        'rh20t_converted': 0,
        'droid_skipped': 0,
        'rh20t_skipped': 0,
    }

    # Store sample data in dry_run mode
    sample_droid = None
    sample_rh20t = None

    merged_keys = []

    # Create LMDB (skipped in dry_run mode)
    env = None
    txn = None
    if not dry_run:
        os.makedirs(os.path.dirname(output_lmdb_path) if os.path.dirname(output_lmdb_path) else ".", exist_ok=True)
        env = lmdb.open(output_lmdb_path, map_size=map_size)
        txn = env.begin(write=True)

    # Process DROID data
    print("\nConverting droid data...")
    for key in tqdm(list(droid_data.keys()), desc="Processing droid"):
        inter_data = droid_data[key]
        converted = convert_single_item(key, inter_data, data_lmdb_path)

        if converted is not None:
            if not dry_run:
                txn.put(key.encode("utf-8"), pickle.dumps(converted))
            elif sample_droid is None:
                sample_droid = (key, converted)
            merged_keys.append(key)
            stats['droid_converted'] += 1
        else:
            stats['droid_skipped'] += 1

    # Process RH20T data
    print("\nConverting rh20t data...")
    for key in tqdm(list(rh20t_data.keys()), desc="Processing rh20t"):
        inter_data = rh20t_data[key]
        converted = convert_single_item(key, inter_data, data_lmdb_path)

        if converted is not None:
            if not dry_run:
                txn.put(key.encode("utf-8"), pickle.dumps(converted))
            elif sample_rh20t is None:
                sample_rh20t = (key, converted)
            merged_keys.append(key)
            stats['rh20t_converted'] += 1
        else:
            stats['rh20t_skipped'] += 1

    # Store metadata and close (skipped in dry_run mode)
    if not dry_run:
        # metadata = {
        #     "total_items": len(merged_keys),
        #     "droid_items": stats['droid_converted'],
        #     "rh20t_items": stats['rh20t_converted'],
        #     "all_keys": merged_keys
        # }
        # txn.put(b"__metadata__", pickle.dumps(metadata))
        txn.commit()
        env.close()

    # Print statistics
    print("\n" + "="*60)
    print("Conversion Statistics:" + (" [DRY RUN]" if dry_run else ""))
    print(f"  DROID total: {stats['droid_total']}")
    print(f"  DROID converted: {stats['droid_converted']}")
    print(f"  DROID skipped (unable to infer frame count): {stats['droid_skipped']}")
    print("-"*60)
    print(f"  RH20T total: {stats['rh20t_total']}")
    print(f"  RH20T converted: {stats['rh20t_converted']}")
    print(f"  RH20T skipped (unable to infer frame count): {stats['rh20t_skipped']}")
    print("-"*60)
    print(f"  Total converted: {stats['droid_converted'] + stats['rh20t_converted']}")
    if not dry_run:
        print(f"  Output LMDB path: {output_lmdb_path}")
    print("="*60)

    # Print sample data in dry_run mode
    if dry_run:
        print("\n" + "="*60)
        print("Sample Data (first successfully converted item):")
        print("="*60)

        if sample_droid:
            key, data = sample_droid
            print(f"\n[DROID] key: {key}")
            print(f"  Total frames: {len([k for k in data.keys() if isinstance(k, int)])}")
            # Print frame 0 data
            if 0 in data:
                print(f"  Frame 0 data:")
                for k, v in data[0].items():
                    if v is None:
                        print(f"    {k}: None")
                    elif isinstance(v, list) and len(v) > 3:
                        print(f"    {k}: list[{len(v)}] = {v[:3]}...")
                    else:
                        print(f"    {k}: {v}")

        if sample_rh20t:
            key, data = sample_rh20t
            print(f"\n[RH20T] key: {key}")
            print(f"  Total frames: {len([k for k in data.keys() if isinstance(k, int)])}")
            if 0 in data:
                print(f"  Frame 0 data:")
                for k, v in data[0].items():
                    if v is None:
                        print(f"    {k}: None")
                    elif isinstance(v, list) and len(v) > 3:
                        print(f"    {k}: list[{len(v)}] = {v[:3]}...")
                    else:
                        print(f"    {k}: {v}")

        print("="*60)

    return stats


def main():
    parser = argparse.ArgumentParser(description="Convert merged pkl files to frame-level LMDB")
    parser.add_argument(
        "--droid_pkl",
        type=str,
        default="",
        help="Path to the merged DROID annotation pkl"
    )
    parser.add_argument(
        "--rh20t_pkl",
        type=str,
        default="",
        help="Path to the merged RH20T annotation pkl"
    )
    parser.add_argument(
        "--output_lmdb",
        type=str,
        default="",
        help="Output LMDB path"
    )

    parser.add_argument(
        "--data_lmdb_path",
        type=str,
        default="",
        help="Path to the action dataset episode directories"
    )

    parser.add_argument(
        "--map_size",
        type=int,
        default=100 * 1024**3,
        help="LMDB map size (bytes)"
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Debug mode: process data without writing, print statistics and samples"
    )

    args = parser.parse_args()

    convert_to_lmdb(
        droid_pkl_path=args.droid_pkl,
        rh20t_pkl_path=args.rh20t_pkl,
        output_lmdb_path=args.output_lmdb,
        map_size=args.map_size,
        data_lmdb_path=args.data_lmdb_path,
        dry_run=False,
    )


if __name__ == "__main__":
    main()
