# ==================== Configuration ====================
import lmdb
import pickle
import os

# LMDB database path
LMDB_PATH = "demo_data"

# Video file root directory - modify this to your video storage path
# Video files should be named as: {video_name}.mp4
# Example: 11947_exterior_image_1_left.mp4
VIDEO_ROOT = "videos"

# Coordinate scaling function
def scale_coordinates(annotations, scale_x=4.0, scale_y=4.0):
    """
    Scale coordinates in annotation data
    From (180, 320) to (720, 1280), scaling factor is 4
    Only apply scaling to videos where video_name does not contain 'rh20t'
    """
    scaled_annotations = {}

    for video_name, video_data in annotations.items():
        # Check if video_name contains 'rh20t' (case-insensitive)
        if 'rh20t' in video_name.lower():
            # If it contains rh20t, do not scale, use original data directly
            scaled_annotations[video_name] = video_data
            continue

        scaled_video_data = {}

        for frame_idx, frame_data in video_data.items():
            scaled_frame_data = frame_data.copy()

            # Scale box-type coordinates (format: [[x1, y1], [x2, y2]])
            for box_key in ['object_box', 'placement_proposal', 'gripper_box', 'affordance_box']:
                if box_key in scaled_frame_data and scaled_frame_data[box_key] is not None:
                    box = scaled_frame_data[box_key]
                    if isinstance(box, list) and len(box) == 2:
                        scaled_frame_data[box_key] = [
                            [box[0][0] * scale_x, box[0][1] * scale_y],
                            [box[1][0] * scale_x, box[1][1] * scale_y]
                        ]

            # Scale trace coordinates (format: [[x1, y1], [x2, y2], ...])
            if 'trace' in scaled_frame_data and scaled_frame_data['trace'] is not None:
                trace = scaled_frame_data['trace']
                if isinstance(trace, list):
                    scaled_frame_data['trace'] = [
                        [point[0] * scale_x, point[1] * scale_y] for point in trace
                    ]

            # Scale contact_points coordinates (format: [[x1, y1], [x2, y2], ...])
            if 'contact_points' in scaled_frame_data and scaled_frame_data['contact_points'] is not None:
                points = scaled_frame_data['contact_points']
                if isinstance(points, list):
                    scaled_frame_data['contact_points'] = [
                        [point[0] * scale_x, point[1] * scale_y] for point in points
                    ]

            # Scale grasp_pose coordinates (format: [[x1, y1], [x2, y2], ...])
            if 'grasp_pose' in scaled_frame_data and scaled_frame_data['grasp_pose'] is not None:
                gp = scaled_frame_data['grasp_pose']
                if isinstance(gp, list):
                    scaled_frame_data['grasp_pose'] = [
                        [point[0] * scale_x, point[1] * scale_y] for point in gp
                    ]

            scaled_video_data[frame_idx] = scaled_frame_data

        scaled_annotations[video_name] = scaled_video_data

    return scaled_annotations


# Load annotation data from LMDB
def load_annotations_from_lmdb(lmdb_path):
    """
    Load all annotation data from LMDB database
    Return format:
    {
        "video_name": {
            frame_idx: {
                "time_clip": list or None,
                "instruction_add": str or None,
                "substask": str or None,
                "primitive_skill": str or None,
                "segmentation": np.ndarray or None,
                "object_box": list or None,
                "placement_proposal": list or None,
                "trace": list or None,
                "gripper_box": list or None,
                "contact_frame": int or None,
                "state_affordance": list or None,
                "affordance_box": list or None,
                "contact_points": list or None,
            },
            ...
        },
        ...
    }
    """
    if not os.path.exists(lmdb_path):
        print(f"Error: LMDB path {lmdb_path} does not exist!")
        return {}

    annotations = {}
    env = lmdb.open(lmdb_path, readonly=True)

    with env.begin() as txn:
        cursor = txn.cursor()
        for key, value in cursor:
            video_name = key.decode('utf-8') if isinstance(key, bytes) else str(key)
            try:
                video_data = pickle.loads(value)
                annotations[video_name] = video_data
            except Exception as e:
                print(f"Error loading data for {video_name}: {e}")

    env.close()
    # print(f"Loaded {len(annotations)} videos from LMDB")

    # Apply coordinate scaling (from 180x320 to 720x1280)
    annotations = scale_coordinates(annotations, scale_x=4.0, scale_y=4.0)
    # print(f"Applied 4x coordinate scaling to all annotations")

    return annotations

# Load data
ANNOTATIONS = load_annotations_from_lmdb(LMDB_PATH)