"""
Annotation Server

A Flask server for managing video annotation workflows.
"""

import io
import zipfile
import json
import os
import pickle
import numpy as np
import yaml
from flask import Flask, request, send_file
import multiprocessing
import portalocker
from multiprocessing import Lock


lock = Lock()
app = Flask(__name__)

# Global configuration
CONFIG = None
ROOT_DIR = None
PATHS = None


def load_server_config(config_path="./config/config.yaml"):
    """Load server configuration from yaml file."""
    global CONFIG, ROOT_DIR, PATHS
    with open(config_path) as f:
        CONFIG = yaml.load(f, Loader=yaml.FullLoader)

    server_config = CONFIG.get("server", {})
    ROOT_DIR = server_config.get("root_dir", "./data")
    PATHS = {
        "no_annotation_sam": os.path.join(ROOT_DIR, server_config.get("no_annotation_sam", "no_annotation_sam.json")),
        "has_annotation_sam": os.path.join(ROOT_DIR, server_config.get("has_annotation_sam", "has_annotation_sam.json")),
        "no_annotation_lang": os.path.join(ROOT_DIR, server_config.get("no_annotation_lang", "no_annotation.json")),
        "has_annotation_lang": os.path.join(ROOT_DIR, server_config.get("has_annotation_lang", "has_annotation.json")),
        "user_list_file": os.path.join(ROOT_DIR, server_config.get("user_list_file", "user_list.txt")),
        "user_history_dir": os.path.join(ROOT_DIR, server_config.get("user_history_dir", "user_config")),
    }

    return CONFIG


def get_user_history(user_name, mode, suffix=""):
    """Get user annotation history."""
    os.makedirs(os.path.join(PATHS["user_history_dir"], mode), exist_ok=True)
    history_file = os.path.join(PATHS["user_history_dir"], mode, f"{user_name}{suffix}.txt")
    if not os.path.exists(history_file):
        with open(history_file, 'w') as f:
            f.write('')
        return []
    with open(history_file, 'r') as f:
        return f.readlines()


def save_user_history(user_name, mode, history, suffix=""):
    """Save user annotation history."""
    os.makedirs(os.path.join(PATHS["user_history_dir"], mode), exist_ok=True)
    history_file = os.path.join(PATHS["user_history_dir"], mode, f"{user_name}{suffix}.txt")
    with open(history_file, 'w') as f:
        portalocker.lock(f, portalocker.LOCK_EX)
        f.writelines(history)
        f.flush()
        os.fsync(f.fileno())


def get_diff(a, b):
    """Get items in a but not in b (by filename)."""
    a_dict = {i.split('/')[-1].strip(): i for i in a}
    a_set = set(list(a_dict.keys()))
    b_set = set([i.split('/')[-1].strip() for i in b])
    diff = list(a_set.difference(b_set))
    return [a_dict[i] for i in diff]


def get_available(a, b):
    """Get items in both a and b (by filename)."""
    a_dict = {i.split('/')[-1].strip(): i for i in a}
    b = set([i.split('/')[-1].strip() for i in b])
    return [a_dict[i] for i in b if i in a_dict]


@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint."""
    return {"status": "ok"}


@app.route("/is_available_user", methods=["POST"])
def is_available_user():
    """Check if a user is valid against the user list file."""
    config = json.loads(request.data)
    user_name = config['user_name']

    user_list_file = PATHS["user_list_file"]
    if os.path.exists(user_list_file):
        with open(user_list_file, 'r') as f:
            user_list = [i.strip() for i in f.readlines() if i.strip()]
        if user_name not in user_list:
            user_name = ''
    # If no user list file exists, all users are accepted

    zip_io = io.BytesIO()
    with zipfile.ZipFile(zip_io, "w") as zf:
        zf.writestr("user_name", user_name)

    zip_io.seek(0)
    return send_file(
        zip_io,
        mimetype="application/zip",
        as_attachment=True,
        download_name="user_name.zip",
    )


@app.route("/get_video_lang", methods=["POST"])
def get_video_lang():
    """
    Get next video for annotation.

    Request body:
    {
        "username": "user1",
        "mode": "next" | "pre",
        "last_video_path": "/path/to/last/video.mp4"
    }
    """
    config = json.loads(request.data)
    user_name = config.get('username', '').strip()
    mode = config.get('mode', 'next')
    last_video_path = config.get('last_video_path', '')

    # Get user history
    history = get_user_history(user_name, 'lang')

    with lock:
        with open(PATHS["no_annotation_lang"], 'r+') as f1:
            portalocker.lock(f1, portalocker.LOCK_EX)
            no_annotation = json.load(f1)

            with open(PATHS["has_annotation_lang"], 'r+') as f2:
                portalocker.lock(f2, portalocker.LOCK_EX)
                has_annotation = json.load(f2)

                # User-specific annotation pool
                user_pool = no_annotation.get(user_name, {})
                user_has = has_annotation.get(user_name, {})

                if len(user_pool) == 0:
                    is_finished = True
                    video_path = None
                else:
                    is_finished = False

                    if mode == 'pre' and last_video_path:
                        # Return to previous video
                        video_path = history[-1].strip() if history else None
                        if last_video_path in user_has:
                            no_annotation.setdefault(user_name, {})[last_video_path] = user_has[last_video_path].copy()
                            del has_annotation[user_name][last_video_path]
                    else:
                        # Get next video
                        video_path = list(user_pool.keys())[0]
                        has_annotation.setdefault(user_name, {})[video_path] = user_pool[video_path].copy()
                        del no_annotation[user_name][video_path]

                f2.seek(0)
                f2.truncate()
                json.dump(has_annotation, f2)
                f2.flush()
                os.fsync(f2.fileno())

            f1.seek(0)
            f1.truncate()
            json.dump(no_annotation, f1)
            f1.flush()
            os.fsync(f1.fileno())

    # Build response
    zip_io = io.BytesIO()
    with zipfile.ZipFile(zip_io, "w") as zf:
        zf.writestr("is_finished", str(is_finished))

        if not is_finished and video_path:
            # Read video file
            if os.path.exists(video_path):
                with zf.open("video.mp4", "w") as f:
                    with open(video_path, "rb") as video_file:
                        f.write(video_file.read())

            # Get annotation info
            video_info = has_annotation.get(user_name, {}).get(video_path, {})

            # Load annotation if exists
            anno_path = video_info.get('anno_path')
            if anno_path and os.path.exists(anno_path):
                npz_io = io.BytesIO()
                anno_file = np.load(anno_path, allow_pickle=True)
                np.savez_compressed(npz_io, anno_file=anno_file.get('data', anno_file))
                npz_io.seek(0)
                zf.writestr("anno.npz", npz_io.getvalue())

            # Include paths
            save_path = video_info.get('save_path', '')
            zf.writestr("save_path", save_path)
            zf.writestr("video_path", video_path)
            zf.writestr("history_number", str(len(history)))

    zip_io.seek(0)
    return send_file(
        zip_io,
        mimetype="application/zip",
        as_attachment=True,
        download_name="video.zip",
    )


@app.route("/get_video_sam", methods=["POST"])
def get_video_and_anno_sam():
    config = json.loads(request.data)
    user_name = config['username'].strip()
    mode = config['mode']
    re_anno = int(config['re_anno'])
    last_video_path = config['last_video_path']
    history_finished = get_user_history(user_name, 'sam', '_finish')

    history_all = get_user_history(user_name, 'sam', '')
    history = get_diff(history_all, history_finished)
    
    history_1 = get_user_history(user_name, 'sam', '_1')
    history_1 = get_diff(history_1, history_finished)
    usable_1 = get_diff(history, history_1)
    all_one_anno_num = len(usable_1)
    
    history_2 = get_user_history(user_name, 'sam', '_2')
    history_2 = get_diff(history_2, history_finished)
    usable_2 = get_diff(history_1, history_2)
    all_two_anno_num = len(usable_2)
    
    history_3 = get_user_history(user_name, 'sam', '_3')
    usable_3 = get_diff(history_2, history_3)
    all_three_anno_num = len(usable_3)
    
    with lock:
        with open(PATHS["no_annotation_sam"], 'r+') as f1:
            portalocker.lock(f1, portalocker.LOCK_EX)
            no_annotation = json.load(f1)
            with open(PATHS["has_annotation_sam"], 'r+') as f2:
                portalocker.lock(f2, portalocker.LOCK_EX)
                has_annotation = json.load(f2)
                if len(no_annotation[user_name]) == 0:
                    is_finished = True
                else:
                    is_finished = False
                
                available_0 = [i for i in no_annotation[user_name].keys() if 'ann_human' not in i]
                available_1 = get_available(list(no_annotation[user_name].keys()), usable_1)
                available_2 = get_available(list(no_annotation[user_name].keys()), usable_2)
                available_3 = get_available(list(no_annotation[user_name].keys()), usable_3)
                
                
                if re_anno == 1 and len(available_1) == 0:
                    is_finished = True
                
                if re_anno == 2 and len(available_2) == 0:
                    is_finished = True
                
                if re_anno == 3 and len(available_3) == 0:
                    is_finished = True

                if re_anno > 0 and not is_finished:
                    if re_anno == 1:
                        video_path = available_1[0]
                    elif re_anno == 2:
                        video_path = available_2[0]
                    elif re_anno == 3:
                        video_path = available_3[0]
                    assert mode == 'next'
                    has_annotation[user_name][video_path] = no_annotation[user_name][video_path].copy()
                    del no_annotation[user_name][video_path]
                else:
                    if mode == 'pre':
                        video_path = history[-1].strip()
                        no_annotation[user_name][last_video_path] = has_annotation[user_name][last_video_path].copy()
                        del has_annotation[user_name][last_video_path]
                        is_finished = False
                    elif not is_finished: 
                        video_path = available_0[0]
                        has_annotation[user_name][video_path] = no_annotation[user_name][video_path].copy()
                        del no_annotation[user_name][video_path]
                    else:
                        video_path = '{}_{}'.format(re_anno, is_finished)
                
                f2.seek(0)
                f2.truncate()
                json.dump(has_annotation, f2)
                f2.flush()
                os.fsync(f2.fileno())
            
            f1.seek(0)
            f1.truncate()
            json.dump(no_annotation, f1)
            f1.flush()
            os.fsync(f1.fileno())
    
    zip_io = io.BytesIO()
    with zipfile.ZipFile(zip_io, "w") as zf:
        if not is_finished:
            with zf.open("video.mp4", "w") as f:
                with open(video_path, "rb") as video_file:
                    f.write(video_file.read())
            # send save path
            save_path = has_annotation[user_name][video_path]['save_path'].rsplit('/', 1)[0]
            save_file_name = has_annotation[user_name][video_path]['save_path'].split('/')[-1].split('.')[0]
            save_path = os.path.join(save_path, save_file_name)
            zf.writestr("save_path", save_path)
            zf.writestr("video_path", video_path)
            zf.writestr("history_number", str(len(history_all)))
        
        zf.writestr("is_finished", str(is_finished))
        zf.writestr("all_one_anno_num", str(all_one_anno_num))
        zf.writestr("one_anno_num", str(len(available_1)))
        zf.writestr("all_two_anno_num", str(all_two_anno_num))
        zf.writestr("two_anno_num", str(len(available_2)))
        zf.writestr("all_three_anno_num", str(all_three_anno_num))
        zf.writestr("three_anno_num", str(len(available_3)))
    
    zip_io.seek(0)
    return send_file(
        zip_io,
        mimetype="application/zip",
        as_attachment=True,
        download_name="video_and_anno_sam.zip",
    )


@app.route("/save_anno", methods=["POST"])
def save_anno():
    """
    Save annotation result.

    Form data:
    - file: annotation npz file
    - save_path: where to save the annotation
    """
    file = request.files.get('file')
    if not file:
        return {"error": "No file provided"}, 400

    file_content = file.read()
    with np.load(io.BytesIO(file_content), allow_pickle=True) as data:
        anno = data['anno_file'].item()

    save_path = request.form.get('save_path')
    user_name = anno.get('user', 'unknown')
    video_path = anno.get('video_path', '')
    
    if '/0/' in video_path:
        time = '_1'
    elif '/1/' in video_path:
        time = '_2'
    elif '/2/' in video_path:
        time = '_3'
    else:
        time = ''

    # Save annotation file
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.savez(save_path, pickle.dumps(anno))
    
    if 'sam' in save_path:
        mode = 'sam'
    else:
        mode = 'lang'

    # Update user history
    history = get_user_history(user_name, mode, time)
    if history and video_path == history[-1].strip():
        history = history[:-1]
    history.append(video_path + '\n')
    
    save_user_history(user_name, mode, history, time)

    # Handle finished/hard sample markers
    if anno.get('is_finished', False):
        finished_history = get_user_history(user_name, mode, '_finish')
        finished_history.append(video_path.strip() + '\n')
        save_user_history(user_name, mode, finished_history, '_finish')

    if anno.get('is_hard_sample', False):
        sample_type = anno.get('hard_sample_type', '')
        if sample_type == '困难样本':
            suffix = '_hard'
        elif sample_type == '问题样本':
            suffix = '_question'

        hard_history = get_user_history(user_name, mode, suffix)
        hard_history.append(video_path.strip() + '\n')
        save_user_history(user_name, mode, hard_history, suffix)
    

    return "success"


@app.route("/drawback_lang", methods=["POST"])
def drawback_video_lang():
    """
    Withdraw a video back to the unannotated pool (lang mode).

    Request body:
    {
        "video_path": "/path/to/video.mp4",
        "username": "user1"
    }
    """
    config = json.loads(request.data)
    video_path = config['video_path']
    user_name = config.get('username', '').strip()

    with lock:
        with open(PATHS["no_annotation_lang"], 'r+') as f1:
            portalocker.lock(f1, portalocker.LOCK_EX)
            no_annotation = json.load(f1)

            with open(PATHS["has_annotation_lang"], 'r+') as f2:
                portalocker.lock(f2, portalocker.LOCK_EX)
                has_annotation = json.load(f2)

                if user_name in has_annotation and video_path in has_annotation[user_name]:
                    no_annotation.setdefault(user_name, {})[video_path] = has_annotation[user_name][video_path].copy()
                    del has_annotation[user_name][video_path]

                f2.seek(0)
                f2.truncate()
                json.dump(has_annotation, f2)
                f2.flush()
                os.fsync(f2.fileno())

            f1.seek(0)
            f1.truncate()
            json.dump(no_annotation, f1)
            f1.flush()
            os.fsync(f1.fileno())

    return "success"


@app.route("/drawback_sam", methods=["POST"])
def drawback_video_sam():
    """
    Withdraw a video back to the unannotated pool (sam mode).

    Request body:
    {
        "video_path": "/path/to/video.mp4",
        "username": "user1"
    }
    """
    config = json.loads(request.data)
    video_path = config['video_path']
    user_name = config.get('username', '').strip()

    with lock:
        with open(PATHS["no_annotation_sam"], 'r+') as f1:
            portalocker.lock(f1, portalocker.LOCK_EX)
            no_annotation = json.load(f1)

            with open(PATHS["has_annotation_sam"], 'r+') as f2:
                portalocker.lock(f2, portalocker.LOCK_EX)
                has_annotation = json.load(f2)

                if user_name in has_annotation and video_path in has_annotation[user_name]:
                    no_annotation.setdefault(user_name, {})[video_path] = has_annotation[user_name][video_path].copy()
                    del has_annotation[user_name][video_path]

                f2.seek(0)
                f2.truncate()
                json.dump(has_annotation, f2)
                f2.flush()
                os.fsync(f2.fileno())

            f1.seek(0)
            f1.truncate()
            json.dump(no_annotation, f1)
            f1.flush()
            os.fsync(f1.fileno())

    return "success"


@app.route("/stats_lang", methods=["GET"])
def get_stats_lang():
    """Get annotation statistics (lang mode)."""
    try:
        with open(PATHS["no_annotation_lang"], 'r') as f:
            no_annotation = json.load(f)
        with open(PATHS["has_annotation_lang"], 'r') as f:
            has_annotation = json.load(f)

        per_user = {}
        total_remaining = 0
        total_annotated = 0

        for user, videos in no_annotation.items():
            if isinstance(videos, dict):
                count = len(videos)
                total_remaining += count
                per_user.setdefault(user, {"remaining": 0, "in_progress": 0})
                per_user[user]["remaining"] = count

        for user, videos in has_annotation.items():
            if isinstance(videos, dict):
                count = len(videos)
                total_annotated += count
                per_user.setdefault(user, {"remaining": 0, "in_progress": 0})
                per_user[user]["in_progress"] = count

        return {
            "remaining": total_remaining,
            "in_progress": total_annotated,
            "total": total_remaining + total_annotated,
            "per_user": per_user
        }
    except Exception as e:
        return {"error": str(e)}, 500


@app.route("/stats_sam", methods=["GET"])
def get_stats_sam():
    """Get annotation statistics (sam mode)."""
    try:
        with open(PATHS["no_annotation_sam"], 'r') as f:
            no_annotation = json.load(f)
        with open(PATHS["has_annotation_sam"], 'r') as f:
            has_annotation = json.load(f)

        per_user = {}
        total_remaining = 0
        total_annotated = 0

        for user, videos in no_annotation.items():
            if isinstance(videos, dict):
                count = len(videos)
                total_remaining += count
                per_user.setdefault(user, {"remaining": 0, "in_progress": 0})
                per_user[user]["remaining"] = count

        for user, videos in has_annotation.items():
            if isinstance(videos, dict):
                count = len(videos)
                total_annotated += count
                per_user.setdefault(user, {"remaining": 0, "in_progress": 0})
                per_user[user]["in_progress"] = count

        return {
            "remaining": total_remaining,
            "in_progress": total_annotated,
            "total": total_remaining + total_annotated,
            "per_user": per_user
        }
    except Exception as e:
        return {"error": str(e)}, 500


def run_on_port(port, config_path):
    """Run Flask app on specified port."""
    load_server_config(config_path)
    app.run(host="0.0.0.0", port=port)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Annotation Server")
    parser.add_argument('--config', type=str, default="./config/config.yaml",
                        help="Path to config file")
    parser.add_argument('--port', type=int, default=None,
                        help="Single port to run on (overrides multi-process mode)")
    parser.add_argument('--processes', type=int, default=1,
                        help="Number of server processes")
    parser.add_argument('--base-port', type=int, default=10086,
                        help="Base port for multi-process mode")
    args = parser.parse_args()

    if args.port:
        # Single process mode
        run_on_port(args.port, args.config)
    else:
        # Multi-process mode
        process_list = []
        for i in range(args.processes):
            process = multiprocessing.Process(
                target=run_on_port,
                args=(args.base_port + i, args.config)
            )
            process_list.append(process)
            process.start()

        for process in process_list:
            process.join()
