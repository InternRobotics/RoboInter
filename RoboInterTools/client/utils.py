import pickle
import json
import numpy as np
import requests, io, zipfile
import imageio

base_url = 'http://{ip}:{port}'


def get_available_username(ip, port, username):
    """Validate user against server's user list.

    Returns username if valid, empty string if invalid, None on error.
    """
    root_url = base_url.format(ip=ip, port=port)
    url = f"{root_url}/is_available_user"
    config = {
        "user_name": username,
    }
    response = requests.post(
        url, data=json.dumps(config), headers={"content-type": "application/json"}
    )
    if response.status_code == 200:
        zip_io = io.BytesIO(response.content)
        with zipfile.ZipFile(zip_io, "r") as zf:
            with zf.open("user_name") as f:
                username = f.read().decode("utf-8")
        return username.strip()
    else:
        print("Error:", response)
        return None


# Keep old name as alias for backward compatibility
get_avaiable_username = get_available_username


def request_video_and_anno(ip, port, mode, username, button_mode, last_video_path, re_anno=0):
    """Fetch next video and annotation from server.

    Args:
        ip: Server IP
        port: Server port
        mode: 'lang' or 'sam'
        username: Annotator username
        button_mode: "next" or "pre"
        last_video_path: Path of the last video (used when mode="pre")
        re_anno: Re-annotation round (0-3, only used in sam mode)

    Returns:
        0 if all videos are finished.
        For lang mode:
            (frames, anno, save_path, video_path, history_number)
        For sam mode:
            (frames, save_path, video_path, history_number,
             one_anno_num, all_one_anno_num, two_anno_num, all_two_anno_num,
             three_anno_num, all_three_anno_num)
        None on error.
    """
    root_url = base_url.format(ip=ip, port=port)
    url = f"{root_url}/get_video_{mode}"

    config = {
        "username": username,
        "mode": button_mode,
        "last_video_path": last_video_path,
    }
    if mode == 'sam':
        config["re_anno"] = re_anno

    response = requests.post(
        url, data=json.dumps(config), headers={"content-type": "application/json"}, stream=True
    )
    if response.status_code != 200:
        print("Error:", response)
        return None

    zip_io = io.BytesIO(response.content)
    with zipfile.ZipFile(zip_io, "r") as zf:
        with zf.open("is_finished") as f:
            is_finished = f.read().decode("utf-8") == 'True'

        if mode == 'sam':
            # Annotation count fields are always present in SAM response
            with zf.open("one_anno_num") as f:
                one_anno_num = int(f.read().decode("utf-8"))
            with zf.open("all_one_anno_num") as f:
                all_one_anno_num = int(f.read().decode("utf-8"))
            with zf.open("two_anno_num") as f:
                two_anno_num = int(f.read().decode("utf-8"))
            with zf.open("all_two_anno_num") as f:
                all_two_anno_num = int(f.read().decode("utf-8"))
            with zf.open("three_anno_num") as f:
                three_anno_num = int(f.read().decode("utf-8"))
            with zf.open("all_three_anno_num") as f:
                all_three_anno_num = int(f.read().decode("utf-8"))

        if is_finished:
            return 0

        with zf.open("video.mp4") as f:
            video = f.read()
        with zf.open("save_path") as f:
            save_path = f.read().decode("utf-8")
        with zf.open("video_path") as f:
            video_path = f.read().decode("utf-8")
        with zf.open("history_number") as f:
            history_number = int(f.read().decode("utf-8"))

        if mode == 'lang':
            # Load annotation if present
            anno = None
            if "anno.npz" in zf.namelist():
                with zf.open("anno.npz") as f:
                    anno_data = np.load(f, allow_pickle=True)['anno_file']
                    anno = pickle.loads(anno_data)

    frames = []
    reader = imageio.get_reader(video, "mp4")
    for _, im in enumerate(reader):
        frames.append(np.array(im))

    if mode == 'sam':
        return np.stack(frames), save_path, video_path, history_number, \
               one_anno_num, all_one_anno_num, two_anno_num, all_two_anno_num, \
               three_anno_num, all_three_anno_num
    else:
        return np.stack(frames), anno, save_path, video_path, history_number


def save_anno(ip, port, save_path, anno):
    """Save annotation result to server.

    Args:
        ip: Server IP
        port: Server port
        save_path: Where the server should save the annotation
        anno: Annotation dict to save

    Returns:
        True on success, False on error.
    """
    root_url = base_url.format(ip=ip, port=port)
    url = f"{root_url}/save_anno"
    anno_bytes = io.BytesIO()
    np.savez_compressed(anno_bytes, anno_file=anno)
    anno_bytes.seek(0)
    files = {
        "file": ("anno.npz", anno_bytes, "application/octet-stream"),
        "save_path": (None, save_path),
    }
    response = requests.post(url, files=files)
    if response.status_code == 200:
        return True
    else:
        print("Error:", response)
        return False


def drawback_video(ip, port, video_path, mode, username=''):
    """Withdraw a video back to the unannotated pool.

    Args:
        ip: Server IP
        port: Server port
        video_path: Path of the video to withdraw
        mode: 'lang' or 'sam'
        username: Annotator username

    Returns:
        True on success, False on error.
    """
    root_url = base_url.format(ip=ip, port=port)
    url = f"{root_url}/drawback_{mode}"
    config = {
        "video_path": video_path,
        "username": username,
    }
    response = requests.post(
        url, data=json.dumps(config), headers={"content-type": "application/json"}
    )
    if response.status_code == 200:
        return True
    else:
        print("Error:", response)
        return False


def request_sam(ip, port, config, mode):
    """Request SAM prediction from SAM server.

    Args:
        ip: SAM server IP
        port: SAM server port
        config: SAM config dict
        mode: "online" for real-time prediction, "offline" for cached mask

    Returns:
        masks (online mode) or (config, masks) (offline mode), None on error.
    """
    root_url = base_url.format(ip=ip, port=port)
    if mode == "online":
        url = f"{root_url}/predict_sam"
    else:
        url = f"{root_url}/get_mask"
    response = requests.post(
        url, data=json.dumps(config), headers={"content-type": "application/json"}
    )
    if response.status_code == 200:
        zip_io = io.BytesIO(response.content)
        with zipfile.ZipFile(zip_io, "r") as zf:
            if mode == "online":
                with zf.open("masks.npy") as f:
                    masks = np.load(f)
                return masks
            if mode == "offline":
                with zf.open("config.json") as f:
                    config = json.load(f)
                with zf.open("masks.npy") as f:
                    masks = np.load(f)['masks']
                return config, masks
    else:
        print("Error:", response)
        return None
