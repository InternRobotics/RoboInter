import argparse
import json
import os
import subprocess
from typing import List


DROID_SHEET_PATH = "path/to/RoboInter_Data/RoboInter_Data_Qsheet.json"
DROID_RAW_PREFIX = "gs://gresearch/robotics/droid_raw/1.0.1"


def run_cmd(cmd: List[str], check: bool = True) -> str:
    """Run command and return stdout (str)."""
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if check and p.returncode != 0:
        raise RuntimeError(
            f"Command failed: {' '.join(cmd)}\n"
            f"Return code: {p.returncode}\n"
            f"STDOUT:\n{p.stdout}\n"
            f"STDERR:\n{p.stderr}\n"
        )
    return p.stdout.strip()


def extract_episode_from_ori_path(ori_path: str) -> str:
    """
    ori_path example:
      gs://xembodiment_data/r2d2/r2d2-data-full/TRI/success/2024-02-07/Wed_Feb__7_16:52:46_2024/trajectory.h5

    return:
      TRI/success/2024-02-07/Wed_Feb__7_16:52:46_2024
    """
    if not isinstance(ori_path, str) or not ori_path.startswith("gs://"):
        raise ValueError(f"Invalid ori_path: {ori_path}")

    # Split and find "r2d2-data-full" as anchor
    parts = ori_path.split("/")
    # parts: ['gs:', '', 'xembodiment_data', 'r2d2', 'r2d2-data-full', 'TRI', 'success', '2024-02-07', 'Wed_Feb__7_16:52:46_2024', 'trajectory.h5']
    try:
        idx = parts.index("r2d2-data-full")
    except ValueError:
        raise ValueError(f"Cannot find 'r2d2-data-full' in ori_path: {ori_path}")

    # episode components after r2d2-data-full: org/success_or_failure/date/session_name
    # ensure length
    if idx + 4 >= len(parts):
        raise ValueError(f"ori_path too short to extract episode: {ori_path}")

    episode = "/".join(parts[idx + 1 : idx + 5])  # 4 parts
    return episode


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--key", required=True, help="Key in droid_sheet_merged.json, e.g. '74200_exterior_image_2_left'")
    parser.add_argument("--out_dir", default="./hr_video_reader/downloaded_mp4", help="Local output directory")
    parser.add_argument("--sheet_path", default=DROID_SHEET_PATH, help="Path to droid_sheet_merged.json")
    args = parser.parse_args()
    args.out_dir = os.path.join(args.out_dir)
    os.makedirs(args.out_dir, exist_ok=True)

    # 1) load json
    with open(args.sheet_path, "r", encoding="utf-8") as f:
        sheet = json.load(f)

    # args.key = [
    #     "13060_exterior_image_1_left",
    #     "13601_exterior_image_2_left",
    #     "13916_exterior_image_1_left"
    # ] # episode you want to utilize

    for i_key in args.key:
        if i_key not in sheet:
            raise KeyError(f"Key not found in sheet: {i_key}")

        item = sheet[i_key]
        if not isinstance(item, dict):
            raise TypeError(f"sheet[{i_key}] is not a dict")

        ori_path = item.get("ori_path", None)
        if ori_path is None:
            raise ValueError(f"ori_path is None for key: {i_key}")

        # 2) extract episode
        episode = extract_episode_from_ori_path(ori_path)

        # 3) construct mp4 dir
        mp4_dir = f"{DROID_RAW_PREFIX}/{episode}/recordings/MP4/"
        print(f"[INFO] key      : {i_key}")
        print(f"[INFO] ori_path : {ori_path}")
        print(f"[INFO] episode  : {episode}")
        print(f"[INFO] mp4_dir  : {mp4_dir}")
        print(f"[INFO] out_dir  : {args.out_dir+'/'+i_key}")

        # 4) list mp4 files in the directory
        # gsutil ls returns full gs:// paths
        listing = run_cmd(["gsutil", "ls", mp4_dir])
        files = [line.strip() for line in listing.splitlines() if line.strip()]

        # keep only .mp4 and exclude -stereo.mp4
        mp4_files = [p for p in files if p.endswith(".mp4") and (not p.endswith("-stereo.mp4"))]
        if len(mp4_files) == 0:
            print("[WARN] No non-stereo mp4 found.")
            return

        print(f"[INFO] Found {len(mp4_files)} non-stereo mp4 files:")
        for p in mp4_files:
            print("  -", p)
        os.makedirs(args.out_dir+'/'+i_key, exist_ok=True)
        # 5) download (parallel with -m)
        # gsutil -m cp <file1> <file2> ... <out_dir>
        # If too many files, you can batch; here usually only a few.
        cmd = ["gsutil", "-m", "cp"] + mp4_files + [args.out_dir+'/'+i_key]
        print(f"[INFO] Running: {' '.join(cmd)}")
        run_cmd(cmd, check=True)

        print("[DONE] Download completed.")


if __name__ == "__main__":
    main()