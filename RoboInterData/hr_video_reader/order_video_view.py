import os
import shutil
import re

src_root = "./hr_video_reader/downloaded_mp4"
dst_root = "./hr_video_reader/demo_mp4"

os.makedirs(dst_root, exist_ok=True)

def extract_number(filename):
    # Extract the number before the mp4 file name
    return int(re.findall(r'\d+', filename)[0])

for folder in os.listdir(src_root):
    folder_path = os.path.join(src_root, folder)
    if not os.path.isdir(folder_path):
        continue

    mp4_files = [f for f in os.listdir(folder_path) if f.endswith(".mp4")]
    if len(mp4_files) < 3:
        print(f"[SKIP] {folder} mp4 insufficient quantity")
        continue

    # Sort by number from large to small
    mp4_files.sort(key=extract_number, reverse=True)

    if folder.endswith("_exterior_image_1_left"):
        target_file = mp4_files[1]  # the second largest
    elif folder.endswith("_exterior_image_2_left"):
        target_file = mp4_files[0]  # the third largest
    else:
        print(f"[SKIP] {folder} conflict with rules")
        continue

    src_file_path = os.path.join(folder_path, target_file)
    new_name = f"{folder}.mp4"
    dst_file_path = os.path.join(dst_root, new_name)

    shutil.copy2(src_file_path, dst_file_path)
    print(f"[OK] {src_file_path} -> {dst_file_path}")