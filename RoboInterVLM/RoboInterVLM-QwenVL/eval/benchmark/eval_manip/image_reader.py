import lmdb
import pickle
from pathlib import Path
from PIL import Image
import io, os
import cv2
import numpy as np


class EpisodeReader:
    def __init__(self):
        """
        episodes_dir: Path to the root folder that contains all episodes
        """
        self.episodes_dir = "#TODO/episodes"  # Path to LMDB episodes directory

    def get_images(self, episode_key: str, index, img_type: str = "primary_image"):
        """
        Load episode meta_info, read image(s) and return as PIL.Image.
        episode_key: e.g. '91970_exterior_image_1_left' 'RH20T_cfg7_task_0210_user_0066_scene_0003_cfg_0007_104122064161'
        index: int or list of int
        img_type: 'primary_image' or 'wrist_image'
        """
        # Load meta_info
        if 'exterior_image' in episode_key:
            assert img_type in ['primary_image', 'wrist_image'], f"image type {img_type} not supported"
        elif 'RH20T' in episode_key:
            assert img_type == 'primary_image', f"image type {img_type} not supported"
        else:
            raise NotImplementedError(f"dataset and episode {episode_key} not supported")
        
        meta_info_path = os.path.join(self.episodes_dir, episode_key, "meta_info.pkl")
        with open(meta_info_path, "rb") as f:
            meta_info = pickle.load(f)

        lmdb_path = os.path.join(self.episodes_dir, episode_key, "lmdb")

        # Open LMDB env for this episode
        with lmdb.open(str(lmdb_path), readonly=True, lock=False) as env:
            with env.begin(write=False) as txn:
                if isinstance(index, int):
                    indices = [index]
                else:
                    indices = list(index)

                images = []
                for n in indices:
                    key = meta_info["keys"][img_type][n]
                    img_bytes = pickle.loads(txn.get(key))
                    if 'RH20T' in episode_key:
                        arr = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR_RGB)
                    else:
                        arr = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
                    img = Image.fromarray(arr)
                    images.append(img)

        if isinstance(index, int):
            return images[0]
        return images
    
    def get_images_for_VQA(self, image_path: str):
        assert '#' in image_path
        if '/' in image_path:
            image_path = image_path.split('/')[-1]
        episode_key, index = image_path.rsplit('#', 1)
        index = int(index)
        return self.get_images(episode_key, index)


def get_reader():
    reader = EpisodeReader()
    return reader
