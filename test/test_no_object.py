import cv2
import numpy as np
import json
import time
from PIL import Image

import pytest

from olt.tracker import Tracker
from olt.localizer import Localizer
from olt.utils import Kres2intrinsics, print_mem_usage
from olt.config import OBJ_MODEL_DIRS, MEGAPOSE_DATA_DIR, TrackerConfig, LocalizerConfig


def test_obj_in_img():
    #############################
    """
        Image + info loading

        For this example, manually get one image + cam intrinsics of the views from a bop dataset
        To load a full scene, use BOPDatasetReader in evaluation_tools.py
    """
    DS_NAME = 'ycbv'
    SCENE_ID = 48
    VIEW_ID = 1

    scene_id_str = '{SCENE_ID:06}'
    view_id_str = '{VIEW_ID:06}'
    rgb_full_path = MEGAPOSE_DATA_DIR / 'bop_datasets/ycbv/test/000048/rgb/000001.png'
    scene_cam_full_path = MEGAPOSE_DATA_DIR / 'bop_datasets/ycbv/test/000048/scene_camera.json'
    d_scene_camera = json.loads(scene_cam_full_path.read_text())
    K = d_scene_camera[str(VIEW_ID)]['cam_K']
    K = np.array(K).reshape((3,3))

    im = Image.open(rgb_full_path)
    rgb = np.array(im, dtype=np.uint8)

    height, width, _ = rgb.shape
    intrinsics = Kres2intrinsics(K, width, height)
    #############################


    ## HAPPYPOSE

    lcfg = LocalizerConfig
    lcfg.n_workers = 2
    localizer = Localizer(DS_NAME, lcfg)

    poses = localizer.predict(rgb, K, n_coarse=1, n_refiner=3)

    assert 'ycbv-obj_000006' in poses.keys()
    print("objects are detected")

    # rgb[:] = 42
    # poses = localizer.predict(rgb, K, n_coarse=1, n_refiner=3)

    # assert len(poses.keys()) == 0
    # print("objects are detected")



