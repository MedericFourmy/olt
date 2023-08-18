import cv2
import numpy as np
import json
import time
from PIL import Image

from olt.tracker import Tracker
from olt.localizer import Localizer
from olt.utils import Kres2intrinsics, print_mem_usage
from olt.config import OBJ_MODEL_DIRS, MEGAPOSE_DATA_DIR, TrackerConfig, LocalizerConfig

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
THRESHOLD_DETECTOR = 0.6

accepted_objs = 'all'
tcfg = TrackerConfig()
tcfg.n_corr_iterations
tcfg.viewer_display = True
tcfg.viewer_save = True
tracker = Tracker(intrinsics, OBJ_MODEL_DIRS[DS_NAME], accepted_objs, tcfg)
tracker.init()

lcfg = LocalizerConfig
lcfg.n_workers = 2
localizer = Localizer(DS_NAME, lcfg)

poses = localizer.predict(rgb, K, n_coarse=1, n_refiner=3)

tracker.detected_bodies(poses)
tracker.set_image(rgb)
tracker.track()
t = time.time()
tracker.update_viewers()
print('update_viewers (ms)', 1000*(time.time() - t))
print_mem_usage()

# cv2.waitKey(0)

