import numpy as np
import json
from PIL import Image

from olt.tracker import Tracker
from olt.localizer import Localizer
from olt.utils import Kres2intrinsics
from olt.config import OBJ_MODEL_DIRS, MEGAPOSE_DATA_DIR, LocalizerConfig


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

    scene_id_str = f'{SCENE_ID:06}'
    view_id_str = f'{VIEW_ID:06}'
    rgb_full_path = MEGAPOSE_DATA_DIR / f'bop_datasets/ycbv/test/{scene_id_str}/rgb/{view_id_str}.png'
    scene_cam_full_path = MEGAPOSE_DATA_DIR / f'bop_datasets/ycbv/test/{scene_id_str}/scene_camera.json'
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

    assert 'obj_000006' in poses.keys()
    print("objects are detected")

    # rgb[:] = 42
    # poses = localizer.predict(rgb, K, n_coarse=1, n_refiner=3)

    # assert len(poses.keys()) == 0
    # print("objects are detected")




