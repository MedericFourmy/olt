import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image

from olt.utils import Kres2intrinsics


# BOP data READER
# - for each new scene, load all images in memory

"""
BOPDataReader:
- for each new scene, load all images in memory

"""
class BOPDataReader:

    def __init__(self, bop_datasets_dir: Path, ds_split='test') -> None:
        self.bop_datasets_dir = bop_datasets_dir
        self.scenes_dir = self.bop_datasets_dir / ds_split 
        
        assert(self.bop_datasets_dir.exists())
        assert(self.scenes_dir.exists())

        # NOT USED
        # self.test_targets_path = self.bop_datasets_dir / 'test_targets_bop19.json'
        # assert(self.test_targets_path.exists())
        # self.df_targets = pd.read_json(self.test_targets_path.as_posix(), orient='records')
        

        # get available scene ids
        self.scene_ids_str = list((d.name for d in self.scenes_dir.iterdir()))
        self.scene_ids = [int(sid) for sid in self.scene_ids_str]
        
        self.img_lst = None
        # Make assumption of constant camera params for one scene
        # Ok for YCBV, not for T-less
        self.K = None
        self.width, self.heigth = None, None

    def load_scene(self, scene_id, nb_img_loaded=-1):
        if isinstance(scene_id, int):
            scene_id = f'{scene_id:06}'    
        assert(scene_id in self.scene_ids_str)
        
        scene_path = self.scenes_dir / scene_id

        # Extract camera data
        # For some datasets (e.g. ycbv) camera matrix can vary for image to image due to recroping
        scene_camera_path = scene_path / 'scene_camera.json'
        df_camera = pd.read_json(scene_camera_path.as_posix(), orient='index')
        # self.K_lst = [np.array(kvec).reshape((3,3)) for kvec in df_camera.cam_K.to_list()]
        kvec = df_camera.iloc[0].cam_K
        self.K = np.array(kvec).reshape((3,3))

        # Extract scene images
        rgb_imgs_dir = scene_path / 'rgb'
        rgb_imgs_paths = [p.as_posix() for p in sorted(rgb_imgs_dir.iterdir())]

        # Limit nb of imgs for quicker tests
        rgb_imgs_paths = rgb_imgs_paths[:nb_img_loaded]
        # self.K_lst = self.K_lst[:nb]

        # open image files, lazy operation -> do not put the image in memory
        self.img_lst = [Image.open(rgb_path) for rgb_path in rgb_imgs_paths]
        assert(len(self.img_lst) > 0)

        # Recover camera resolution from images (assuming equal resolutions for entire scene)
        self.width, self.heigth = self.img_lst[0].width, self.img_lst[0].height

        self.img_index = -1

    def get_img(self, img_index):
        if 0 < img_index > len(self.img_lst):
            raise ValueError(f'Index {img_index} not in loaded images')
        return np.array(self.img_lst[self.img_index], dtype=np.uint8)

    def get_next_img(self):
        self.img_index += 1
        print(self.img_lst[self.img_index].filename)
        return self.img_index, self.get_img(self.img_index)
    
    def get_intrinsics(self):
        return Kres2intrinsics(self.K, self.width, self.heigth)