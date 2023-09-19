import os
import numpy as np
import pandas as pd
from pathlib import Path

from happypose.toolbox.datasets.bop_scene_dataset import BOPDataset
from happypose.toolbox.datasets.scene_dataset import ObservationInfos

from olt.config import BOP_DS_DIRS
from olt.utils import Kres2intrinsics, obj_id2name

import bop_toolkit_lib
import subprocess




class BOPDatasetReader:
    """
    sid: scene_id
    vid: vid
    """

    def __init__(self, ds_name: str, ds_split='test', enforce_targets=False, load_depth=False):
        self.ds_name = ds_name

        self.bs = BOPDataset(
            ds_dir=BOP_DS_DIRS[ds_name],
            label_format=ds_name + "-{label}",
            split=ds_split,
            load_depth=load_depth
        )

        targets_filename = BOP_DS_DIRS[ds_name] / "test_targets_bop19.json"
        if targets_filename.exists():
            self.df_targets = pd.read_json(targets_filename)
        else:
            assert not enforce_targets, f'{targets_filename} is required and not found'
            self.df_targets = None

        # keys: sids
        # values: pd.Index of vids
        self.map_sids_vids = self.bs.frame_index.groupby('scene_id').view_id.apply(list).to_dict()

    def get_targets(self, sid, vid):
        return self.df_targets[(self.df_targets['scene_id'] == sid) & (self.df_targets['im_id'] == vid)]

    def get_object_ids_in_scene(self, sid):
        return self.df_targets[self.df_targets['scene_id'] == sid].obj_id.unique()

    def get_object_names_in_scene(self, sid):
        return [obj_id2name(object_id) for object_id in self.get_object_ids_in_scene(sid)]

    def check_if_in_targets(self, sid, vid):
        if self.df_targets is None:
            # No target associated to the dataset
            return True
        return len(self.get_targets(sid, vid)) > 0
    
    def get_obs(self, sid, vid):
        infos = ObservationInfos(sid, vid)
        return self.bs._load_scene_observation(infos)

    def get_rgb(self, sid, vid):
        return self.get_obs(sid, vid).rgb

    def get_depth(self, sid, vid):
        return self.get_obs(sid, vid).depth

    def get_Kres(self, sid, vid):
        obs = self.get_obs(sid, vid)
        K = obs.camera_data.K
        height, width = obs.camera_data.resolution

        return K, height, width
    
    def get_intrinsics(self, sid, vid):
        K, height, width = self.get_Kres(sid, vid)
        return Kres2intrinsics(K, width, height)

    def predict_gt(self, sid: int, vid: int):
        # Mimics Localizer predict
        sid_str = f'{sid:06}'
        vid_str = str(vid)

        # keys: int, object ids
        # values: np.ndarray, 4x4 T_m2c transformations
        preds = {}
        gt = self.bs.annotations[sid_str]['scene_gt'][vid_str]
        for obj_pred in gt:
            obj_name = obj_id2name(obj_pred['obj_id'])

            T_m2c = np.eye(4)
            T_m2c[:3,:3] = np.array(obj_pred['cam_R_m2c']).reshape((3,3))
            T_m2c[:3, 3] = np.array(obj_pred['cam_t_m2c']) * 0.001   # mm->m

            preds[obj_name] = T_m2c

        return preds


def append_result(results, scene_id, obj_id, view_id, score, TCO, dt):

    t = TCO[:3, -1] * 1e3  # m -> mm conversion
    R = TCO[:3, :3]
    pred = dict(
        scene_id=scene_id,
        im_id=view_id,
        obj_id=obj_id,
        score=score,
        t=t,
        R=R,
        time=dt,
    )

    results.append(pred)

def run_bop_evaluation(filename, results_dir_name, evaluations_dir_name):
    myenv = os.environ.copy()

    BOP_TOOLKIT_DIR = Path(bop_toolkit_lib.__file__).parent.parent
    POSE_EVAL_SCRIPT_PATH = BOP_TOOLKIT_DIR / "scripts/eval_bop19_pose.py"

    # Put results in current directory
    root_dir = Path(os.getcwd())
    results_path = root_dir / results_dir_name
    eval_path = root_dir / evaluations_dir_name

    renderer_type = 'vispy'  # other options: 'cpp', 'python'
    cmd = [
        "python",
        str(POSE_EVAL_SCRIPT_PATH),
        "--result_filenames",
        filename,
        "--results_path",
        results_path,
        "--renderer_type",
        renderer_type,
        "--eval_path",
        eval_path,
    ]
    # subprocess.call(cmd, env=myenv, cwd=BOP_TOOLKIT_DIR.as_posix())
    subprocess.call(cmd, env=myenv, cwd=os.getcwd())
