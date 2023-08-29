import os
import numpy as np
import pandas as pd
from pathlib import Path

from happypose.toolbox.datasets.bop_scene_dataset import BOPDataset
from happypose.toolbox.datasets.scene_dataset import ObservationInfos
from happypose.pose_estimators.megapose.src.megapose.evaluation.bop import run_evaluation
from happypose.pose_estimators.megapose.src.megapose.evaluation.eval_config import BOPEvalConfig

from olt.config import BOP_DS_DIRS
from olt.utils import Kres2intrinsics, obj_id2name

import bop_toolkit_lib
import subprocess




class BOPDatasetReader:
    """
    sid: scene_id
    vid: vid
    """

    def __init__(self, ds_name: str, ds_split='test', enforce_targets=False):
        self.ds_name = ds_name

        self.bs = BOPDataset(
            ds_dir=BOP_DS_DIRS[ds_name],
            label_format=ds_name + "-{label}",
            split=ds_split
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

    def get_cam_data(self, sid, vid):
        infos = ObservationInfos(sid, vid)
        obs = self.bs._load_scene_observation(infos)
        K = obs.camera_data.K
        height, width = obs.camera_data.resolution

        return K, height, width
    
    def check_if_in_bop19_targets(self, scene_id, view_id):
        if self.df_targets is None:
            # No target associated to the dataset
            return True
        return len(self.df_targets[(self.df_targets['scene_id'] == scene_id) & (self.df_targets['im_id'] == view_id)]) > 0
    
    def get_img(self, sid, vid):
        infos = ObservationInfos(sid, vid)
        obs = self.bs._load_scene_observation(infos)

        return obs.rgb

    def get_intrinsics(self, sid, vid):
        K, height, width = self.get_cam_data(sid, vid)
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




def run_bop_evaluation(filename, eval_dir=''):
    myenv = os.environ.copy()

    BOP_TOOLKIT_DIR = Path(bop_toolkit_lib.__file__).parent.parent
    POSE_EVAL_SCRIPT_PATH = BOP_TOOLKIT_DIR / "scripts/eval_bop19_pose.py"

    # 
    root_dir = os.getcwd()
    results_path = os.path.join(root_dir, 'results')
    eval_path = 'evals'

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


# # Folder with the BOP datasets.
# datasets_path = str(MEGAPOSE_DATA_DIR / 'bop_datasets')

# # Folder with pose results to be evaluated.
# results_path = str(MEGAPOSE_DATA_DIR / 'results')

# # Folder for the calculated pose errors and performance scores.
# eval_path = str(MEGAPOSE_DATA_DIR / 'bop_eval_outputs')


# parser = argparse.ArgumentParser()
# parser.add_argument('--renderer_type', default=p['renderer_type'])
# parser.add_argument('--result_filenames',
#                     default=','.join(p['result_filenames']),
#                     help='Comma-separated names of files with results.')
# parser.add_argument('--results_path', default=p['results_path'])
# parser.add_argument('--eval_path', default=p['eval_path'])
# parser.add_argument('--targets_filename', default=p['targets_filename'])
# args = parser.parse_args()



if __name__ == '__main__':
    from bop_toolkit_lib import inout  # noqa

    preds = [] 
    scene_id = 1 
    obj_id = 3
    view_id = 0 
    score = 0.6
    TCO = np.eye(4) 
    dt = 1.0

    append_result(preds, scene_id, obj_id, view_id, score, TCO, dt)
    view_id+=1
    append_result(preds, scene_id, obj_id, view_id, score, TCO, dt)
    view_id+=1
    append_result(preds, scene_id, obj_id, view_id, score, TCO, dt)
    view_id+=1

    inout.save_bop_results('results_test.csv', preds)

