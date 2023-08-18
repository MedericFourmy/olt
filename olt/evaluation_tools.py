import numpy as np

from happypose.toolbox.datasets.bop_scene_dataset import BOPDataset
from happypose.toolbox.datasets.scene_dataset import ObservationInfos
from happypose.pose_estimators.megapose.src.megapose.evaluation.bop import run_evaluation
from happypose.pose_estimators.megapose.src.megapose.evaluation.eval_config import BOPEvalConfig

from olt.config import BOP_DS_DIRS
from olt.utils import Kres2intrinsics



class BOPDatasetReader:
    """
    sid: scene_id
    vid: vid
    """

    def __init__(self, ds_name: str, ds_split='test'):
        self.ds_name = ds_name

        self.bs = BOPDataset(
            ds_dir=BOP_DS_DIRS[ds_name],
            label_format=ds_name + "-{label}",
            split=ds_split
        )

        # keys: sids
        # values: pd.Index of vids
        self.map_sids_vids = self.bs.frame_index.groupby('scene_id').view_id.apply(list).to_dict()

    def get_cam_data(self, sid, vid):
        infos = ObservationInfos(sid, vid)
        obs = self.bs._load_scene_observation(infos)
        K = obs.camera_data.K
        height, width = obs.camera_data.resolution

        return K, height, width
    
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
            obj_id = obj_pred['obj_id']
            obj_id = f'{self.ds_name}-obj_{obj_id:06}'

            T_m2c = np.eye(4)
            T_m2c[:3,:3] = np.array(obj_pred['cam_R_m2c']).reshape((3,3))
            T_m2c[:3, 3] = np.array(obj_pred['cam_t_m2c']) * 0.001   # mm->m

            preds[obj_id] = T_m2c

        return preds
