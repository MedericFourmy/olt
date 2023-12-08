import numpy as np
import pandas as pd
import torch

from happypose.toolbox.inference.types import ObservationTensor
from happypose.pose_estimators.cosypose.cosypose.utils.cosypose_wrapper import CosyPoseWrapper
from happypose.pose_estimators.cosypose.cosypose.utils.tensor_collection import TensorCollection as tc
from happypose.pose_estimators.cosypose.cosypose.config import LOCAL_DATA_DIR

from olt.config import LocalizerConfig
from olt.utils import obj_label2name


def dist_objects(T1, T2):
    # return np.linalg.norm(T1[:3,3] - T2[:3,3])
    return np.linalg.norm(T1[:3,3] - T2[:3,3])


class Localizer:

    def __init__(self, obj_dataset, cfg: LocalizerConfig) -> None:
        self.obj_dataset = obj_dataset
        self.cfg = cfg
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Cosypose
        cosy_wrapper = CosyPoseWrapper(dataset_name=self.obj_dataset, 
                                       n_workers=cfg.n_workers, 
                                       )
        self.pose_estimator = cosy_wrapper.pose_predictor

        # Megapose (TODO)
    


    def get_cosy_predictions(self, rgb, K, n_coarse=None, n_refiner=None, TCO_init=None, run_detector=True):
        n_coarse = self.cfg.n_coarse if n_coarse is None else n_coarse
        n_refiner = self.cfg.n_refiner if n_refiner is None else n_refiner
        observation = ObservationTensor.from_numpy(rgb, None, K)
        if self.device.type == 'cuda':
            observation.cuda()

        coarse_estimates = None if run_detector else 'NotNone'  # this argument has to be None 

        # labels returned by cosypose are the same as the "local_data/urdfs/ycbv" ones
        # -> obj_000010 == banana
        # Exception handling: if no object detected in the image, cosypose currently throws an AttributeError error
        try:
            data_TCO, extra_data = self.pose_estimator.run_inference_pipeline(observation,
                                                                              data_TCO_init=TCO_init,
                                                                              run_detector=run_detector,
                                                                              n_coarse_iterations=n_coarse, 
                                                                              n_refiner_iterations=n_refiner,
                                                                              detection_th=self.cfg.detector_threshold,
                                                                              coarse_estimates=coarse_estimates)
        except AttributeError as e:
            return None, None

        return data_TCO, extra_data

    def track(self, rgb, K, TCO_init, n_refiner=1):
        data_TCO, extra_data = self.get_cosy_predictions(rgb, K, n_coarse=0, n_refiner=n_refiner, TCO_init=TCO_init, run_detector=False)
        
        return self.cosypreds2posesscores(data_TCO, extra_data)
        
    @staticmethod
    def cosypreds2posesscores(data_TCO=None, extra_data=None):
        """
        Turn the raw cosypose predictions into object poses and scores usable by other systems.
        """


        if data_TCO is None:
            return {}, {}

        # Send all poses to cpu to be able to process them
        poses = data_TCO.poses.cpu()

        preds = {}
        scores = {}
        # returns best detection of each image type 
        # EXCEPT those are the ycbv clamps
        # clamps_obj_name = ['obj_000019', 'obj_000020']
        # MIN_DIST = 0.0

        # get indexes sorted by decreasing scores
        score_list = list(data_TCO.infos['score'])
        scores_decreasing_indexes = sorted(range(len(score_list)), key=lambda k: score_list[k], reverse=True)
        for i in scores_decreasing_indexes:
            obj_label = data_TCO.infos['label'][i]
            obj_name = obj_label2name(obj_label, 'ycbv')
            score = data_TCO.infos['score'][i]
            # print(score)

            # if obj_name in clamps_obj_name:
            #     print('!!!! one_clamp_with_higher_score_already_detected')
            #     clamps_already_detected = [name for name in preds if name in clamps_obj_name]
            #     if len(clamps_already_detected) > 0: 
            #         dist = dist_objects(poses[i].numpy(), preds[clamps_already_detected[0]])
            #         print('dist:', dist)
            #         if dist < MIN_DIST:
            #             print('REMOVE!!!!!!!', clamps_already_detected[0])
            #             continue


            # if obj_name in scores:
            #     print(obj_name)
            #     if score > scores[obj_name]:

            #         scores[obj_name] = score
            #         preds[obj_name] = poses[i].numpy()

            if obj_name not in scores:
                scores[obj_name] = score
                preds[obj_name] = poses[i].numpy()
        
        assert len(preds) == len(scores)
        return preds, scores


    def predict(self, rgb, K, n_coarse=None, n_refiner=None):

        data_TCO, extra_data = self.get_cosy_predictions(rgb, K, n_coarse=n_coarse, n_refiner=n_refiner)
        return self.cosypreds2posesscores(data_TCO, extra_data)
