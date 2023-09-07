import torch

from happypose.toolbox.inference.types import ObservationTensor
from happypose.pose_estimators.cosypose.cosypose.utils.cosypose_wrapper import CosyPoseWrapper
from happypose.pose_estimators.cosypose.cosypose.config import LOCAL_DATA_DIR

from olt.config import LocalizerConfig
from olt.utils import obj_label2name





class Localizer:

    def __init__(self, obj_dataset, cfg: LocalizerConfig) -> None:
        self.obj_dataset = obj_dataset
        self.cfg = cfg
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Cosypose
        cosy_wrapper = CosyPoseWrapper(dataset_name=self.obj_dataset, n_workers=cfg.n_workers, renderer_name=cfg.renderer_name)
        self.pose_estimator = cosy_wrapper.pose_predictor

        # Megapose (TODO)

    def predict(self, rgb, K, n_coarse=1, n_refiner=3):
        observation = ObservationTensor.from_numpy(rgb, None, K)
        if self.device.type == 'cuda':
            observation.cuda()

        # labels returned by cosypose are the same as the "local_data/urdfs/ycbv" ones
        # -> obj_000010 == banana
        # Exception handling: if no object detected in the image, cosypose currently throws an AttributeError error
        try:
            predictions, _ = self.pose_estimator.run_inference_pipeline(observation,
                                                                        run_detector=True,
                                                                        n_coarse_iterations=n_coarse, 
                                                                        n_refiner_iterations=n_refiner,
                                                                        detection_th=self.cfg.detector_threshold)
        except AttributeError as e:
            return {}

        # Send all poses to cpu to be able to process them
        poses = predictions.poses.cpu()

        preds = {}
        for i in range(len(predictions)):
            obj_label = predictions.infos['label'][i]
            obj_name = obj_label2name(obj_label, 'ycbv')
            preds[obj_name] = poses[i].numpy() 
        
        return preds
