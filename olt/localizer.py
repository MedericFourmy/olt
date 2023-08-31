import torch

from happypose.toolbox.inference.types import ObservationTensor
from happypose.pose_estimators.cosypose.cosypose.utils.cosypose_wrapper import CosyPoseWrapper
from happypose.pose_estimators.cosypose.cosypose.config import LOCAL_DATA_DIR

from olt.config import LocalizerConfig
from olt.utils import obj_label2name





class Localizer:

    def __init__(self, obj_dataset, cfg: LocalizerConfig) -> None:
        self.obj_dataset = obj_dataset
        self.detector_threshold = cfg.detector_threshold
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Cosypose
        cosy_wrapper = CosyPoseWrapper(dataset_name=self.obj_dataset, n_workers=cfg.n_workers)
        self.pose_estimator = cosy_wrapper.pose_predictor

        # Megapose (TODO)


    def predict(self, rgb, K, n_coarse=1, n_refiner=3):
        observation = ObservationTensor.from_numpy(rgb, None, K)
        if self.device.type == 'cuda':
            observation.cuda()

        # labels returned by cosypose are the same as the "local_data/urdfs/ycbv" ones
        # -> obj_000010 == banana
        predictions, _ = self.pose_estimator.run_inference_pipeline(observation,
                                                                    run_detector=True,
                                                                    n_coarse_iterations=n_coarse, 
                                                                    n_refiner_iterations=n_refiner)
        # Send all poses to cpu to be able to process them
        poses = predictions.poses.cpu()
        preds = {}
        # print('# detections: ', len(predictions.infos))
        for i in range(len(predictions)):
            # print('Det score/label: ', predictions.infos['score'][i], predictions.infos['label'][i])
            if predictions.infos['score'][i] > self.detector_threshold:
                obj_label = predictions.infos['label'][i]
                obj_name = obj_label2name(obj_label, 'ycbv')
                preds[obj_name] = poses[i].numpy() 
            else:
                print('-> Skip detection: ', predictions.infos['label'][i])
        
        return preds
