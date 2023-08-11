
from happypose.pose_estimators.cosypose.cosypose.utils.cosypose_wrapper import CosyPoseWrapper
from happypose.toolbox.inference.types import ObservationTensor
from happypose.pose_estimators.cosypose.cosypose.config import LOCAL_DATA_DIR


class Localizer:

    def __init__(self, obj_dataset, threshold_detector: float = 0.0) -> None:
        self.obj_dataset = obj_dataset
        self.threshold_detector = threshold_detector

        # Cosypose
        cosy_wrapper = CosyPoseWrapper(dataset_name=self.obj_dataset, n_workers=8)
        self.pose_estimator = cosy_wrapper.pose_predictor

        # Megapose (TODO)


    def predict(self, rgb, K, n_coarse=1, n_refiner=3):
        observation = ObservationTensor.from_numpy(rgb, None, K)

        # labels returned by cosypose are the same as the "local_data/urdfs/ycbv" ones
        # -> obj_000010 == banana
        predictions, _ = self.pose_estimator.run_inference_pipeline(observation,
                                                                    run_detector=True,
                                                                    n_coarse_iterations=n_coarse, 
                                                                    n_refiner_iterations=n_refiner)
        
        poses = {}
        for i in range(len(predictions)):
            if predictions.infos['score'][i] > self.threshold_detector:
                obj_id = predictions.infos['label'][i]
                poses[obj_id] = predictions.poses[i].numpy() 
            else:
                print('Skip detection: ', predictions.infos['label'][i])
        
        return poses
